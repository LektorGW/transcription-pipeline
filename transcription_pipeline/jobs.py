"""Work distribution between parallel workers, and resumability.

* An episode is *done* when `<output>/transcripts/<id>.json` exists (it is written last and atomically).
* A worker *claims* an episode by creating `<state>/claims/<id>.lock` with O_CREAT | O_EXCL. The OS
  guarantees only one process succeeds, so any number of workers (several GPUs, several processes per
  GPU, a CPU/API worker on a laptop) can share one episode list without coordination.
* A skip file (one id per line) lists episodes finished elsewhere, e.g. on another machine; it is re-read
  before every claim, so a background sync can update it while workers run.
"""
import json
import os
import time
import traceback

from .formats import is_done


class WorkQueue:
    def __init__(self, episodes, out_dir, state_dir, worker, skip_file=""):
        self.episodes = episodes
        self.out_dir = out_dir
        self.worker = worker
        self.skip_file = skip_file
        self.claims = os.path.join(state_dir, "claims")
        self.failed = os.path.join(state_dir, "failed")
        os.makedirs(self.claims, exist_ok=True)
        os.makedirs(self.failed, exist_ok=True)

    def done_elsewhere(self):
        if not self.skip_file:
            return set()
        try:
            with open(self.skip_file, encoding="utf-8") as f:
                return set(f.read().split())
        except OSError:
            return set()

    def pending(self):
        remote = self.done_elsewhere()
        return [e for e in self.episodes if e["id"] not in remote and not is_done(self.out_dir, e["id"])]

    def claim(self):
        for ep in self.pending():
            try:
                fd = os.open(os.path.join(self.claims, ep["id"] + ".lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.write(fd, f"{self.worker} {time.time():.0f}".encode())
            os.close(fd)
            return ep
        return None

    def mark_failed(self, ep):
        with open(os.path.join(self.failed, ep["id"] + ".txt"), "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())


def reset_claims(state_dir):
    """Remove stale locks (e.g. after a crash). Only call this when no worker is running."""
    d = os.path.join(state_dir, "claims")
    n = 0
    if os.path.isdir(d):
        for name in os.listdir(d):
            os.remove(os.path.join(d, name))
            n += 1
    return n


class RunLog:
    """Append-only JSONL log shared by workers: one line per event (ready / done / failed / exit)."""

    def __init__(self, path, worker):
        self.path, self.worker = path, worker
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def write(self, **rec):
        rec.update(worker=self.worker, time=time.time())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_log(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)
