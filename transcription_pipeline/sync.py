"""Optional sync of finished transcripts with any rclone remote (S3, Google Drive, Dropbox, SFTP ...).

Lets several machines (e.g. Kaggle + Colab + a laptop) share one archive: each pushes its new files and
pulls the list of ids finished elsewhere into the skip file read by `WorkQueue`. Credentials stay in the
user's own rclone config (RCLONE_CONFIG or ~/.config/rclone/rclone.conf), never in this repository.
"""
import os
import shutil
import subprocess
import threading
import time

from .formats import EXTS


def rclone_bin():
    return shutil.which("rclone")


def push(out_dir, remote, log_names=()):
    rc = rclone_bin()
    cmd = [rc, "copy", os.path.join(out_dir, "transcripts"), remote.rstrip("/") + "/transcripts", "--ignore-existing",
           "--transfers", "8"]
    for ext in EXTS:
        cmd += ["--include", "*" + ext]
    subprocess.run(cmd, capture_output=True)
    for name in log_names:
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            subprocess.run([rc, "copyto", p, remote.rstrip("/") + "/logs/" + name], capture_output=True)


def remote_done(remote):
    r = subprocess.run([rclone_bin(), "lsf", remote.rstrip("/") + "/transcripts", "--include", "*.json"],
                       capture_output=True, text=True)
    return {n[:-5] for n in r.stdout.split()}


def update_skip_file(skip_file, ids):
    with open(skip_file + ".tmp", "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(ids)))
    os.replace(skip_file + ".tmp", skip_file)


def sync_once(out_dir, remote, skip_file, log_names=()):
    if not rclone_bin():
        return -1
    push(out_dir, remote, log_names)
    ids = remote_done(remote)
    update_skip_file(skip_file, ids)
    return len(ids)


def start_background(out_dir, remote, skip_file, every_s=120, log_names=()):
    def loop():
        while True:
            try:
                sync_once(out_dir, remote, skip_file, log_names)
            except Exception as ex:   # sync problems must not stop transcription
                print("sync:", ex, flush=True)
            time.sleep(every_s)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t
