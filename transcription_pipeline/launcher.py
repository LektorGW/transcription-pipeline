"""Multi-GPU launcher: N worker processes per GPU, progress reports, a hard deadline for time-limited
sessions (Kaggle and Colab kill notebooks after a fixed number of hours) and optional rclone sync."""
import glob
import json
import os
import subprocess
import sys
import time

from . import sync
from .jobs import read_log, reset_claims


def gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except (OSError, subprocess.CalledProcessError):
        return 0


def cuda_lib_path():
    """CTranslate2 needs cuBLAS/cuDNN; on Kaggle/Colab they come from the nvidia-* pip wheels and are not
    on the loader path by default."""
    libs = []
    for mod in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            m = __import__(mod, fromlist=["_"])
            libs.append(os.path.join(list(m.__path__)[0], "lib"))
        except ImportError:
            pass
    return ":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")])


def report(out_dir, log_name, total, t_start):
    recs = [r for r in read_log(os.path.join(out_dir, log_name)) if r.get("time", 0) >= t_start]
    done = [r for r in recs if r.get("event") == "done"]
    failed = [r for r in recs if r.get("event") == "failed"]
    audio_h = sum(r["audio_s"] for r in done) / 3600
    elapsed_h = max((time.time() - t_start) / 3600, 1e-6)
    left = max(0, total - len(done) - len(failed))
    eta = left * elapsed_h / len(done) if done else 0
    on_disk = len(glob.glob(os.path.join(out_dir, "transcripts", "*.json")))
    print(time.strftime("%H:%M"), f"| this session: {len(done)} episodes, {audio_h:.1f} h audio "
          f"({audio_h / elapsed_h:.0f}x real time) | failed: {len(failed)} | on disk: {on_disk} | "
          f"left: {left} (~{eta:.1f} h)", flush=True)


def launch(cfg, config_path, episodes_path, workers_per_gpu=1, max_hours=0, gpus=None, reset=False, tag="gpu",
           shards="", shard_count=1):
    out, state = cfg["output_dir"], cfg["state_dir"]
    os.makedirs(state, exist_ok=True)
    if reset:
        print("removed stale claims:", reset_claims(state))
    n_gpu = gpus if gpus is not None else gpu_count()
    if n_gpu == 0:
        raise SystemExit("no GPU found (use the `local` command with device=cpu, or the `groq` backend)")
    t_start = time.time()
    deadline = t_start + max_hours * 3600 if max_hours else 0
    log_name = f"run_log_{tag}.jsonl"
    skip_file = os.path.join(state, "done_elsewhere.txt")
    if cfg["sync"]["remote"]:
        print("synced ids:", sync.sync_once(out, cfg["sync"]["remote"], skip_file, [log_name]))
        sync.start_background(out, cfg["sync"]["remote"], skip_file, cfg["sync"]["every_s"], [log_name])

    with open(episodes_path, encoding="utf-8") as f:
        total = len(json.load(f))
    procs = []
    for g in range(n_gpu):
        for w in range(workers_per_gpu):
            name = f"{tag}{g}-w{w}"
            cmd = [sys.executable, "-m", "transcription_pipeline", "local", "--episodes", episodes_path,
                   "--worker", name, "--deadline", str(deadline), "--skip-file", skip_file, "--log-name", log_name]
            if config_path:
                cmd += ["--config", config_path]
            if shards:
                cmd += ["--shards", shards, "--shard-count", str(shard_count)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1", LD_LIBRARY_PATH=cuda_lib_path())
            logf = open(os.path.join(state, f"{name}.log"), "w", encoding="utf-8")
            procs.append((name, subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT), logf))
            time.sleep(20)   # stagger model loading to avoid RAM spikes
    last = 0.0
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(30)
        if time.time() - last > 300:
            last = time.time()
            report(out, log_name, total, t_start)
    report(out, log_name, total, t_start)
    for name, p, logf in procs:
        logf.close()
        if p.returncode:
            with open(os.path.join(state, f"{name}.log"), encoding="utf-8") as f:
                tail = f.readlines()[-15:]
            print(f"!! {name} exited with code {p.returncode}:\n" + "".join(tail))
    if cfg["sync"]["remote"]:
        sync.sync_once(out, cfg["sync"]["remote"], skip_file, [log_name])
