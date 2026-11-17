"""Command line interface: python -m transcription_pipeline <command> [options]"""
import argparse
import os
import sys

from . import episodes as E
from .config import load_config, load_dotenv


def _common(p, episodes=True):
    p.add_argument("--config", default=os.environ.get("TP_CONFIG", ""), help="config.yaml / config.json")
    if episodes:
        p.add_argument("--episodes", default="episodes.json", help="episode list from the `episodes` command")
        p.add_argument("--shards", default="", help="e.g. 0,1: take these shards first (multi-machine runs)")
        p.add_argument("--shard-count", type=int, default=1)
        p.add_argument("--no-help-others", action="store_true", help="stop after own shards")
        p.add_argument("--worker", default="")
        p.add_argument("--max-episodes", type=int, default=0)
        p.add_argument("--skip-file", default="", help="ids finished elsewhere, one per line (re-read continuously)")


def _ordered(a):
    eps = E.load(a.episodes)
    shards = {int(x) for x in a.shards.split(",") if x.strip()} if a.shards else None
    return E.work_order(eps, shards, a.shard_count, not a.no_help_others)


def main(argv=None):
    load_dotenv()
    ap = argparse.ArgumentParser(prog="transcription_pipeline",
                                 description="Podcast / audio archive transcription pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("episodes", help="build episodes.json from an RSS feed, a folder or a URL list")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--rss", help="RSS feed URL")
    src.add_argument("--rss-file", help="RSS feed saved to a local file")
    src.add_argument("--dir", help="folder with audio files")
    src.add_argument("--urls", help="text file with one audio URL per line")
    p.add_argument("-o", "--output", default="episodes.json")
    p.add_argument("--since", default="", help="YYYY-MM-DD")
    p.add_argument("--until", default="", help="YYYY-MM-DD")
    p.add_argument("--limit", type=int, default=0, help="keep only the newest N episodes")

    p = sub.add_parser("local", help="transcribe with faster-whisper (GPU or CPU)")
    _common(p)
    p.add_argument("--model", default="", help="override local.model (e.g. large-v3, medium.en, small)")
    p.add_argument("--device", default="", help="cuda / cpu")
    p.add_argument("--compute-type", default="", help="float16 / int8 / int8_float16")
    p.add_argument("--deadline", type=float, default=0, help="unix time after which no new episode is started")
    p.add_argument("--limit-seconds", type=int, default=0, help="transcribe only the first N seconds (testing)")
    p.add_argument("--log-name", default="run_log.jsonl")

    p = sub.add_parser("groq", help="transcribe through the Groq Whisper API (needs GROQ_API_KEY)")
    _common(p)
    p.add_argument("--model", default="", help="whisper-large-v3 or whisper-large-v3-turbo")
    p.add_argument("--log-name", default="run_log_groq.jsonl")

    p = sub.add_parser("launch", help="run several local workers per GPU with progress reports")
    _common(p, episodes=False)
    p.add_argument("--episodes", default="episodes.json")
    p.add_argument("--workers-per-gpu", type=int, default=1)
    p.add_argument("--max-hours", type=float, default=0, help="stop taking new episodes after this many hours")
    p.add_argument("--gpus", type=int, default=None, help="override GPU detection")
    p.add_argument("--reset-claims", action="store_true", help="clear stale locks before starting")
    p.add_argument("--shards", default="", help="passed to every worker, see `local --help`")
    p.add_argument("--shard-count", type=int, default=1)

    p = sub.add_parser("report", help="episodes.csv/.xlsx, qa_flags.csv and transcripts.zip")
    _common(p, episodes=False)

    p = sub.add_parser("site", help="static archive site with full-text search")
    _common(p, episodes=False)
    p.add_argument("-o", "--out", default="site")
    p.add_argument("--audio-dir", default="", help="copy local audio files into the site")

    p = sub.add_parser("reset-claims", help="remove lock files left by crashed workers")
    _common(p, episodes=False)

    a = ap.parse_args(argv)
    cfg = load_config(getattr(a, "config", "") or None)

    if a.cmd == "episodes":
        if a.rss:
            import requests
            r = requests.get(a.rss, headers={"User-Agent": cfg["download"]["user_agent"]}, timeout=60)
            r.raise_for_status()
            eps = E.from_rss(r.content)
        elif a.rss_file:
            with open(a.rss_file, "rb") as f:
                eps = E.from_rss(f.read())
        elif a.dir:
            eps = E.from_directory(a.dir)
        else:
            with open(a.urls, encoding="utf-8") as f:
                eps = E.from_url_list(f)
        eps = E.filter_dates(eps, a.since, a.until)
        eps.sort(key=lambda e: (e["date"], e["id"]), reverse=True)
        if a.limit:
            eps = eps[:a.limit]
        E.save(eps, a.output)
        print(f"{len(eps)} episodes -> {a.output}")

    elif a.cmd == "local":
        for k, attr in (("model", "model"), ("device", "device"), ("compute_type", "compute_type")):
            if getattr(a, attr):
                cfg["local"][k] = getattr(a, attr)
        from .local_backend import run_worker
        run_worker(cfg, _ordered(a), a.worker or "local", a.max_episodes, a.deadline, a.limit_seconds,
                   a.skip_file, a.log_name)

    elif a.cmd == "groq":
        from .groq_backend import run_worker
        run_worker(cfg, _ordered(a), a.worker or "groq", a.max_episodes, a.skip_file, a.log_name, a.model or None)

    elif a.cmd == "launch":
        from .launcher import launch
        launch(cfg, a.config, a.episodes, a.workers_per_gpu, a.max_hours, a.gpus, a.reset_claims,
               shards=a.shards, shard_count=a.shard_count)

    elif a.cmd == "report":
        from .report import build
        build(cfg["output_dir"])

    elif a.cmd == "site":
        from .site import build
        build(cfg["output_dir"], a.out, cfg["site"], a.audio_dir)

    elif a.cmd == "reset-claims":
        from .jobs import reset_claims
        print("removed:", reset_claims(cfg["state_dir"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
