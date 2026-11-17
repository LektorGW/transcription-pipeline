"""Groq backend: Whisper large-v3 (or -turbo) through Groq's OpenAI-compatible audio API.

Useful without a GPU: a laptop process can work through the same episode list as GPU workers, because
claims and output format are shared. Audio is re-encoded to mono 16 kHz 32 kb/s to stay under the upload
limit. HTTP 429 (rate limit) is handled by sleeping for the `retry-after` the server asks for.
"""
import os
import tempfile
import time

from . import audio as A
from .cleanup import compile_replacements, hallucination_regex
from .formats import save_episode
from .jobs import RunLog, WorkQueue, log
from .repair import fill_regions, find_regions


def api_key():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise SystemExit("GROQ_API_KEY is not set (export it or put it in .env, see .env.example)")
    return key


class GroqClient:
    def __init__(self, key, model="whisper-large-v3", api_url="https://api.groq.com/openai/v1/audio/transcriptions",
                 language="en", prompt="", max_wait_s=3600, session=None):
        self.key, self.model, self.api_url = key, model, api_url
        self.language, self.prompt, self.max_wait_s = language, prompt, max_wait_s
        self.session = session

    def transcribe(self, path):
        http = self.session
        if http is None:
            import requests
            http = requests
        data = [("model", self.model), ("response_format", "verbose_json"), ("language", self.language),
                ("temperature", "0"), ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment")]
        if self.prompt:
            data.append(("prompt", self.prompt[:800]))
        while True:
            with open(path, "rb") as f:
                r = http.post(self.api_url, headers={"Authorization": f"Bearer {self.key}"}, timeout=600,
                              files={"file": (os.path.basename(path), f, "audio/mpeg")}, data=data)
            if r.status_code == 429:
                wait = float(r.headers.get("retry-after", 60))
                log(f"rate limited, waiting {wait / 60:.1f} min")
                time.sleep(min(max(wait, 30), self.max_wait_s))
                continue
            r.raise_for_status()
            return r.json()


def to_segments(resp, offset=0.0):
    """Convert a verbose_json response to internal segments, attaching word timings to segments.
    A word on the boundary of two segments is kept only in the first one."""
    words = resp.get("words") or []
    segs, seen = [], set()
    for s in resp.get("segments") or []:
        ws = []
        for w in words:
            if s["start"] - 0.05 <= w["start"] < s["end"] + 0.05:
                key = (w["start"], w["word"])
                if key in seen:
                    continue
                seen.add(key)
                ws.append({"w": " " + w["word"].strip(), "s": round(w["start"] + offset, 2),
                           "e": round(w["end"] + offset, 2)})
        segs.append({"start": round(s["start"] + offset, 2), "end": round(s["end"] + offset, 2),
                     "text": s["text"].strip(), "words": ws})
    return segs


def transcribe_episode(client, src, tmp_dir, cfg, hallu):
    """Full pass + repair pass for one local audio file. Returns (segments, duration, flags, words_added)."""
    small = A.compress(src, os.path.join(tmp_dir, "small.mp3"), cfg["groq"]["bitrate"])
    duration = A.probe_duration(small)
    segs = to_segments(client.transcribe(small))
    flags, added = [], 0
    if cfg["repair"]["enabled"]:
        def clip(a, b):
            part = A.compress(small, os.path.join(tmp_dir, "clip.mp3"), cfg["groq"]["bitrate"], a, b)
            return to_segments(client.transcribe(part), offset=a)
        segs, regions = find_regions(segs, duration, flags, client.prompt, hallu, **cfg["repair"])
        segs, added = fill_regions(regions, segs, duration, flags, clip, client.prompt, hallu)
    return segs, duration, flags, added


def run_worker(cfg, episodes, worker="groq", max_episodes=0, skip_file="", log_name="run_log_groq.jsonl",
               model=None):
    out = cfg["output_dir"]
    g = cfg["groq"]
    client = GroqClient(api_key(), model or g["model"], g["api_url"], cfg["language"], cfg["vocabulary"],
                        g.get("max_wait_s", 3600))
    hallu = hallucination_regex(cfg.get("extra_hallucinations"))
    repl = compile_replacements(cfg.get("replacements"))
    wq = WorkQueue(episodes, out, cfg["state_dir"], worker, skip_file)
    runlog = RunLog(os.path.join(out, log_name), worker)
    n = 0
    while not max_episodes or n < max_episodes:
        ep = wq.claim()
        if ep is None:
            break
        n += 1
        t0 = time.time()
        try:
            with tempfile.TemporaryDirectory() as td:
                src, _ = A.fetch(ep, td, cfg.get("audio_dir", ""), cfg.get("download"))
                segs, duration, flags, added = transcribe_episode(client, src, td, cfg, hallu)
            meta = {"engine": f"groq {client.model}", "worker": worker, "took_s": round(time.time() - t0, 1),
                    "qa_flags": flags, "repaired_words": added}
            words = save_episode(out, ep, segs, duration, meta, repl)
            runlog.write(event="done", id=ep["id"], audio_s=round(duration, 1), took_s=round(time.time() - t0, 1),
                         words=words, flags=len(flags), repaired_words=added)
            log(f'{worker} {ep["id"]}: {duration / 60:.0f} min, {words} words, {len(flags)} QA flags, {added} recovered')
        except Exception as ex:
            wq.mark_failed(ep)
            runlog.write(event="failed", id=ep["id"], error=repr(ex)[:300])
            log(f'{worker}: failed {ep["id"]}: {ex!r}')
    runlog.write(event="exit")
