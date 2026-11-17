"""Output formats (TXT with timestamped paragraphs, SRT, WebVTT, JSON with word timings) and atomic saving.

Segment shape used everywhere:
    {"start": 12.3, "end": 15.9, "text": "...", "words": [{"w": " Hello", "s": 12.3, "e": 12.6, "p": 0.98}, ...]}
"""
import json
import os

from .cleanup import clean_text

EXTS = (".txt", ".srt", ".vtt", ".json")   # JSON is written last: its presence marks the episode as done


def _hms(t):
    t = max(0.0, float(t))
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return h, m, s, ms


def ts_srt(t):
    h, m, s, ms = _hms(t)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ts_vtt(t):
    h, m, s, ms = _hms(t)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def ts_short(t):
    h, m, s, _ = _hms(int(t))
    return f"{h:02d}:{m:02d}:{s:02d}"


def paragraphs(segs, max_len_s=60.0, pause_s=2.5):
    """Group segments into readable paragraphs: a new one every ~60 s or after a pause > 2.5 s."""
    out, cur, start, last = [], [], None, 0.0
    for s in segs:
        if cur and (s["start"] - start > max_len_s or s["start"] - last > pause_s):
            out.append((start, " ".join(cur)))
            cur = []
        if not cur:
            start = s["start"]
        cur.append(s["text"])
        last = s["end"]
    if cur:
        out.append((start, " ".join(cur)))
    return out


def to_txt(segs, header=""):
    body = "\n\n".join(f"[{ts_short(a)}] {t}" for a, t in paragraphs(segs))
    return (header + "\n\n" if header else "") + body + "\n"


def to_srt(segs):
    return "".join(f'{i}\n{ts_srt(s["start"])} --> {ts_srt(s["end"])}\n{s["text"]}\n\n' for i, s in enumerate(segs, 1))


def to_vtt(segs):
    return "WEBVTT\n\n" + "".join(f'{ts_vtt(s["start"])} --> {ts_vtt(s["end"])}\n{s["text"]}\n\n' for s in segs)


def episode_header(ep):
    lines = [" | ".join(x for x in (ep.get("title", ""), ep.get("date", "")) if x)]
    if ep.get("source_url"):
        lines.append(f'Source: {ep["source_url"]}')
    return "\n".join(x for x in lines if x)


def transcript_path(out_dir, ep_id, ext=".json"):
    return os.path.join(out_dir, "transcripts", ep_id + ext)


def is_done(out_dir, ep_id):
    return os.path.exists(transcript_path(out_dir, ep_id))


def save_episode(out_dir, ep, segs, duration, meta, replacements=()):
    """Write TXT/SRT/VTT/JSON for one episode. Each file goes to `<name>.part` first and is renamed only
    when all of them are complete, so an interrupted run never leaves a half-written transcript behind."""
    os.makedirs(os.path.join(out_dir, "transcripts"), exist_ok=True)
    for s in segs:
        s["text"] = clean_text(s["text"], replacements)
    segs = [s for s in segs if s["text"]]
    record = {
        "id": ep["id"], "date": ep.get("date", ""), "title": ep.get("title", ""),
        "source_url": ep.get("source_url", ""), "description": ep.get("description", ""),
        "audio_url": ep["audio"] if str(ep.get("audio", "")).startswith(("http://", "https://")) else "",
        "engine": meta.get("engine", ""), "duration": round(duration, 2), "meta": meta, "segments": segs,
    }
    content = {
        ".txt": to_txt(segs, episode_header(ep)),
        ".srt": to_srt(segs),
        ".vtt": to_vtt(segs),
        ".json": json.dumps(record, ensure_ascii=False),
    }
    base = transcript_path(out_dir, ep["id"], "")
    for ext in EXTS:
        with open(base + ext + ".part", "w", encoding="utf-8") as f:
            f.write(content[ext])
    for ext in EXTS:
        os.replace(base + ext + ".part", base + ext)
    return sum(len(s["text"].split()) for s in segs)
