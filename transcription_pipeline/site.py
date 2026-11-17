"""Static archive site: one index.html with client-side full-text search, plus segments.json.

No server and no build tools: the output folder can be opened locally or uploaded to any static host.
Episode cards show title, summary, keywords and key moments; every search hit and timestamp seeks the
audio player to that second. Transcripts (the heavy part) load in the background from segments.json.

Per-episode metadata is optional: `<meta_dir>/<id>.json` with any of
    {"title": "...", "summary": "...", "keywords": ["..."], "people": "...", "key_moments": ["[00:12:30] ..."]}
Without it, the title and description come from the episode list and keywords from TF-IDF.
"""
import html
import json
import os
import re
import shutil
from importlib import resources

from . import keywords as KW
from .audio import local_copy
from .report import load_transcripts

_MOMENT = re.compile(r"\s*\[(\d{1,2}):(\d{2}):(\d{2})\]\s*(.+)")


def parse_moments(lines):
    out = []
    for line in lines or []:
        m = _MOMENT.match(str(line))
        if m:
            out.append([int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)), m.group(4).strip()])
    return out


def _read_meta(meta_dir, ep_id):
    p = os.path.join(meta_dir, ep_id + ".json") if meta_dir else ""
    if p and os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def build(out_dir, site_dir, site_cfg, audio_dir="", top_keywords=8):
    items = load_transcripts(out_dir)
    os.makedirs(os.path.join(site_dir, "transcripts"), exist_ok=True)
    auto_kw = KW.extract({j["id"]: " ".join(s["text"] for s in j["segments"]) for j in items}, top_keywords)

    web, segs_all = [], {}
    for j in items:
        meta = _read_meta(site_cfg.get("meta_dir", ""), j["id"])
        audio = j.get("audio_url", "")
        local = local_copy({"id": j["id"], "audio": ""}, audio_dir) if audio_dir else None
        if local:
            os.makedirs(os.path.join(site_dir, "audio"), exist_ok=True)
            dst = os.path.join(site_dir, "audio", os.path.basename(local))
            if not os.path.exists(dst):
                shutil.copy2(local, dst)
            audio = "audio/" + os.path.basename(local)
        for ext in (".txt", ".srt", ".vtt", ".json"):
            src = os.path.join(out_dir, "transcripts", j["id"] + ext)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(site_dir, "transcripts", j["id"] + ext))
        summary = meta.get("summary") or j.get("description", "")
        if len(summary) > 600:
            summary = summary[:600].rsplit(" ", 1)[0] + " ..."
        web.append({
            "id": j["id"], "date": j.get("date", ""), "title": meta.get("title") or j.get("title") or j["id"],
            "summary": summary, "keywords": meta.get("keywords") or auto_kw.get(j["id"], []),
            "people": meta.get("people", ""), "moments": parse_moments(meta.get("key_moments")),
            "audio": audio, "minutes": round(j["duration"] / 60, 1),
            "words": sum(len(s["text"].split()) for s in j["segments"]),
        })
        segs_all[j["id"]] = [[round(s["start"], 1), s["text"]] for s in j["segments"] if s["text"].strip()]

    with open(os.path.join(site_dir, "segments.json"), "w", encoding="utf-8") as f:
        json.dump(segs_all, f, ensure_ascii=False, separators=(",", ":"))
    downloads = []
    for name, label in (("episodes.xlsx", "Episode list (Excel)"), ("episodes.csv", "Episode list (CSV)"),
                        ("transcripts.zip", "All transcripts (ZIP)")):
        if os.path.exists(os.path.join(out_dir, name)):
            shutil.copy2(os.path.join(out_dir, name), os.path.join(site_dir, name))
            downloads.append([name, label])

    tpl = resources.files("transcription_pipeline").joinpath("templates/archive.html").read_text(encoding="utf-8")
    total_min = sum(e["minutes"] for e in web)
    total_words = sum(e["words"] for e in web)
    esc = html.escape
    page = (tpl.replace("__DATA__", json.dumps(web, ensure_ascii=False).replace("</", "<\\/"))
            .replace("__DOWNLOADS__", json.dumps(downloads).replace("</", "<\\/"))
            .replace("__TITLE__", esc(site_cfg.get("title", "Podcast Archive")))
            .replace("__SUBTITLE__", esc(site_cfg.get("subtitle", "")))
            .replace("__FOOTER__", esc(site_cfg.get("footer", "")))
            .replace("__COUNT__", str(len(web)))
            .replace("__HOURS__", f"{total_min / 60:,.0f} h")
            .replace("__WORDS__", f"{total_words:,}"))
    with open(os.path.join(site_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print(f"site: {len(web)} episodes -> {os.path.join(site_dir, 'index.html')}")
    return web
