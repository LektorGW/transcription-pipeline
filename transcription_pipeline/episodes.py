"""Building the episode list from an RSS feed, a folder of audio files or a plain list of URLs.

Episode record (what every other stage consumes):
    {"id": "2025-03-14_episode-title", "date": "2025-03-14", "title": "...",
     "audio": "<http(s) URL or local path>", "source_url": "<page or audio URL>", "description": "..."}
"""
import email.utils
import html
import json
import os
import re
import unicodedata
import xml.etree.ElementTree as ET

from .audio import AUDIO_EXTS

_DATE_IN_NAME = re.compile(r"(\d{4})[-_.]?(\d{2})[-_.]?(\d{2})")


def slugify(text, max_len=60):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)[:max_len].strip("-_")


def strip_html(text):
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or ""))).strip()


def _date_from_name(name):
    m = _DATE_IN_NAME.search(name)
    if m and 1900 < int(m.group(1)) < 2100 and 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return ""


def assign_ids(episodes):
    """Stable, filesystem-safe, unique ids: `<date>_<title-slug>`, with -2, -3 ... on collisions."""
    seen = {}
    for ep in episodes:
        if ep.get("id"):
            base = slugify(ep["id"], 100) or "episode"
        else:
            base = "_".join(x for x in (ep.get("date", ""), slugify(ep.get("title", ""))) if x) or "episode"
        n = seen.get(base, 0) + 1
        seen[base] = n
        ep["id"] = base if n == 1 else f"{base}-{n}"
    return episodes


def from_rss(xml_text):
    """Parse a podcast RSS 2.0 feed: one episode per <item> with an audio <enclosure>."""
    root = ET.fromstring(xml_text)
    eps = []
    for item in root.iter("item"):
        enc = item.find("enclosure")
        url = enc.get("url") if enc is not None else ""
        if not url:
            continue
        date = ""
        pub = item.findtext("pubDate")
        if pub:
            try:
                date = email.utils.parsedate_to_datetime(pub).date().isoformat()
            except (TypeError, ValueError):
                date = ""
        eps.append({
            "date": date or _date_from_name(url),
            "title": strip_html(item.findtext("title") or ""),
            "audio": url,
            "source_url": (item.findtext("link") or url).strip(),
            "description": strip_html(item.findtext("description") or ""),
        })
    return assign_ids(eps)


def from_directory(path):
    eps = []
    for name in sorted(os.listdir(path)):
        stem, ext = os.path.splitext(name)
        if ext.lower() in AUDIO_EXTS:
            eps.append({"id": stem, "date": _date_from_name(stem), "title": stem.replace("_", " "),
                        "audio": os.path.abspath(os.path.join(path, name)), "source_url": "", "description": ""})
    return assign_ids(eps)


def from_url_list(lines):
    eps = []
    for line in lines:
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        stem = os.path.splitext(os.path.basename(url.split("?")[0]))[0]
        eps.append({"id": stem, "date": _date_from_name(url), "title": stem.replace("_", " "),
                    "audio": url, "source_url": url, "description": ""})
    return assign_ids(eps)


def filter_dates(episodes, since="", until=""):
    return [e for e in episodes if (not since or e["date"] >= since) and (not until or (e["date"] and e["date"] <= until))]


def work_order(episodes, my_shards=None, count=1, help_others=True):
    """My shards first (newest first), then the remaining episodes from the other end of the list,
    so a machine that finishes early helps the others instead of idling."""
    eps = sorted(episodes, key=lambda e: (e.get("date", ""), e["id"]), reverse=True)
    if not my_shards or count <= 1:
        return eps
    mine = [e for i, e in enumerate(eps) if i % count in my_shards]
    rest = [e for i, e in enumerate(eps) if i % count not in my_shards][::-1]
    return mine + (rest if help_others else [])


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(episodes, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, ensure_ascii=False, indent=1)
