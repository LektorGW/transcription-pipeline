"""Archive summary: episodes.csv (+ .xlsx if openpyxl is installed), qa_flags.csv for reviewers, and a ZIP
with all transcripts."""
import csv
import glob
import json
import os
import zipfile
from collections import Counter

from .formats import EXTS, ts_short


def load_transcripts(out_dir):
    items = []
    for p in sorted(glob.glob(os.path.join(out_dir, "transcripts", "*.json"))):
        with open(p, encoding="utf-8") as f:
            items.append(json.load(f))
    return sorted(items, key=lambda j: (j.get("date", ""), j["id"]), reverse=True)


def build(out_dir, zip_name="transcripts.zip"):
    rows, flags = [], []
    for j in load_transcripts(out_dir):
        meta = j.get("meta", {})
        rows.append({
            "ID": j["id"], "Date": j.get("date", ""), "Title": j.get("title", ""),
            "Source": j.get("source_url", ""), "Duration (min)": round(j["duration"] / 60, 1),
            "Word Count": sum(len(s["text"].split()) for s in j["segments"]), "Engine": j.get("engine", ""),
            "QA flags": len(meta.get("qa_flags", [])), "Recovered words": meta.get("repaired_words", 0),
        })
        for fl in meta.get("qa_flags", []):
            flags.append({"ID": j["id"], "Time": ts_short(fl["t"]), "Type": fl["type"], "Text": fl["text"]})

    for name, data in (("episodes.csv", rows), ("qa_flags.csv", flags)):
        if data:
            with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    try:
        import openpyxl
        if rows:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Episodes"
            ws.append(list(rows[0]))
            for r in rows:
                ws.append(list(r.values()))
            ws.freeze_panes = "A2"
            wb.save(os.path.join(out_dir, "episodes.xlsx"))
    except ImportError:
        pass

    with zipfile.ZipFile(os.path.join(out_dir, zip_name), "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(glob.glob(os.path.join(out_dir, "transcripts", "*"))):
            if p.endswith(EXTS):
                z.write(p, "transcripts/" + os.path.basename(p))

    hours = sum(r["Duration (min)"] for r in rows) / 60
    print(f"episodes: {len(rows)} | audio: {hours:.1f} h | words: {sum(r['Word Count'] for r in rows):,} | "
          f"QA flags: {len(flags)} {Counter(f['Type'] for f in flags).most_common(6)}")
    return rows, flags
