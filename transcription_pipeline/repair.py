"""Second-pass repair, shared by all backends.

1. `find_regions` removes bad segments (hallucinations, loops, prompt leaks), records QA flags and returns
   the time regions worth a second look: removed segments and suspicious gaps between segments.
2. `fill_regions` re-transcribes each region with a backend-supplied callable and keeps the result only
   if it looks like real speech (`plausible_fill`). Everything it does is recorded as a QA flag, so a
   reviewer can jump to the exact timestamps later.
"""
from .cleanup import HALLU, bad_segment, plausible_fill


def find_regions(segs, duration, flags, prompt="", hallu=HALLU, gap_s=6.0, edge_gap_s=20.0,
                 long_word_s=4.0, include_edges=True, **_):
    regions, kept = [], []
    for s in segs:
        if bad_segment(s, prompt, hallu):
            flags.append({"t": s["start"], "type": "hallucination_removed", "text": s["text"][:200]})
            regions.append([s["start"], s["end"], "hallucination"])
        else:
            kept.append(s)

    # (start, end) pairs with sentinels for the beginning and the end of the recording
    bounds = [(0.0, None)] + [(s["start"], s["end"]) for s in kept] + [(None, duration)]
    for (_, a_end), (b_start, _) in zip(bounds, bounds[1:]):
        is_edge = a_end is None or b_start is None
        gap_from = 0.0 if a_end is None else a_end
        gap_to = duration if b_start is None else b_start
        if is_edge and not include_edges:
            continue
        if gap_to - gap_from > (edge_gap_s if is_edge else gap_s):
            regions.append([gap_from, gap_to, "gap"])

    for s in kept:
        for w in s.get("words", []):
            if w["e"] - w["s"] > long_word_s:
                flags.append({"t": w["s"], "type": "long_word", "text": f'"{w["w"].strip()}" {w["e"] - w["s"]:.0f} s'})
        ps = [w["p"] for w in s.get("words", []) if "p" in w]
        if len(ps) >= 5 and sum(ps) / len(ps) < 0.45:
            flags.append({"t": s["start"], "type": "low_confidence", "text": s["text"][:200]})

    return kept, merge_regions(regions)


def merge_regions(regions, slack=0.5):
    merged = []
    for r in sorted(regions):
        if merged and r[0] <= merged[-1][1] + slack:
            merged[-1][1] = max(merged[-1][1], r[1])
        else:
            merged.append(list(r))
    return merged


def fill_regions(regions, kept, duration, flags, transcribe_clip, prompt="", hallu=HALLU):
    """`transcribe_clip(a, b)` must return segments with absolute timestamps for audio between a and b."""
    added = 0
    for a, b, why in regions:
        a2, b2 = max(0.0, a + 0.1), min(duration, b - 0.1)
        if b2 - a2 < 1.0:
            continue
        try:
            new = transcribe_clip(a2, b2)
        except Exception as ex:  # a failed repair must never lose the main transcript
            flags.append({"t": a, "type": "repair_error", "text": f"{type(ex).__name__}: {ex}"[:200]})
            continue
        good = []
        for s in new:
            if bad_segment(s, prompt, hallu):
                continue
            ws = [w for w in s.get("words", []) if a2 - 0.3 <= (w["s"] + w["e"]) / 2 <= b2 + 0.3]
            if not ws:
                continue
            good.append({"start": ws[0]["s"], "end": ws[-1]["e"], "text": "".join(w["w"] for w in ws).strip(),
                         "words": ws, "repaired": True})
        if good and plausible_fill(good, b - a):
            n = sum(len(s["words"]) for s in good)
            kept += good
            added += n
            flags.append({"t": a, "type": f"{why}_filled",
                          "text": f'{b - a:.0f} s, {n} words: ' + " ".join(s["text"] for s in good)[:160]})
        else:
            flags.append({"t": a, "type": f"{why}_no_speech", "text": f"{b - a:.0f} s"})
    kept.sort(key=lambda s: s["start"])
    return kept, added
