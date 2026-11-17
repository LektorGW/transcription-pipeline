from transcription_pipeline.repair import fill_regions, find_regions, merge_regions


def seg(start, end, text, p=0.9):
    words = text.split()
    step = (end - start) / max(len(words), 1)
    return {"start": start, "end": end, "text": text,
            "words": [{"w": " " + w, "s": start + i * step, "e": start + (i + 1) * step, "p": p}
                      for i, w in enumerate(words)]}


def test_find_regions_detects_gaps_and_hallucinations():
    segs = [seg(1, 10, "first part of the talk"),
            seg(10, 12, "Thanks for watching!"),        # hallucination -> removed, becomes a region
            seg(20, 30, "second part of the talk"),     # 10 s gap before it (> 6 s)
            seg(31, 40, "third part of the talk")]
    flags = []
    kept, regions = find_regions(segs, duration=100.0, flags=flags)
    assert [s["start"] for s in kept] == [1, 20, 31]
    # hallucination 10-12 and the gap 10-20 merge into one region; the 60 s tail is an edge gap > 20 s
    assert regions == [[10, 20, "hallucination"], [40, 100.0, "gap"]]
    assert flags[0]["type"] == "hallucination_removed"


def test_short_gaps_and_edges_are_ignored():
    segs = [seg(5, 10, "a b c"), seg(14, 20, "d e f")]
    _, regions = find_regions(segs, duration=30.0, flags=[])
    assert regions == []
    _, regions = find_regions(segs, duration=60.0, flags=[], include_edges=False)
    assert regions == []


def test_low_confidence_and_long_word_flags():
    s = seg(0, 30, "one two three four five", p=0.2)
    flags = []
    find_regions([s], duration=30.0, flags=flags)
    types = {f["type"] for f in flags}
    assert {"low_confidence", "long_word"} <= types


def test_merge_regions():
    assert merge_regions([[5, 6, "gap"], [0, 2, "gap"], [2.3, 4, "hallucination"]]) == [[0, 4, "gap"], [5, 6, "gap"]]


def test_fill_regions_keeps_speech_and_rejects_music():
    kept = [seg(0, 10, "start of the episode here"), seg(20, 30, "end of the episode here")]
    flags = []

    def fake_transcribe(a, b):
        if a < 20:   # region 10-20: speech the first pass lost
            return [seg(12, 18, "this speech was lost by the first pass")]
        return [seg(a, b, "la")]   # region 30-90: music

    out, added = fill_regions([[10, 20, "gap"], [30, 90, "gap"]], kept, 90.0, flags, fake_transcribe)
    assert added == 8
    assert [s["start"] for s in out] == [0, 12, 20]
    assert out[1]["repaired"] is True
    assert [f["type"] for f in flags] == ["gap_filled", "gap_no_speech"]


def test_fill_regions_survives_backend_errors():
    flags = []

    def boom(a, b):
        raise RuntimeError("network down")

    out, added = fill_regions([[10, 40, "gap"]], [], 90.0, flags, boom)
    assert out == [] and added == 0
    assert flags[0]["type"] == "repair_error"
