import json
import os

from transcription_pipeline.formats import (EXTS, is_done, paragraphs, save_episode, to_srt, to_txt, to_vtt, ts_srt,
                                            ts_vtt)

SEGS = [
    {"start": 0.0, "end": 2.5, "text": "Hello and welcome.", "words": []},
    {"start": 2.6, "end": 5.0, "text": "Today we talk about storage.", "words": []},
    {"start": 9.0, "end": 3725.4567, "text": "A long one.", "words": []},
]


def test_timestamps():
    assert ts_srt(0) == "00:00:00,000"
    assert ts_srt(3725.4567) == "01:02:05,457"
    assert ts_vtt(59.9996) == "00:01:00.000"     # rounding carries into minutes
    assert ts_vtt(-1) == "00:00:00.000"


def test_srt_and_vtt():
    srt = to_srt(SEGS)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,500\nHello and welcome.\n\n2\n")
    assert "3\n00:00:09,000 --> 01:02:05,457\nA long one." in srt
    vtt = to_vtt(SEGS)
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nHello and welcome.\n")


def test_paragraphs_split_on_pause():
    paras = paragraphs(SEGS)
    assert paras[0] == (0.0, "Hello and welcome. Today we talk about storage.")
    assert paras[1][0] == 9.0
    assert to_txt(SEGS, "Header").startswith("Header\n\n[00:00:00] Hello and welcome.")


def test_save_episode_writes_all_formats_atomically(tmp_path):
    ep = {"id": "2025-01-02_test", "date": "2025-01-02", "title": "Test", "audio": "https://example.com/a.mp3"}
    segs = [dict(s) for s in SEGS] + [{"start": 4000, "end": 4001, "text": "   ", "words": []}]
    assert not is_done(str(tmp_path), ep["id"])
    words = save_episode(str(tmp_path), ep, segs, 4001.0, {"engine": "test"})
    assert words == 11
    for ext in EXTS:
        assert os.path.exists(tmp_path / "transcripts" / (ep["id"] + ext))
    assert not list((tmp_path / "transcripts").glob("*.part"))
    data = json.loads((tmp_path / "transcripts" / (ep["id"] + ".json")).read_text(encoding="utf-8"))
    assert data["audio_url"] == "https://example.com/a.mp3"
    assert len(data["segments"]) == 3                        # empty segment dropped
    assert is_done(str(tmp_path), ep["id"])
