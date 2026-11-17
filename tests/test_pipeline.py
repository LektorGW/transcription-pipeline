"""End-to-end checks without network or models: a fake Groq HTTP session, a generated test tone (FFmpeg),
then the report and the static site built from the result."""
import json
import shutil
import subprocess

import pytest

from transcription_pipeline import groq_backend as G
from transcription_pipeline import keywords, report, site
from transcription_pipeline.config import DEFAULTS, deep_merge, load_config

HAS_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def test_to_segments_assigns_boundary_words_once():
    resp = {"segments": [{"start": 0.0, "end": 2.0, "text": " Hello there "}, {"start": 2.0, "end": 4.0, "text": "General"}],
            "words": [{"word": "Hello", "start": 0.1, "end": 0.5}, {"word": "there", "start": 1.98, "end": 2.02},
                      {"word": "General", "start": 2.5, "end": 3.0}]}
    segs = G.to_segments(resp, offset=10)
    assert segs[0]["text"] == "Hello there" and segs[0]["start"] == 10.0
    assert [w["w"] for w in segs[0]["words"]] == [" Hello", " there"]
    assert [w["w"] for w in segs[1]["words"]] == [" General"]          # "there" not duplicated
    assert segs[1]["words"][0]["s"] == 12.5


def test_config_merge(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"language": "de", "repair": {"gap_s": 9}}))
    cfg = load_config(str(p))
    assert cfg["language"] == "de" and cfg["repair"]["gap_s"] == 9 and cfg["repair"]["edge_gap_s"] == 20.0
    assert DEFAULTS["repair"]["gap_s"] == 6.0                          # defaults untouched
    assert deep_merge({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}


def test_keywords_prefer_distinctive_terms():
    docs = {"a": "solar panels solar panels on the roof, solar panels again. weather weather",
            "b": "heat pumps and heat pumps in winter. weather weather"}
    kw = keywords.extract(docs, top_n=2)
    assert kw["a"][0] == "solar panels"
    assert kw["b"][0] == "heat pumps"


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload
        self.headers = {}

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    """First upload (the whole file): one real segment and one hallucination.
    Later uploads (repair clips): nothing, as on music or silence."""

    def __init__(self):
        self.calls = 0

    def post(self, url, **kw):
        self.calls += 1
        assert kw["headers"]["Authorization"] == "Bearer test-key"
        if self.calls > 1:
            return FakeResponse({"segments": [], "words": []})
        return FakeResponse({
            "segments": [{"start": 0.2, "end": 1.4, "text": "Welcome to the example show."},
                         {"start": 1.5, "end": 2.8, "text": "Thanks for watching!"}],
            "words": [{"word": w, "start": 0.2 + i * 0.2, "end": 0.35 + i * 0.2}
                      for i, w in enumerate("Welcome to the example show.".split())],
        })


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_groq_episode_to_report_and_site(tmp_path):
    wav = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3", str(wav)], check=True)
    cfg = deep_merge(DEFAULTS, {"output_dir": str(tmp_path / "out"), "state_dir": str(tmp_path / "state")})
    client = G.GroqClient("test-key", session=FakeSession())
    segs, duration, flags, added = G.transcribe_episode(client, str(wav), str(tmp_path), cfg,
                                                        G.hallucination_regex())
    assert 2.9 < duration < 3.2
    assert [s["text"] for s in segs] == ["Welcome to the example show."]
    assert [f["type"] for f in flags] == ["hallucination_removed", "hallucination_no_speech"]
    assert client.session.calls == 2

    from transcription_pipeline.formats import save_episode
    ep = {"id": "2025-05-01_example", "date": "2025-05-01", "title": "Example", "audio": str(wav)}
    save_episode(cfg["output_dir"], ep, segs, duration, {"engine": "groq test", "qa_flags": flags})
    rows, qa = report.build(cfg["output_dir"])
    assert rows[0]["ID"] == "2025-05-01_example" and qa[0]["Type"] == "hallucination_removed"

    web = site.build(cfg["output_dir"], str(tmp_path / "site"), cfg["site"])
    html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert "__DATA__" not in html and "Podcast Archive" in html
    segs_json = json.loads((tmp_path / "site" / "segments.json").read_text(encoding="utf-8"))
    assert segs_json["2025-05-01_example"][0][1] == "Welcome to the example show."
    assert web[0]["words"] == 5
    assert (tmp_path / "site" / "transcripts" / "2025-05-01_example.vtt").exists()


def test_parse_moments():
    assert site.parse_moments(["[00:01:05] Intro", "[1:00:00] Late point", "no timestamp"]) == \
        [[65, "Intro"], [3600, "Late point"]]


def test_example_config_is_valid():
    pytest.importorskip("yaml")
    import os

    from transcription_pipeline.cleanup import clean_text, compile_replacements, hallucination_regex
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config.example.yaml"))
    assert clean_text("Jon Roe on net meetering", compile_replacements(cfg["replacements"])) == "John Roe on net metering"
    assert hallucination_regex(cfg["extra_hallucinations"]).match("See you next week.")
    assert cfg["local"]["model"] == "large-v3" and cfg["groq"]["bitrate"] == "32k"
