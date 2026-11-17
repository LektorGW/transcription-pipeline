import os

from transcription_pipeline.episodes import (assign_ids, filter_dates, from_directory, from_rss, from_url_list, slugify,
                                             work_order)

FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Example Show</title>
<item><title>Episode 2: Heat &amp; Power</title><link>https://example.com/ep2</link>
  <pubDate>Tue, 04 Mar 2025 10:00:00 +0000</pubDate>
  <description>&lt;p&gt;Second &lt;b&gt;episode&lt;/b&gt;&lt;/p&gt;</description>
  <enclosure url="https://cdn.example.com/ep2.mp3" type="audio/mpeg" length="1"/></item>
<item><title>Episode 1</title><pubDate>Mon, 03 Mar 2025 10:00:00 +0000</pubDate>
  <enclosure url="https://cdn.example.com/ep1.mp3?x=1" type="audio/mpeg" length="1"/></item>
<item><title>No audio</title></item>
</channel></rss>"""


def test_from_rss():
    eps = from_rss(FEED)
    assert len(eps) == 2
    assert eps[0] == {"date": "2025-03-04", "title": "Episode 2: Heat & Power", "audio": "https://cdn.example.com/ep2.mp3",
                      "source_url": "https://example.com/ep2", "description": "Second episode",
                      "id": "2025-03-04_episode-2-heat-power"}
    assert eps[1]["source_url"] == "https://cdn.example.com/ep1.mp3?x=1"


def test_ids_are_unique_and_safe():
    eps = assign_ids([{"date": "2025-01-01", "title": "Same"}, {"date": "2025-01-01", "title": "Same"},
                      {"title": "Café über / Q&A"}])
    assert [e["id"] for e in eps] == ["2025-01-01_same", "2025-01-01_same-2", "cafe-uber-qa"]
    assert slugify("  Hello,   World!  ") == "hello-world"


def test_from_directory_and_url_list(tmp_path):
    for name in ("show_2024-12-31.mp3", "notes.txt", "B.wav"):
        (tmp_path / name).write_bytes(b"")
    eps = from_directory(str(tmp_path))
    assert [(e["id"], e["date"]) for e in eps] == [("b", ""), ("show_2024-12-31", "2024-12-31")]
    assert os.path.isabs(eps[0]["audio"])
    urls = from_url_list(["# comment", "", "https://x.example/a/20250105_talk.mp3"])
    assert urls[0]["id"] == "20250105_talk" and urls[0]["date"] == "2025-01-05"


def test_filter_and_work_order():
    eps = [{"id": f"e{i}", "date": f"2025-01-{i + 1:02d}"} for i in range(6)]
    assert [e["id"] for e in filter_dates(eps, "2025-01-03", "2025-01-04")] == ["e2", "e3"]
    order = [e["id"] for e in work_order(eps, {0}, 2)]
    # newest first: e5..e0; shard 0 = e5, e3, e1, then the other shard from the oldest end
    assert order == ["e5", "e3", "e1", "e0", "e2", "e4"]
    assert [e["id"] for e in work_order(eps, {1}, 2, help_others=False)] == ["e4", "e2", "e0"]
