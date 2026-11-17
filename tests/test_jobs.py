from transcription_pipeline.jobs import RunLog, WorkQueue, read_log, reset_claims

EPS = [{"id": f"ep{i}"} for i in range(3)]


def test_claims_are_exclusive_between_workers(tmp_path):
    out, state = str(tmp_path / "out"), str(tmp_path / "state")
    a = WorkQueue(EPS, out, state, "a")
    b = WorkQueue(EPS, out, state, "b")
    got = [a.claim()["id"], b.claim()["id"], a.claim()["id"]]
    assert sorted(got) == ["ep0", "ep1", "ep2"]
    assert a.claim() is None and b.claim() is None


def test_resume_skips_finished_and_remote_episodes(tmp_path):
    out, state = tmp_path / "out", tmp_path / "state"
    (out / "transcripts").mkdir(parents=True)
    (out / "transcripts" / "ep0.json").write_text("{}")
    skip = tmp_path / "skip.txt"
    skip.write_text("ep1\n")
    q = WorkQueue(EPS, str(out), str(state), "w", str(skip))
    assert [e["id"] for e in q.pending()] == ["ep2"]
    assert q.claim()["id"] == "ep2"
    assert q.claim() is None


def test_reset_claims(tmp_path):
    state = str(tmp_path / "state")
    q = WorkQueue(EPS, str(tmp_path / "out"), state, "w")
    q.claim()
    q.claim()
    assert reset_claims(state) == 2
    assert q.claim()["id"] == "ep0"


def test_run_log_roundtrip(tmp_path):
    p = str(tmp_path / "log" / "run.jsonl")
    log = RunLog(p, "w1")
    log.write(event="done", id="ep0", audio_s=60)
    log.write(event="exit")
    recs = read_log(p)
    assert [r["event"] for r in recs] == ["done", "exit"]
    assert recs[0]["worker"] == "w1"
