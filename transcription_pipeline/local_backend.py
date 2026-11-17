"""Local backend: faster-whisper (CTranslate2), one model per process.

Each worker process:
  * claims the next free episode (lock file, see jobs.py),
  * downloads and decodes the NEXT episode in a background thread while the GPU works on the current one,
  * transcribes with BatchedInferencePipeline (VAD + batched decoding, word timestamps),
  * runs the repair pass on gaps and removed segments with plain beam search and no VAD,
  * saves TXT/SRT/VTT/JSON immediately, so stopping the process loses at most the current episode.
"""
import os
import queue
import tempfile
import threading
import time

from . import audio as A
from .cleanup import compile_replacements, hallucination_regex
from .formats import save_episode
from .jobs import RunLog, WorkQueue, log
from .repair import fill_regions, find_regions


def seg_dict(s, offset=0.0):
    return {"start": round(s.start + offset, 2), "end": round(s.end + offset, 2), "text": s.text.strip(),
            "words": [{"w": w.word, "s": round(w.start + offset, 2), "e": round(w.end + offset, 2),
                       "p": round(w.probability, 3)} for w in (s.words or [])]}


def run_worker(cfg, episodes, worker="local", max_episodes=0, deadline=0, limit_seconds=0, skip_file="",
               log_name="run_log.jsonl"):
    lc, rc = cfg["local"], cfg["repair"]
    out, lang, vocab = cfg["output_dir"], cfg["language"], cfg["vocabulary"]
    hallu = hallucination_regex(cfg.get("extra_hallucinations"))
    repl = compile_replacements(cfg.get("replacements"))
    tmp = cfg.get("tmp_dir") or os.path.join(tempfile.gettempdir(), "tp_audio")
    os.makedirs(tmp, exist_ok=True)
    wq = WorkQueue(episodes, out, cfg["state_dir"], worker, skip_file)
    runlog = RunLog(os.path.join(out, log_name), worker)

    q = queue.Queue(maxsize=1)   # at most one decoded episode waiting: bounded memory
    taken = [0]

    def producer():
        while True:
            if deadline and time.time() > deadline:
                log(f"{worker}: deadline reached, not taking new episodes")
                break
            if max_episodes and taken[0] >= max_episodes:
                break
            ep = wq.claim()
            if ep is None:
                break
            taken[0] += 1
            try:
                t0 = time.time()
                path, is_tmp = A.fetch(ep, tmp, cfg.get("audio_dir", ""), cfg.get("download"))
                pcm = A.load_audio(path, limit_seconds)
                if is_tmp:
                    os.remove(path)
                q.put((ep, pcm, time.time() - t0))
            except Exception as ex:
                wq.mark_failed(ep)
                runlog.write(event="failed", id=ep["id"], stage="fetch", error=repr(ex)[:300])
                log(f'{worker}: fetch failed for {ep["id"]}: {ex!r}')
        q.put(None)

    from faster_whisper import BatchedInferencePipeline, WhisperModel
    t0 = time.time()
    model = WhisperModel(lc["model"], device=lc["device"], compute_type=lc["compute_type"],
                         cpu_threads=lc.get("cpu_threads", 2))
    pipe = BatchedInferencePipeline(model=model)
    log(f'{worker}: model {lc["model"]} ({lc["compute_type"]}) loaded in {time.time() - t0:.0f} s')
    runlog.write(event="ready", load_s=round(time.time() - t0, 1))
    threading.Thread(target=producer, daemon=True).start()

    def clip(a, b):
        it, _ = model.transcribe(pcm[int(a * A.SR):int(b * A.SR)], language=lang, beam_size=lc["beam_size"],
                                 vad_filter=False, word_timestamps=True, condition_on_previous_text=False,
                                 initial_prompt=vocab or None)
        return [seg_dict(s, a) for s in it]

    while True:
        item = q.get()
        if item is None:
            break
        ep, pcm, fetch_s = item
        duration = len(pcm) / A.SR
        t0 = time.time()
        try:
            kw = dict(language=lang, batch_size=lc["batch_size"], beam_size=lc["beam_size"], word_timestamps=True,
                      vad_filter=lc["vad_filter"])
            try:
                it, _ = pipe.transcribe(pcm, initial_prompt=vocab or None, **kw)
            except TypeError:     # older faster-whisper: batched pipeline has no initial_prompt
                it, _ = pipe.transcribe(pcm, hotwords=vocab or None, **kw)
            segs = [seg_dict(s) for s in it]
            t_main = time.time() - t0
            flags, added = [], 0
            if rc["enabled"]:
                segs, regions = find_regions(segs, duration, flags, vocab, hallu, **rc)
                segs, added = fill_regions(regions, segs, duration, flags, clip, vocab, hallu)
            n_words = sum(len(s["text"].split()) for s in segs)
            if n_words < 50 and duration > 120:
                raise RuntimeError(f"suspiciously short transcript ({n_words} words for {duration:.0f} s)")
            took = time.time() - t0
            meta = {"engine": f'faster-whisper {os.path.basename(str(lc["model"]))}', "worker": worker,
                    "took_s": round(took, 1), "main_pass_s": round(t_main, 1), "fetch_s": round(fetch_s, 1),
                    "repaired_words": added, "qa_flags": flags, "limit_seconds": limit_seconds}
            n_words = save_episode(out, ep, segs, duration, meta, repl)
            runlog.write(event="done", id=ep["id"], audio_s=round(duration, 1), took_s=round(took, 1),
                         words=n_words, flags=len(flags), repaired_words=added)
            log(f'{worker} {ep["id"]}: {duration / 60:.0f} min in {took:.0f} s ({duration / max(took, 1):.0f}x), '
                f'{n_words} words, {len(flags)} QA flags, {added} words recovered')
        except Exception as ex:
            wq.mark_failed(ep)
            runlog.write(event="failed", id=ep["id"], stage="transcribe", error=repr(ex)[:300])
            log(f'{worker}: transcription failed for {ep["id"]}: {ex!r}')
            if "out of memory" in str(ex).lower():
                import gc
                gc.collect()
        del pcm
    runlog.write(event="exit")
    log(f"{worker}: finished")
