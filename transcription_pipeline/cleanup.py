"""Detection of typical Whisper failure modes and text normalisation.

Whisper is very good on clean speech but has well-known failure modes on long recordings:
  * invented sentences on music or silence ("Thanks for watching", "Subtitles by ..."),
  * repetition loops ("the the the ..." or one phrase repeated many times),
  * low-diversity garbage on music beds,
  * "prompt leaks": the model reads back the vocabulary prompt instead of the audio.
All checks here are pure functions so they can be unit-tested without a model.
"""
import re

_HALLU_DEFAULT = [
    r"transcription by castingwords",
    r"thank(s| you) (so much )?for watching",
    r"subtitles? by.*",
    r".*amara\.org.*",
    r"please subscribe.*",
    r"like and subscribe.*",
    r"bye\.?",
]


def hallucination_regex(extra=()):
    parts = _HALLU_DEFAULT + list(extra or [])
    return re.compile(r"^(" + "|".join(parts) + r")$", re.I)


HALLU = hallucination_regex()
_WORD = re.compile(r"[\w']+")


def words(text):
    return _WORD.findall(text.lower())


def is_loop(text):
    """A repetition loop: one word repeated >= 10 times in a row, or a 2-6 word phrase >= 5 times.
    Normal speech such as "blah, blah, blah" (3-6 repetitions) is not a loop."""
    w = words(text)
    for n in range(1, 7):
        reps = 10 if n == 1 else 5
        for i in range(0, max(0, len(w) - n * reps) + 1):
            g = w[i:i + n]
            if len(g) == n and all(w[i + k * n:i + (k + 1) * n] == g for k in range(reps)):
                return True
    return False


def low_diversity(text):
    """Garbage on music: 'November November October November ...'."""
    w = words(text)
    return len(w) >= 8 and len(set(w)) / len(w) < 0.45


def is_prompt_leak(text, prompt, run=5):
    """True if the segment repeats `run` consecutive words of the vocabulary prompt verbatim."""
    if not prompt:
        return False
    p, t = words(prompt), words(text)
    if len(p) < run or len(t) < run:
        return False
    grams = {tuple(p[i:i + run]) for i in range(len(p) - run + 1)}
    return any(tuple(t[i:i + run]) in grams for i in range(len(t) - run + 1))


def bad_segment(seg, prompt="", hallu=HALLU):
    t = re.sub(r"[^\w.' ]+", "", seg["text"]).strip().strip(".").strip()
    return (not t) or bool(hallu.match(t)) or is_loop(seg["text"]) or low_diversity(seg["text"]) \
        or is_prompt_leak(seg["text"], prompt)


def plausible_fill(segs, seconds):
    """A re-transcribed gap counts as speech only if it has >= 3 words and, for gaps >= 15 s,
    at least 0.5 words per second. Otherwise it is music/silence and is discarded."""
    n = sum(len(s["text"].split()) for s in segs)
    return n >= 3 and (seconds < 15 or n / seconds >= 0.5)


def compile_replacements(pairs):
    return [(re.compile(p), r) for p, r in (pairs or [])]


def clean_text(text, replacements=()):
    for rx, rep in replacements:
        text = rx.sub(rep, text)
    return re.sub(r"\s{2,}", " ", text).strip()
