"""Lightweight keyword extraction (TF-IDF over the whole archive, no external services).

Used as a fallback for the archive site when no hand-written or LLM-written metadata is provided.
Picks terms that are frequent in one episode but rare across the archive, preferring two-word phrases.
"""
import math
import re
from collections import Counter

STOP = set("""a about above after again against all also am an and any are aren't as at be because been before
being below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
during each even ever every few for from further get gets getting go goes going gone gonna got gotta had hadn't
has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd
i'll i'm i've if in into is isn't it it's its itself just know let let's like lot many may maybe me mean might
more most much must my myself need no nor not now of off oh ok okay on once one only or other ought our ours
ourselves out over own really right said same say says see she she'd she'll she's should shouldn't so some
something still such sure take talk tell than thank thanks that that's the their theirs them themselves then
there there's these they they'd they'll they're they've thing things think this those though through time to
today too two under until up upon us very want was wasn't way we we'd we'll we're we've well went were weren't
what what's when when's where where's whether which while who who's whom why why's will with won't would
wouldn't yeah yes yet you you'd you'll you're you've your yours yourself yourselves""".split())

_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def terms(text):
    toks = [t.lower().strip("'-") for t in _TOKEN.findall(text)]
    toks = [t for t in toks if len(t) > 2]
    uni = [t for t in toks if t not in STOP]
    bi = [f"{a} {b}" for a, b in zip(toks, toks[1:]) if a not in STOP and b not in STOP]
    return uni + bi


def extract(docs, top_n=8, min_count=2):
    """docs: {id: text}. Returns {id: [keyword, ...]} ranked by TF-IDF (bigrams weighted x1.5)."""
    counts = {k: Counter(terms(v)) for k, v in docs.items()}
    df = Counter()
    for c in counts.values():
        df.update(c.keys())
    n = max(len(docs), 1)
    out = {}
    for k, c in counts.items():
        total = sum(c.values()) or 1
        scored = []
        for term, tf in c.items():
            if tf < min_count:
                continue
            idf = math.log((1 + n) / (1 + df[term])) + 1
            scored.append(((tf / total) * idf * (1.5 if " " in term else 1.0), term))
        picked = []
        for _, term in sorted(scored, reverse=True):
            # skip single words already covered by a chosen phrase, and vice versa
            if any(term in p.split() or p in term.split() for p in picked):
                continue
            picked.append(term)
            if len(picked) == top_n:
                break
        out[k] = picked
    return out
