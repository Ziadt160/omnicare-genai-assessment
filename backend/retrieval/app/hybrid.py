"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion.

RRF is used rather than score normalization because the two retrievers produce
scores on incompatible scales (cosine similarity vs BM25 term saturation);
fusing on *rank* sidesteps the calibration problem entirely and is a pure
function of the two orderings.

Honest scope note for the README: with a two-section corpus both sections fit
inside ``top_k=3``, so hybrid retrieval cannot be shown to move recall here.
It is present because BM25 genuinely helps on exact tokens - POL-1092,
$25,000, "deductible" - which dense embeddings blur, not because it changes
the number on this fixture.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from libs.contracts import Chunk

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = RRF_K
) -> dict[str, float]:
    """Fuse ranked id lists into one score map.

    score(d) = sum over rankings of 1 / (k + rank(d)), rank starting at 1.

    Args:
        rankings: Each inner list is one retriever's ids, best first.
        k: Damping constant. 60 is the value from the original RRF paper and
            keeps any single retriever from dominating on rank-1 alone.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


# Two alternatives, money first: "$25,000" and "2,500" keep their separators,
# but "electronics," yields "electronics". A single character class cannot do
# both - it glues trailing punctuation onto words and silently breaks matching.
# Two alternatives, money first: "$25,000" and "2,500" keep their separators,
# but "electronics," yields "electronics". A single character class cannot do
# both - it glues trailing punctuation onto words and silently breaks matching.
_TOKEN_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?|[a-z][a-z0-9]*(?:-[a-z0-9]+)*")

# Deliberately small and English-only. On a two-section corpus, IDF cannot do
# this job: a term present in one of two documents still scores log(2), so a
# stray "a" in the query is enough to pull the wrong section to rank 1. That is
# a real failure this list fixes - see EV-06 in the eval suite.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here
i me my we our you your he she it its they them their
is am are was were be been being do does did doing done have has had having
can could will would shall should may might must
of in on at to for from by with about into over under up down out off
as no not so such only own same too very just
what which who whom when where why how
me my mine i'd i'm need want like get got tell show me please
""".split())


def _stem(token: str) -> str:
    """Crude, symmetric suffix stripping.

    Not linguistically correct, and deliberately so. A lexical matcher needs
    the *same* transformation on both sides, not a correct one: "cause" and
    "caused" both reducing to "caus" makes them match, which is the entire
    point. Classic Porter actually gets that pair wrong (cause->cause,
    caused->caus) because it optimises for readable stems.

    Only alphabetic tokens are stemmed, so "$25,000" and "pol-1092" pass
    through untouched - preserving exactly the advantage BM25 has over dense
    retrieval on this corpus.
    """
    if not token.isalpha() or len(token) <= 3:
        return token

    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("sses"):
        token = token[:-2]
    elif token.endswith("s") and not token.endswith(("ss", "us")):
        token = token[:-1]

    if token.endswith("ing") and len(token) > 5:
        token = token[:-3]
    elif token.endswith("ed") and not token.endswith("eed") and len(token) > 4:
        # The "eed" guard is Porter's: without it "exceeds" reduces to
        # "exceed" and then to "exc", which matches nothing.
        token = token[:-2]

    # Symmetric trailing-e drop. Unifies cause/caused, damage/damages,
    # exclude/excluded - the pairs that actually occur in this policy.
    if token.endswith("e") and len(token) > 3:
        token = token[:-1]

    return token


def tokenize(text: str) -> list[str]:
    """Split into tokens, keeping money and identifiers intact.

    Splitting on $ , - would destroy exactly the advantage BM25 has here.
    This is tokenization only; BM25 uses `analyze`.
    """
    return _TOKEN_RE.findall(text.lower())


def analyze(text: str) -> list[str]:
    """The full BM25 pipeline: tokenize, drop stopwords, stem.

    Separated from `tokenize` the way Lucene separates a tokenizer from an
    analyzer, so each can be tested for what it alone is responsible for.

    If a query is nothing but stopwords the terms are kept rather than
    returning an empty list - matching badly beats matching nothing.
    """
    tokens = tokenize(text)
    kept = [t for t in tokens if t not in STOPWORDS]
    return [_stem(t) for t in (kept or tokens)]


class BM25:
    """Okapi BM25 over an in-memory corpus. Small enough to hold entirely."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.chunks = chunks
        self.docs = [analyze(c.text) for c in chunks]
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.docs else 0.0
        self.freqs = [Counter(d) for d in self.docs]
        self.df: Counter[str] = Counter()
        for d in self.docs:
            self.df.update(set(d))
        self.n = len(self.docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def rank(self, query: str, top_k: int) -> list[str]:
        """Return chunk ids, best first."""
        terms = analyze(query)
        scored: list[tuple[float, str]] = []
        for i, chunk in enumerate(self.chunks):
            score = 0.0
            for term in terms:
                tf = self.freqs[i].get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1)
                )
                score += self._idf(term) * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((score, chunk.chunk_id))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [cid for _, cid in scored[:top_k]]


def fuse(
    dense_ids: list[str],
    sparse_ids: list[str],
    by_id: dict[str, Chunk],
    top_k: int,
) -> list[Chunk]:
    """Fuse two rankings and return chunks with ``score`` set to the RRF score."""
    scores = reciprocal_rank_fusion([dense_ids, sparse_ids])
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[Chunk] = []
    for chunk_id, score in ordered[:top_k]:
        chunk = by_id.get(chunk_id)
        if chunk is not None:
            out.append(chunk.model_copy(update={"score": score}))
    return out
