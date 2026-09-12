# AI use statement: docs/AI_USE.md
##
# @file bm25_index.py
# @brief Lexical (BM25) retrieval — §4.4 bullet 3.
#
"""
Lexical (BM25) retrieval — §4.4 bullet 3.

Recovers exact engineering tokens (valve IDs, part numbers, error codes)
that dense embeddings blur. Built over the same segment set as the graph
and dense indices; tokenization is kept simple and case-preserving for
alphanumeric part/error codes.
"""
import re
from rank_bm25 import BM25Okapi
from schema import Segment

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\._]*")


def tokenize(text: str) -> list[str]:
    # Lowercase for words, but keep digit-bearing tokens (e.g. "SOV-204", "E-0147")
    # intact so codes aren't destroyed by splitting on '-'.
    return [t if any(c.isdigit() for c in t) else t.lower()
            for t in _TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, segments: list[Segment]):
        self.segments = segments
        self._corpus_tokens = [tokenize(s.search_text()) for s in segments]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, k: int = 20) -> list[tuple[Segment, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.segments, scores), key=lambda x: x[1], reverse=True)
        return [(seg, score) for seg, score in ranked[:k] if score > 0]
