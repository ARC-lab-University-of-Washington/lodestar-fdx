# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file retriever.py
# @brief BM25 lexical retrieval over the distilled corpus, driven by fault salience.
#
# Deterministic and CPU-only. No embedding model, no torch.
#
"""BM25 lexical retrieval over the distilled corpus, driven by fault SALIENCE.

Why BM25-only on the edge: on the OCR'd Apollo manual corpus, dense embeddings (nomic over
scanned text) and graph routing were empirically found to POLLUTE retrieval — surfacing
unrelated boilerplate — while lexical BM25 on the extracted fault salience recovers the exact
engineering tokens (bus IDs, part/alarm names) that matter. Dropping the dense channel also
removes the embedding model + torch from the Jetson image (smaller, lower power).
"""
import re
from rank_bm25 import BM25Okapi

# ---- boilerplate filter (front-matter / illustration lists / near-empty pages) ----
_MARKERS = ("list of illustrations", "list of tables", "table of contents",
            "list of effective pages", "index of", "illustrations (cont")


def is_boilerplate(raw: str) -> bool:
    """@brief Detect front-matter, illustration lists and near-empty pages.

    Applied at index time; drops 222 of the 3,370 corpus segments, leaving 3,148 retrievable.

    @param raw Verbatim OCR text of a segment.
    @return True if the segment is boilerplate and should not be indexed.
    """
    t = (raw or "").lower()
    words = re.findall(r"[a-z]{3,}", t)
    if len(words) < 6:
        return True
    if any(m in t for m in _MARKERS):
        return True
    if (t.count("figure") + t.count(" page ")) >= 3 and len(words) < 50:
        return True
    if ("basic date" in t or "changed" in t) and len(words) < 12:
        return True
    return False


# ---- fault salience: keep the indication tokens, drop the conversational filler ----
_CAPS = re.compile(r"\b(?:[A-Z0-9][A-Z0-9/\-]*(?:\s+[A-Z0-9][A-Z0-9/\-]*){0,4})\b")
_LEX = ["undervolt", "venting", "bang", "barber", "amp spike", "restart", "zero",
        "quantity", "pressure", "fuel cell", "bus", "cryo", "oxygen", "helium",
        "gimbal lock", "vent", "cabin", "leak", "short", "alarm", "temperature"]
_STOP = {"OK", "OKAY", "I", "A", "AND", "THE", "WE", "HOUSTON", "JACK", "JIM", "FRED",
         "RIGHT", "NOW", "MAIN", "STAND", "BY", "ROGER", "YES", "NO", "GO", "OVER"}


def salience(text: str) -> str:
    """@brief Extract the fault-indication tokens from a symptom report.

    Keeps capitalised annunciator strings (bus IDs, part and alarm names) plus a fixed
    fault lexicon; drops conversational filler and crew first names.

    @param text Crew symptom report, verbatim.
    @return Space-joined salient tokens, de-duplicated, order preserved. "" if none match.
    """
    caps = [m.group().strip() for m in _CAPS.finditer(text or "")]
    caps = [c for c in caps if len(c) >= 3 and c.upper() not in _STOP and not c.isdigit()]
    lex = [w for w in _LEX if w in (text or "").lower()]
    seen, out = set(), []
    for tok in caps + lex:
        k = tok.lower()
        if k not in seen:
            seen.add(k); out.append(tok)
    return " ".join(out)


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\._]*")


def tokenize(text: str) -> list:
    """@brief Tokenise for BM25 — lowercase, except tokens containing digits.

    Part numbers and bus IDs keep their case so `MAIN B` and `main b` do not collide
    with unrelated prose.

    @param text Any string.
    @return list of token strings.
    """
    return [t if any(c.isdigit() for c in t) else t.lower()
            for t in _TOKEN_RE.findall(text)]


# conversational / narration filler stripped from a raw symptom when salience yields no annunciator
# tokens — so a plain-English fault report ("water leak observed at the B-nut connection") still
# retrieves the right manuals instead of auto-declining.
_FALLBACK_STOP = set("the a an of to in on and or with at from for was were is are be been being this "
    "that these those it its they them their has had have will would could should may might not no nor "
    "due most probable caused cause failure failed fault anomaly report system during after before "
    "reported observed crew when then approximately about around also only which while into onto over "
    "under out off up down but if so as than very more less each any all some there here we i he she "
    "his her him our your at-the".split())


def fallback_tokens(text: str) -> list:
    """@brief Content terms of a raw symptom, used when salience yields nothing.

    Lets a plain-English fault report retrieve instead of auto-declining.

    @param text Crew symptom report, verbatim.
    @return list of content tokens with conversational filler removed.
    """
    return [t for t in tokenize(text) if len(t) >= 3 and t not in _FALLBACK_STOP]


class SalienceBM25Retriever:
    """@brief Boilerplate-filtered BM25 index; queries retrieve on extracted fault salience.

    @var segments   The indexed segments, after the boilerplate filter.
    @var n_dropped  How many input segments the filter removed.
    @var pos        Map of segment id to its row in the BM25 matrix.
    """

    def __init__(self, segments: list):
        """@brief Build the index.

        @param segments list of @ref Segment, typically from @ref load_corpus.
        """
        self.segments = [s for s in segments if not is_boilerplate(s.raw_text)]
        self._bm25 = BM25Okapi([tokenize(s.search_text()) for s in self.segments])
        self.n_dropped = len(segments) - len(self.segments)
        self.pos = {s.id: i for i, s in enumerate(self.segments)}   # id -> row in the BM25 matrix

    def raw_scores(self, query_text: str):
        """@brief BM25 score of every indexed segment against a literal query.

        @param query_text Query string, tokenised as-is with no salience extraction.
        @return list[float] aligned to self.segments; index it with `self.pos[seg.id]`.
        """
        q = tokenize(query_text)
        if not q:
            return [0.0] * len(self.segments)
        return [float(x) for x in self._bm25.get_scores(q)]

    def effective_scores(self, symptom: str):
        """@brief The score array actually used for retrieval on this symptom.

        Salience if it matches, else the raw-symptom fallback. The graph chase scores
        next-hop endpoints with THIS, so the chase follows the same query that found the
        entry, including the fallback case.

        @param symptom Crew symptom report, verbatim.
        @return list[float] aligned to self.segments.
        """
        sal = salience(symptom)
        scores = self._bm25.get_scores(tokenize(sal)) if sal else None
        if scores is None or max(scores, default=0.0) <= 0:
            toks = fallback_tokens(symptom)
            scores = self._bm25.get_scores(toks) if toks else [0.0] * len(self.segments)
        return [float(x) for x in scores]

    def score_of(self, scores, seg) -> float:
        """@brief Look up a segment's precomputed BM25 score.

        @param scores Score array from @ref effective_scores or @ref raw_scores.
        @param seg    A @ref Segment.
        @return Its score, or 0.0 if the segment is not indexed.
        """
        i = self.pos.get(seg.id)
        return scores[i] if i is not None else 0.0

    def search_scored(self, symptom: str, k: int = 6) -> list:
        """@brief Retrieve the top-k entry segments for a symptom, with scores.

        @param symptom Crew symptom report, verbatim.
        @param k       Maximum segments to return.
        @return list of (@ref Segment, score) with score > 0; [] if nothing matched.

        Retrieve top-k by BM25 as [(segment, bm25_score)] (score>0). Query = the fault SALIENCE
        (caps annunciators + fault lexicon); if salience is empty or matches nothing, FALL BACK to the
        raw symptom's content terms so a plain-English fault report still retrieves rather than
        auto-declining. Returns [] only when even the fallback matches nothing. (Whether the symptom is
        actually in coverage is signalled by the annunciation coverage dial, not by declining here.)"""
        sal = salience(symptom)
        scores = self._bm25.get_scores(tokenize(sal)) if sal else None
        if scores is None or max(scores, default=0.0) <= 0:
            toks = fallback_tokens(symptom)
            if not toks:
                return []
            scores = self._bm25.get_scores(toks)
        ranked = sorted(zip(self.segments, scores), key=lambda x: x[1], reverse=True)
        return [(seg, float(sc)) for seg, sc in ranked[:k] if sc > 0]

    def search(self, symptom: str, k: int = 6) -> list:
        """@brief Retrieve the top-k entry segments for a symptom.

        @param symptom Crew symptom report, verbatim.
        @param k       Maximum segments to return.
        @return list of @ref Segment. Scores are discarded; see @ref search_scored.
        """
        return [seg for seg, _ in self.search_scored(symptom, k)]
