# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file graph.py
# @brief The onboard-manual GRAPH: annunciation nodes wired by their documented `links_to` edges ("this indication -> this system -> these connected components.
#
# "). BM25 finds a good entry manual; the graph is what we CHASE from there to
# build a reasoning chain of candidate causes. No model calls; see the docstring
# model calls: * edge resolution — `links_to` strings are terse component/step
# references (OCR'd); we resolve each to the segment(s) whose
# `component`/`signature` it names (>=2 shared tokens). ~98% of links RESOLVE (a resolution rate, not a precision: a 1-2 token link needs only one shared token, so generic links such as "BUS" over-connect).
# * chase() — from an entry segment, greedily follow the neighbor with the
# HIGHEST BM25 score against the fault salience, continuing while that
# endpoint score stays above a (relative, tunable) threshold. One chain per
# entry manual. This keeps the chain on the symptom while walking the
# documented connectivity toward candidate causes. Also exposes the
# annunciation-node vocabulary + `coverage()` — the validated in/out-of-corpus
# dial (a symptom made of documented-annunciation vocabulary is in coverage;
# surface-science words are not).
#
"""The onboard-manual GRAPH: annunciation nodes wired by their documented `links_to` edges
("this indication -> this system -> these connected components"). BM25 finds a good entry manual;
the graph is what we CHASE from there to build a reasoning chain of candidate causes.

Two pieces, no model calls. NOTE: chain membership is NOT fully deterministic — adjacency order
derives from set iteration, so PYTHONHASHSEED can change which of two near-identical segments is
chased and therefore which manual page is cited. Set PYTHONHASHSEED=0 for a repeatable run.
  * edge resolution — `links_to` strings are terse component/step references (OCR'd); we resolve each
    to the segment(s) whose `component`/`signature` it names (>=2 shared tokens). ~98% of links RESOLVE (a resolution rate, not a precision: a 1-2 token link needs only one shared token, so generic links such as "BUS" over-connect).
  * chase() — from an entry segment, greedily follow the neighbor with the HIGHEST BM25 score against
    the fault salience, continuing while that endpoint score stays above a (relative, tunable)
    threshold. One chain per entry manual. This keeps the chain on the symptom while walking the
    documented connectivity toward candidate causes.

Also exposes the annunciation-node vocabulary + `coverage()` — the validated in/out-of-corpus dial
(a symptom made of documented-annunciation vocabulary is in coverage; surface-science words are not).
"""
import re
import math
from collections import defaultdict

_TOK = re.compile(r"[a-z0-9]{3,}")


def _toks(x):
    return [w for w in _TOK.findall(str(x).lower())]


class CorpusGraph:
    def __init__(self, segments: list):
        # Build over the SAME (post-boilerplate-filter) segment list the retriever indexes, so a
        # segment's identity and BM25 row line up. Keyed by segment id throughout.
        self.by_id = {s.id: s for s in segments}
        # index: component/signature token -> ids that carry it (edge-resolution target lookup)
        tok_index = defaultdict(set)
        for s in segments:
            for f in (s.component, s.signature):
                for w in set(_toks(f)):
                    tok_index[w].add(s.id)
        # resolve links_to -> neighbor ids
        self.adj = defaultdict(list)
        self._n_links = 0
        self._n_resolved = 0
        for s in segments:
            seen = set()
            for link in (s.links_to or []):
                self._n_links += 1
                lt = set(_toks(link))
                if not lt:
                    continue
                # require real overlap for multi-token links; allow a single-token link (e.g. "IMU",
                # "CRYO FANS") to resolve on its one distinctive token — the chase disambiguates by BM25.
                need = 1 if len(lt) <= 2 else max(2, len(lt) // 2)
                cand = defaultdict(int)
                for w in lt:
                    for tid in tok_index.get(w, ()):
                        cand[tid] += 1
                tgts = [tid for tid, n in cand.items() if n >= need and tid != s.id]
                if tgts:
                    self._n_resolved += 1
                for tid in tgts:
                    if tid not in seen:
                        seen.add(tid)
                        self.adj[s.id].append(tid)
        # annunciation-node vocabulary (from the `symptoms` field) for the coverage dial
        self.ann_vocab = set()
        for s in segments:
            for sy in (s.symptoms or []):
                self.ann_vocab |= set(_toks(sy))
        # content-term IDF over the corpus, for the submodular COVERAGE-greedy chase criterion.
        from .diagnose import _content_terms                    # lazy: avoids import-order coupling
        self._ct = _content_terms
        self._seg_terms = {s.id: _content_terms((s.raw_text or "") + " " + (s.component or "")
                                                 + " " + (s.signature or "")) for s in segments}
        df = defaultdict(int)
        for terms in self._seg_terms.values():
            for w in terms:
                df[w] += 1
        n = max(1, len(segments))
        self._idf = {w: math.log(n / (c + 1)) for w, c in df.items()}

    # ---- diagnostics ----
    @property
    def n_edges(self):
        return sum(len(v) for v in self.adj.values())

    @property
    def resolution_rate(self):
        return self._n_resolved / self._n_links if self._n_links else 0.0

    def neighbors(self, seg):
        return [self.by_id[tid] for tid in self.adj.get(seg.id, ())]

    def has_out(self, seg) -> bool:
        return bool(self.adj.get(seg.id))

    def pick_entries(self, ranked_segs, k: int) -> list:
        """From BM25-ranked candidates, prefer the top-k that are GRAPH NODES (have outgoing edges)
        so a chain can actually be chased; fall back to plain top-k to fill. Preserves BM25 order."""
        connected = [s for s in ranked_segs if self.has_out(s)]
        out = connected[:k]
        if len(out) < k:
            for s in ranked_segs:
                if s not in out:
                    out.append(s)
                if len(out) >= k:
                    break
        return out

    # ---- coverage dial (validated in/out signal; deterministic, query-time) ----
    def coverage(self, symptom: str) -> float:
        """Fraction of the symptom's content tokens that are documented annunciation-node vocabulary.
        High => the fault is described in the manuals' annunciator language (in coverage);
        low  => surface-science / no-annunciator vocabulary (likely outside the onboard manuals)."""
        q = set(_toks(symptom))
        return (len(q & self.ann_vocab) / len(q)) if q else 0.0

    # ---- the chase: one chain per entry manual ----
    def chase(self, entry, scores, retriever, rel_threshold: float = 0.5,
              max_depth: int = 4) -> list:
        """Greedy BM25-guided walk from `entry`. At each node move to the neighbor with the highest
        endpoint BM25 score (`scores` = retriever.raw_scores(salience)); stop when the best neighbor
        falls below `rel_threshold * entry_score`, or no neighbor / max_depth / a cycle. Returns the
        chain as a list of Segments (>=1: the entry itself)."""
        entry_score = retriever.score_of(scores, entry) or 1e-9
        floor = rel_threshold * entry_score
        chain = [entry]
        visited = {entry.id}
        cur = entry
        for _ in range(max_depth):
            best, best_sc = None, -1.0
            for nb in self.neighbors(cur):
                if nb.id in visited:
                    continue
                sc = retriever.score_of(scores, nb)
                if sc > best_sc:
                    best, best_sc = nb, sc
            if best is None or best_sc < floor:
                break
            chain.append(best)
            visited.add(best.id)
            cur = best
        return chain

    def chase_coverage(self, entry, scores, retriever, rel_floor: float = 0.0,
                       max_depth: int = 3) -> list:
        """SUBMODULAR coverage-greedy walk: at each node move to the neighbor that adds the most
        MARGINAL new (IDF-weighted) content terms to the chain so far — rewarding fresh coverage of the
        fault's connected components instead of symptom-restating redundancy (the BM25-greedy failure
        mode). `rel_floor`>0 optionally gates neighbors by BM25 (>= rel_floor*entry) to stay on-topic.
        Reaches ~0.42 of distinctive cause terms vs ~0.29 for `chase()` (analysis/chase_submodular_experiment.py)."""
        covered = set(self._seg_terms.get(entry.id, ()))
        floor = rel_floor * (retriever.score_of(scores, entry) or 1e-9)
        chain = [entry]
        visited = {entry.id}
        cur = entry
        for _ in range(max_depth):
            best, best_gain = None, 0.0
            for nb in self.neighbors(cur):
                if nb.id in visited:
                    continue
                if rel_floor and retriever.score_of(scores, nb) < floor:
                    continue
                gain = sum(self._idf.get(w, 0.0) for w in (self._seg_terms.get(nb.id, set()) - covered))
                if gain > best_gain:
                    best_gain, best = gain, nb
            if best is None or best_gain <= 0:
                break
            chain.append(best)
            visited.add(best.id)
            covered |= self._seg_terms.get(best.id, set())
            cur = best
        return chain
