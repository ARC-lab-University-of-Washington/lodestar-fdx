# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file chase_submodular_experiment.py
# @brief Submodular COVERAGE-greedy chase vs the shipped BM25-RELEVANCE-greedy chase.
#
# The current chase picks the neighbor with max symptom-BM25 (a
# modular/relevance objective -> rewards symptom-restating, redundant hops).
# The coverage variant picks the neighbor with max MARGINAL NEW coverage (IDF-
# weighted new content terms not yet in the chain) -> submodular, penalizes
# redundancy. Measures cause-term coverage of the resulting chains.
# Deterministic, no model calls.
#
"""Submodular COVERAGE-greedy chase vs the shipped BM25-RELEVANCE-greedy chase. The current chase picks
the neighbor with max symptom-BM25 (a modular/relevance objective -> rewards symptom-restating,
redundant hops). The coverage variant picks the neighbor with max MARGINAL NEW coverage (IDF-weighted
new content terms not yet in the chain) -> submodular, penalizes redundancy. Measures cause-term
coverage of the resulting chains. Deterministic, no model calls."""
import sys
import json, os, sys, math, importlib, statistics as st
from collections import Counter
EDGE = ".."; sys.path.insert(0, EDGE)
from lodestar_edge.segment import load_corpus
from lodestar_edge.retriever import SalienceBM25Retriever
from lodestar_edge.graph import CorpusGraph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import _d, _DATA_ROOT   # noqa: E402

D = importlib.import_module("lodestar_edge.diagnose"); ct = D._content_terms

recs = [json.loads(l) for l in open(_d("full_log.jsonl"), encoding="utf-8") if l.strip()]
segs = load_corpus(os.path.join(EDGE, "lodestar_edge", "corpus.json"))
retr = SalienceBM25Retriever(segs); g = CorpusGraph(retr.segments)

df = Counter()
for s in retr.segments:
    for w in set(ct(s.search_text())): df[w] += 1
NS = len(retr.segments)
def idf(w): return math.log(NS / (df.get(w, 0) + 1))
def seg_terms(s): return ct((s.raw_text or "") + " " + (s.component or "") + " " + (s.signature or ""))

CDF = Counter()
for r in recs:
    for w in ct(r["documented_cause"]): CDF[w] += 1
def cause_terms(r):
    return {w for w in ct(r["documented_cause"]) if math.log(len(recs)/(CDF.get(w,0)+1)) >= 2.5}

def chase_bm25(entry, escores, max_depth=3, rel=0.4):
    return g.chase(entry, escores, retr, rel_threshold=rel, max_depth=max_depth)

def chase_cov(entry, escores, max_depth=3, rel_floor=0.0):
    """submodular: max marginal IDF-weighted NEW terms; optional light relevance guard rel_floor*entry."""
    covered = set(seg_terms(entry)); chain = [entry]; visited = {entry.id}; cur = entry
    escore_entry = retr.score_of(escores, entry) or 1e-9
    floor = rel_floor * escore_entry
    for _ in range(max_depth):
        best, best_gain = None, 0.0
        for nb in g.neighbors(cur):
            if nb.id in visited: continue
            if rel_floor and retr.score_of(escores, nb) < floor: continue
            gain = sum(idf(w) for w in (seg_terms(nb) - covered))
            if gain > best_gain: best_gain, best = gain, nb
        if best is None or best_gain <= 0: break
        chain.append(best); visited.add(best.id); covered |= seg_terms(best); cur = best
    return chain

def cov_of(chains, terms):
    if not terms: return None
    txt = set()
    for ch in chains:
        for s in ch: txt |= seg_terms(s)
    return len(terms & txt) / len(terms)

K = 8
variants = {"bm25 (shipped)": lambda e, sc: chase_bm25(e, sc),
            "coverage pure": lambda e, sc: chase_cov(e, sc, rel_floor=0.0),
            "coverage +relguard.2": lambda e, sc: chase_cov(e, sc, rel_floor=0.2)}
res = {k: [] for k in variants}
lens = {k: [] for k in variants}
for r in recs:
    ranked = [s for s, _ in retr.search_scored(r["symptom"], k=K + 8)]
    if not ranked: continue
    entries = g.pick_entries(ranked, K)
    escores = retr.effective_scores(r["symptom"])
    cts = cause_terms(r)
    for name, fn in variants.items():
        chains = [fn(e, escores) for e in entries]
        c = cov_of(chains, cts)
        if c is not None: res[name].append(c)
        lens[name].append(st.mean([len(ch) for ch in chains]))

print(f"n={len(res['bm25 (shipped)'])}  (flat top-8 cause coverage baseline = 0.147)")
print(f"{'chase variant':24}{'cause-coverage':>16}{'mean chain len':>16}")
for name in variants:
    print(f"{name:24}{st.mean(res[name]):>16.3f}{st.mean(lens[name]):>16.2f}")
