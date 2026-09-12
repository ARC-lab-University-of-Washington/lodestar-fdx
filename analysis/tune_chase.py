# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file tune_chase.py
# @brief Tune the chase: sweep rel_threshold x max_depth on the SHIPPED graph/retriever.
#
# Measure mean chain length, cause-term reachability of the chains, and the
# degenerate (length-1) rate, vs flat top-8 BM25. Deterministic, no model
# calls. Picks defaults for diagnose_chain.
#
"""Tune the chase: sweep rel_threshold x max_depth on the SHIPPED graph/retriever. Measure mean chain
length, cause-term reachability of the chains, and the degenerate (length-1) rate, vs flat top-8 BM25.
Deterministic, no model calls. Picks defaults for diagnose_chain."""
import sys
import json, os, sys, math, importlib, statistics as st
from collections import Counter
EDGE = ".."; sys.path.insert(0, EDGE)
from lodestar_edge.segment import load_corpus
from lodestar_edge.retriever import SalienceBM25Retriever, salience
from lodestar_edge.graph import CorpusGraph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import _d, _DATA_ROOT   # noqa: E402

D = importlib.import_module("lodestar_edge.diagnose"); ct = D._content_terms

CORPUS = os.path.join(EDGE, "lodestar_edge", "corpus.json")
recs = [json.loads(l) for l in open(_d("full_log.jsonl"), encoding="utf-8") if l.strip()]
segs = load_corpus(CORPUS)
retr = SalienceBM25Retriever(segs)
g = CorpusGraph(retr.segments)
print(f"graph: {len(retr.segments)} indexed segs, resolution {g.resolution_rate:.0%}, "
      f"{g.n_edges} edges, ann_vocab {len(g.ann_vocab)}")

CDF = Counter()
for r in recs:
    for w in ct(r["documented_cause"]): CDF[w] += 1
def cause_terms(r):
    return {w for w in ct(r["documented_cause"]) if math.log(len(recs)/(CDF.get(w,0)+1)) >= 2.5}
def seg_text(s): return (s.raw_text or "") + " " + (s.component or "") + " " + (s.signature or "")
def cov(segments, terms):
    if not terms: return None
    txt = set()
    for s in segments: txt |= ct(seg_text(s))
    return len(terms & txt) / len(terms)

K = 3   # entry manuals -> up to 3 chains
print(f"(re-check) resolution {g.resolution_rate:.0%}, {g.n_edges} edges")
# precompute entries + query scores per anomaly; entries = graph-biased top-K from a top-12 pool
prep = []
for r in recs:
    pool = [seg for seg, _ in retr.search_scored(r["symptom"], k=12)]
    entries = g.pick_entries(pool, K)
    scores = retr.raw_scores(salience(r["symptom"])) if entries else None
    prep.append((r, entries, scores))

# baseline: flat top-8 raw BM25 cause coverage
flat = []
for r in recs:
    segs8 = [seg for seg, _ in retr.search_scored(r["symptom"], k=8)]
    c = cov(segs8, cause_terms(r))
    if c is not None: flat.append(c)
print(f"\nbaseline flat top-8 cause coverage: {st.mean(flat):.3f}")

print(f"\n{'rel_thr':>8}{'depth':>7}{'meanLen':>9}{'len=1%':>8}{'chainCov':>10}{'lift vs flat':>14}")
for rel in (0.3, 0.4, 0.5, 0.6, 0.7):
    for depth in (2, 3, 4):
        lens = []; covs = []; deg = 0; n = 0
        for r, entries, scores in prep:
            if not entries:
                continue
            n += 1
            chain_segs = set()
            clens = []
            for e in entries:
                chain = g.chase(e, scores, retr, rel_threshold=rel, max_depth=depth)
                clens.append(len(chain))
                for s in chain: chain_segs.add(s.id)
            id2 = {s.id: s for s in retr.segments}
            merged = [id2[i] for i in chain_segs]
            lens.append(st.mean(clens))
            if max(clens) == 1: deg += 1
            c = cov(merged, cause_terms(r))
            if c is not None: covs.append(c)
        print(f"{rel:>8.1f}{depth:>7}{st.mean(lens):>9.2f}{deg/n:>8.0%}{st.mean(covs):>10.3f}"
              f"{st.mean(covs)-st.mean(flat):>+14.3f}")
