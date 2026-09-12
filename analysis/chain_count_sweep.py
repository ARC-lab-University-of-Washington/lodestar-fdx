# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file chain_count_sweep.py
# @brief Sweep the number of chains (k_entries) and measure right-direction accuracy + latency, to find how many chains are needed to hit the target.
#
# Honest metric unchanged: the decision names >=1 distinctive documented-cause
# term (IDF>=2.5). Needs Ollama. Deterministic sample + temp 0. Env: SWEEP_N
# (sample size, default 30), SWEEP_KS (comma list, default 8,12,16,20,24).
#
"""Sweep the number of chains (k_entries) and measure right-direction accuracy + latency, to find how
many chains are needed to hit the target. Honest metric unchanged: the decision names >=1 distinctive
documented-cause term (IDF>=2.5). Needs Ollama. Deterministic sample + temp 0.
Env: SWEEP_N (sample size, default 30), SWEEP_KS (comma list, default 8,12,16,20,24)."""
import sys
import json, os, sys, math, time, importlib
from collections import Counter
EDGE = ".."; sys.path.insert(0, EDGE)
from lodestar_edge.segment import load_corpus
from lodestar_edge.retriever import SalienceBM25Retriever
from lodestar_edge.graph import CorpusGraph
from lodestar_edge.diagnose import diagnose_chain

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import _d, _DATA_ROOT   # noqa: E402

Dd = importlib.import_module("lodestar_edge.diagnose"); ct = Dd._content_terms

N = int(os.environ.get("SWEEP_N", "30"))
KS = [int(x) for x in os.environ.get("SWEEP_KS", "8,12,16,20,24").split(",")]
recs = [json.loads(l) for l in open(_d("full_log.jsonl"), encoding="utf-8") if l.strip()]
CDF = Counter()
for r in recs:
    for w in ct(r["documented_cause"]): CDF[w] += 1
def cause_terms(r):
    return {w for w in ct(r["documented_cause"]) if math.log(len(recs)/(CDF.get(w,0)+1)) >= 2.5}
def right(text, terms):
    if not terms: return None
    t = (text or "").lower()
    return any(w in t for w in terms)

elig = [r for r in recs if cause_terms(r)]
step = max(1, len(elig) // N)
sample = elig[::step][:N]

segs = load_corpus(os.path.join(EDGE, "lodestar_edge", "corpus.json"))
retr = SalienceBM25Retriever(segs)
g = CorpusGraph(retr.segments)
BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
print(f"sweep N={len(sample)}  ks={KS}  graph {g.n_edges} edges", flush=True)

print(f"\n{'k_entries':>10}{'accuracy':>12}{'mean_lat':>10}{'mean_chains':>13}")
for k in KS:
    nr = ok = 0; t = 0.0; nch = 0
    for r in sample:
        ctm = cause_terms(r)
        t0 = time.perf_counter()
        res = diagnose_chain(r["symptom"], retr, g, base_url=BASE, k_entries=k, pool=k + 8)
        t += time.perf_counter() - t0
        nch += res.get("n_chains", 0)
        txt = res.get("diagnosis", "") if not res["declined"] else ""
        rr = right(txt, ctm)
        if rr is not None:
            nr += 1; ok += 1 if rr else 0
    acc = ok / nr if nr else 0
    print(f"{k:>10}{ok}/{nr} = {acc:.0%}{t/len(sample):>10.1f}s{nch/len(sample):>13.1f}", flush=True)
