# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file eval_coverage_convergence.py
# @brief Does the ANNUNCIATION coverage dial converge with GENERAL corpus coverage? For each of the 146 crew-observed anomalies recompute (deterministically): .
#
# the dial (annunciation-vocab match), and two INDEPENDENT general-coverage
# signals — retrieval strength (BM25 top-1) and cause-in-chains (fraction of
# the fault's distinctive documented-cause terms actually present in the
# retrieved chain segments). Then: correlate the dial with general coverage,
# and rate subsystem accuracy low->high by BOTH, to see if accuracy converges
# the same way. No model calls (chains rebuilt deterministically).
#
"""Does the ANNUNCIATION coverage dial converge with GENERAL corpus coverage? For each of the 146
crew-observed anomalies recompute (deterministically): the dial (annunciation-vocab match), and two
INDEPENDENT general-coverage signals — retrieval strength (BM25 top-1) and cause-in-chains (fraction
of the fault's distinctive documented-cause terms actually present in the retrieved chain segments).
Then: correlate the dial with general coverage, and rate subsystem accuracy low->high by BOTH, to see
if accuracy converges the same way. No model calls (chains rebuilt deterministically)."""
import sys
import json, os, sys, math, re, statistics as st
from collections import Counter
EDGE = ".."; sys.path.insert(0, EDGE)
from lodestar_edge.segment import load_corpus
from lodestar_edge.retriever import SalienceBM25Retriever
from lodestar_edge.graph import CorpusGraph
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import _d, _DATA_ROOT   # noqa: E402

D = importlib.import_module("lodestar_edge.diagnose"); ct = D._content_terms

# reuse the subsystem matcher from the breakdown script
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_breakdown.py")).read().split("recs_full =")[0])   # ALIASES, aliases_for, names, is_in

rows = [json.loads(l) for l in open(_d("decisions_146_k8.jsonl"), encoding="utf-8") if l.strip()]
recs_full = [json.loads(l) for l in open(_d("full_log.jsonl"), encoding="utf-8") if l.strip()]
CDF = Counter()
for r in recs_full:
    for w in ct(r["documented_cause"]): CDF[w] += 1
def cause_terms(dc):
    return {w for w in ct(dc) if math.log(len(recs_full)/(CDF.get(w,0)+1)) >= 2.5}

segs = load_corpus(os.path.join(EDGE, "lodestar_edge", "corpus.json"))
retr = SalienceBM25Retriever(segs); g = CorpusGraph(retr.segments)
def seg_text(s): return (s.raw_text or "") + " " + (s.component or "") + " " + (s.signature or "")

for r in rows:
    sym = r["symptom"]
    scored = retr.search_scored(sym, k=16)
    r["_strength"] = scored[0][1] if scored else 0.0
    entries = g.pick_entries([s for s, _ in scored], 8)
    escores = retr.effective_scores(sym)
    chain_ids = set()
    for e in entries:
        for s in g.chase(e, escores, retr, 0.4, 3): chain_ids.add(s.id)
    txt = set()
    for sid in chain_ids: txt |= ct(seg_text(g.by_id[sid]))
    cts = cause_terms(r["documented_cause"])
    r["_causecov"] = (len(cts & txt) / len(cts)) if cts else None
    r["_dial"] = r.get("coverage")
    r["_sub"] = (not r["declined"]) and names(r.get("decision",""), aliases_for(r.get("subsystem")))

def pear(xs, ys):
    n = len(xs); mx = st.mean(xs); my = st.mean(ys)
    num = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a-mx)**2 for a in xs) * sum((b-my)**2 for b in ys))
    return num/den if den else 0.0
def auc(pos, neg):
    if not pos or not neg: return .5
    c = t = 0
    for a in pos:
        for b in neg:
            t += 1; c += 1 if a > b else .5 if a == b else 0
    return c/t

# --- convergence of the dial with general coverage ---
have = [r for r in rows if r["_causecov"] is not None]
d = [r["_dial"] for r in have]; cc = [r["_causecov"] for r in have]; strg = [r["_strength"] for r in have]
print(f"=== Does the annunciation dial converge with GENERAL coverage? (n={len(have)}) ===")
print(f"  corr(dial, cause-in-chains)   = {pear(d, cc):+.2f}")
print(f"  corr(dial, retrieval-strength)= {pear(d, strg):+.2f}")
print(f"  corr(cause-cov, retr-strength)= {pear(cc, strg):+.2f}")
print(f"  mean cause-in-chains coverage = {st.mean(cc):.2f}  (this is the ~0.3 corpus ceiling)")

def rate(key, label):
    rs = sorted([r for r in rows if r.get(key) is not None], key=lambda r: r[key])
    q = len(rs) // 5
    print(f"\nRATED LOW -> HIGH by {label} (quintiles)")
    print(f"  {'bin':12}{'range':>14}{'n':>4}{'subsystem acc':>15}{'mean dial':>11}")
    for i in range(5):
        seg = rs[i*q:(i+1)*q] if i < 4 else rs[4*q:]
        if not seg: continue
        sub = sum(r["_sub"] for r in seg); sn = len(seg)
        dl = st.mean([r["_dial"] for r in seg if r["_dial"] is not None])
        print(f"  {['Q1 low','Q2','Q3','Q4','Q5 high'][i]:12}{f'{seg[0][key]:.2f}-{seg[-1][key]:.2f}':>14}"
              f"{sn:>4}{f'{sub}/{sn}={sub/sn:.0%}':>15}{dl:>11.2f}")
    pos = [r[key] for r in rs if r["_sub"]]; neg = [r[key] for r in rs if not r["_sub"]]
    print(f"  {label} predicts subsystem-correct: AUC={auc(pos, neg):.2f}")

rate("_causecov", "GENERAL coverage (cause-in-chains)")
rate("_strength", "retrieval strength (BM25 top-1)")
rate("_dial", "annunciation dial (recap)")
