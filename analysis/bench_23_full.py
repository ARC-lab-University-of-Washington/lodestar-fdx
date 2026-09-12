# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file bench_23_full.py
# @brief COMBINED bench on the 23 live-loop anomalies: reports BOTH (a) single-shot per-diagnosis latency and (b) MULTI-TURN convergence — incremental disclosu.
#
# re, terminate on the EXACT fault (distinctive documented-cause term, not
# subsystem), recording turns-to-converge and cumulative multi-turn time.
# Needs Ollama. temp 0, k_entries=8. Env: LODESTAR_EDGE_DIR, ATLAS, CALL_GAP
# (pace on tight Jetson), BENCH_LABEL.
#
"""COMBINED bench on the 23 live-loop anomalies: reports BOTH (a) single-shot per-diagnosis latency
and (b) MULTI-TURN convergence — incremental disclosure, terminate on the EXACT fault (distinctive
documented-cause term, not subsystem), recording turns-to-converge and cumulative multi-turn time.
Needs Ollama. temp 0, k_entries=8. Env: LODESTAR_EDGE_DIR, ATLAS, CALL_GAP (pace on tight Jetson), BENCH_LABEL."""
import json, os, sys, re, math, time, importlib, statistics as st
from collections import Counter
EDGE = os.environ.get("LODESTAR_EDGE_DIR", ".."); sys.path.insert(0, EDGE)
from lodestar_edge.segment import load_corpus
from lodestar_edge.retriever import SalienceBM25Retriever
from lodestar_edge.graph import CorpusGraph
from lodestar_edge.diagnose import diagnose_chain
Dd = importlib.import_module("lodestar_edge.diagnose"); ct = Dd._content_terms

ATLAS = os.environ.get("LODESTAR_ATLAS", "../../apollo-anomaly-atlas/data/atlas_parsed.json")
BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
K = int(os.environ.get("EVAL_K", "8")); GAP = float(os.environ.get("CALL_GAP", "0"))
LABEL = os.environ.get("BENCH_LABEL", "dev-box")

atlas = json.load(open(ATLAS, encoding="utf-8"))
flat = [an for m in atlas["missions"] for an in m.get("anomalies", [])]
loop = [x for x in flat if x.get("crew_initiated") and x.get("diagnosed_in_flight")]
CDF = Counter()
for x in flat:
    for w in ct(x.get("documented_cause", "")): CDF[w] += 1
def exact_terms(x):
    return {w for w in ct(x.get("documented_cause", "")) if math.log(len(flat)/(CDF.get(w,0)+1)) >= 2.5}
def hits(text, terms):
    t = (text or "").lower(); return bool(terms) and any(w in t for w in terms)
def turns_of(s):
    p = [q.strip() for q in re.split(r"(?<=[.;])\s+", s) if len(q.strip()) >= 12]
    return p or [s]

segs = load_corpus(os.path.join(EDGE, "lodestar_edge", "corpus.json"))
retr = SalienceBM25Retriever(segs); g = CorpusGraph(retr.segments)
print(f"combined bench [{LABEL}]: {len(loop)} live-loop anomalies, k={K}", flush=True)
def diag(q):
    for _ in range(3):
        r = diagnose_chain(q, retr, g, base_url=BASE, k_entries=K, pool=K + 8)
        if r["declined"] or (r.get("diagnosis") and "unavailable" not in r["diagnosis"]): return r
        time.sleep(3)
    return r

diag(loop[0]["symptom"]); print("warmup done", flush=True)
rows = []
for i, a in enumerate(loop, 1):
    terms = exact_terms(a)
    # (a) single-shot
    t0 = time.perf_counter(); ss = diag(a["symptom"]); ss_lat = time.perf_counter() - t0
    if GAP: time.sleep(GAP)
    # (b) multi-turn incremental, terminate on exact fault
    turns = turns_of(a["symptom"]); conv, cum = None, 0.0
    for k in range(1, len(turns) + 1):
        t0 = time.perf_counter(); r = diag(" ".join(turns[:k])); cum += time.perf_counter() - t0
        if GAP: time.sleep(GAP)
        if hits(r.get("diagnosis", ""), terms):
            conv = k; break
    rows.append({"ss_lat": ss_lat, "ss_exact": hits(ss.get("diagnosis", ""), terms),
                 "n_turns": len(turns), "converged_at": conv, "mt_cum": cum, "has_target": bool(terms)})
    print(f"  [{i}/{len(loop)}] ss={ss_lat:5.1f}s ss_exact={hits(ss.get('diagnosis',''),terms)} "
          f"| mt: turns={len(turns)} conv@{conv or 'never'} cum={cum:.0f}s | {a['symptom'][:34]}", flush=True)

ss = sorted(r["ss_lat"] for r in rows)
p95 = ss[min(len(ss)-1, int(0.95*len(ss)))]
scored = [r for r in rows if r["has_target"]]; conv = [r for r in scored if r["converged_at"]]
print(f"\n===== BENCH [{LABEL}] — 23 live-loop anomalies (k={K}, warm) =====")
print("(a) SINGLE-SHOT per-diagnosis latency:")
print(f"    mean {st.mean(ss):.1f}s  median {st.median(ss):.1f}s  p95 {p95:.1f}s  min {ss[0]:.1f}s  max {ss[-1]:.1f}s")
print(f"    exact-fault named in one shot: {sum(1 for r in rows if r['ss_exact'])}/{len(rows)}")
print("(b) MULTI-TURN, terminate on EXACT fault:")
print(f"    converged: {len(conv)}/{len(scored)} = {len(conv)/max(len(scored),1):.0%}")
if conv:
    print(f"    turns-to-converge: mean {st.mean([r['converged_at'] for r in conv]):.1f} median {st.median([r['converged_at'] for r in conv])} "
          f"dist {dict(sorted(Counter(r['converged_at'] for r in conv).items()))}")
    print(f"    multi-turn cumulative time-to-converge: mean {st.mean([r['mt_cum'] for r in conv]):.1f}s median {st.median([r['mt_cum'] for r in conv]):.1f}s")
print(f"    (single-sentence/forced-1-turn: {sum(1 for r in rows if r['n_turns']==1)}/{len(rows)})")
print("HUMAN air-to-ground baseline: mean 4.4 turns to diagnosis")
