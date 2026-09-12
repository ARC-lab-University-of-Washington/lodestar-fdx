# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file latency_bench_23.py
# @brief LATENCY bench on EXACTLY the 23 anomalies that carry a human loop — crew_initiated AND diagnosed_in_flight (diagnosed over a live air-to-ground exchan.
#
# ge). Measures Lodestar's per-diagnosis latency (single-shot = 1 turn) and
# pairs it with the human turns/time baseline for those that have it. Needs
# Ollama. temp 0, k_entries=8. Env: LODESTAR_EDGE_DIR, ATLAS, CALL_GAP (pace
# on tight Jetson), BENCH_LABEL.
#
"""LATENCY bench on EXACTLY the 23 anomalies that carry a human loop — crew_initiated AND
diagnosed_in_flight (diagnosed over a live air-to-ground exchange). Measures Lodestar's per-diagnosis
latency (single-shot = 1 turn) and pairs it with the human turns/time baseline for those that have it.
Needs Ollama. temp 0, k_entries=8. Env: LODESTAR_EDGE_DIR, ATLAS, CALL_GAP (pace on tight Jetson),
BENCH_LABEL."""
import json, os, sys, time, statistics as st
EDGE = os.environ.get("LODESTAR_EDGE_DIR", ".."); sys.path.insert(0, EDGE)
from lodestar_edge.segment import load_corpus
from lodestar_edge.retriever import SalienceBM25Retriever
from lodestar_edge.graph import CorpusGraph
from lodestar_edge.diagnose import diagnose_chain

ATLAS = os.environ.get("LODESTAR_ATLAS", "../../apollo-anomaly-atlas/data/atlas_parsed.json")
BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
K = int(os.environ.get("EVAL_K", "8"))
GAP = float(os.environ.get("CALL_GAP", "0"))
LABEL = os.environ.get("BENCH_LABEL", "dev-box")

atlas = json.load(open(ATLAS, encoding="utf-8"))
flat = [an for m in atlas["missions"] for an in m.get("anomalies", [])]
loop = [x for x in flat if x.get("crew_initiated") and x.get("diagnosed_in_flight")]
print(f"latency bench [{LABEL}]: {len(loop)} live-human-loop anomalies (crew_initiated & diagnosed_in_flight), k={K}", flush=True)

segs = load_corpus(os.path.join(EDGE, "lodestar_edge", "corpus.json"))
retr = SalienceBM25Retriever(segs); g = CorpusGraph(retr.segments)

def one(sym):
    for _ in range(3):
        t0 = time.perf_counter()
        r = diagnose_chain(sym, retr, g, base_url=BASE, k_entries=K, pool=K + 8)
        dt = time.perf_counter() - t0
        if r["declined"] or (r.get("diagnosis") and "unavailable" not in r["diagnosis"]):
            return dt, r
        time.sleep(3)
    return dt, r

_ = one(loop[0]["symptom"]); print("warmup done", flush=True)   # exclude cold-load
lats = []
for i, a in enumerate(loop, 1):
    dt, r = one(a["symptom"])
    lats.append(dt)
    ht = a.get("human_turns_to_diagnosis"); hs = a.get("human_time_s")
    print(f"  [{i}/{len(loop)}] {dt:5.1f}s  cov={r.get('coverage')}  human_turns={ht if isinstance(ht,(int,float)) and ht>0 else '-'} "
          f"human_time={hs if isinstance(hs,(int,float)) and hs>0 else '-'}s  {a['symptom'][:40]}", flush=True)
    if GAP: time.sleep(GAP)

lats.sort()
p95 = lats[min(len(lats)-1, int(0.95*len(lats)))]
hturns = [a["human_turns_to_diagnosis"] for a in loop if isinstance(a.get("human_turns_to_diagnosis"),(int,float)) and a["human_turns_to_diagnosis"]>0]
htime = [a["human_time_s"] for a in loop if isinstance(a.get("human_time_s"),(int,float)) and a["human_time_s"]>0]
print(f"\n===== LATENCY [{LABEL}] over the {len(loop)} live-loop anomalies (k={K}, warm) =====")
print("Lodestar time-to-diagnosis (single-shot, 1 turn):")
print(f"  mean {st.mean(lats):.1f}s  median {st.median(lats):.1f}s  p95 {p95:.1f}s  min {lats[0]:.1f}s  max {lats[-1]:.1f}s")
if hturns:
    print(f"Human air-to-ground baseline: turns for {len(hturns)} of the {len(loop)} (mean {st.mean(hturns):.1f}, "
          f"range {min(hturns)}-{max(hturns)}); time for {len(htime)} (mean {st.mean(htime):.0f}s)")
else:
    print("Human baseline: no turns/time extracted for these anomalies")
