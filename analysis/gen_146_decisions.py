# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file gen_146_decisions.py
# @brief Generation pass: run diagnose_chain over the 146 crew_initiated anomalies and DUMP each decision (+ metadata) to data/decisions_146_k{K}.jsonl, so acc.
#
# uracy can be scored/re-scored offline without re-generating. Resumable
# (skips symptoms already in the file). Needs Ollama. temp 0, k_entries=8.
#
"""Generation pass: run diagnose_chain over the 146 crew_initiated anomalies and DUMP each decision
(+ metadata) to data/decisions_146_k{K}.jsonl, so accuracy can be scored/re-scored offline without
re-generating. Resumable (skips symptoms already in the file). Needs Ollama. temp 0, k_entries=8."""
import sys
import json, os, sys, time, importlib
EDGE = os.environ.get("LODESTAR_EDGE_DIR", ".."); sys.path.insert(0, EDGE)
from lodestar_edge.segment import load_corpus
from lodestar_edge.retriever import SalienceBM25Retriever
from lodestar_edge.graph import CorpusGraph
from lodestar_edge.diagnose import diagnose_chain

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import _d, _DATA_ROOT   # noqa: E402


FULL_LOG = os.environ.get("FULL_LOG", _d("full_log.jsonl"))
OUT_DIR = os.environ.get("OUT_DIR", "data")
CALL_GAP = float(os.environ.get("CALL_GAP", "0"))   # inter-call pause; set >0 on memory-tight Jetson
CRIT = os.environ.get("CHASE_CRITERION", "bm25")    # 'bm25' or 'coverage' (submodular)
recs = [json.loads(l) for l in open(FULL_LOG, encoding="utf-8") if l.strip()]
ci = [r for r in recs if r.get("crew_initiated")]
K = int(os.environ.get("EVAL_K", "8"))
OUT = os.path.join(OUT_DIR, f"decisions_146_k{K}{'' if CRIT == 'bm25' else '_' + CRIT}.jsonl")
BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")

done = set()
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        try: done.add(json.loads(l)["symptom"])
        except Exception: pass

segs = load_corpus(os.path.join(EDGE, "lodestar_edge", "corpus.json"))
retr = SalienceBM25Retriever(segs); g = CorpusGraph(retr.segments)
print(f"gen: {len(ci)} crew_initiated, {len(done)} already done, k={K} -> {OUT}", flush=True)

def robust_diagnose(sym):
    """Retry on an empty/failed generation (transient Ollama blips zero out diagnose_chain silently)."""
    for attempt in range(3):
        res = diagnose_chain(sym, retr, g, base_url=BASE, k_entries=K, pool=K + 8, chase_criterion=CRIT)
        if res["declined"] or (res.get("diagnosis") and "generation unavailable" not in res["diagnosis"]):
            return res
        time.sleep(3 + 3 * attempt)   # let Ollama recover
    return res

# warm the model so the first real diagnosis doesn't race a cold/loading server
_ = robust_diagnose(ci[0]["symptom"]); print("warmup done", flush=True)
fh = open(OUT, "a", encoding="utf-8")
for i, r in enumerate(ci, 1):
    if r["symptom"] in done:
        continue
    t0 = time.perf_counter()
    res = robust_diagnose(r["symptom"])
    rec = {"mission": r["mission"], "symptom": r["symptom"], "subsystem": r.get("subsystem"),
           "in_corpus_domain": r.get("in_corpus_domain"), "documented_cause": r.get("documented_cause", ""),
           "declined": res["declined"], "decision": res.get("diagnosis", ""),
           "n_chains": res.get("n_chains", 0), "coverage": res.get("coverage"),
           "latency_s": round(time.perf_counter() - t0, 2)}
    fh.write(json.dumps(rec) + "\n"); fh.flush()
    if i % 10 == 0: print(f"  [{i}/{len(ci)}] {r['mission']} {res.get('n_chains',0)}ch {rec['latency_s']}s", flush=True)
    if CALL_GAP: time.sleep(CALL_GAP)   # let a memory-tight box settle between large-prompt calls
fh.close()
print(f"gen done -> {OUT}")
