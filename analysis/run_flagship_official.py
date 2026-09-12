# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file run_flagship_official.py
# @brief OFFICIAL flagship multi-turn run, graph-chain generation.
#
# Replays the Apollo-13 O2 crew air-to-ground report one utterance per turn
# (cumulative), runs diagnose_chain each turn, and scores with the OFFICIAL
# metric — transcript_metrics.diagnosis_coverage: fraction of the curated
# `component_terms` matched by strict normalized substring;
# 'reached/characterized' = coverage >= COV_BAR (0.25). Records the first turn
# reached (turns-to-diagnosis) and cumulative time; iterates the fixed
# utterance list (bounded, no infinite loop) and reports reached-or-not vs the
# human baseline. Needs Ollama. Env: LODESTAR_EDGE_DIR, EPISODE, CALL_GAP,
# BENCH_LABEL, COV_BAR.
#
"""OFFICIAL flagship multi-turn run, graph-chain generation. Replays the Apollo-13 O2 crew air-to-ground
report one utterance per turn (cumulative), runs diagnose_chain each turn, and scores with the OFFICIAL
metric — transcript_metrics.diagnosis_coverage: fraction of the curated `component_terms` matched by
strict normalized substring; 'reached/characterized' = coverage >= COV_BAR (0.25). Records the first
turn reached (turns-to-diagnosis) and cumulative time; iterates the fixed utterance list (bounded, no
infinite loop) and reports reached-or-not vs the human baseline. Needs Ollama.
Env: LODESTAR_EDGE_DIR, EPISODE, CALL_GAP, BENCH_LABEL, COV_BAR."""
import json, os, sys, re, time, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subsystem_match import subsystem_match, aliases_for  # our success criterion
EDGE = os.environ.get("LODESTAR_EDGE_DIR", ".."); sys.path.insert(0, EDGE)
from lodestar_edge.segment import load_corpus
from lodestar_edge.retriever import SalienceBM25Retriever
from lodestar_edge.graph import CorpusGraph
from lodestar_edge.diagnose import diagnose_chain

EPISODE = os.environ.get("LODESTAR_EPISODE", "../eval/data/apollo13_o2_episode.json")
BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
K = int(os.environ.get("EVAL_K", "8")); GAP = float(os.environ.get("CALL_GAP", "0"))
COV_BAR = float(os.environ.get("COV_BAR", "0.25")); LABEL = os.environ.get("BENCH_LABEL", "dev-box")
SUBSYS = os.environ.get("SUBSYS", "EPS")            # ground-truth subsystem (atlas labels A13 O2 = EPS)
SCORE_MODE = os.environ.get("SCORE_MODE", "coverage")  # 'coverage' (12-term bar) or 'subsystem' (our metric)

# --- OFFICIAL metric, inlined verbatim from transcript_metrics.diagnosis_coverage ---
def _norm(t): return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower()).split())
def coverage(output, terms):
    text = _norm(output)
    hits = [t for t in terms if _norm(t) in text]
    return (len(hits) / len(terms) if terms else 0.0), hits

ep = json.load(open(EPISODE, encoding="utf-8"))
utt = [u["text"] for u in ep["input_utterances"]]
terms = ep["resolution_anchor"]["component_terms"]
hb = ep.get("human_baseline", {})
human_turn = (hb.get("turns_to_diagnosis") or {}).get("to_diagnostic_recognition") or (hb.get("turns_to_diagnosis") or {}).get("to_diagnosis")

segs = load_corpus(os.path.join(EDGE, "lodestar_edge", "corpus.json"))
r = SalienceBM25Retriever(segs); g = CorpusGraph(r.segments)
print(f"OFFICIAL flagship run [{LABEL}]: {len(utt)} crew utterances, {len(terms)} component_terms, "
      f"reached bar = coverage >= {COV_BAR}", flush=True)

def diag(q):
    for _ in range(3):
        res = diagnose_chain(q, r, g, base_url=BASE, k_entries=K, pool=K + 8)
        if res["declined"] or (res.get("diagnosis") and "unavailable" not in res["diagnosis"]): return res
        time.sleep(3)
    return res

# gently cold-load the model on a TINY prompt first — a full-symptom warmup + cold-load spikes the
# KV-cache and OOMs the 8 GB Jetson on a fresh reboot.
try:
    requests.post(f"{BASE.rstrip('/')}/api/chat", json={"model": "llama3.2:3b", "stream": False,
        "keep_alive": -1, "messages": [{"role": "user", "content": "ok"}],
        "options": {"num_predict": 1}}, timeout=(10, 200))
except Exception:
    pass
print(f"model preloaded (scoring by {SCORE_MODE}; subsystem target = {SUBSYS})", flush=True)
acc, cum, first_reached, first_sub, best = "", 0.0, None, None, 0.0
for t, u in enumerate(utt, 1):
    acc = (acc + " " + u).strip()
    t0 = time.perf_counter(); res = diag(acc); cum += time.perf_counter() - t0
    if GAP: time.sleep(GAP)
    decision = res.get("diagnosis", "") if not res["declined"] else ""
    cov, hits = coverage(decision, terms)
    best = max(best, cov)
    sub_ok = subsystem_match(decision, SUBSYS, res["declined"])
    # which component_terms of the target subsystem did it name (for reporting)
    named = [h for h in hits]
    if cov >= COV_BAR and first_reached is None:
        first_reached = t
    if sub_ok and first_sub is None:
        first_sub = t
    tag = "<SUBSYS MATCH>" if sub_ok else ("<COV REACHED>" if cov >= COV_BAR else "")
    print(f"  turn {t}: subsystem={'EPS-MATCH' if sub_ok else 'no'} cov={cov:.2f} {tag} "
          f"cum={cum:.0f}s | names={named}", flush=True)
    stop = (sub_ok if SCORE_MODE == "subsystem" else cov >= COV_BAR)
    if stop and os.environ.get("STOP_AT_BAR") == "1":
        print(f"  -> stopping at the finish (reached at turn {t}); not replaying further utterances.",
              flush=True)
        break

print(f"\n===== FLAGSHIP [{LABEL}] Apollo-13 O2 replay — SUBSYSTEM-MATCH scoring =====")
print(f"subsystem target: {SUBSYS} (atlas label for this event); aliases e.g. {aliases_for(SUBSYS)[:6]}")
if first_sub:
    print(f"NAMES THE RIGHT SUBSYSTEM (component) at turn {first_sub} — vs the human loop at turn {human_turn}")
else:
    print(f"never named the {SUBSYS} subsystem across {len(utt)} turns")
print(f"(reference) coverage bar {COV_BAR}: {'reached at turn '+str(first_reached) if first_reached else 'not reached'}; peak coverage {best:.2f}")
print(f"HUMAN baseline: characterized at turn {human_turn}")
