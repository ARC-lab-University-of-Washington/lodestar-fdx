# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file flagship_ablation.py
# @brief ABLATION on the flagship A13 O2 replay: is human speech as important to the LLM as raw fault indicators? Each verbatim crew utterance is hand-decompos.
#
# ed into SIGNALS (panel/annunciator nomenclature + instrument readings: MAIN
# B BUS UNDERVOLT, O2 QUANTITY, PGNCS, barber pole, numeric readings) vs
# SPEECH (the crew's sensory/subjective narration: 'a pretty large bang',
# 'venting something out into space', 'gas of some sort'). We replay
# cumulatively (turn k = utterances 1..k) under FULL / SIGNALS / SPEECH, score
# subsystem-match(EPS) + documented-cause coverage(component_terms) per turn,
# over R seeds. Question: same conclusion (EPS / O2 tank), same turn, same
# coverage — with speech removed? Run from analysis/ with
# CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=-1. Env: REPS (default 3).
#
"""ABLATION on the flagship A13 O2 replay: is human speech as important to the LLM as raw fault
indicators? Each verbatim crew utterance is hand-decomposed into SIGNALS (panel/annunciator nomenclature
+ instrument readings: MAIN B BUS UNDERVOLT, O2 QUANTITY, PGNCS, barber pole, numeric readings) vs
SPEECH (the crew's sensory/subjective narration: 'a pretty large bang', 'venting something out into
space', 'gas of some sort'). We replay cumulatively (turn k = utterances 1..k) under FULL / SIGNALS /
SPEECH, score subsystem-match(EPS) + documented-cause coverage(component_terms) per turn, over R seeds.
Question: same conclusion (EPS / O2 tank), same turn, same coverage — with speech removed?
Run from analysis/ with CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=-1. Env: REPS (default 3)."""
import sys
import json, os, re, sys, time
sys.path.insert(0, "scripts")
from subsystem_match import subsystem_match
EDGE = os.environ.get("LODESTAR_EDGE_DIR", ".."); sys.path.insert(0, EDGE)
from lodestar_edge.segment import load_corpus
from lodestar_edge.retriever import SalienceBM25Retriever
from lodestar_edge.graph import CorpusGraph
from lodestar_edge.diagnose import diagnose_chain

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import _d, _DATA_ROOT   # noqa: E402


TE = os.environ.get("LODESTAR_ATLAS_DIR", "../../apollo-anomaly-atlas/data")
BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("LODESTAR_EDGE_MODEL", "llama3.2:3b")
REPS = int(os.environ.get("REPS", "3"))
ep = json.load(open(f"{TE}/apollo13_o2_episode.json", encoding="utf-8"))
TERMS = ep["resolution_anchor"]["component_terms"]

# hand decomposition of each verbatim utterance: (FULL, SIGNALS-only, SPEECH-only)
TURNS = [
 ("Okay, Houston, we've had a problem here.",
  "",
  "we've had a problem here."),
 ("Houston, we've had a problem. We've had a MAIN B BUS UNDERVOLT.",
  "MAIN B BUS UNDERVOLT.",
  "we've had a problem."),
 ("Right now the voltage is looking good. And we had a pretty large bang associated with the caution and warning. And as I recall, MAIN B was the one that had an amp spike on it once before.",
  "voltage. caution and warning. MAIN B. amp spike.",
  "the voltage is looking good. a pretty large bang. as I recall it was the one that had it once before."),
 ("And, Houston, we had a RESTART on our computer and we had a PGNCS light and the RESTART RESET.",
  "RESTART computer. PGNCS light. RESTART RESET.",
  "we had one on ours and we had a and it."),
 ("We're looking at our SERVICE MODULE RCS HELIUM 1. We have B is barber poled and D is barber poled.",
  "SERVICE MODULE RCS HELIUM 1. B barber poled. D barber poled.",
  "we're looking at ours. we have one and another."),
 ("MAIN A UNDERVOLT now, too. It's reading about 25-1/2, MAIN B is reading zip right now.",
  "MAIN A UNDERVOLT. 25-1/2 volts. MAIN B reading zip.",
  "it's like that now too. it's reading about that. it's nothing right now."),
 ("And, Jack, our O2 QUANTITY number 2 tank is reading zero.",
  "O2 QUANTITY number 2 tank reading zero.",
  "and ours is reading nothing."),
 ("And it looks to me, looking out the hatch, that we are venting something. We are venting something out into space. It's a gas of some sort, coming out of window 1.",
  "window 1.",
  "it looks to me, looking out the hatch, that we are venting something. we are venting something out into space. it's a gas of some sort, coming out."),
]

def _norm(t): return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower()).split())
def coverage(text):
    tn = _norm(text); hits = [t for t in TERMS if _norm(t) in tn]
    return len(hits) / len(TERMS), hits

segs = load_corpus(os.path.join(EDGE, "lodestar_edge", "corpus.json"))
r = SalienceBM25Retriever(segs); g = CorpusGraph(r.segments)
# warm
diagnose_chain("MAIN B BUS UNDERVOLT", r, g, model=MODEL, base_url=BASE, k_entries=8, pool=16)
print(f"flagship ablation: {len(TURNS)} turns x 3 conditions x {REPS} reps; subsystem target EPS; {len(TERMS)} component_terms\n", flush=True)

COND = {"FULL": 0, "SIGNALS": 1, "SPEECH": 2}
res = {c: {"first_eps": [], "cov_by_turn": [[] for _ in TURNS]} for c in COND}
OUT = open(os.environ.get("OUT", _d("flagship_ablation.jsonl")), "w", encoding="utf-8")
for rep in range(REPS):
    for c, ci in COND.items():
        first = None
        for k in range(len(TURNS)):
            q = " ".join(TURNS[j][ci] for j in range(k + 1)).strip()
            if not q:
                res[c]["cov_by_turn"][k].append(0.0); continue
            d = diagnose_chain(q, r, g, model=MODEL, base_url=BASE, k_entries=8, pool=16)
            txt = "" if d["declined"] else d.get("diagnosis", "")
            eps = subsystem_match(txt, "EPS", d["declined"])
            cov, hits = coverage(txt)
            res[c]["cov_by_turn"][k].append(cov)
            if eps and first is None: first = k + 1
            OUT.write(json.dumps({"rep": rep, "cond": c, "turn": k + 1, "eps": bool(eps),
                                  "cov": round(cov, 3), "hits": hits, "q": q[:150]}) + "\n"); OUT.flush()
        res[c]["first_eps"].append(first)
        print(f"  rep{rep} {c:<8}: first EPS turn = {first}", flush=True)
OUT.close()

def summ(xs):
    got = [x for x in xs if x]
    return (f"reached {len(got)}/{len(xs)} reps, median turn {sorted(got)[len(got)//2]}" if got else "NEVER reached")
print("\n===== FLAGSHIP ABLATION RESULT =====")
for c in COND:
    print(f"\n{c}:")
    print(f"  turns-to-EPS: {summ(res[c]['first_eps'])}  (raw {res[c]['first_eps']})")
    covs = [sum(t)/len(t) if t else 0 for t in res[c]['cov_by_turn']]
    print(f"  mean coverage by turn: " + " ".join(f"t{i+1}={v:.2f}" for i, v in enumerate(covs)))
    print(f"  peak mean coverage: {max(covs):.2f}")
