# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file run_capcom_crossmission.py
# @brief Run LODESTAR on the cross-mission CAPCOM faults to PAIR them (extends the LODESTAR-vs-CAPCOM data).
#
# These faults carry a curated CAPCOM turn/time but no subsystem label, so the
# subsystem ground-truth is assigned here from the (unambiguous) symptom —
# noted as [assigned]. LODESTAR is run multi-turn until subsystem match (same
# criterion as the 23). Dumps data/crossmission_lodestar.jsonl. Needs Ollama.
#
"""Run LODESTAR on the cross-mission CAPCOM faults to PAIR them (extends the LODESTAR-vs-CAPCOM data).
These faults carry a curated CAPCOM turn/time but no subsystem label, so the subsystem ground-truth is
assigned here from the (unambiguous) symptom — noted as [assigned]. LODESTAR is run multi-turn until
subsystem match (same criterion as the 23). Dumps data/crossmission_lodestar.jsonl. Needs Ollama."""
import sys
import json, os, sys, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subsystem_match import subsystem_match
EDGE = os.environ.get("LODESTAR_EDGE_DIR", ".."); sys.path.insert(0, EDGE)
from lodestar_edge.segment import load_corpus
from lodestar_edge.retriever import SalienceBM25Retriever
from lodestar_edge.graph import CorpusGraph
from lodestar_edge.diagnose import diagnose_chain, preload, unload

from _paths import _d, _DATA_ROOT   # noqa: E402


TE = os.environ.get("LODESTAR_ATLAS_DIR", "../../apollo-anomaly-atlas/data")
BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("LODESTAR_EDGE_MODEL", "llama3.2:3b")   # derived lodestar-3b-ctx4k on Jetson
FLUSH_EVERY = int(os.environ.get("FLUSH_EVERY", "0"))          # Jetson: unload+reload every N to cap KV
OUTFILE = os.environ.get("OUT", _d("crossmission_lodestar.jsonl"))

# subsystem assigned from the symptom (domain-obvious). key = distinctive substring.
ASSIGN = [
    ("FDAI attitude jumped", "GNC"), ("docking probe would not extend", "structures"),
    ("cabin pressure dropped rapidly after LM jettison", "structures"),
    ("docking probe will not capture", "structures"), ("landing radar will not lock", "GNC"),
    ("water leaking from the chlorine", "ECLSS"),
    ("AC bus 2 and DC bus B undervolt", "EPS"), ("SPS secondary yaw gimbal servo", "SPS"),
    ("erroneous gimbal-lock indication", "GNC"), ("spurious master alarms with no caution", "EPS"),
    ("EMS accelerometer null-bias shift", "GNC"),
]
def assign(sym):
    for sub, ss in ((k, v) for k, v in ASSIGN):
        if sub.lower() in (sym or "").lower(): return ss
    return None

def num(x): return isinstance(x, (int, float)) and x >= 0
CM_FILE = os.environ.get("CM_FILE", f"{TE}/crossmission_human_turns.json")
cm = json.load(open(CM_FILE, encoding="utf-8")).get("anomalies", [])
usable = [a for a in cm if num(a.get("human_turns_to_diagnosis")) or num(a.get("human_time_s"))]

segs = load_corpus(os.path.join(EDGE, "lodestar_edge", "corpus.json"))
r = SalienceBM25Retriever(segs); g = CorpusGraph(r.segments)
preload(BASE, MODEL); print(f"model preloaded ({MODEL}); {len(usable)} cross-mission faults, FLUSH_EVERY={FLUSH_EVERY}", flush=True)

def turns_of(s):
    parts = [p.strip() for p in re.split(r"(?<=[.;])\s+", s) if len(p.strip()) >= 12]
    return parts or [s]

OUTP = OUTFILE
done = set()
if os.path.exists(OUTP):
    for l in open(OUTP, encoding="utf-8"):
        try: done.add(json.loads(l)["symptom"])
        except Exception: pass
out = open(OUTP, "a", encoding="utf-8"); processed = 0
todo = [a for a in usable if a["symptom"] not in done]
for i, a in enumerate(usable, 1):
    sym = a["symptom"]; sub = assign(sym)
    if sym in done:
        continue
    if sub is None:
        print(f"  [{i}] NO subsystem assigned, skip: {sym[:50]}", flush=True); continue
    turns = turns_of(sym); matched_at, cum, final = None, 0.0, ""
    for k in range(1, len(turns) + 1):
        q = " ".join(turns[:k]); t0 = time.perf_counter()
        res = diagnose_chain(q, r, g, model=MODEL, base_url=BASE, k_entries=8, pool=16); cum += time.perf_counter() - t0
        final = res.get("diagnosis", "") if not res["declined"] else ""
        if subsystem_match(final, sub, res["declined"]): matched_at = k; break
    rec = {"symptom": sym, "subsystem_assigned": sub, "matched": matched_at is not None,
           "matched_at": matched_at, "cum_s": round(cum, 2),
           "capcom_turns": a.get("human_turns_to_diagnosis"), "capcom_time_s": a.get("human_time_s")}
    out.write(json.dumps(rec) + "\n"); out.flush(); processed += 1
    if FLUSH_EVERY and processed % FLUSH_EVERY == 0 and a is not todo[-1]:
        unload(BASE, MODEL); time.sleep(2); preload(BASE, MODEL)
    print(f"  [{i}/{len(usable)}] {sub:<11} L_match@{matched_at or 'never'} L_cum={cum:.0f}s "
          f"| C_turns={a.get('human_turns_to_diagnosis')} C_time={a.get('human_time_s')} | {sym[:38]}", flush=True)
out.close()
print("done -> data/crossmission_lodestar.jsonl")
