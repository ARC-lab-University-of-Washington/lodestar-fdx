# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file ablate_speech_vs_signals.py
# @brief ABLATION: is human speech as important to the LLM as raw fault indicators? Split each crew symptom into two channels and re-run the full pipeline on e.
#
# ach: SIGNALS = documented-annunciator vocabulary (graph.ann_vocab) + CAPS
# panel phrases (MAIN BUS B, O2, UNDERVOLT, FDAI, program-alarm codes) +
# number+unit tokens -> the 'raw fault indicators' SPEECH = the crew's plain-
# language remainder (bang, venting, thump, snowflakes, noise, felt ...) i.e.
# content tokens NOT in the annunciator vocabulary and not part of a CAPS
# phrase. FULL = the original symptom. Score subsystem-match + coverage for
# FULL / SIGNALS / SPEECH. If SPEECH ~ SIGNALS the speech carries comparable
# diagnostic signal; if SIGNALS ~ FULL >> SPEECH, raw indicators dominate.
# DETERMINISTIC scoring (subsystem_match); LLM only for the decision. Modes:
# DEMO=1 -> just print the split for N symptoms (no LLM). else run the
# ablation. Run from analysis/. Env: CUDA_VISIBLE_DEVICES=0
# HIP_VISIBLE_DEVICES=-1, LIMIT (default all 146).
#
"""ABLATION: is human speech as important to the LLM as raw fault indicators?
Split each crew symptom into two channels and re-run the full pipeline on each:
  SIGNALS = documented-annunciator vocabulary (graph.ann_vocab) + CAPS panel phrases (MAIN BUS B, O2,
            UNDERVOLT, FDAI, program-alarm codes) + number+unit tokens  -> the 'raw fault indicators'
  SPEECH  = the crew's plain-language remainder (bang, venting, thump, snowflakes, noise, felt ...)
            i.e. content tokens NOT in the annunciator vocabulary and not part of a CAPS phrase.
FULL = the original symptom.  Score subsystem-match + coverage for FULL / SIGNALS / SPEECH.
If SPEECH ~ SIGNALS the speech carries comparable diagnostic signal; if SIGNALS ~ FULL >> SPEECH,
raw indicators dominate.  DETERMINISTIC scoring (subsystem_match); LLM only for the decision.
Modes:  DEMO=1 -> just print the split for N symptoms (no LLM).   else run the ablation.
Run from analysis/.  Env: CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=-1, LIMIT (default all 146)."""
import sys
import json, os, re, sys, time
sys.path.insert(0, "scripts")
from subsystem_match import aliases_for, names
EDGE = os.environ.get("LODESTAR_EDGE_DIR", ".."); sys.path.insert(0, EDGE)
from lodestar_edge import retriever as R
from lodestar_edge.segment import load_corpus
from lodestar_edge.retriever import SalienceBM25Retriever, tokenize
from lodestar_edge.graph import CorpusGraph
from lodestar_edge.diagnose import diagnose_chain

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import _d, _DATA_ROOT   # noqa: E402


TE = os.environ.get("LODESTAR_ATLAS_DIR", "../../apollo-anomaly-atlas/data")
BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("LODESTAR_EDGE_MODEL", "llama3.2:3b")
atlas = json.load(open(f"{TE}/atlas_parsed.json", encoding="utf-8"))
ci = [a for m in atlas["missions"] for a in m.get("anomalies", []) if a.get("crew_initiated")]

segs = load_corpus(os.path.join(EDGE, "lodestar_edge", "corpus.json"))
r = SalienceBM25Retriever(segs); g = CorpusGraph(r.segments)
ANN = g.ann_vocab                     # documented annunciator vocabulary (the coverage-dial vocab)
_STOP = set("the a an of to in on and or with at from for was were is are be been being this that these "
    "those it its they them their has had have will would could should may might not no nor due most "
    "probable caused cause failure failed fault anomaly report system during after before reported "
    "observed crew when then approximately about around also only which while into onto over under out "
    "off up down but if so as than very more less each any all some there here we i he she his her him "
    "our your near following occurred showed show reading indication indications".split())

def split(symptom: str):
    caps = [m.group().strip() for m in R._CAPS.finditer(symptom or "")]
    caps = [c for c in caps if len(c) >= 2 and c.upper() not in R._STOP and not c.isdigit()]
    caps_words = set(w.lower() for c in caps for w in re.split(r"[^A-Za-z0-9]+", c) if w)
    sig, spe = [], []
    for t in tokenize(symptom or ""):
        tl = t.lower()
        if tl in _STOP or len(tl) < 2:
            continue
        has_digit = any(ch.isdigit() for ch in t)
        if tl in ANN or tl in caps_words or has_digit:
            sig.append(t)
        else:
            spe.append(t)
    signals_text = " ".join(caps + [w for w in sig if w.lower() not in caps_words])
    speech_text = " ".join(spe)
    return signals_text.strip(), speech_text.strip()

if os.environ.get("DEMO"):
    import random
    idx = list(range(len(ci)))
    # deterministic sample: flagship-like + spread
    picks = [ci[i] for i in idx[:: max(1, len(idx)//12)]][:12]
    for a in picks:
        s, p = split(a["symptom"])
        print(f"[{a.get('subsystem')}] {a['symptom'][:96]}")
        print(f"   SIGNALS: {s[:110]}")
        print(f"   SPEECH : {p[:110]}\n")
    sys.exit(0)

LIMIT = int(os.environ.get("LIMIT", str(len(ci))))
OUT = os.environ.get("OUT", _d("ablate_speech_signals.jsonl"))
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["symptom"] for l in open(OUT, encoding="utf-8") if l.strip()}
out = open(OUT, "a", encoding="utf-8")
tally = {c: {"n": 0, "sub": 0, "cov": 0.0} for c in ("full", "signals", "speech")}
for i, a in enumerate(ci[:LIMIT], 1):
    sym = a["symptom"]; sub = a.get("subsystem") or ""
    if sym in done:
        continue
    sig, spe = split(sym)
    rec = {"symptom": sym, "subsystem": sub}
    for cond, q in (("full", sym), ("signals", sig), ("speech", spe)):
        if not q.strip():
            rec[cond] = {"declined": True, "sub_ok": False, "cov": 0.0, "text": q}; continue
        res = diagnose_chain(q, r, g, model=MODEL, base_url=BASE, k_entries=8, pool=16)
        txt = "" if res["declined"] else res.get("diagnosis", "")
        ok = names(txt, aliases_for(sub))
        rec[cond] = {"declined": res["declined"], "sub_ok": bool(ok),
                     "cov": res.get("coverage"), "text": q[:120]}
    out.write(json.dumps(rec) + "\n"); out.flush()
    for c in tally:
        tally[c]["n"] += 1; tally[c]["sub"] += rec[c]["sub_ok"]
        tally[c]["cov"] += (rec[c]["cov"] or 0.0)
    if i % 10 == 0:
        line = "  ".join(f"{c}:{tally[c]['sub']}/{tally[c]['n']}" for c in tally)
        print(f"[{i}/{LIMIT}] subsystem so far -> {line}", flush=True)
out.close()
print("\n=== ABLATION RESULT (crew-observed set) ===")
for c in tally:
    n = tally[c]["n"] or 1
    print(f"  {c:<8}: subsystem {tally[c]['sub']}/{n} = {tally[c]['sub']/n:.0%}   mean coverage {tally[c]['cov']/n:.3f}")
