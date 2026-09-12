# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file where_only.py
# @brief WHERE-ONLY scoring.
#
# Ignore WHY (mechanism). Success = does the decision name the fault's WHERE
# (component/location terms from the documented cause)? Reports (a) WHERE-
# coverage fraction and (b) WHERE-hit rate (named >=1 correct WHERE term), on
# all 146 and on the FAIR set (drop out-of-corpus + no-knowable-cause).
# Deterministic; scores the existing dev-box decisions.
#
"""WHERE-ONLY scoring. Ignore WHY (mechanism). Success = does the decision name the fault's WHERE
(component/location terms from the documented cause)? Reports (a) WHERE-coverage fraction and (b)
WHERE-hit rate (named >=1 correct WHERE term), on all 146 and on the FAIR set (drop out-of-corpus +
no-knowable-cause). Deterministic; scores the existing dev-box decisions."""
import sys
import json, os, re, math, statistics as st
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import _d, _DATA_ROOT   # noqa: E402

recs = [json.loads(l) for l in open(_d("full_log.jsonl"), encoding="utf-8") if l.strip()]
dec = [json.loads(l) for l in open(_d("decisions_146_k8.jsonl"), encoding="utf-8") if l.strip()]

WHERE = re.compile(r"\b(switch|valve|tank|cell|\bbus\b|batter|capacitor|relay|connector|o-?ring|seal|"
 r"wir|cable|motor|pump|transducer|sensor|antenna|transponder|gyro|regulator|filter|bearing|actuator|"
 r"coil|module|assembly|breaker|nozzle|diode|resistor|solenoid|heater|\bfan\b|thruster|quad|inverter|"
 r"commutator|timer|gauge|indicator|\blamp|\blight|window|hatch|shade|strut|bracket|fitting|gasket|"
 r"bellows|diaphragm|orifice|chamber|housing|shaft|\bgear|spring|\bcam\b|lever|\bpin\b|bolt|\bweld|"
 r"joint|\btube|pipe|insulation|potting|\bfilm\b|\bgrid\b|filament|electrode|\bplate|contact|circuit|"
 r"harness|terminal|converter|amplifier|stylus|condenser|radiator|injector|bladder|canister|cartridge)", re.I)
AFT = re.compile(r"postflight|bench test|x-ray|temperature test|duplicat|ground test|review board|"
 r"teardown|closed\b|no hardware|warranted|redesign|modif|relocated|improved|corrective|recommend|"
 r"powered down|lifeboat|recycl", re.I)
_T = re.compile(r"[a-z][a-z\-]{3,}")
GEN = set("the a an of to in on and or with at from for was were is are be this that these those it its "
    "they them their has had have will would could should may might not no non due most probable caused "
    "cause failure failed fault anomaly report system module condition which whose".split())
def ct(s): return {w for w in _T.findall((s or "").lower()) if w not in GEN}

NC = len(recs); cdf = Counter()
for r in recs:
    for w in ct(r["documented_cause"]): cdf[w] += 1
def cidf(w): return math.log(NC / (cdf.get(w, 0) + 1))
def where_terms(cause):
    diag = " ".join(s for s in re.split(r"(?<=[.;])\s+", cause) if len(s.strip()) >= 8 and not AFT.search(s))
    return {w for w in ct(diag) if cidf(w) >= 2.5 and WHERE.search(w)}
def _norm(t): return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower()).split())

_OUT = ("eva","emu","suit","alsep","experiment","camera","lrv","rover","biomed","crew-health","sim-bay",
        "gfe","crew equipment","seismic","geolog","magnetometer","lpm","thumper","side","cplee","leam")
UNK = re.compile(r"pending|tentativ|not duplicat|not reproduc|single-point|spurious|no cause|unknown|"
 r"not determin|gas bubble|momentar|transient|nuisance|no hardware|benign|state of charge|no anomaly", re.I)
def fair(d):
    s = (d["subsystem"] or "").lower()
    if any(w in s for w in _OUT): return False
    if UNK.search(d["documented_cause"]): return False
    return True

def score(group, label):
    covs, hits, n = [], 0, 0
    for d in group:
        tgt = where_terms(d["documented_cause"])
        if not tgt: continue
        n += 1
        txt = _norm(d["decision"]) if not d["declined"] else ""
        m = sum(1 for w in tgt if _norm(w) in txt)
        covs.append(m / len(tgt)); hits += 1 if m >= 1 else 0
    print(f"=== {label} (n={n}, scoreable = has WHERE terms) ===")
    print(f"  WHERE-coverage fraction: mean {st.mean(covs):.2f}  median {st.median(covs):.2f}")
    print(f"  WHERE-HIT rate (named >=1 correct component): {hits}/{n} = {hits/n:.0%}")
    print(f"  mean WHERE terms/cause: {st.mean([len(where_terms(d['documented_cause'])) for d in group if where_terms(d['documented_cause'])]):.1f}\n")

score(dec, "ALL 146")
score([d for d in dec if fair(d)], "FAIR (in-corpus, knowable cause)")
