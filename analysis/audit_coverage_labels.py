# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file audit_coverage_labels.py
# @brief AUDIT of the coverage in/out labels.
#
# The in/out label comes from a subsystem-keyword heuristic (exp_graph.lab)
# that DROPS subsystems matching neither list. This audits those labels: (1)
# confirms the out-class is genuinely out-of-corpus, (2) resolves the
# dropped/ambiguous subsystems by the real rule — is the fault's system
# covered by the Apollo-13 onboard corpus? — and (3) re-computes the ann_frac
# AUC on the audited labels, plus a robustness check. Deterministic, no model
# calls. Mirrors exp_graph.py's ann_frac exactly.
#
"""AUDIT of the coverage in/out labels.
The in/out label comes from a subsystem-keyword heuristic (exp_graph.lab) that DROPS
subsystems matching neither list. This audits those labels: (1) confirms the out-class is genuinely
out-of-corpus, (2) resolves the dropped/ambiguous subsystems by the real rule — is the fault's system
covered by the Apollo-13 onboard corpus? — and (3) re-computes the ann_frac AUC on the audited labels,
plus a robustness check. Deterministic, no model calls. Mirrors exp_graph.py's ann_frac exactly."""
import json, os, sys, math, importlib, random, statistics as st
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = os.path.dirname(HERE)
sys.path.insert(0, EDGE)
D = importlib.import_module("lodestar_edge.diagnose"); ct = D._content_terms
from lodestar_edge.retriever import salience  # noqa: E402

recs = [json.loads(l) for l in open(os.path.join(os.path.dirname(HERE), "data", "full_log.jsonl"),
        encoding="utf-8") if l.strip()]
c = json.load(open(os.path.join(EDGE, "lodestar_edge", "corpus.json"), encoding="utf-8"))
segs = c if isinstance(c, list) else c.get("segments", c)

# annunciation-node vocabulary (exactly exp_graph.py)
ANN = set()
for s in segs:
    for x in (s.get("symptoms") or []):
        ANN |= ct(x)
def ann_frac(r):
    q = ct(r["symptom"]) | ct(salience(r["symptom"]))
    return (len(q & ANN) / len(q)) if q else None

# ---- ORIGINAL heuristic (exp_graph.lab) ----
_OUT = ("alsep","experiment","eva","emu","suit","camera","lrv","rover","drill","sample","geolog",
        "science","seismic","magnetometer","spectrometer","sounder","cplee","side","leam"," sep","lpm","surface")
_IN = ("eclss","ecs","eps","sps","rcs","gnc","comm","instrumentation","structure","propuls","fuel",
       "cryo","batter","electrical","guidance","navigation","sequential","docking","s-band","vhf","pyro","environ")
def lab0(x):
    x = (x or "").lower()
    if any(w in x for w in _OUT): return False
    if any(w in x for w in _IN): return True
    return None

# ---- AUDITED labeling: resolve the dropped subsystems by "is this system in the A13 corpus?" ----
# IN (Apollo-13 CSM/LM manuals cover these): descent/ascent propulsion, caution&warning, lighting,
#   the LM proper, the CM recovery antenna (comms hardware). OUT: SIM-bay science, GFE camera part,
#   crew medical. Everything the original list already decided is kept.
_OUT2 = _OUT + ("sim-bay", "sim bay", "gfe", "government", "crew-health", "crew health", "biomed", "medical")
_IN2 = _IN + ("dps", "aps", " lm", "lm ", "c&w", "caution", "warning", "lighting", "light",
              "descent", "ascent", "recovery")
def lab1(x):
    x = (x or "").lower()
    if any(w in x for w in _OUT2): return False
    if any(w in x for w in _IN2): return True
    return None

def auc(p, n):
    if not p or not n: return 0.5
    c = t = 0
    for a in p:
        for b in n:
            t += 1; c += 1 if a > b else 0.5 if a == b else 0
    return c / t

def build(labf):
    rows = []
    for r in recs:
        v = ann_frac(r)
        if v is None: continue
        lb = labf(r["subsystem"])
        if lb is None: continue
        rows.append({"v": v, "in": lb, "sub": r["subsystem"], "sym": r["symptom"]})
    return rows

def report(rows, name):
    inc = [x for x in rows if x["in"]]; out = [x for x in rows if not x["in"]]
    full = auc([x["v"] for x in inc], [x["v"] for x in out])
    rng = random.Random(20260711); ta = []
    for _ in range(200):
        tr = [x for x in rows if rng.random() >= 0.7]
        ti = [x["v"] for x in tr if x["in"]]; to = [x["v"] for x in tr if not x["in"]]
        if ti and to: ta.append(auc(ti, to))
    ta.sort()
    # permutation null
    labels = [x["in"] for x in rows]; vals = [x["v"] for x in rows]
    rp = random.Random(7); B = 2000; ge = 0
    obs = full
    for _ in range(B):
        rp.shuffle(labels)
        pi = [v for v, l in zip(vals, labels) if l]; po = [v for v, l in zip(vals, labels) if not l]
        if auc(pi, po) >= obs: ge += 1
    print(f"[{name}] in {len(inc)} / out {len(out)}  |  full-set AUC {full:.3f}  "
          f"held-out median {ta[len(ta)//2]:.3f}  5th pct {ta[int(.05*len(ta))]:.3f}  "
          f"perm p {(ge+1)/(B+1):.4f}")
    return inc, out

print(f"ann_vocab = {len(ANN)} tokens\n")
r0 = build(lab0); report(r0, "ORIGINAL heuristic")
r1 = build(lab1); report(r1, "AUDITED labels    ")

# what changed: subsystems the audit newly labelled (were None), and any flips
d0 = {(x["sub"], x["sym"]): x["in"] for x in r0}
print("\n=== newly labelled by the audit (were dropped as ambiguous) ===")
newlab = Counter()
for x in r1:
    if (x["sub"], x["sym"]) not in d0:
        newlab[(x["sub"], "IN" if x["in"] else "OUT")] += 1
for (sub, lb), n in sorted(newlab.items()):
    print(f"  {lb:3}  {sub:<14} x{n}")
print("\n=== any label FLIPS (in<->out) between original and audited ===")
flips = [x for x in r1 if (x["sub"], x["sym"]) in d0 and d0[(x["sub"], x["sym"])] != x["in"]]
print(f"  {len(flips)} flips" + ("" if not flips else ":"))
for x in flips[:10]:
    print(f"    {x['sub']}: {'->IN' if x['in'] else '->OUT'}  {x['sym'][:50]}")
