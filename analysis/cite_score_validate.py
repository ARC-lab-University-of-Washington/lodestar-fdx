# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file cite_score_validate.py
# @brief Independent validation of the CITE-SCORE feature dictated in IMPLEMENTATION_NOTE.md.
#
# Re-derives everything from data/full_log.jsonl — does NOT trust the claimed
# numbers. CITE_SCORE = (g - MU_G)/SD_G + (sup - MU_SUP)/SD_SUP g = mean over
# kept points of extr, extr = |o & src| / |o| (o = _content_terms(model_out -
# [S#])) top = most-common ECHO-EXCLUDED agreement term (echo =
# _content_terms(salience)) sup = # retrieved sources whose src_text contains
# top Reports: (1) reproduced MU/SD vs the frozen constants; (2) held-out
# in/out AUC over N random splits (median + 5th pct); (3) a label-permutation
# null for the median-split AUC. Deterministic given the seed. No LLM, no
# trained classifier.
#
"""Independent validation of the CITE-SCORE feature dictated in IMPLEMENTATION_NOTE.md.
Re-derives everything from data/full_log.jsonl — does NOT trust the claimed numbers.

CITE_SCORE = (g - MU_G)/SD_G + (sup - MU_SUP)/SD_SUP
  g   = mean over kept points of extr, extr = |o & src| / |o|   (o = _content_terms(model_out - [S#]))
  top = most-common ECHO-EXCLUDED agreement term (echo = _content_terms(salience))
  sup = # retrieved sources whose src_text contains top
Reports: (1) reproduced MU/SD vs the frozen constants; (2) held-out in/out AUC over N random
splits (median + 5th pct); (3) a label-permutation null for the median-split AUC.
Deterministic given the seed. No LLM, no trained classifier.
"""
import json, os, re, math, random, statistics as st
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
recs = [json.loads(l) for l in open(os.path.join(DATA, "full_log.jsonl"), encoding="utf-8") if l.strip()]

# ---- term function: EXACTLY diagnose.py's _GEN + the 6 additions the note requires ----
_GEN = set("the a an of to in on and or with at from for during after before was were is are be this "
    "that these those it its they them their has had have will would could should may might not no non "
    "due most probable caused cause failure failed fault anomaly report system module assembly unit "
    "device component part condition normal nominal operation over under within candidate check next "
    "based given symptom source potential possible verify may causing related issue reading high low "
    "value indicates suggests one concrete drawn ending output relevant crew".split())
_GEN |= {"which", "whose", "indicated", "potentially", "malfunction", "insufficient"}
_TERM = re.compile(r"[a-z][a-z\-]{3,}")   # note diagnose.py uses {3,}: 4+ char alpha tokens
def content_terms(t):
    return {w for w in _TERM.findall((t or "").lower()) if w not in _GEN}

TAG = re.compile(r"\[s\d+\]", re.I)
def points_of(r):
    """kept points = answered sources; (o = model-output terms sans [S#] tag, src = source terms)."""
    pts = []
    for s in r["sources"]:
        if s["skip"]:
            continue
        o = content_terms(TAG.sub("", s["model_out"]))
        src = content_terms(s["src_text"])
        pts.append((o, src))
    return pts

def cite_parts(r):
    pts = points_of(r)
    if not pts:
        return None                      # declined -> cite_score None (acceptance check #2)
    extrs = [len(o & src) / len(o) if o else 0.0 for (o, src) in pts]
    g = st.mean(extrs)
    echo = content_terms(r.get("salience") or "")
    vote = Counter()
    for (o, src) in pts:
        for w in (o - echo):
            vote[w] += 1
    top = vote.most_common(1)[0][0] if vote else None
    if top is None:
        sup = 0
    else:
        # "raw_text contains top" == substring match (reproduces the note's frozen MU_SUP=1.60)
        sup = sum(1 for s in r["sources"] if top in (s["src_text"] or "").lower())
    return g, sup, top

# ---- clean in/out label: subsystem field (feature_lab's clean_label; the mad-dog-approved one) ----
_OUT_SUB = ("alsep", "experiment", "eva", "emu", "suit", "camera", "lrv", "rover", "drill", "sample",
            "geolog", "science", "seismic", "magnetometer", "spectrometer", "sounder", "cplee", "side",
            "leam", " sep", "lpm", "surface", "lunar-portable")
_IN_SUB = ("eclss", "ecs", "eps", "sps", "rcs", "gnc", "g&c", "comm", "instrumentation", "structure",
           "propuls", "fuel", "cryo", "batter", "electrical", "guidance", "navigation", "sequential",
           "docking", "s-band", "vhf", "pyro", "environ")
def clean_label(subsys):
    s = (subsys or "").lower()
    if any(w in s for w in _OUT_SUB): return False
    if any(w in s for w in _IN_SUB): return True
    return None

# ---- assemble answered rows with (g, sup, top, label) ----
rows = []
for r in recs:
    parts = cite_parts(r)
    if parts is None:
        continue                          # declined: excluded from in/out AUC (they carry no cite_score)
    lb = clean_label(r["subsystem"])
    if lb is None:
        continue                          # ambiguous subsystem: excluded from the metric
    g, sup, top = parts
    rows.append({"g": g, "sup": sup, "in": lb, "top": top})

gs = [x["g"] for x in rows]
sups = [x["sup"] for x in rows]
# reproduce constants over ALL answered (note says 173 answered; here restricted to labelled answered)
allparts = [cite_parts(r) for r in recs]
allparts = [p for p in allparts if p is not None]
G = [p[0] for p in allparts]; SUP = [float(p[1]) for p in allparts]
MU_G, SD_G = st.mean(G), st.pstdev(G)
MU_SUP, SD_SUP = st.mean(SUP), st.pstdev(SUP)
print(f"ANSWERED (all labelled+unlabelled): n={len(allparts)}")
print(f"  reproduced   MU_G={MU_G:.4f} SD_G={SD_G:.4f}  MU_SUP={MU_SUP:.4f} SD_SUP={SD_SUP:.4f}")
print(f"  note frozen  MU_G=0.3745 SD_G=0.1408  MU_SUP=1.6012 SD_SUP=2.0534")

# use the NOTE's frozen constants for the score (that is what ships)
NG, NSG, NSU, NSSU = 0.3745, 0.1408, 1.6012, 2.0534
def cite_score(g, sup): return (g - NG) / NSG + (sup - NSU) / NSSU
for x in rows:
    x["cs"] = cite_score(x["g"], x["sup"])

def auc(pos, neg):
    if not pos or not neg: return 0.5
    c = t = 0
    for p in pos:
        for n in neg:
            t += 1; c += 1 if p > n else 0.5 if p == n else 0
    return c / t

inc = [x for x in rows if x["in"]]; out = [x for x in rows if not x["in"]]
print(f"\nLABELLED answered: in {len(inc)} / out {len(out)}")
print(f"  FULL-SET AUC (in/out) on cite_score: {auc([x['cs'] for x in inc], [x['cs'] for x in out]):.3f}")
print(f"  FULL-SET AUC on g alone:   {auc([x['g'] for x in inc], [x['g'] for x in out]):.3f}")
print(f"  FULL-SET AUC on sup alone: {auc([x['sup'] for x in inc], [x['sup'] for x in out]):.3f}")

# ---- held-out: N random 70/30 splits, fit direction on train, score test (median + 5th pct) ----
rng = random.Random(20260711)
N = 200
test_aucs = []
for _ in range(N):
    tr, te = [], []
    for x in rows:
        (te if rng.random() >= 0.7 else tr).append(x)
    tin = [x for x in te if x["in"]]; tout = [x for x in te if not x["in"]]
    if not tin or not tout:
        continue
    test_aucs.append(auc([x["cs"] for x in tin], [x["cs"] for x in tout]))
test_aucs.sort()
med = test_aucs[len(test_aucs) // 2]
p5 = test_aucs[int(0.05 * len(test_aucs))]
print(f"\nHELD-OUT over {len(test_aucs)} random 30% test splits:")
print(f"  median AUC {med:.3f}   5th pct {p5:.3f}   min {test_aucs[0]:.3f}   max {test_aucs[-1]:.3f}")

# ---- permutation null: shuffle labels, recompute full-set AUC, how often >= observed ----
obs = auc([x["cs"] for x in inc], [x["cs"] for x in out])
labels = [x["in"] for x in rows]; css = [x["cs"] for x in rows]
rngp = random.Random(7)
B = 2000; ge = 0
for _ in range(B):
    perm = labels[:]; rngp.shuffle(perm)
    pi = [c for c, l in zip(css, perm) if l]; po = [c for c, l in zip(css, perm) if not l]
    if auc(pi, po) >= obs: ge += 1
print(f"\nPERMUTATION null (label shuffle, B={B}): observed AUC {obs:.3f}, "
      f"p = {(ge + 1) / (B + 1):.4f}")

# ---- operating point the note claims: flag at cite_score <= -1.510 ----
THR = -1.510
fl_out = sum(1 for x in out if x["cs"] <= THR); fl_in = sum(1 for x in inc if x["cs"] <= THR)
tot_fl = fl_out + fl_in
print(f"\nFLAG @ cite_score <= {THR}:")
print(f"  catches {fl_out}/{len(out)} out ({fl_out/len(out):.0%}), "
      f"false-flags {fl_in}/{len(inc)} in ({fl_in/len(inc):.0%}), "
      f"precision {fl_out/tot_fl:.0%}" if tot_fl else "  no flags")
