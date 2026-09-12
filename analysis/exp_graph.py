# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file exp_graph.py
# @brief PROOF OF CONCEPT: does the corpus's STRUCTURAL graph (annunciation `symptoms` nodes + `signature` + `component`) separate in/out better than flat BM25.
#
# grounding? An out-of-corpus fault (surface science, no annunciator/bus)
# should fail to match a documented annunciation node, whereas a CSM/LM fault
# should hit one. We score each anomaly's symptom against the annunciation-
# node vocabulary (IDF-weighted) and against best structural-segment overlap,
# and compare in/out AUC to grounding g. No model calls.
#
"""PROOF OF CONCEPT: does the corpus's STRUCTURAL graph (annunciation `symptoms` nodes + `signature`
+ `component`) separate in/out better than flat BM25 grounding? An out-of-corpus fault (surface
science, no annunciator/bus) should fail to match a documented annunciation node, whereas a CSM/LM
fault should hit one. We score each anomaly's symptom against the annunciation-node vocabulary
(IDF-weighted) and against best structural-segment overlap, and compare in/out AUC to grounding g.
No model calls."""
import json, os, sys, math, importlib, statistics as st
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = os.path.dirname(HERE)
sys.path.insert(0, EDGE)
D = importlib.import_module("lodestar_edge.diagnose"); ct, TAG = D._content_terms, D._TAG
from lodestar_edge.retriever import salience

recs = [json.loads(l) for l in open(os.path.join(os.path.dirname(HERE), "data", "full_log.jsonl"),
        encoding="utf-8") if l.strip()]
c = json.load(open(os.path.join(EDGE, "lodestar_edge", "corpus.json"), encoding="utf-8"))
segs = c if isinstance(c, list) else c.get("segments", c)

def seg_struct(s):
    parts = []
    sym = s.get("symptoms") or []
    if isinstance(sym, list): parts += sym
    for f in ("signature", "component", "subsystem"):
        if s.get(f): parts.append(str(s.get(f)))
    return ct(" ".join(parts))

# annunciation-node vocabulary: terms appearing in any segment's `symptoms` field, with node-IDF
ann_df = Counter(); NSEG = len(segs)
struct_index = []
for s in segs:
    stt = seg_struct(s)
    struct_index.append(stt)
    symt = set()
    sym = s.get("symptoms") or []
    if isinstance(sym, list):
        for x in sym: symt |= ct(x)
    for w in symt: ann_df[w] += 1
def ann_idf(w): return math.log(NSEG / (ann_df.get(w, 0) + 1))
ANN_VOCAB = set(ann_df)

# corpus IDF over struct terms (for the best-match score)
sdf = Counter()
for stt in struct_index:
    for w in stt: sdf[w] += 1
def sidf(w): return math.log(NSEG / (sdf.get(w, 0) + 1))

_OUT = ("alsep","experiment","eva","emu","suit","camera","lrv","rover","drill","sample","geolog",
        "science","seismic","magnetometer","spectrometer","sounder","cplee","side","leam"," sep","lpm","surface")
_IN = ("eclss","ecs","eps","sps","rcs","gnc","comm","instrumentation","structure","propuls","fuel",
       "cryo","batter","electrical","guidance","navigation","sequential","docking","s-band","vhf","pyro","environ")
def lab(x):
    x = (x or "").lower()
    if any(w in x for w in _OUT): return False
    if any(w in x for w in _IN): return True
    return None

def signals(r):
    q = ct(r["symptom"]) | ct(salience(r["symptom"]))
    if not q: return None
    # S1: annunciation-vocabulary overlap, IDF-weighted (does symptom name documented annunciations?)
    hit = q & ANN_VOCAB
    ann_score = sum(ann_idf(w) for w in hit)
    ann_frac = len(hit) / len(q)
    # S2: best structural-segment match (max IDF-weighted overlap vs any segment's structure)
    best = 0.0
    for stt in struct_index:
        ov = q & stt
        if ov:
            v = sum(sidf(w) for w in ov)
            if v > best: best = v
    # grounding g (baseline), k=8
    gs = []
    for s in r["sources"][:8]:
        if s["skip"]: continue
        o = ct(TAG.sub("", s["model_out"])); src = ct(s["src_text"])
        if o: gs.append(len(o & src) / len(o))
    g = st.mean(gs) if gs else 0.0
    return {"ann_score": ann_score, "ann_frac": ann_frac, "struct_best": best, "g": g}

rows = []
for r in recs:
    s = signals(r)
    if s is None: continue
    lb = lab(r["subsystem"])
    if lb is None: continue
    rows.append({"s": s, "in": lb})

def auc(p, n):
    if not p or not n: return .5
    c = t = 0
    for a in p:
        for b in n:
            t += 1; c += 1 if a > b else .5 if a == b else 0
    return c / t
inc = [x for x in rows if x["in"]]; out = [x for x in rows if not x["in"]]
print(f"n: in {len(inc)} / out {len(out)}   ann_vocab={len(ANN_VOCAB)} tokens")
print(f"{'signal':14}{'IN mean':>10}{'OUT mean':>10}{'in/out AUC':>12}")
for k in ("g", "ann_score", "ann_frac", "struct_best"):
    iv = [x["s"][k] for x in inc]; ov = [x["s"][k] for x in out]
    print(f"{k:14}{st.mean(iv):>10.2f}{st.mean(ov):>10.2f}{auc(iv, ov):>12.3f}")

# ---- rigorous validation of the winner (ann_frac): held-out splits + permutation null ----
import random
KEY = "ann_frac"
rng = random.Random(20260711); ta = []
for _ in range(200):
    tr = [x for x in rows if rng.random() >= 0.7]
    ti = [x["s"][KEY] for x in tr if x["in"]]; to = [x["s"][KEY] for x in tr if not x["in"]]
    if ti and to: ta.append(auc(ti, to))
ta.sort()
obs = auc([x["s"][KEY] for x in inc], [x["s"][KEY] for x in out])
print(f"\n{KEY}: full-set AUC {obs:.3f}   held-out median {ta[len(ta)//2]:.3f}   5th pct {ta[int(.05*len(ta))]:.3f}")
labels = [x["in"] for x in rows]; vals = [x["s"][KEY] for x in rows]
rp = random.Random(7); B = 2000; ge = 0
for _ in range(B):
    perm = labels[:]; rp.shuffle(perm)
    pi = [v for v, l in zip(vals, perm) if l]; po = [v for v, l in zip(vals, perm) if not l]
    if auc(pi, po) >= obs: ge += 1
print(f"{KEY}: permutation p = {(ge+1)/(B+1):.4f}")

# combine ann_frac with grounding g (z-sum) — does generation add anything on top of retrieval?
af = [x["s"]["ann_frac"] for x in rows]; gg = [x["s"]["g"] for x in rows]
maf, saf = st.mean(af), st.pstdev(af); mg, sg = st.mean(gg), st.pstdev(gg)
for x in rows:
    x["combo"] = (x["s"]["ann_frac"] - maf) / saf + (x["s"]["g"] - mg) / sg
print(f"ann_frac + g (z-sum): AUC {auc([x['combo'] for x in inc], [x['combo'] for x in out]):.3f}")
