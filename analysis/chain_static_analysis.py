# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file chain_static_analysis.py
# @brief STATIC ANALYSIS of the reasoning chains themselves (no model in the loop).
#
# For each of the 146 crew_initiated symptoms, rebuild the 8 graph chains
# (pick_entries -> chase, shipped defaults) and profile: (A) structure, (B)
# what information each chain node carries, (C) what the chains cover of the
# target, (D) the reasoning-path shape (annunciation -> component), (E) cross-
# chain redundancy. Deterministic, zero-GPU. Mirrors diagnose_chain's chain
# construction exactly.
#
"""STATIC ANALYSIS of the reasoning chains themselves (no model in the loop). For each of the 146
crew_initiated symptoms, rebuild the 8 graph chains (pick_entries -> chase, shipped defaults) and
profile: (A) structure, (B) what information each chain node carries, (C) what the chains cover of the
target, (D) the reasoning-path shape (annunciation -> component), (E) cross-chain redundancy.
Deterministic, zero-GPU. Mirrors diagnose_chain's chain construction exactly."""
import sys
import json, os, sys, re, math, statistics as st
from collections import Counter
EDGE = ".."; sys.path.insert(0, EDGE)
from lodestar_edge.segment import load_corpus
from lodestar_edge.retriever import SalienceBM25Retriever, salience
from lodestar_edge.graph import CorpusGraph
import importlib
D = importlib.import_module("lodestar_edge.diagnose"); ct = D._content_terms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subsystem_match import _OUT  # in-scope filter

from _paths import _d, _DATA_ROOT   # noqa: E402

recs = [json.loads(l) for l in open(_d("full_log.jsonl"), encoding="utf-8") if l.strip()]
def _outscope(r): return any(w in (r.get("subsystem") or "").lower() for w in _OUT)
ci = [r for r in recs if r.get("crew_initiated") and not _outscope(r)]   # the 108 in-scope
segs = load_corpus(os.path.join(EDGE, "lodestar_edge", "corpus.json"))
retr = SalienceBM25Retriever(segs); g = CorpusGraph(retr.segments)
K, POOL, REL, DEPTH = 8, 16, 0.4, 3

# distinctive documented-cause terms (IDF>=2.5 over all causes) + WHERE/WHY split for the coverage view
CDF = Counter()
for r in recs:
    for w in set(ct(r["documented_cause"])): CDF[w] += 1
NC = len(recs)
def cidf(w): return math.log(NC / (CDF.get(w, 0) + 1))
def cause_terms(r): return {w for w in ct(r["documented_cause"]) if cidf(w) >= 2.5}
WHERE = re.compile(r"\b(switch|valve|tank|cell|bus|batter|relay|connector|o-?ring|seal|wir|cable|motor|"
    r"pump|transducer|sensor|antenna|gyro|regulator|filter|bearing|coil|module|assembly|breaker|nozzle|"
    r"diode|heater|fan|thruster|quad|inverter|commutator|gauge|lamp|light|window|hatch|line|circuit|"
    r"harness|terminal|converter|amplifier|capacitor|solenoid|actuator|bellows|diaphragm|filament)", re.I)

def seg_terms(s): return ct((s.raw_text or "") + " " + (s.component or "") + " " + (s.signature or ""))

# ---- accumulators ----
n_entries, chain_lens, uniq_segs, visits = [], [], [], []
degen_chain = 0; tot_chain = 0; degen_diag = 0
has_component = has_signature = has_symptoms = has_links = seg_count = 0
distinct_comps, distinct_docs = [], []
cause_cov, where_cov, ann_cov = [], [], []
entry_is_ann = 0; entry_tot = 0
term_kind = Counter()   # of the chains' distinctive terms: where/annunciation/other
n_done = 0

for r in ci:
    ranked = [s for s, _ in retr.search_scored(r["symptom"], k=POOL)]
    if not ranked: continue
    n_done += 1
    entries = g.pick_entries(ranked, K)
    scores = retr.effective_scores(r["symptom"])
    chains = [g.chase(e, scores, retr, rel_threshold=REL, max_depth=DEPTH) for e in entries]
    n_entries.append(len(entries))
    ids, comps, docs, chain_txt = set(), set(), set(), set()
    vtot = 0; maxlen = 0
    for chain in chains:
        tot_chain += 1; chain_lens.append(len(chain))
        if len(chain) == 1: degen_chain += 1
        maxlen = max(maxlen, len(chain))
        for j, s in enumerate(chain):
            seg_count += 1; vtot += 1; ids.add(s.id)
            if s.component: has_component += 1; comps.add(s.component.strip().lower()[:40])
            if s.signature: has_signature += 1
            if s.symptoms: has_symptoms += 1
            if g.has_out(s): has_links += 1
            docs.add(s.source_doc)
            chain_txt |= seg_terms(s)
        e = chain[0]; entry_tot += 1
        if e.symptoms: entry_is_ann += 1
    if maxlen == 1: degen_diag += 1
    uniq_segs.append(len(ids)); visits.append(vtot)
    distinct_comps.append(len(comps)); distinct_docs.append(len(docs))
    tgt = cause_terms(r)
    if tgt:
        cause_cov.append(len(tgt & chain_txt) / len(tgt))
        wt = {w for w in tgt if WHERE.search(w)}
        if wt: where_cov.append(len(wt & chain_txt) / len(wt))
    # annunciation coverage: symptom terms that are documented annunciation vocab, found in chains
    qann = ct(r["symptom"]) & g.ann_vocab
    if qann: ann_cov.append(len(qann & chain_txt) / len(qann))
    # term-kind composition of the chain content (distinctive chain terms)
    for w in {w for w in chain_txt if cidf(w) >= 2.5}:
        term_kind["where" if WHERE.search(w) else ("annunc" if w in g.ann_vocab else "other")] += 1

print(f"STATIC CHAIN ANALYSIS over {n_done} crew_initiated diagnoses (k=8, rel=0.4, depth=3)\n")
print("== A. STRUCTURE ==")
print(f"  entries chased/diag:   mean {st.mean(n_entries):.2f}")
print(f"  chain length:          mean {st.mean(chain_lens):.2f}  dist {dict(sorted(Counter(chain_lens).items()))}")
print(f"  unique segments/diag:  mean {st.mean(uniq_segs):.1f}  (visits {st.mean(visits):.1f} -> redundancy {st.mean(visits)/st.mean(uniq_segs):.2f}x)")
print(f"  degenerate: per-chain (len 1) {degen_chain}/{tot_chain}={degen_chain/tot_chain:.0%}  |  per-diag (all len 1) {degen_diag}/{n_done}={degen_diag/n_done:.0%}")
print("\n== B. INFORMATION PER CHAIN NODE (what each segment carries) ==")
print(f"  has component:  {has_component/seg_count:.0%}")
print(f"  has signature:  {has_signature/seg_count:.0%}")
print(f"  has symptoms(annunciation node): {has_symptoms/seg_count:.0%}")
print(f"  is graph node (has links_to):    {has_links/seg_count:.0%}")
print("\n== C. WHAT THE CHAINS COVER (per diagnosis) ==")
print(f"  distinct components named: mean {st.mean(distinct_comps):.1f}")
print(f"  distinct source docs:      mean {st.mean(distinct_docs):.1f}")
print(f"  documented-cause term coverage: mean {st.mean(cause_cov):.2f}")
print(f"  WHERE (component) cause-term coverage: mean {st.mean(where_cov):.2f}")
print(f"  annunciation coverage (symptom's annunciator terms found in chains): mean {st.mean(ann_cov):.2f}")
print("\n== D. REASONING-PATH SHAPE ==")
print(f"  entries that are annunciation nodes (have symptoms): {entry_is_ann}/{entry_tot}={entry_is_ann/entry_tot:.0%}")
tk = sum(term_kind.values())
print(f"  chain distinctive-term composition: WHERE {term_kind['where']/tk:.0%} / annunciation {term_kind['annunc']/tk:.0%} / other {term_kind['other']/tk:.0%}")
