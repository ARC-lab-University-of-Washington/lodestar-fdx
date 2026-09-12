# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file chains_experiment.py
# @brief Can we CHASE the graph to reach the right outcome from a symptom, and what should we index? Compares, per anomaly, how well each strategy reaches (a) .
#
# the fault's SUBSYSTEM and (b) the DISTINCTIVE documented-cause terms: R1
# BM25 over RAW_TEXT (deployed) R2 BM25 over GRAPH FIELDS
# (symptoms+links_to+component+signature+subsystem) R3 BM25 over SIMPLIFIED
# (signature only) R4 CHAIN traversal: symptom-entry node -> follow links_to
# (<=3 hops) No model calls. rank_bm25 tokenizer = simple lowercase word split
# (underscore/dash kept).
#
"""Can we CHASE the graph to reach the right outcome from a symptom, and what should we index?
Compares, per anomaly, how well each strategy reaches (a) the fault's SUBSYSTEM and (b) the
DISTINCTIVE documented-cause terms:
   R1 BM25 over RAW_TEXT (deployed)         R2 BM25 over GRAPH FIELDS (symptoms+links_to+component+signature+subsystem)
   R3 BM25 over SIMPLIFIED (signature only) R4 CHAIN traversal: symptom-entry node -> follow links_to (<=3 hops)
No model calls. rank_bm25 tokenizer = simple lowercase word split (underscore/dash kept)."""
import sys
import json, os, sys, re, importlib, statistics as st
from collections import Counter, defaultdict
from rank_bm25 import BM25Okapi
EDGE = ".."; sys.path.insert(0, EDGE)
D = importlib.import_module("lodestar_edge.diagnose"); ct = D._content_terms
from lodestar_edge.retriever import salience
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import _d, _DATA_ROOT   # noqa: E402


corp = json.load(open(os.path.join(EDGE, "lodestar_edge", "corpus.json"), encoding="utf-8"))
segs = corp if isinstance(corp, list) else corp.get("segments", corp)
recs = [json.loads(l) for l in open(_d("full_log.jsonl"), encoding="utf-8") if l.strip()]

def norm(x): return re.sub(r"[^a-z0-9 ]", " ", str(x).lower())
def tok(x): return [w for w in norm(x).split() if len(w) >= 3]
def graph_text(s):
    parts = []
    for f in ("symptoms", "links_to"):
        v = s.get(f) or []
        parts += v if isinstance(v, list) else [str(v)]
    for f in ("component", "signature", "subsystem"):
        if s.get(f): parts.append(str(s[f]))
    return " ".join(parts)

RAW = [tok(s.get("raw_text") or "") for s in segs]
GRAPH = [tok(graph_text(s)) for s in segs]
SIMP = [tok(s.get("signature") or "") for s in segs]
bm_raw, bm_graph, bm_simp = BM25Okapi(RAW), BM25Okapi(GRAPH), BM25Okapi(SIMP)

# resolved edges (tighter: >=2 shared component/signature tokens)
comp_index = defaultdict(list)
for i, s in enumerate(segs):
    for f in ("component", "signature"):
        for w in set(tok(s.get(f) or "")): comp_index[w].append(i)
edges = defaultdict(set)
for i, s in enumerate(segs):
    for link in (s.get("links_to") or []):
        lt = set(tok(link));  need = max(2, len(lt) // 2)
        if not lt: continue
        cand = Counter()
        for w in lt:
            for j in comp_index.get(w, []): cand[w and j] += 1
        for j, n in cand.items():
            if n >= need and j != i: edges[i].add(j)
sym_nodes = [i for i, s in enumerate(segs) if s.get("symptoms")]
SYM_TOK = {i: set(tok(graph_text(segs[i]))) for i in sym_nodes}

def chain_reach(qtok, hops=3, entries=3, fanout=5):
    qs = set(qtok)
    scored = sorted(sym_nodes, key=lambda i: -len(qs & SYM_TOK[i]))
    starts = [i for i in scored[:entries] if qs & SYM_TOK[i]]
    seen = set(starts); frontier = set(starts)
    for _ in range(hops):
        nxt = set()
        for u in frontier:
            nxt |= set(list(edges.get(u, set()))[:fanout])
        nxt -= seen; seen |= nxt; frontier = nxt
        if not frontier: break
    return starts, seen

# targets
def cause_terms(r):
    cdf = CDF
    return {w for w in ct(r["documented_cause"]) if math.log(len(recs)/(cdf.get(w,0)+1)) >= 2.5}
CDF = Counter()
for r in recs:
    for w in ct(r["documented_cause"]): CDF[w] += 1
def subsys_tokens(r): return set(tok(r.get("subsystem") or ""))

def coverage(idxs, terms):
    if not terms: return None
    txt = set()
    for i in idxs: txt |= ct(segs[i].get("raw_text") or "") | set(tok(graph_text(segs[i])))
    return len(terms & txt) / len(terms)

res = {k: {"cause": [], "subsys": []} for k in ("raw", "graph", "simp", "chain")}
for r in recs:
    q = salience(r["symptom"]) + " " + r["symptom"]
    qt = tok(q)
    top = {"raw": bm_raw.get_top_n(qt, list(range(len(segs))), n=8),
           "graph": bm_graph.get_top_n(qt, list(range(len(segs))), n=8),
           "simp": bm_simp.get_top_n(qt, list(range(len(segs))), n=8)}
    _, reached = chain_reach(qt)
    top["chain"] = list(reached)
    ctм = cause_terms(r); sst = subsys_tokens(r)
    for k, idxs in top.items():
        cv = coverage(idxs, ctм)
        if cv is not None: res[k]["cause"].append(cv)
        if sst:
            hit = any(sst & set(tok(segs[i].get("subsystem") or "")) for i in idxs)
            res[k]["subsys"].append(1.0 if hit else 0.0)

print(f"anomalies={len(recs)}  sym_nodes={len(sym_nodes)}  chain edges={sum(len(v) for v in edges.values())}")
print(f"{'strategy':8}{'cause-term coverage':>22}{'right-subsystem hit rate':>28}{'reach size':>12}")
for k in ("raw", "graph", "simp", "chain"):
    cc = st.mean(res[k]["cause"]); ss = st.mean(res[k]["subsys"])
    print(f"{k:8}{cc:>22.3f}{ss:>28.1%}")
# chain reach size
rs = []
for r in recs:
    _, reached = chain_reach(tok(salience(r["symptom"]) + " " + r["symptom"])); rs.append(len(reached))
print(f"\nchain reach set size: mean {st.mean(rs):.1f} median {st.median(rs):.0f} "
      f"(entries with 0 reach: {sum(1 for x in rs if x==0)}/{len(rs)})")
