# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file coverage_vs_accuracy.py
# @brief Does the coverage dial track SUBSYSTEM accuracy, or only in/out-of-corpus scope? Compute mean coverage in subsystem-correct vs wrong, AUC of coverage-.
#
# >subsystem-correct, and accuracy by coverage bin — OVERALL and WITHIN the
# in-scope set. The within-scope AUC is the honest test: if coverage were a
# confidence signal it would still discriminate once in/out is held fixed; it
# does not (~chance). Deterministic; scores the shipped single-shot dump
# (carries the query-time `coverage`). Run from analysis/.
#
"""Does the coverage dial track SUBSYSTEM accuracy, or only in/out-of-corpus scope? Compute mean
coverage in subsystem-correct vs wrong, AUC of coverage->subsystem-correct, and accuracy by coverage
bin — OVERALL and WITHIN the in-scope set. The within-scope AUC is the honest test: if coverage were a
confidence signal it would still discriminate once in/out is held fixed; it does not (~chance).
Deterministic; scores the shipped single-shot dump (carries the query-time `coverage`). Run from analysis/."""
import sys
import os
import json, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subsystem_match import aliases_for, names, is_in_corpus

from _paths import _d, _DATA_ROOT   # noqa: E402


CUT = 0.38   # shipped coverage_flag threshold (diagnose.py COVERAGE_CUT)
dec = [json.loads(l) for l in open(_d("decisions_146_k8.jsonl"), encoding="utf-8") if l.strip()]
rows = []
for r in dec:
    txt = "" if r["declined"] else r.get("decision", "")
    ok = names(txt, aliases_for(r.get("subsystem") or ""))
    rows.append({"cov": r.get("coverage"), "ok": ok, "inscope": is_in_corpus(r.get("subsystem") or "")})
rows = [x for x in rows if x["cov"] is not None]

def auc(group):
    pos = [x["cov"] for x in group if x["ok"]]; neg = [x["cov"] for x in group if not x["ok"]]
    if not pos or not neg: return None, len(pos), len(neg)
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg)), len(pos), len(neg)

def report(group, label):
    a, npos, nneg = auc(group)
    cok = [x["cov"] for x in group if x["ok"]]; cno = [x["cov"] for x in group if not x["ok"]]
    print(f"=== {label} (n={len(group)}) ===")
    print(f"  mean coverage | CORRECT {st.mean(cok):.3f} (n={npos})  vs  WRONG {st.mean(cno):.3f} (n={nneg})")
    print(f"  AUC(coverage -> subsystem-correct): {a:.3f}" if a is not None else "  AUC: n/a")
    hi = [x for x in group if x["cov"] >= CUT]; lo = [x for x in group if x["cov"] < CUT]
    for nm, g in [("coverage >= 0.38", hi), ("coverage <  0.38", lo)]:
        if g: print(f"  {nm}: acc {sum(x['ok'] for x in g)}/{len(g)} = {sum(x['ok'] for x in g)/len(g):.0%}")
    print()

report(rows, "ALL 146")
report([x for x in rows if x["inscope"]], "IN-SCOPE only (107)")
