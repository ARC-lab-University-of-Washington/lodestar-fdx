# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file score_subsystem.py
# @brief Score subsystem-level accuracy from a decisions dump (data/decisions_146_k8.jsonl) — no model calls, re-runnable.
#
# subsystem-match = the decision names the anomaly's subsystem (aliases incl.
# the terse OCR abbreviations the chains use). Reports match (overall +
# in/out), a random-wrong-subsystem CONTROL, strict distinctive-cause-term
# rate, and decline rate. `--inspect` dumps mismatches to calibrate.
#
"""Score subsystem-level accuracy from a decisions dump (data/decisions_146_k8.jsonl) — no model calls,
re-runnable. subsystem-match = the decision names the anomaly's subsystem (aliases incl. the terse OCR
abbreviations the chains use). Reports match (overall + in/out), a random-wrong-subsystem CONTROL,
strict distinctive-cause-term rate, and decline rate. `--inspect` dumps mismatches to calibrate."""
import sys
import json, os, sys, math, re, random
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subsystem_match import aliases_for, names, is_in_corpus as is_in  # shared success criterion

from _paths import _d, _DATA_ROOT   # noqa: E402


DUMP = _d(os.environ.get("DUMP", "decisions_146_k8.jsonl"))
rows = [json.loads(l) for l in open(DUMP, encoding="utf-8") if l.strip()]

recs_full = [json.loads(l) for l in open(_d("full_log.jsonl"), encoding="utf-8") if l.strip()]
CDF = Counter()
for r in recs_full:
    for w in re.findall(r"[a-z][a-z\-]{3,}", (r["documented_cause"] or "").lower()): CDF[w] += 1
def strict_terms(r):
    return {w for w in re.findall(r"[a-z][a-z\-]{3,}", (r["documented_cause"] or "").lower())
            if math.log(len(recs_full)/(CDF.get(w,0)+1)) >= 2.5}

allsubs = sorted(set((r.get("subsystem") or "").strip() for r in rows if r.get("subsystem")))
rng = random.Random(20260711)
sub_ok = ctrl_ok = strict_ok = strict_n = declined = 0
in_ok = in_n = out_ok = out_n = 0
inspect = "--inspect" in sys.argv; misses = []
for r in rows:
    txt = "" if r["declined"] else r.get("decision", "")
    if r["declined"]: declined += 1
    sub = r.get("subsystem") or ""
    m = names(txt, aliases_for(sub)); sub_ok += 1 if m else 0
    inc = is_in(sub)
    if inc: in_n += 1; in_ok += 1 if m else 0
    else: out_n += 1; out_ok += 1 if m else 0
    wrong = rng.choice([s for s in allsubs if s != sub] or [sub])
    ctrl_ok += 1 if names(txt, aliases_for(wrong)) else 0
    st_terms = strict_terms(r)
    if st_terms:
        strict_n += 1; strict_ok += 1 if any(w in txt.lower() for w in st_terms) else 0
    if inspect and not m and not r["declined"]:
        misses.append((sub, r["symptom"][:50], txt[:120]))

n = len(rows)
print(f"scored {n} decisions from {DUMP}")
print(f"SUBSYSTEM match:  {sub_ok}/{n} = {sub_ok/n:.0%}   (in-corpus {in_ok}/{in_n}={in_ok/max(in_n,1):.0%}, out {out_ok}/{out_n}={out_ok/max(out_n,1):.0%})")
print(f"  random-wrong CONTROL: {ctrl_ok}/{n} = {ctrl_ok/n:.0%}   signal (match-control) = {(sub_ok-ctrl_ok)/n:+.0%}")
print(f"strict distinctive-term: {strict_ok}/{strict_n} = {strict_ok/max(strict_n,1):.0%}")
print(f"declined: {declined}/{n} = {declined/n:.0%}")
if inspect:
    print(f"\n-- {len(misses)} subsystem MISSES (calibrate aliases) --")
    for sub, sym, txt in misses[:25]:
        print(f"  [{sub}] {sym}\n      -> {txt}")
