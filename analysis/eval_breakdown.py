# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file eval_breakdown.py
# @brief Accuracy breakdown on the 146 dev-box decisions: by FAILURE TYPE, by IN/OUT of corpus, and RATED from LOW to HIGH coverage (does the coverage dial pre.
#
# dict accuracy?). No model calls.
#
"""Accuracy breakdown on the 146 dev-box decisions: by FAILURE TYPE, by IN/OUT of corpus, and RATED
from LOW to HIGH coverage (does the coverage dial predict accuracy?). No model calls."""
import sys
import json, os, math, re
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import _d, _DATA_ROOT   # noqa: E402


DUMP = _d(os.environ.get("DUMP", "decisions_146_k8.jsonl"))
rows = [json.loads(l) for l in open(DUMP, encoding="utf-8") if l.strip()]

ALIASES = {
 "gnc": ["gnc","guidance","navigation","\\bimu\\b","gyro","attitude","fdai","gimbal","program alarm",
         "dsky","\\bagc\\b","\\bcmc\\b","platform","coas","optics","star","align","\\bems\\b","entry monitor",
         "pgns","\\bags\\b","accelerometer","noun","verb"],
 "eclss": ["eclss","\\becs\\b","environmental","cool","glycol","water","oxygen","\\bo2\\b","cabin","co2",
           "suit","humidity","sublimator","evaporator","potable","surge tank","\\bwms\\b","freon"],
 "eps": ["\\beps\\b","electrical","fuel cell","batter","\\bbat\\b","inverter","undervolt","main bus","mn bus",
         "\\bdc\\b","\\bac bus","bus tie","pyro","volt","\\bamp","d-c","a-c"],
 "comms": ["comm","s-band","\\bvhf\\b","telemetry","antenna","transponder","\\bpcm\\b","voice","subcarrier",
           "uplink","downlink","high gain","\\bhga\\b","\\busb\\b","omni"],
 "rcs": ["\\brcs\\b","reaction control","thruster","quad","propellant isol","isol valve","\\bhe 1","\\bhe 2","\\btca"],
 "sps": ["\\bsps\\b","service propulsion","ball valve","gimbal motor","\\bpugs\\b"],
 "instrumentation": ["instrumentation","commutator","transducer","signal condition","readout","sensor","\\bpcm\\b"],
 "structures": ["structur","hatch","window","docking","\\bseal","shade","tunnel","probe","drogue","crack"],
 "lighting": ["lighting","spotlight","floodlight","integral light","\\blamp","\\blight"],
 "c&w": ["caution","warning","master alarm","c&w","annunciator"],
 "lm dps": ["descent propulsion","descent engine","\\bdps\\b"], "dps": ["descent propulsion","descent engine","\\bdps\\b"],
 "aps": ["ascent propulsion","ascent engine","\\baps\\b"], "lm": ["lunar module","\\blm cabin","\\blm "],
 "eva": ["\\beva\\b","\\bemu\\b","\\bsuit","purge","\\bops\\b","plss","umbilical","opgs"],
 "alsep": ["alsep","experiment","seismic","cplee","\\bside\\b","\\bleam\\b","central station","\\brtg\\b","magnetometer"],
 "lrv": ["\\blrv\\b","rover","steering","traction","wheel"], "camera": ["camera","magazine","film","lens"],
 "sim-bay": ["sim bay","sim-bay","scientific instrument","mapping camera","panoramic"],
 "biomed": ["biomed","bioinstrument","\\becg\\b","physiolog"], "crew-health": ["crew health","medical","fatigue","illness"],
 "gfe": ["government furnished","\\bgfe\\b"], "crew equipment": ["stowage","restraint","crew equipment"],
}
def aliases_for(s):
    k = (s or "").strip().lower()
    if k in ALIASES: return ALIASES[k]
    return [re.escape(t) for t in re.split(r"[^a-z0-9]+", k) if len(t) >= 3] or [re.escape(k)]
def names(text, al):
    t = (text or "").lower(); return any(re.search(a, t) for a in al)
_IN = ("eclss","ecs","eps","sps","rcs","gnc","comm","instrumentation","structure","propuls","fuel","cryo",
       "batter","electrical","guidance","navigation","docking","s-band","vhf","pyro","lighting","c&w","dps","aps"," lm")
def is_in(s): s=(s or "").lower(); return any(w in s for w in _IN)

recs_full = [json.loads(l) for l in open(_d("full_log.jsonl"), encoding="utf-8") if l.strip()]
CDF = Counter()
for r in recs_full:
    for w in re.findall(r"[a-z][a-z\-]{3,}", (r["documented_cause"] or "").lower()): CDF[w] += 1
def strict_hit(r):
    cts = {w for w in re.findall(r"[a-z][a-z\-]{3,}", (r["documented_cause"] or "").lower())
           if math.log(len(recs_full)/(CDF.get(w,0)+1)) >= 2.5}
    if not cts: return None
    return any(w in (r["decision"] or "").lower() for w in cts)

for r in rows:
    r["_in"] = is_in(r.get("subsystem"))
    r["_sub"] = (not r["declined"]) and names(r.get("decision",""), aliases_for(r.get("subsystem")))
    r["_strict"] = strict_hit(r)
    r["_err"] = ("unavailable" in r.get("decision","")) or (not r["declined"] and not r.get("decision","").strip())
    r["_cov"] = r.get("coverage")

n = len(rows)
def pct(a, b): return f"{a}/{b} = {a/b:.0%}" if b else "n/a"
print(f"=== 146 crew-observed anomalies (dev-box, k=8, +fallback) ===\n")

# 1. IN vs OUT of corpus
inc = [r for r in rows if r["_in"]]; out = [r for r in rows if not r["_in"]]
print("IN vs OUT OF CORPUS")
print(f"  in-corpus  (n={len(inc)}): subsystem {pct(sum(r['_sub'] for r in inc),len(inc))} · "
      f"strict {pct(sum(1 for r in inc if r['_strict']),sum(1 for r in inc if r['_strict'] is not None))}")
print(f"  out-of-corp(n={len(out)}): subsystem {pct(sum(r['_sub'] for r in out),len(out))} · "
      f"strict {pct(sum(1 for r in out if r['_strict']),sum(1 for r in out if r['_strict'] is not None))}")

# 2. FAILURE-TYPE taxonomy (subsystem x strict) + what the misses look like
print("\nOUTCOME TAXONOMY (subsystem-match x strict-term)")
both = sum(1 for r in rows if r["_sub"] and r["_strict"])
subonly = sum(1 for r in rows if r["_sub"] and not r["_strict"])
termonly = sum(1 for r in rows if (not r["_sub"]) and r["_strict"])
neither = sum(1 for r in rows if (not r["_sub"]) and not r["_strict"])
print(f"  both right (subsystem+term): {both} ({both/n:.0%})")
print(f"  right system, wrong/no term: {subonly} ({subonly/n:.0%})   <- correct triage, cause not named")
print(f"  right term, wrong system:    {termonly} ({termonly/n:.0%})   <- named the culprit but framed wrong")
print(f"  both wrong:                  {neither} ({neither/n:.0%})   <- true miss")
miss = [r for r in rows if not r["_sub"]]
err = sum(1 for r in miss if r["_err"]); decl = sum(1 for r in miss if r["declined"])
print(f"  of the {len(miss)} subsystem misses: {decl} declined, {err} empty/error, "
      f"{len(miss)-decl-err} answered-but-wrong-system (chase strayed)")

# 3. RATE LOW -> HIGH coverage (quintiles); does the coverage dial predict accuracy?
cov_rows = [r for r in rows if r["_cov"] is not None]
cov_rows.sort(key=lambda r: r["_cov"])
q = len(cov_rows) // 5
print(f"\nRATED LOW -> HIGH COVERAGE (quintiles, n={len(cov_rows)})")
print(f"  {'bin':16}{'cov range':>14}{'n':>4}{'subsystem':>12}{'strict':>9}{'%in-corpus':>12}")
labels = ["Q1 lowest","Q2","Q3","Q4","Q5 highest"]
for i in range(5):
    seg = cov_rows[i*q:(i+1)*q] if i < 4 else cov_rows[4*q:]
    if not seg: continue
    sub = sum(r["_sub"] for r in seg); sn = len(seg)
    st_ = [r for r in seg if r["_strict"] is not None]; sth = sum(1 for r in st_ if r["_strict"])
    lo, hi = seg[0]["_cov"], seg[-1]["_cov"]
    print(f"  {labels[i]:16}{f'{lo:.2f}-{hi:.2f}':>14}{sn:>4}{f'{sub}/{sn}={sub/sn:.0%}':>12}"
          f"{f'{sth}/{len(st_)}={sth/max(len(st_),1):.0%}':>9}{sum(r['_in'] for r in seg)/sn:>11.0%}")

# coverage as a predictor of correctness (AUC)
def auc(pos, neg):
    if not pos or not neg: return .5
    c = t = 0
    for a in pos:
        for b in neg:
            t += 1; c += 1 if a > b else .5 if a == b else 0
    return c / t
cp = [r["_cov"] for r in cov_rows if r["_sub"]]; cn = [r["_cov"] for r in cov_rows if not r["_sub"]]
import statistics as st
print(f"\n  coverage predicts subsystem-correct: mean cov correct={st.mean(cp):.2f} vs wrong={st.mean(cn):.2f}, "
      f"AUC={auc(cp,cn):.2f}")
# latency
lat = [r["latency_s"] for r in rows]
print(f"\nLatency (dev-box): mean {st.mean(lat):.1f}s median {st.median(lat):.1f}s")
