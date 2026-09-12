# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file failure_wherewhy.py
# @brief FAILURE-ONLY analysis.
#
# Take the subsystem-match FAILURES (Lodestar named the wrong system), and
# characterize the diagnostic content it missed as WHERE
# (component/location/hardware) vs WHY (failure mechanism). Tests the
# assumption that the diagnostic terms are mostly where + why, and asks: in a
# failure, did Lodestar at least name SOME where/why (right kind, wrong
# specifics) or nothing? Deterministic; scores the existing dev-box decisions.
# No model calls.
#
"""FAILURE-ONLY analysis. Take the subsystem-match FAILURES (Lodestar named the wrong system), and
characterize the diagnostic content it missed as WHERE (component/location/hardware) vs WHY
(failure mechanism). Tests the assumption that the diagnostic terms are mostly where + why, and asks:
in a failure, did Lodestar at least name SOME where/why (right kind, wrong specifics) or nothing?
Deterministic; scores the existing dev-box decisions. No model calls."""
import sys
import json, os, re, math
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import _d, _DATA_ROOT   # noqa: E402


recs = [json.loads(l) for l in open(_d("full_log.jsonl"), encoding="utf-8") if l.strip()]
dec = [json.loads(l) for l in open(_d("decisions_146_k8.jsonl"), encoding="utf-8") if l.strip()]

# subsystem matcher (from score_subsystem.py) to split success/failure
ALIASES = {
 "gnc": ["gnc","guidance","navigation","\\bimu\\b","gyro","attitude","fdai","gimbal","program alarm","dsky","\\bagc\\b","\\bcmc\\b","platform","coas","optics","star","align","\\bems\\b","entry monitor","pgns","\\bags\\b","accelerometer","noun","verb"],
 "eclss": ["eclss","\\becs\\b","environmental","cool","glycol","water","oxygen","\\bo2\\b","cabin","co2","suit","humidity","sublimator","evaporator","potable","surge tank","\\bwms\\b","freon"],
 "eps": ["\\beps\\b","electrical","fuel cell","batter","\\bbat\\b","inverter","undervolt","main bus","mn bus","\\bdc\\b","\\bac bus","bus tie","pyro","volt","\\bamp","d-c","a-c"],
 "comms": ["comm","s-band","\\bvhf\\b","telemetry","antenna","transponder","\\bpcm\\b","voice","subcarrier","uplink","downlink","high gain","\\bhga\\b","\\busb\\b","omni"],
 "rcs": ["\\brcs\\b","reaction control","thruster","quad","propellant isol","isol valve","\\bhe 1","\\bhe 2","\\btca"],
 "sps": ["\\bsps\\b","service propulsion","ball valve","gimbal motor","\\bpugs\\b"],
 "instrumentation": ["instrumentation","commutator","transducer","signal condition","readout","sensor","\\bpcm\\b"],
 "structures": ["structur","hatch","window","docking","\\bseal","shade","tunnel","probe","drogue","crack"],
 "lighting": ["lighting","spotlight","floodlight","integral light","\\blamp","\\blight"], "c&w": ["caution","warning","master alarm","c&w","annunciator"],
 "lm dps": ["descent propulsion","descent engine","\\bdps\\b"], "dps": ["descent propulsion","descent engine","\\bdps\\b"], "aps": ["ascent propulsion","ascent engine","\\baps\\b"], "lm": ["lunar module","\\blm cabin","\\blm "],
 "eva": ["\\beva\\b","\\bemu\\b","\\bsuit","purge","\\bops\\b","plss","umbilical","opgs"], "alsep": ["alsep","experiment","seismic","cplee","\\bside\\b","\\bleam\\b","central station","\\brtg\\b","magnetometer"],
 "lrv": ["\\blrv\\b","rover","steering","traction","wheel"], "camera": ["camera","magazine","film","lens"], "sim-bay": ["sim bay","sim-bay","scientific instrument","mapping camera","panoramic"],
 "biomed": ["biomed","bioinstrument","\\becg\\b","physiolog"], "crew-health": ["crew health","medical","fatigue","illness"], "gfe": ["government furnished","\\bgfe\\b"], "crew equipment": ["stowage","restraint","crew equipment"]}
def aliases_for(s):
    k=(s or "").strip().lower()
    return ALIASES.get(k) or [re.escape(t) for t in re.split(r"[^a-z0-9]+",k) if len(t)>=3] or [re.escape(k)]
def names(text, al): t=(text or "").lower(); return any(re.search(a,t) for a in al)

# WHERE (component/location/hardware) vs WHY (failure mechanism) lexicons
WHERE = re.compile(r"\b(switch|valve|tank|cell|\bbus\b|batter|capacitor|relay|connector|o-?ring|seal|"
 r"wir|cable|motor|pump|transducer|sensor|antenna|transponder|gyro|regulator|filter|bearing|actuator|"
 r"coil|module|assembly|breaker|nozzle|diode|resistor|solenoid|heater|\bfan\b|thruster|quad|inverter|"
 r"commutator|timer|gauge|indicator|\blamp|\blight|window|hatch|shade|strut|bracket|fitting|gasket|"
 r"bellows|diaphragm|orifice|chamber|housing|shaft|\bgear|spring|\bcam\b|lever|\bpin\b|bolt|\bweld|"
 r"joint|\btube|pipe|insulation|potting|\bfilm\b|\bgrid\b|filament|electrode|\bplate|contact|circuit|"
 r"transducer|harness|terminal|bus tie|converter|amplifier|stylus)", re.I)
WHY = re.compile(r"\b(arc|corona|short|shorted|leak|ignit|combust|swell|damag|broke|broken|breakage|"
 r"crack|worn|\bwear\b|corro|contaminat|overheat|stuck|stick|seiz|embrittle|hammer|degrad|oversize|"
 r"misalign|fatigue|ruptur|burn|melt|deform|block|clog|intermittent|transient|spike|surge|oscillat|"
 r"\bdrift|vibrat|\bshock\b|\bopen\b|separat|loose|disconnect|inadvert|erroneous|noise|marginal)", re.I)

AFT = re.compile(r"postflight|post-flight|bench test|vibrat test|x-ray|temperature test|duplicat|"
 r"ground test|review board|teardown|closed\b|no hardware|no performance|warranted|redesign|modif|"
 r"relocated|improved|corrective|for future|recommend|powered down|lifeboat|recycl", re.I)
def diag_terms(cause):
    sents=[s for s in re.split(r"(?<=[.;])\s+",cause) if len(s.strip())>=8 and not AFT.search(s)]
    return " ".join(sents)

def classify(term):
    w=WHERE.search(term); y=WHY.search(term)
    return "both" if (w and y) else "where" if w else "why" if y else "other"

# split into success / failure by subsystem match
fails, succ = [], []
for d in dec:
    ok = (not d["declined"]) and names(d.get("decision",""), aliases_for(d.get("subsystem")))
    (succ if ok else fails).append(d)
print(f"subsystem SUCCESS {len(succ)} / FAILURE {len(fails)} (of {len(dec)})\n")

# characterize the WHERE/WHY content of the documented cause in FAILURES vs SUCCESSES
def profile(group, label):
    cats = Counter(); named_where = named_why = has_where = has_why = 0
    for d in group:
        dtxt = diag_terms(d["documented_cause"]).lower()
        w_present = bool(WHERE.search(dtxt)); y_present = bool(WHY.search(dtxt))
        has_where += w_present; has_why += y_present
        # tally cause content by WHERE/WHY (word matches in diagnostic text)
        cats["where"] += len(WHERE.findall(dtxt)); cats["why"] += len(WHY.findall(dtxt))
        # did the DECISION name any where / why?
        dec_txt = (d.get("decision","") if not d["declined"] else "").lower()
        if w_present and WHERE.search(dec_txt): named_where += 1
        if y_present and WHY.search(dec_txt): named_why += 1
    n=len(group); tot=cats["where"]+cats["why"] or 1
    print(f"=== {label} (n={n}) ===")
    print(f"  documented cause has WHERE terms: {has_where}/{n}={has_where/n:.0%}   WHY terms: {has_why}/{n}={has_why/n:.0%}")
    print(f"  cause content composition: WHERE {cats['where']/tot:.0%} / WHY {cats['why']/tot:.0%}")
    print(f"  decision named SOME where (of those with where): {named_where}/{has_where}={named_where/max(has_where,1):.0%}")
    print(f"  decision named SOME why   (of those with why):   {named_why}/{has_why}={named_why/max(has_why,1):.0%}\n")

profile(fails, "FAILURES (wrong subsystem)")
profile(succ, "SUCCESSES (right subsystem)")

# concrete failure examples
print("=== failure examples: [subsystem] cause(where/why) -> decision head ===")
for d in fails[:8]:
    print(f"  [{d['subsystem']}] {d['symptom'][:44]}")
    print(f"      cause: {d['documented_cause'][:90]}")
    print(f"      said:  {(d['decision'][:80] if not d['declined'] else 'DECLINED')}\n")
