# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file exact_component.py
# @brief EXACT-COMPONENT accuracy: not "right subsystem" but "right THING" — does the decision name the specific failed part the documented cause identifies (O.
#
# -ring, switching relay, breakout microswitch, dc-dc converter, RTV, B-nut
# seal ...)? Ground truth = component nouns extracted from the DIAGNOSTIC
# portion of the atlas documented_cause (distinctive IDF>=2.5 AND in a broad
# component lexicon). No LLM judging — deterministic substring, two strictness
# levels. Reports primary-component hit, any-component hit, and component
# coverage, overall + by subsystem, and dumps a calibration sample to eyeball
# fairness. Scores the single-shot decisions (full symptom, one call) as
# primary; multi-turn final for reference. Env: DUMP (default
# decisions_146_k8.jsonl), SAMPLE (default 25).
#
"""EXACT-COMPONENT accuracy: not "right subsystem" but "right THING" — does the decision name the
specific failed part the documented cause identifies (O-ring, switching relay, breakout microswitch,
dc-dc converter, RTV, B-nut seal ...)? Ground truth = component nouns extracted from the DIAGNOSTIC
portion of the atlas documented_cause (distinctive IDF>=2.5 AND in a broad component lexicon). No LLM
judging — deterministic substring, two strictness levels. Reports primary-component hit, any-component
hit, and component coverage, overall + by subsystem, and dumps a calibration sample to eyeball fairness.
Scores the single-shot decisions (full symptom, one call) as primary; multi-turn final for reference.
Env: DUMP (default decisions_146_k8.jsonl), SAMPLE (default 25)."""
import sys
import json, os, re, math, sys, statistics as st
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subsystem_match import is_in_corpus, fair as fair_fn  # shared subset definitions

from _paths import _d, _DATA_ROOT   # noqa: E402


ATLAS = os.environ.get("LODESTAR_ATLAS", "../../apollo-anomaly-atlas/data/atlas_parsed.json")
atlas = json.load(open(ATLAS, encoding="utf-8"))
flat = [an for m in atlas["missions"] for an in m.get("anomalies", [])]

dec = [json.loads(l) for l in open(_d(os.environ.get("DUMP", "decisions_146_k8.jsonl")),
                                   encoding="utf-8") if l.strip()]
mt = {}
mtf = _d("mt_subsystem_ci146.jsonl")
if os.path.exists(mtf):
    for l in open(mtf, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); mt[r["symptom"]] = r.get("final_decision", "")

# --- component lexicon (broad; the specific failed part, not the mechanism) ---
COMPONENT = re.compile(r"\b("
 r"switch|microswitch|relay|valve|poppet|spool|solenoid|actuator|regulator|diverter|"
 r"connector|wire|cable|harness|terminal|contact|plug|socket|receptacle|jumper|splice|"
 r"o-?ring|oring|seal|gasket|b-?nut|bellows|diaphragm|packing|grommet|"
 r"converter|inverter|transducer|sensor|amplifier|capacitor|diode|resistor|transistor|resolver|"
 r"potentiometer|oscillator|filter|circuit|board|condenser|rectifier|coil|winding|transformer|"
 r"commutator|brush|stator|rotor|armature|magnet|electrode|filament|grid|"
 r"bearing|\bgear|\bcam\b|shaft|spring|lever|bolt|screw|\bweld|joint|hinge|latch|strut|bracket|"
 r"fitting|clamp|retainer|roller|pulley|sprocket|detent|"
 r"tank|\bcell\b|batter|\bbus\b|breaker|\bfuse|module|assembly|housing|chamber|nozzle|orifice|"
 r"\btube|pipe|hose|duct|\bline\b|\bvent|\bport\b|screen|membrane|"
 r"heater|radiator|evaporator|sublimat|pump|motor|\bfan\b|blower|injector|bladder|canister|cartridge|wick|"
 r"lamp|\blight|bulb|lens|window|glass|mirror|prism|gauge|indicator|\bmeter\b|dial|stylus|"
 r"rtv|potting|insulation|adhesive|epoxy|teflon|solder|\bfilm\b|coating|lubricant|sealant|"
 r"antenna|transponder|transmitter|receiver|waveguide|"
 r"gyro|accelerometer|gimbal|encoder|thruster|quad|nut|pin\b"
 r")", re.I)
MECH = re.compile(r"\b(short|arc|leak|corro|swell|crack|worn|wear|stuck|stick|seiz|broke|broken|ruptur|"
 r"burn|melt|deform|block|clog|contaminat|oversize|misalign|fatigue|intermittent|drift|open\b|"
 r"loose|disconnect|degrad|embrittle|separat|overheat)", re.I)
AFT = re.compile(r"postflight|post-flight|bench test|vibrat|x-ray|temperature test|\bduplicat|ground test|"
 r"review board|teardown|\binspect|closed\b|no hardware|warranted|redesign|\bmodif|relocated|improved|"
 r"corrective|for future|future mission|recommend|changed to|material changed|was changed|replaced", re.I)

_T = re.compile(r"[a-z][a-z\-]{3,}")
GEN = set("the a an of to in on and or with at from for was were is are be this that these those it its "
    "they them their has had have will would could should may might not no non due most probable caused "
    "cause failure failed fault anomaly report system module condition which whose found made used later "
    "when than then also into more over under between".split())
def ct(s): return [w for w in _T.findall((s or "").lower()) if w not in GEN]

NC = len(flat); CDF = Counter()
for x in flat:
    for w in set(ct(x.get("documented_cause", ""))): CDF[w] += 1
def idf(w): return math.log(NC / (CDF.get(w, 0) + 1))

def diag_text(cause):
    sents = [s.strip() for s in re.split(r"(?<=[.;])\s+", cause or "") if len(s.strip()) >= 8]
    keep = [s for s in sents if not AFT.search(s)]
    return " ".join(keep) if keep else (cause or "")

def norm(t): return re.sub(r"[^a-z0-9]", "", (t or "").lower())        # strip hyphens/spaces
_GERUND = re.compile(r"(ing|ed)$")
_ALLOW = ("o-ring", "b-nut", "dc-dc", "quick-disconnect")               # keep these compounds whole
def components(cause):
    """ground-truth component set: distinctive (IDF>=2.5) part nouns in the diagnostic text. Hyphenated
    compounds are split to their component head (motor-operated->motor, environmental-seal->seal) except
    a small allowlist kept whole (o-ring); gerund/participle modifiers (switching, operated) dropped.
    Returns list sorted by IDF desc (primary = most specific part first)."""
    dt = diag_text(cause)
    cand = {}
    for w in ct(dt):
        if idf(w) < 2.5:
            continue
        pieces = [w] if (w in _ALLOW or "-" not in w) else [p for p in w.split("-") if len(p) >= 4]
        for p in pieces:
            if COMPONENT.search(p) and not MECH.fullmatch(p) and not _GERUND.search(p):
                cand[p] = max(cand.get(p, 0), idf(w))
    return sorted(cand, key=lambda k: cand[k], reverse=True)

def named(term, decision, loose=False):
    d = (decision or "").lower(); t = term.lower()
    stems = {t, t[:-1] if t.endswith("s") else t}                     # plural-insensitive
    for s in stems:
        if s and (s in d or (norm(s) and norm(s) in norm(d))):        # strict + hyphen/space-insensitive
            return True
    if loose:                                                          # head-token match (o-ring -> ring)
        toks = [tk for tk in re.split(r"[^a-z0-9]+", t) if len(tk) >= 4]
        return any(tk in d or (tk.endswith("s") and tk[:-1] in d) for tk in toks)
    return False

# index atlas causes by symptom to align decisions -> ground truth
cause_of = {x["symptom"]: x.get("documented_cause", "") for x in flat}
subsys_of = {x["symptom"]: x.get("subsystem", "") for x in flat}

def score(decisions, label, sample=0):
    recs = []; by = defaultdict(lambda: [0, 0]); samples = []
    icd_of = {d["symptom"]: d.get("in_corpus_domain") for d in decisions}
    for d in decisions:
        sym = d["symptom"]; cause = cause_of.get(sym, d.get("documented_cause", ""))
        comps = components(cause)
        if not comps: continue
        text = "" if d.get("declined") else d.get("decision") or d.get("final_decision") or ""
        sub = subsys_of.get(sym, "?")
        rec = {"sub": sub, "icd": bool(d.get("in_corpus_domain")), "inscope": is_in_corpus(sub),
               "fair": fair_fn(sub, cause), "prim": named(comps[0], text),
               "any": any(named(c, text) for c in comps),
               "anyl": any(named(c, text, loose=True) for c in comps),
               "cov": sum(1 for c in comps if named(c, text)) / len(comps)}
        recs.append(rec)
        by[sub][0] += 1; by[sub][1] += 1 if rec["any"] else 0
        if sample and len(samples) < sample:
            samples.append((sub, comps, text, rec["prim"], rec["any"], rec["cov"]))

    def line(subset, name):
        if not subset:
            print(f"  {name:<24} n=0"); return
        n = len(subset)
        print(f"  {name:<24} n={n:<3}  primary {sum(r['prim'] for r in subset)/n:.0%}  "
              f"any-part {sum(r['any'] for r in subset)/n:.0%}  "
              f"any-loose {sum(r['anyl'] for r in subset)/n:.0%}  "
              f"cov {st.mean([r['cov'] for r in subset]):.2f}")

    print(f"\n=== EXACT-COMPONENT [{label}] (scoreable = cause names an extractable part) ===")
    line(recs, "ALL 146")
    line([r for r in recs if r["inscope"]], "in-scope subsystem (107)")
    line([r for r in recs if r["icd"]], "in_corpus_domain (101)")
    line([r for r in recs if r["fair"]], "fair in-scope+knowable (89)")
    print(f"  by subsystem (any-part hit): " + "  ".join(
        f"{s}:{v[1]}/{v[0]}" for s, v in sorted(by.items(), key=lambda kv: -kv[1][0]) if v[0] >= 3))
    return samples

s = score(dec, "single-shot (full symptom)", sample=int(os.environ.get("SAMPLE", "25")))
if mt:
    mt_rows = [{"symptom": k, "final_decision": v, "declined": False} for k, v in mt.items()]
    for r in mt_rows: r["decision"] = r["final_decision"]
    score(mt_rows, "multi-turn final (early-terminated)")

print("\n=== CALIBRATION SAMPLE (subsystem | components[primary first] | decision head | prim/any/cov) ===")
for sub, comps, text, p, a, cov in s:
    print(f"[{sub}] comp={comps}")
    print(f"    said: {(text[:110] or 'DECLINED')}")
    print(f"    -> primary={'Y' if p else 'n'} any={'Y' if a else 'n'} cov={cov:.2f}")
