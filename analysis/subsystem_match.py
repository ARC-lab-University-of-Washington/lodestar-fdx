# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file subsystem_match.py
# @brief SINGLE SOURCE OF TRUTH for the success criterion: a decision SUCCEEDS when it names the anomaly's subsystem (alias-aware, incl.
#
# the terse OCR abbreviations the chains use). Used by the multi-turn runner
# (termination = subsystem match) and the offline scorer. Deterministic, no
# model calls. The `fair()` filter defines the scoreable-fault subset (returns 89 on the 146; the '125' in earlier drafts is stale) (drop out-
# of-scope payload/EVA + no-knowable-cause).
#
"""SINGLE SOURCE OF TRUTH for the success criterion: a decision SUCCEEDS when it names the anomaly's
subsystem (alias-aware, incl. the terse OCR abbreviations the chains use). Used by the multi-turn
runner (termination = subsystem match) and the offline scorer. Deterministic, no model calls.
The `fair()` filter defines the scoreable-fault subset (returns 89 on the 146; the '125' in earlier drafts is stale) (drop out-of-scope payload/EVA + no-knowable-cause)."""
import re

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
 "lm dps": ["descent propulsion","descent engine","\\bdps\\b"],
 "dps": ["descent propulsion","descent engine","\\bdps\\b"],
 "aps": ["ascent propulsion","ascent engine","\\baps\\b"],
 "lm": ["lunar module","\\blm cabin","\\blm "],
 "eva": ["\\beva\\b","\\bemu\\b","\\bsuit","purge","\\bops\\b","plss","umbilical","opgs"],
 "alsep": ["alsep","experiment","seismic","cplee","\\bside\\b","\\bleam\\b","central station","\\brtg\\b","magnetometer"],
 "lrv": ["\\blrv\\b","rover","steering","traction","wheel"],
 "camera": ["camera","magazine","film","lens"],
 "sim-bay": ["sim bay","sim-bay","scientific instrument","mapping camera","panoramic"],
 "biomed": ["biomed","bioinstrument","\\becg\\b","physiolog"],
 "crew-health": ["crew health","medical","fatigue","illness"],
 "gfe": ["government furnished","\\bgfe\\b"],
 "crew equipment": ["stowage","restraint","crew equipment"],
}


def aliases_for(subsys):
    key = (subsys or "").strip().lower()
    if key in ALIASES:
        return ALIASES[key]
    toks = [re.escape(t) for t in re.split(r"[^a-z0-9]+", key) if len(t) >= 3]
    return toks or [re.escape(key)]


def names(text, alist):
    t = (text or "").lower()
    return any(re.search(a, t) for a in alist)


def subsystem_match(decision, subsystem, declined=False):
    """The success test: a non-declined decision that names the subsystem (alias-aware)."""
    if declined:
        return False
    return names(decision, aliases_for(subsystem))


_IN = ("eclss","ecs","eps","sps","rcs","gnc","comm","instrumentation","structure","propuls","fuel",
       "cryo","batter","electrical","guidance","navigation","docking","s-band","vhf","pyro","lighting",
       "c&w","dps","aps"," lm")


def is_in_corpus(subsystem):
    s = (subsystem or "").lower()
    return any(w in s for w in _IN)


# --- FAIR set (the scoreable-fault subset (returns 89 on the 146; the '125' in earlier drafts is stale)): drop out-of-scope payload/EVA and no-knowable-cause ---
_OUT = ("eva","emu","suit","alsep","experiment","camera","lrv","rover","biomed","crew-health","sim-bay",
        "gfe","crew equipment","seismic","geolog","magnetometer","lpm","thumper","side","cplee","leam")
_UNK = re.compile(r"pending|tentativ|not duplicat|not reproduc|single-point|spurious|no cause|unknown|"
                  r"not determin|gas bubble|momentar|transient|nuisance|no hardware|benign|"
                  r"state of charge|no anomaly", re.I)


def fair(subsystem, documented_cause):
    """True if this crew_initiated fault is in-scope AND has a knowable cause (the scoreable set)."""
    s = (subsystem or "").lower()
    if any(w in s for w in _OUT):
        return False
    if _UNK.search(documented_cause or ""):
        return False
    return True
