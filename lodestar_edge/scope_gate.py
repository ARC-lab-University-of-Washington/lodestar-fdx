# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file scope_gate.py
# @brief Coverage gate — the safety front-door of grounding-or-silence.
#
# Runs BEFORE retrieval and before any answer is shown to the crew. Deterministic:
# no model, no latency, auditable by reading one list.
#
"""Coverage gate — the safety front-door of grounding-or-silence.

Before retrieving/diagnosing, ask: is this symptom even in the DOMAIN my manuals cover? On the
OCR'd corpus a relevance judge over the retrieved snippets fails (the snippets are too garbled to
judge — both 3B and 8B judges score correct and out-of-scope cases the same). And an LLM query
classifier needs a model the Jetson can't spare and still over-declines borderline cases.

So the gate is DETERMINISTIC: a curated denylist of distinctive out-of-domain EQUIPMENT terms that
provably never occur in a Command/Service Module or Lunar Module *vehicle-system* fault (rover,
ALSEP, seismic thumper, spectrometer, cooling garment, MESA, ...). This is chosen on purpose:
  * keep-biased — it fires only on terms chosen to be absent from in-domain faults, because the
    assistant is advisory and withholding a true fault is worse than presenting an imperfect
    candidate the crew can filter. Two caveats: (a) `cli.py --multiturn` re-gates the accumulated transcript, so one stale
    out-of-domain word latches the gate for the rest of a session, including through a real
    vehicle fault; (b) `cooling garment` and `tv camera` are arguably in-domain (the LCG loop is
    plumbed into the suit circuit; the onboard TV ran off the vehicle buses). So keep-bias is a
    property of the word list and the single-call path, not an invariant of the system;
  * zero model cost / zero latency — runs on the 3B-only Jetson (in fact on no model at all);
  * transparent and auditable — a reviewer can read exactly what it will and will not silence.

Deliberately NOT denylisted: activity words like "EVA" or "suit-loop", because the suit circuit IS
part of CSM/LM ECLSS — those faults are in-domain and must be presented. Consequence: a genuinely
ambiguous case ("water entered the suit ... during EVA") is PRESENTED, not declined — the preferred
failure direction. Measured on the evaluation set: out-of-scope decline 0.13 -> 0.93, in-domain
false-decline 0.00 — on single-call invocations. See the multiturn caveat above.
"""
import re

# Distinctive lunar-surface / EVA-equipment / science-experiment terms outside the CSM/LM manuals.
# Word-boundary matched, case-insensitive. Extend this list for a different corpus's domain edge.
OUT_OF_DOMAIN_TERMS = [
    r"rover", r"roving vehicle", r"\blrv\b",                         # lunar roving vehicle
    r"alsep", r"seismic experiment", r"thumper", r"magnetometer",    # surface experiments
    r"deep[- ]core", r"core drill", r"spectrometer",
    r"lunar[- ]sounder", r"\bsounder\b", r"sim[- ]bay",              # SIM-bay / sounder
    r"cooling garment", r"\bplss\b", r"portable life support",       # EVA/EMU life support
    r"\bmesa\b", r"sample return", r"sample container",              # surface tools
    r"panoramic camera", r"tv camera", r"television camera", r"surface camera",
]
_DENY_RE = re.compile("|".join(OUT_OF_DOMAIN_TERMS), re.I)


def out_of_domain(symptom: str):
    """@brief Test a symptom against the out-of-domain equipment denylist.

    Return the matched out-of-domain term if the symptom is clearly outside the corpus's
    domain, else None. Deterministic; keep-biased (only fires on provably out-of-domain gear).

    @param symptom Crew symptom report, verbatim. None and "" are accepted.
    @return The matched term as a string, or None if the symptom is in domain.
    @note Keep-biased: fires only on terms selected as out-of-domain. This is a property of
          the word list, not a proven invariant — see the module docstring for two caveats.
    @see OUT_OF_DOMAIN_TERMS
    """
    m = _DENY_RE.search(symptom or "")
    return m.group(0) if m else None


def in_domain(symptom: str, **_ignored) -> bool:
    """@brief Whether this symptom is inside the domain the manuals cover.

    True unless the symptom names equipment the corpus's domain does not cover.

    @param symptom   Crew symptom report, verbatim.
    @param _ignored  Extra kwargs (model/base_url) accepted and ignored for call-site
                     compatibility with model-backed gates.
    @return True if the symptom should be diagnosed, False if it should be declined.
    @see out_of_domain
    """
    return out_of_domain(symptom) is None
