# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file metrics.py
# @brief Metrics for the transcript-driven evaluation (turns/time-to-diagnosis) + lexical coverage.
#
# Stdlib only. Copied verbatim from the research harness so the on-device eval
# reproduces the published methodology exactly. time_to_diagnosis(human) =
# GET(first CapCom diagnosis) - GET(end of crew fault report)
# turns_to_diagnosis(human) = # air-to-ground exchanges between the two.
# time_to_diagnosis(lodestar) = on-device wall-clock (retrieval + generation).
# turns_to_diagnosis(lodestar) = crew<->Lodestar dialogue turns to the same
# point. Correctness scored against the DOCUMENTED resolution anchor (outcome-
# anchored, non-circular).
#
"""Metrics for the transcript-driven evaluation (turns/time-to-diagnosis) + lexical coverage.
Stdlib only. Copied verbatim from the research harness so the on-device eval reproduces the
published methodology exactly.

    time_to_diagnosis(human)  = GET(first CapCom diagnosis) - GET(end of crew fault report)
    turns_to_diagnosis(human) = # air-to-ground exchanges between the two.
    time_to_diagnosis(lodestar) = on-device wall-clock (retrieval + generation).
    turns_to_diagnosis(lodestar) = crew<->Lodestar dialogue turns to the same point.
Correctness scored against the DOCUMENTED resolution anchor (outcome-anchored, non-circular).
"""
import re


def get_to_seconds(get: str):
    if not get:
        return None
    parts = get.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        h, m, s = (int(p) for p in parts)
    except ValueError:
        return None
    return (h * 60 + m) * 60 + s


def fmt_dur(sec):
    if sec is None:
        return "  (unreconciled)"
    sec = int(round(sec))
    if sec < 60:
        return f"{sec} s"
    if sec < 3600:
        return f"{sec // 60} min {sec % 60} s"
    return f"{sec // 3600} h {(sec % 3600) // 60} min"


def human_time_to_diagnosis(episode, checkpoint_name=None):
    hb = episode["human_baseline"]
    name = checkpoint_name or hb.get("default_diagnosis_checkpoint")
    cp = next((c for c in hb["checkpoints"] if c["name"] == name), None)
    if cp is None:
        return None, False, None
    t_end = get_to_seconds(hb["report_end_get"])
    t_cp = get_to_seconds(cp.get("get"))
    if t_end is not None and t_cp is not None:
        return t_cp - t_end, True, cp
    d = cp.get("delta_from_report_end_s")
    return d, (d is not None), cp


def _norm(t):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower()).split())


def diagnosis_coverage(lodestar_output: str, anchor: dict):
    """Fraction of anchor component_terms present in Lodestar's diagnosis (lexical proxy) + the
    corrective-action gist (power down CM / LM lifeboat)."""
    text = _norm(lodestar_output)
    terms = anchor.get("component_terms", [])
    hits = [t for t in terms if _norm(t) in text]
    cov = len(hits) / len(terms) if terms else 0.0
    gist = {
        "power_down_cm": any(k in text for k in ["power down", "powerdown", "power-down",
                                                 "conserve", "preserve reentry"]),
        "lm_lifeboat": ("lunar module" in text or "lm " in text or "lifeboat" in text) and
                       any(k in text for k in ["lifeboat", "power up", "power-up",
                                               "descent batter", "aquarius"]),
    }
    return {"component_coverage": cov, "components_hit": hits,
            "n_terms": len(terms), "action_gist": gist}
