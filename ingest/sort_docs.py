# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file sort_docs.py
# @brief Sort source PDFs into OPERATIONAL (in-flight crew documents a spacecraft would actually carry — checklists, procedures, flight rules, subsystem handbo.
#
# oks, data cards) vs HISTORICAL (post-hoc / narrative / admin — anomaly
# reports, corrective-action panels, mission reports, transcripts, press).
# Lodestar models an ONBOARD assistant, so its corpus should be operational
# docs, not retrospective accident analyses that wouldn't exist during the
# mission. Usage: python sort_docs.py <src_dir> <dst_dir> (defaults:
# pdfs/curated -> pdfs/operational) Prints the classification and copies
# OPERATIONAL docs to <dst_dir>.
#
"""Sort source PDFs into OPERATIONAL (in-flight crew documents a spacecraft
would actually carry — checklists, procedures, flight rules, subsystem
handbooks, data cards) vs HISTORICAL (post-hoc / narrative / admin — anomaly
reports, corrective-action panels, mission reports, transcripts, press).

Lodestar models an ONBOARD assistant, so its corpus should be operational docs,
not retrospective accident analyses that wouldn't exist during the mission.

Usage:  python sort_docs.py <src_dir> <dst_dir>   (defaults: pdfs/curated -> pdfs/operational)
Prints the classification and copies OPERATIONAL docs to <dst_dir>.
"""
import os
import re
import sys
import shutil

# Strong "historical / not-carried-onboard" markers (checked first, they win).
HISTORICAL = [
    "anomaly", "corrective", "panel", "review", "investigation", "-report", "report-",
    "transcript", "press", "memo", "award", "telegram", "logbook", "console",
    "accident", "naming", "dispersion", "analysis", "fer", "-traj", "config",
    "msfn", "mcc", "stowage", "summary", "article", "presentation", "michoud",
    "construction", "westinghouse", "tv-camera", "fdlog", "-loop", "readiness",
    "postlaunch", "mission-ops-report", "5-day", "30-day", "backup-to-csm", "experiments",
]
# In-flight operational markers.
OPERATIONAL = [
    "checklist", "procedure", "-proc", "procs", "rules", "aoh", "subs",
    "data-card", "datacard", "dictionary", "flightplan", "flight-plan", "fplan",
    "timeline", "flight-manual", "activation", "-nav-",
]


def classify(name: str) -> tuple[str, str]:
    s = name.lower()
    for h in HISTORICAL:
        if h in s:
            return "HISTORICAL", h
    for o in OPERATIONAL:
        if o in s:
            return "OPERATIONAL", o
    return "HISTORICAL", "(no operational marker — default drop)"


def main():
    repo = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo, "pdfs", "curated")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(repo, "pdfs", "operational")
    os.makedirs(dst, exist_ok=True)

    pdfs = []
    for dp, _, fs in os.walk(src):
        for f in fs:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(dp, f))
    pdfs.sort(key=lambda p: os.path.basename(p))

    kept = dropped = 0
    print(f"{'CLASS':<12} {'why':<28} file")
    for p in pdfs:
        name = os.path.basename(p)
        cls, why = classify(name)
        print(f"{cls:<12} {why:<28} {name}")
        if cls == "OPERATIONAL":
            shutil.copy2(p, os.path.join(dst, name))
            kept += 1
        else:
            dropped += 1
    print(f"\n-> {kept} OPERATIONAL copied to {dst}  |  {dropped} HISTORICAL dropped")


if __name__ == "__main__":
    main()
