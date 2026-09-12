# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file build_lodestar_vs_capcom.py
# @brief Build the LODESTAR (=LODESTAR) vs CAPCOM (Mission Control / ground) comparison DATA (no plots).
#
# Per fault, all cases (including LODESTAR misses — NOT best-case). Emits a
# CSV + a summary .md giving the paired data for: (a) turns-to-diagnosis
# LODESTAR vs CAPCOM, (b) time-to-diagnosis LODESTAR vs CAPCOM, (c) accuracy
# LODESTAR vs CAPCOM. Deterministic; reads stored run dumps + the atlas (no
# model). Pairing = the 23 crew-observed AND diagnosed-in-flight anomalies
# (both LODESTAR and CAPCOM attempted them). CAPCOM turns/time are hand-
# curated from the air-to-ground exchange_quotes (sparse). Cross-mission
# CAPCOM turn/time entries are appended as CAPCOM-only context (no subsystem
# label -> not LODESTAR-scoreable).
#
"""Build the LODESTAR (=LODESTAR) vs CAPCOM (Mission Control / ground) comparison DATA (no plots).
Per fault, all cases (including LODESTAR misses — NOT best-case). Emits a CSV + a summary .md giving
the paired data for: (a) turns-to-diagnosis LODESTAR vs CAPCOM, (b) time-to-diagnosis LODESTAR vs
CAPCOM, (c) accuracy LODESTAR vs CAPCOM. Deterministic; reads stored run dumps + the atlas (no model).

Pairing = the 23 crew-observed AND diagnosed-in-flight anomalies (both LODESTAR and CAPCOM attempted
them). CAPCOM turns/time are hand-curated from the air-to-ground exchange_quotes (sparse). Cross-mission
CAPCOM turn/time entries are appended as CAPCOM-only context (no subsystem label -> not LODESTAR-scoreable)."""
import sys
import json, os, csv, statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import _d, _DATA_ROOT   # noqa: E402


TE = os.environ.get("LODESTAR_ATLAS_DIR", "../../apollo-anomaly-atlas/data")
DATA = _DATA_ROOT
atlas = json.load(open(f"{TE}/atlas_parsed.json", encoding="utf-8"))
flat = [a for m in atlas["missions"] for a in m.get("anomalies", [])]
dif = [a for a in flat if a.get("crew_initiated") and a.get("diagnosed_in_flight")]  # the 23

def load(fn):
    """@brief Load a run dump, keyed by symptom.

    @param fn Filename inside DATA.
    @return dict mapping symptom -> row.
    @exception FileNotFoundError if the dump is absent.

    This RAISES on a missing file rather than returning {}. Returning an empty dict
    silently scored every LODESTAR row as incorrect and still printed a summary
    ("LODESTAR 11/37 = 30%") as though it had been measured. Raised in the
    """
    p = os.path.join(DATA, fn)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"required run dump not found: {p}\n"
            "These ship in the benchmark repository under baselines/. Set DATA to point "
            "at it, e.g. DATA=../../apollo-anomaly-atlas/baselines")
    return {r["symptom"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}
dev = load("mt_subsystem_loop23.jsonl")   # LODESTAR dev-box (turns=matched_at, time=cum_s, matched)
jet = load("loop23_jetson.jsonl") if os.path.exists(os.path.join(DATA, "loop23_jetson.jsonl")) else {}   # LODESTAR jetson (time=cum_s)

def num(x): return x if isinstance(x, (int, float)) and x >= 0 else None

rows = []
for a in dif:
    sym = a["symptom"]; d = dev.get(sym, {}); j = jet.get(sym, {})
    rows.append({
        "set": "paired_diagnosed_in_flight", "mission": a.get("mission", ""),
        "subsystem": a.get("subsystem", ""), "symptom": sym[:90],
        # LODESTAR
        "lodestar_correct": int(bool(d.get("matched"))),
        "lodestar_turns": d.get("matched_at") if d.get("matched") else "",   # blank = miss (not best-case)
        "lodestar_time_s_devbox": round(d["cum_s"], 1) if d.get("cum_s") is not None else "",
        "lodestar_time_s_jetson": round(j["cum_s"], 1) if j.get("cum_s") is not None else "",
        # CAPCOM (ground): correct by selection (these were diagnosed in flight)
        "capcom_correct": 1,
        "capcom_turns": num(a.get("human_turns_to_diagnosis")) if num(a.get("human_turns_to_diagnosis")) is not None else "",
        "capcom_time_s": num(a.get("human_time_s")) if num(a.get("human_time_s")) is not None else "",
    })

# cross-mission — now PAIRED (LODESTAR run in run_capcom_crossmission.py; subsystem [assigned] from symptom)
cjet = {}   # jetson cum_s per cross-mission symptom (from the on-device run), if present
cjp = os.path.join(DATA, "crossmission_lodestar_jetson.jsonl")
if os.path.exists(cjp):
    for r in (json.loads(l) for l in open(cjp, encoding="utf-8") if l.strip()):
        if r.get("cum_s") is not None: cjet[r["symptom"]] = r["cum_s"]
clp = os.path.join(DATA, "crossmission_lodestar.jsonl")
if os.path.exists(clp):
    for r in (json.loads(l) for l in open(clp, encoding="utf-8") if l.strip()):
        rows.append({"set": "paired_crossmission", "mission": "",
            "subsystem": (r["subsystem_assigned"] or "") + "(assigned)", "symptom": r["symptom"][:90],
            "lodestar_correct": int(bool(r["matched"])),
            "lodestar_turns": r["matched_at"] if r["matched"] else "",
            "lodestar_time_s_devbox": round(r["cum_s"], 1) if r.get("cum_s") is not None else "",
            "lodestar_time_s_jetson": round(cjet[r["symptom"]], 1) if r["symptom"] in cjet else "",
            "capcom_correct": 1,
            "capcom_turns": num(r.get("capcom_turns")) if num(r.get("capcom_turns")) is not None else "",
            "capcom_time_s": num(r.get("capcom_time_s")) if num(r.get("capcom_time_s")) is not None else ""})

cols = ["set", "mission", "subsystem", "symptom", "lodestar_correct", "lodestar_turns",
        "lodestar_time_s_devbox", "lodestar_time_s_jetson", "capcom_correct", "capcom_turns", "capcom_time_s"]
with open(os.path.join(DATA, "lodestar_vs_capcom.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

# ---- summary (a)/(b)/(c) ----
paired = [r for r in rows if r["set"].startswith("paired")]
turn_pairs = [(r["lodestar_turns"], r["capcom_turns"]) for r in paired if r["lodestar_turns"] != "" and r["capcom_turns"] != ""]
time_pairs_dev = [(r["lodestar_time_s_devbox"], r["capcom_time_s"]) for r in paired if r["lodestar_time_s_devbox"] != "" and r["capcom_time_s"] != ""]
time_pairs_jet = [(r["lodestar_time_s_jetson"], r["capcom_time_s"]) for r in paired if r["lodestar_time_s_jetson"] != "" and r["capcom_time_s"] != ""]
lod_correct = sum(r["lodestar_correct"] for r in paired if r["lodestar_correct"] != "")
n_paired = len(paired)

def stat(vs): return f"n={len(vs)} mean={st.mean(vs):.1f} median={st.median(vs)} range={min(vs)}-{max(vs)}" if vs else "n=0"
with open(os.path.join(DATA, "lodestar_vs_capcom_summary.md"), "w", encoding="utf-8") as f:
    f.write("# LODESTAR vs CAPCOM — comparison data summary\n\n")
    f.write("Data: `lodestar_vs_capcom.csv` (one row per fault; blank LODESTAR-turns = LODESTAR MISS — all cases kept, not best-case).\n")
    n_dif = sum(1 for r in paired if r["set"] == "paired_diagnosed_in_flight")
    f.write(f"Paired set = {n_paired} faults (both LODESTAR + CAPCOM attempted): {n_dif} atlas diagnosed-in-flight "
            f"+ {n_paired - n_dif} cross-mission (subsystem [assigned] from symptom; CAPCOM turn/time hand-curated).\n\n")
    f.write("## (a) turns-to-diagnosis — LODESTAR vs CAPCOM (scatter pairs)\n")
    f.write(f"- pairs (both have a turn count): **{len(turn_pairs)}**\n")
    for lt, ct in turn_pairs: f.write(f"  - LODESTAR {lt} vs CAPCOM {ct}\n")
    if turn_pairs:
        f.write(f"- LODESTAR turns {stat([p[0] for p in turn_pairs])}\n- CAPCOM turns {stat([p[1] for p in turn_pairs])}\n")
    f.write("\n## (b) time-to-diagnosis (s) — LODESTAR vs CAPCOM (scatter pairs)\n")
    f.write(f"- dev-box pairs: **{len(time_pairs_dev)}** | jetson pairs: **{len(time_pairs_jet)}**\n")
    for lt, ct in time_pairs_dev: f.write(f"  - LODESTAR(dev) {lt}s vs CAPCOM {ct}s\n")
    if time_pairs_dev:
        f.write(f"- LODESTAR(dev) time {stat([p[0] for p in time_pairs_dev])}\n")
        f.write(f"- LODESTAR(jetson) time {stat([p[0] for p in time_pairs_jet])}\n")
        f.write(f"- CAPCOM time {stat([p[1] for p in time_pairs_dev])}\n")
    f.write("\n## (c) accuracy — LODESTAR vs CAPCOM\n")
    f.write(f"- On the {n_paired} paired faults: **LODESTAR {lod_correct}/{n_paired} = {lod_correct/n_paired:.0%}** (subsystem match) vs **CAPCOM {n_paired}/{n_paired} = 100%**. LODESTAR misses ({n_paired-lod_correct}) are kept in the CSV (lodestar_correct=0).\n")
    f.write("- ⚠ CAPCOM = 100% is definitional (this set is *selected* as faults CAPCOM diagnosed, with ground resources + time); read as speed-vs-accuracy.\n")

print(f"wrote lodestar_vs_capcom.csv ({len(rows)} rows) + _summary.md")
print(f"(a) turn pairs {len(turn_pairs)} | (b) time pairs dev {len(time_pairs_dev)} jet {len(time_pairs_jet)} | (c) LODESTAR {lod_correct}/{n_paired} vs CAPCOM {n_paired}/{n_paired}")
