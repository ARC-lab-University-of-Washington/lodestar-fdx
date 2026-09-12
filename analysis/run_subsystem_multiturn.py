# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file run_subsystem_multiturn.py
# @brief UNIFIED multi-turn-until-SUBSYSTEM-MATCH runner (the new success criterion).
#
# For each anomaly, reveal the symptom incrementally (one clause/sentence per
# turn, cumulative — the Open WebUI accumulation), run diagnose_chain each
# turn, and TERMINATE as soon as the decision names the anomaly's SUBSYSTEM
# (alias-aware, subsystem_match.py — the single source of truth). Records
# turns-to-match + cumulative time. Two datasets (env DATASET): * ci146 — the
# 146 crew_initiated faults (full_log.jsonl): ACCURACY = fraction reaching
# subsystem match. * loop23 — the 23 live air-to-ground loops
# (atlas_parsed.json): TURN COUNTING vs the human loop. Dumps per-case JSONL
# so any denominator can be scored offline (no re-generation). Needs Ollama.
# Env: DATASET, LODESTAR_EDGE_DIR, OLLAMA_URL, BENCH_LABEL, CALL_GAP, EVAL_K,
# OUT, MAX_TURNS.
#
"""UNIFIED multi-turn-until-SUBSYSTEM-MATCH runner (the new success criterion). For each anomaly, reveal
the symptom incrementally (one clause/sentence per turn, cumulative — the Open WebUI accumulation), run
diagnose_chain each turn, and TERMINATE as soon as the decision names the anomaly's SUBSYSTEM
(alias-aware, subsystem_match.py — the single source of truth). Records turns-to-match + cumulative time.

Two datasets (env DATASET):
  * ci146  — the 146 crew_initiated faults (full_log.jsonl): ACCURACY = fraction reaching subsystem match.
  * loop23 — the 23 live air-to-ground loops (atlas_parsed.json): TURN COUNTING vs the human loop.

Dumps per-case JSONL so any denominator can be scored offline (no re-generation). Needs Ollama.
Env: DATASET, LODESTAR_EDGE_DIR, OLLAMA_URL, BENCH_LABEL, CALL_GAP, EVAL_K, OUT, MAX_TURNS."""
import sys
import json, os, sys, re, time, importlib, statistics as st
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subsystem_match import subsystem_match, is_in_corpus, fair, aliases_for  # noqa: E402

EDGE = os.environ.get("LODESTAR_EDGE_DIR", ".."); sys.path.insert(0, EDGE)
from lodestar_edge.segment import load_corpus            # noqa: E402
from lodestar_edge.retriever import SalienceBM25Retriever  # noqa: E402
from lodestar_edge.graph import CorpusGraph              # noqa: E402
from lodestar_edge.diagnose import diagnose_chain, preload, unload  # noqa: E402
import glob  # noqa: E402

from _paths import _d, _DATA_ROOT   # noqa: E402


FLUSH_EVERY = int(os.environ.get("FLUSH_EVERY", "0"))   # Jetson: unload+reload every N cases to cap the
INSTRUMENT = bool(os.environ.get("INSTRUMENT"))          # known Ollama KV accumulation (#10114/#16698)


def _memstat():
    """Linux-only: available MB, swap-used MB, and summed RSS of ollama processes (the KV holder)."""
    try:
        mi = {}
        for line in open("/proc/meminfo"):
            p = line.split()
            mi[p[0].rstrip(":")] = int(p[1])
        avail = mi.get("MemAvailable", 0) // 1024
        swap = (mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)) // 1024
        orss = 0
        for cl in glob.glob("/proc/[0-9]*/cmdline"):
            try:
                if b"ollama" in open(cl, "rb").read():
                    for l in open(cl.replace("cmdline", "status")):
                        if l.startswith("VmRSS"):
                            orss += int(l.split()[1]) // 1024
            except Exception:
                pass
        return f"avail={avail}MB swap={swap}MB ollamaRSS={orss}MB"
    except Exception as e:
        return f"(memstat n/a: {e})"

DATASET = os.environ.get("DATASET", "ci146")
BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("LODESTAR_EDGE_MODEL", "llama3.2:3b")   # derived model (num_ctx baked) on Jetson
K = int(os.environ.get("EVAL_K", "8")); GAP = float(os.environ.get("CALL_GAP", "0"))
LABEL = os.environ.get("BENCH_LABEL", "dev-box")
MAX_TURNS = int(os.environ.get("MAX_TURNS", "99"))
FULL_LOG = os.environ.get("FULL_LOG", _d("full_log.jsonl"))
ATLAS = os.environ.get("LODESTAR_ATLAS", "../../apollo-anomaly-atlas/data/atlas_parsed.json")
OUT = os.environ.get("OUT", _d(f"mt_subsystem_{DATASET}.jsonl"))


def load_items():
    if DATASET == "loop23":
        atlas = json.load(open(ATLAS, encoding="utf-8"))
        flat = [an for m in atlas["missions"] for an in m.get("anomalies", [])]
        loop = [x for x in flat if x.get("crew_initiated") and x.get("diagnosed_in_flight")]
        return [{"symptom": x["symptom"], "subsystem": x.get("subsystem"),
                 "documented_cause": x.get("documented_cause", ""), "mission": x.get("mission", ""),
                 "human_turns": x.get("human_turns_to_diagnosis")} for x in loop]
    recs = [json.loads(l) for l in open(FULL_LOG, encoding="utf-8") if l.strip()]
    ci = [r for r in recs if r.get("crew_initiated")]
    return [{"symptom": r["symptom"], "subsystem": r.get("subsystem"),
             "documented_cause": r.get("documented_cause", ""), "mission": r.get("mission", ""),
             "in_corpus_domain": r.get("in_corpus_domain"), "human_turns": r.get("human_turns")}
            for r in ci]


def turns_of(symptom):
    parts = [p.strip() for p in re.split(r"(?<=[.;])\s+", symptom) if len(p.strip()) >= 12]
    return (parts or [symptom])[:MAX_TURNS]


items = load_items()
LIMIT = int(os.environ.get("LIMIT", "0"))
if LIMIT:
    items = items[:LIMIT]
segs = load_corpus(os.path.join(EDGE, "lodestar_edge", "corpus.json"))
retr = SalienceBM25Retriever(segs); g = CorpusGraph(retr.segments)
print(f"multi-turn->subsystem-match [{LABEL}] DATASET={DATASET}: {len(items)} anomalies, "
      f"k={K}, terminate on subsystem match -> {OUT}", flush=True)


def diag(q):
    r = diagnose_chain(q, retr, g, model=MODEL, base_url=BASE, k_entries=K, pool=K + 8)
    return r


# RESUMABLE: skip anything already dumped (survives a mid-run reset — write to persistent storage,
# not /tmp). Load prior rows so the final report covers the whole set.
done = set(); rows = []
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        try:
            r = json.loads(l); done.add(r["symptom"]); rows.append(r)
        except Exception:
            pass
remaining = [it for it in items if it["symptom"] not in done]
preload(BASE, MODEL)
if remaining:
    diag(remaining[0]["symptom"])
print(f"model preloaded + warmed ({len(done)} already done, {len(remaining)} remaining)", flush=True)
fh = open(OUT, "a", encoding="utf-8")
processed = 0
for i, it in enumerate(items, 1):
    if it["symptom"] in done:
        continue
    turns = turns_of(it["symptom"])
    matched_at, cum, final = None, 0.0, ""
    for k in range(1, len(turns) + 1):
        q = " ".join(turns[:k])
        t0 = time.perf_counter(); r = diag(q); cum += time.perf_counter() - t0
        if GAP:
            time.sleep(GAP)
        final = r.get("diagnosis", "") if not r["declined"] else ""
        if subsystem_match(final, it["subsystem"], r["declined"]):
            matched_at = k
            break
    rec = {"mission": it["mission"], "symptom": it["symptom"], "subsystem": it["subsystem"],
           "documented_cause": it["documented_cause"], "in_corpus_domain": it.get("in_corpus_domain"),
           "n_turns_available": len(turns), "matched_at": matched_at, "matched": matched_at is not None,
           "cum_s": round(cum, 2), "final_decision": final, "human_turns": it.get("human_turns")}
    rows.append(rec); fh.write(json.dumps(rec) + "\n"); fh.flush()
    processed += 1
    memnote = f" | {_memstat()}" if INSTRUMENT else ""
    print(f"  [{i}/{len(items)}] {it['subsystem']:<14} turns={len(turns)} "
          f"matched@{matched_at or 'never'} cum={cum:.0f}s | {it['symptom'][:38]}{memnote}", flush=True)
    # periodic KV flush: unload + reload to release accumulated Ollama KV (8 GB Jetson stability)
    if FLUSH_EVERY and processed % FLUSH_EVERY == 0 and it is not items[-1]:
        unload(BASE, MODEL); time.sleep(2); preload(BASE, MODEL)
        if INSTRUMENT:
            print(f"    [flush @ case {processed}] {_memstat()}", flush=True)
fh.close()


def report(subset, name):
    if not subset:
        return
    m = [r for r in subset if r["matched"]]
    acc = len(m) / len(subset)
    line = f"  {name:<26} {len(m):>3}/{len(subset):<3} = {acc:.0%}"
    if m:
        ta = [r["matched_at"] for r in m]
        line += f"   turns-to-match: mean {st.mean(ta):.2f} median {st.median(ta)} dist {dict(sorted(Counter(ta).items()))}"
    print(line)


print(f"\n===== MULTI-TURN -> SUBSYSTEM MATCH [{LABEL}] DATASET={DATASET} =====")
report(rows, "ALL")
if DATASET == "ci146":
    report([r for r in rows if is_in_corpus(r["subsystem"])], "in-scope subsystem")
    report([r for r in rows if r.get("in_corpus_domain")], "in_corpus_domain")
    report([r for r in rows if fair(r["subsystem"], r["documented_cause"])], "fair (in-scope+knowable)")
matched = [r for r in rows if r["matched"]]
if matched:
    print(f"\nturns-to-match overall: mean {st.mean([r['matched_at'] for r in matched]):.2f}  "
          f"median {st.median([r['matched_at'] for r in matched])}")
    cums = [r["cum_s"] for r in matched]
    print(f"cumulative time-to-match: mean {st.mean(cums):.1f}s  median {st.median(cums):.1f}s")
n_multi = sum(1 for r in rows if r["n_turns_available"] > 1)
print(f"anomalies with >1 available turn: {n_multi}/{len(rows)}")
if DATASET == "loop23":
    hb = [r["human_turns"] for r in rows if r.get("human_turns")]
    if hb:
        print(f"HUMAN air-to-ground turns (recorded {len(hb)}): mean {st.mean(hb):.1f}")
print(f"target: subsystem-match accuracy >= 85%")
