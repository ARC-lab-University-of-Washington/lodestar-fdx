# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file run_stackup.py
# @brief ON-DEVICE breadth stack-up: run the DEPLOYED lodestar_edge.diagnose (attribute-and-flag) over all 255 documented Apollo anomalies, scoring reach against the.
#
# documented cause. Crash-safe (appends each result to
# results/stackup_result.jsonl immediately) and resumable (skips anomalies
# already done), because the full run is ~255 x k generation calls and takes a
# while on the Orin Nano. Breadth is scored separately by
# distinctive_scorer.py over the stored diagnoses (no model calls), so the
# scorer can change without re-running the model. python run_stackup.py # full
# 255, resume if interrupted python run_stackup.py --limit 5 # smoke test
#
"""ON-DEVICE breadth stack-up: run the DEPLOYED lodestar_edge.diagnose (attribute-and-flag) over all 255
documented Apollo anomalies, scoring reach against the documented cause. Crash-safe (appends each
result to results/stackup_result.jsonl immediately) and resumable (skips anomalies already done),
because the full run is ~255 x k generation calls and takes a while on the Orin Nano.

Breadth is scored separately by distinctive_scorer.py over the stored diagnoses (no model calls),
so the scorer can change without re-running the model.

  python run_stackup.py                 # full 255, resume if interrupted
  python run_stackup.py --limit 5       # smoke test
"""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.environ.get("LODESTAR_EDGE_DIR", os.path.dirname(HERE))   # parent dir holds lodestar_edge/
sys.path.insert(0, EDGE_ROOT)
from lodestar_edge.segment import load_corpus          # noqa: E402
from lodestar_edge.retriever import SalienceBM25Retriever  # noqa: E402
from lodestar_edge.diagnose import diagnose            # noqa: E402

DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
CORPUS = os.environ.get("LODESTAR_CORPUS", os.path.join(EDGE_ROOT, "lodestar_edge", "corpus.json"))

# subsystems the Apollo-13 CSM/LM corpus can plausibly cover (shared vehicle hardware) vs not
_OUT_SUBSYS = ("lrv", "rover", "alsep", "eva", "emu", "suit", "sim-bay", "sim bay", "experiment",
               "seismic", "magnetometer", "spectrometer", "drill", "sounder", "camera", "sample",
               "mesa", "surface", "geolog")


def in_corpus_domain(subsystem, symptom):
    s = f"{subsystem} {symptom}".lower()
    return not any(w in s for w in _OUT_SUBSYS)


def _load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line); done.add((r["mission"], r["symptom"]))
                except json.JSONDecodeError:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("LODESTAR_EDGE_MODEL", "llama3.2:3b"))
    ap.add_argument("--base-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--atlas", default=os.path.join(DATA, "atlas_parsed.json"))
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    ckpt = os.path.join(RESULTS, "stackup_result.jsonl")

    atlas = json.load(open(args.atlas, encoding="utf-8"))
    if isinstance(atlas, dict):
        atlas = atlas.get("missions") or atlas.get("results") or [atlas]
    flat = [(m["mission"], a) for m in atlas for a in m.get("anomalies", [])]
    if args.limit:
        flat = flat[:args.limit]

    retr = SalienceBM25Retriever(load_corpus(CORPUS))
    print(f"[stackup] corpus ready ({retr.n_dropped} boilerplate dropped); model={args.model} "
          f"k={args.k}; {len(flat)} anomalies", flush=True)
    done = _load_done(ckpt)
    fh = open(ckpt, "a", encoding="utf-8")
    n_reached_in = n_in = 0
    for i, (mission, a) in enumerate(flat, 1):
        sym = a["symptom"]
        if (mission, sym) in done:
            continue
        t0 = time.perf_counter()
        res = diagnose(sym, retr, model=args.model, base_url=args.base_url, k=args.k, gate=False)
        dt = time.perf_counter() - t0
        ht = a.get("human_turns_to_diagnosis", -1)
        rec = {
            "mission": mission, "symptom": sym, "subsystem": a.get("subsystem"),
            "in_corpus_domain": in_corpus_domain(a.get("subsystem", ""), sym),
            "crew_initiated": a.get("crew_initiated"),
            "human_turns": (ht if (ht is not None and ht >= 0) else None),
            "human_time_s": (a.get("human_time_s") if (a.get("human_time_s", -1) or -1) >= 0 else None),
            "lodestar_time_s": round(dt, 1),
            "documented_cause": a.get("documented_cause", ""),
            "lodestar_diagnosis": res["diagnosis"] if not res["declined"] else "DECLINE",
            "lodestar_declined": res["declined"],
            "n_points": res.get("n_points", 0),
            "n_grounded": res.get("n_grounded", 0),
            "n_inferred": res.get("n_inferred", 0),
            "confidence": res.get("confidence"),
            "confidence_level": res.get("confidence_level"),
            "retrieval_strength": res.get("retrieval_strength"),
        }
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        if rec["in_corpus_domain"]:
            n_in += 1
        print(f"[{i}/{len(flat)}] {mission:10} {'DECL' if res['declined'] else 'diag'} "
              f"pts={rec['n_points']} conf={rec['confidence']} g/i={rec['n_grounded']}/{rec['n_inferred']} "
              f"{'IN ' if rec['in_corpus_domain'] else 'OUT'} {dt:4.1f}s  {sym[:36]}", flush=True)
    fh.close()
    print(f"\n[stackup] done -> {ckpt}\n  Now score breadth:  python distinctive_scorer.py")


if __name__ == "__main__":
    main()
