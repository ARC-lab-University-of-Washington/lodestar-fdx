# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file cli.py
# @brief Lodestar edge CLI — offline fault diagnosis on the Jetson.
#
# python -m lodestar_edge.cli --symptom "master alarm, MAIN B BUS UNDERVOLT,
# O2 QUANTITY 2 zero" python -m lodestar_edge.cli --multiturn # then type one
# crew utterance per line; blank line = quit Grounding-or-silence: it answers
# with cited candidate causes drawn from the onboard corpus, or it declines.
# It is strictly advisory — the crew acts.
#
"""Lodestar edge CLI — offline fault diagnosis on the Jetson.

  python -m lodestar_edge.cli --symptom "master alarm, MAIN B BUS UNDERVOLT, O2 QUANTITY 2 zero"
  python -m lodestar_edge.cli --multiturn   # then type one crew utterance per line; blank line = quit

Grounding-or-silence: it answers with cited candidate causes drawn from the onboard corpus, or
it declines. It is strictly advisory — the crew acts.
"""
import os, sys, time, argparse
from .segment import load_corpus
from .retriever import SalienceBM25Retriever
from .graph import CorpusGraph
from .diagnose import diagnose_chain

# The output carries a few non-ASCII glyphs (⚠, →). Keep them where the console is UTF-8; never
# crash where it isn't (Windows cp1252 dev console, a mis-locale'd Jetson tty).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS = os.path.join(HERE, "corpus.json")


def build(corpus_path):
    t0 = time.perf_counter()
    segs = load_corpus(corpus_path)
    r = SalienceBM25Retriever(segs)
    g = CorpusGraph(r.segments)
    print(f"[lodestar-edge] corpus {len(segs)} segments ({r.n_dropped} boilerplate dropped); "
          f"graph {g.n_edges} edges ({g.resolution_rate:.0%} of links_to resolved) in "
          f"{time.perf_counter()-t0:.1f}s", flush=True)
    return r, g


def show(res, elapsed):
    if res["declined"]:
        print(f"\nLODESTAR: {res['diagnosis']}")
        print(f"  [{elapsed:.1f}s]")
        return
    cov = res.get("coverage")
    print("\nLODESTAR — reasoning chains from the onboard manuals (crew verifies):")
    if cov is not None:
        band = "in coverage" if not res.get("coverage_flag") else "COVERAGE SUSPECT"
        print(f"  manual coverage {cov:.2f} ({band}) · retrieval strength {res.get('retrieval_strength','?')}")
    if res.get("coverage_flag"):
        print("  ⚠ this symptom uses little documented-annunciator vocabulary — it may lie outside "
              "the onboard manuals; treat leads with extra caution.")
    for c in res.get("chains", []):
        print(f"  chain (entry: {c['entry_doc']}):")
        for s in c["steps"]:
            arrow = "" if s["tag"].endswith(".1]") else "-> "
            print(f"      {arrow}{s['tag']} {s['label']}")
    print("  decision:")
    for line in res["diagnosis"].splitlines():
        print("   ", line)
    print(f"  [{elapsed:.1f}s · {res.get('n_chains',0)} chains, lengths {res.get('chain_lengths',[])}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symptom", default=None)
    ap.add_argument("--multiturn", action="store_true")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--model", default=os.environ.get("LODESTAR_EDGE_MODEL", "llama3.2:3b"))
    ap.add_argument("--no-gate", action="store_true",
                    help="disable the deterministic out-of-domain coverage gate")
    ap.add_argument("--base-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--entries", type=int, default=8, help="entry manuals -> one chain each")
    ap.add_argument("--depth", type=int, default=3, help="max graph-chase hops per chain")
    ap.add_argument("--threshold", type=float, default=0.4,
                    help="chase stops when the best next-hop BM25 < threshold x entry score")
    args = ap.parse_args()

    def run(sym):
        return diagnose_chain(sym, r, g, model=args.model, base_url=args.base_url,
                              k_entries=args.entries, rel_threshold=args.threshold,
                              max_depth=args.depth, gate=not args.no_gate)

    r, g = build(args.corpus)

    if args.multiturn:
        acc = ""
        print("Enter crew utterances (one per line; blank line to exit):")
        while True:
            try:
                line = input("crew> ").strip()
            except EOFError:
                break
            if not line:
                break
            acc = (acc + " " + line).strip()
            t0 = time.perf_counter()
            res = run(acc)
            show(res, time.perf_counter() - t0)
        return

    sym = args.symptom or (sys.stdin.read().strip() if not sys.stdin.isatty() else None)
    if not sym:
        ap.error("provide --symptom, --multiturn, or pipe a symptom on stdin")
    t0 = time.perf_counter()
    res = run(sym)
    show(res, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
