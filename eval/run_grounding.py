# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file run_grounding.py
# @brief ON-DEVICE grounding-or-silence audit: 64 cross-mission scenarios (49 in-scope shared-hardware -> should diagnose; 15 out-of-scope mission-unique -> sh.
#
# ould DECLINE), run through the DEPLOYED lodestar_edge.diagnose (cite-or-
# drop). Uncited leads are attributed-and-flagged, not dropped, so
# impossible: a point the model did not cite is DROPPED (counted as
# n_inferred counts them. So this run
# measures: in-scope reach, in-scope false-decline, out-of-scope decline, and
# how many points were dropped for being uncited (the fabrications the
# deployed system avoids).
#
"""ON-DEVICE grounding-or-silence audit: 64 cross-mission scenarios (49 in-scope shared-hardware ->
should diagnose; 15 out-of-scope mission-unique -> should DECLINE), run through the DEPLOYED
lodestar_edge.diagnose (attribute-and-flag).

Leads the model did not cite are kept and attributed to the source they were generated from, then
flagged grounded=False and rendered "(inferred from source - verify)". Nothing is dropped. So this
run measures: in-scope reach, in-scope false-decline, out-of-scope decline, and how many points were
dropped for being uncited (the fabrications the deployed system avoids).
"""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.environ.get("LODESTAR_EDGE_DIR", os.path.dirname(HERE))
sys.path.insert(0, EDGE_ROOT)
from lodestar_edge.segment import load_corpus              # noqa: E402
from lodestar_edge.retriever import SalienceBM25Retriever  # noqa: E402
from lodestar_edge.diagnose import diagnose                # noqa: E402
from metrics import diagnosis_coverage                    # noqa: E402

DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
CORPUS = os.environ.get("LODESTAR_CORPUS", os.path.join(EDGE_ROOT, "lodestar_edge", "corpus.json"))


def build_anchor_map():
    sc = json.load(open(os.path.join(DATA, "apollo_cross_mission_scenarios.json"),
                        encoding="utf-8"))["scenarios"]
    an = json.load(open(os.path.join(DATA, "resolution_anchors.json"), encoding="utf-8"))["missions"]
    by_mission = {}
    for m in an:
        by_mission.setdefault(m["mission"], []).extend(m["anchors"])
    cursor, mapped = {}, []
    for s in sc:
        if s.get("scope") == "in":
            mis = s["mission"]; idx = cursor.get(mis, 0)
            anchor = by_mission[mis][idx] if idx < len(by_mission.get(mis, [])) else None
            cursor[mis] = idx + 1
            mapped.append((s, anchor))
        else:
            mapped.append((s, None))
    return mapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("LODESTAR_EDGE_MODEL", "llama3.2:3b"))
    ap.add_argument("--base-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--cov-bar", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help="cap scenarios (smoke test)")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    retr = SalienceBM25Retriever(load_corpus(CORPUS))
    mapped = build_anchor_map()
    if args.limit:
        mapped = mapped[:args.limit]
    print(f"[grounding] {len(mapped)} scenarios; model={args.model} k={args.k}", flush=True)
    results = []
    for n, (s, anchor) in enumerate(mapped, 1):
        sym = s["query"]
        t0 = time.perf_counter()
        res = diagnose(sym, retr, model=args.model, base_url=args.base_url, k=args.k, gate=False)
        dt = time.perf_counter() - t0
        cov = (diagnosis_coverage(res["diagnosis"], anchor)["component_coverage"]
               if (anchor and not res["declined"]) else (0.0 if anchor else None))
        row = {"n": n, "mission": s["mission"], "scope": s["scope"], "subsystem": s.get("subsystem"),
               "query": sym, "declined": res["declined"], "n_points": res.get("n_points", 0),
               "n_grounded": res.get("n_grounded", 0), "n_inferred": res.get("n_inferred", 0),
               "confidence": res.get("confidence"), "confidence_level": res.get("confidence_level"),
               "retrieval_strength": res.get("retrieval_strength"), "coverage": cov,
               "time_s": round(dt, 1), "diagnosis": res["diagnosis"]}
        results.append(row)
        flag = "DECLINE" if res["declined"] else f"pts={row['n_points']} cov={cov} conf={row['confidence']}"
        print(f"[{n}/{len(mapped)}] {s['scope']:3} {s['mission']:10} {flag}  {sym[:42]}", flush=True)

    insc = [r for r in results if r["scope"] == "in"]
    out = [r for r in results if r["scope"] == "out"]
    rate = lambda rows, pred: round(sum(1 for r in rows if pred(r)) / len(rows), 3) if rows else 0.0
    mconf = lambda rows: round(sum((r["confidence"] or 0) for r in rows) / len(rows), 3) if rows else 0.0
    summ = {
        "model": args.model, "k": args.k, "cov_bar": args.cov_bar,
        "n_in": len(insc), "n_out": len(out),
        "in_scope_mean_coverage": round(sum(r["coverage"] for r in insc) / len(insc), 3) if insc else 0.0,
        "in_scope_reached_rate": rate(insc, lambda r: (r["coverage"] or 0) >= args.cov_bar),
        "in_scope_decline_rate": rate(insc, lambda r: r["declined"]),
        "out_scope_decline_rate": rate(out, lambda r: r["declined"]),
        # confidence is the lead-first safety signal (not declines): does it flag out-of-scope?
        "in_scope_mean_confidence": mconf(insc),
        "out_scope_mean_confidence": mconf(out),
        "in_scope_low_conf_rate": rate(insc, lambda r: r.get("confidence_level") == "low"),
        "out_scope_low_conf_rate": rate(out, lambda r: r.get("confidence_level") == "low"),
        "total_points": sum(r["n_points"] for r in results),
        "total_inferred": sum(r["n_inferred"] for r in results),
        # Leads the model did not cite are kept and attributed to the source they were
        # generated from, flagged grounded=False (see diagnose.py). Nothing is dropped,
        # so this counts inferred leads.
        "n_inferred_leads": sum(int(r.get("n_inferred") or 0) for r in results),
    }
    print("\n==== CITE-LEAD-FIRST GROUNDING + CONFIDENCE ====")
    for k_, v in summ.items():
        print(f"  {k_:28} {v}")
    json.dump({"summary": summ, "results": results},
              open(os.path.join(RESULTS, "grounding_result.json"), "w", encoding="utf-8"), indent=1)
    print(f"\n-> {os.path.join(RESULTS, 'grounding_result.json')}")


if __name__ == "__main__":
    main()
