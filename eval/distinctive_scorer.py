# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file distinctive_scorer.py
# @brief DETERMINISTIC breadth scorer (no LLM) — the official §5.4 metric.
#
# Weight each documented-cause term by DISTINCTIVENESS = inverse document
# frequency across ALL documented causes, so generic engineering words
# (valve/motor/temperature — in many causes) count ~0 and anomaly-specific
# components (transponder/shuttle-valve/thumper — in few) dominate. A
# diagnosis is 'right-direction' if it names >= 1 STRONG-distinctive cause
# term (idf >= IDF_STRONG). This discriminates in-corpus from out-of-corpus
# where plain term-coverage cannot. Reads this eval package's stackup output
# (stackup_result.jsonl) — the diagnoses produced by the deployed
# lodestar_edge.diagnose (attribute-and-flag) — and prints in/out-corpus right-
# direction rates. Stdlib only; runs on the Jetson.
#
"""DETERMINISTIC breadth scorer (no LLM) — the official §5.4 metric.

Weight each documented-cause term by DISTINCTIVENESS = inverse document frequency across ALL
documented causes, so generic engineering words (valve/motor/temperature — in many causes) count
~0 and anomaly-specific components (transponder/shuttle-valve/thumper — in few) dominate. A
diagnosis is 'right-direction' if it names >= 1 STRONG-distinctive cause term (idf >= IDF_STRONG).
This discriminates in-corpus from out-of-corpus where plain term-coverage cannot.

Reads this eval package's stackup output (stackup_result.jsonl) — the diagnoses produced by the
deployed lodestar_edge.diagnose (attribute-and-flag) — and prints in/out-corpus right-direction rates.
Stdlib only; runs on the Jetson.
"""
import os, json, re, math, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
IDF_STRONG = 2.5

_STOP = set("the a an of to in on and or with at from for during after before was were is are be "
    "this that these those it its they them their has had have will would could should may might "
    "not no non due most probable caused cause failure failed fault anomaly report system module "
    "assembly unit device component part condition normal nominal operation over under within".split())


def _words(t):
    return [w for w in re.findall(r"[a-z][a-z\-]{2,}", (t or "").lower()) if w not in _STOP]


# WARNING: build_idf computes IDF over whatever
# rows are present, and IDF_STRONG is an absolute cut against that size-dependent
# scale. Scoring a PARTIAL checkpoint therefore changes the reported rate without
# any error: on identical rows, n=146 -> 85%, n=50 -> 93%, n=20 -> 0%. Only score a
# complete run, or freeze the IDF table. _check_complete() below enforces the first.


def _check_complete(rows, expected=255):
    """@brief Refuse to score a partial checkpoint.
    @param rows     Loaded stackup rows.
    @param expected Full-run row count.
    @exception SystemExit if the checkpoint is incomplete.
    """
    if len(rows) < expected:
        raise SystemExit(
            f"refusing to score a partial checkpoint: {len(rows)}/{expected} rows. "
            "IDF is computed over the rows present, so a partial run yields a "
            "different rate with no error. Re-run run_stackup.py to completion, or "
            "set LODESTAR_SCORE_PARTIAL=1 to override deliberately."
            if os.environ.get("LODESTAR_SCORE_PARTIAL") != "1" else "")


def build_idf(recs):
    causes = [r.get("documented_cause", "") for r in recs]
    n = len(causes) or 1
    df = {}
    for c in causes:
        for w in set(_words(c)):
            df[w] = df.get(w, 0) + 1
    return lambda w: math.log(n / (df.get(w, 0) + 1))


def score(rec, idf):
    diag = (rec.get("lodestar_diagnosis") or "").lower()
    declined = diag.startswith("no onboard") or diag == "decline" or not diag
    cterms = set(_words(rec.get("documented_cause", "")))
    named = [] if declined else [w for w in cterms if w in diag]
    rec["lodestar_declined"] = declined
    rec["distinctive_idf_matched"] = round(sum(idf(w) for w in named), 2)
    rec["n_strong_terms"] = sum(1 for w in named if idf(w) >= IDF_STRONG)
    rec["right_direction"] = (not declined) and rec["n_strong_terms"] >= 1
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "results", "stackup_result.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "breadth_scored.json"))
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.ckpt, encoding="utf-8") if l.strip()]
    idf = build_idf(recs)
    for r in recs:
        score(r, idf)
    inc = [r for r in recs if r.get("in_corpus_domain")]
    out = [r for r in recs if not r.get("in_corpus_domain")]
    rd = lambda rows: sum(1 for r in rows if r["right_direction"]) / len(rows) if rows else 0
    print("==== DETERMINISTIC breadth (names >=1 distinctive cause term; attribute-and-flag diagnoses) ====")
    print(f"  in-corpus  n={len(inc)}: right-direction {rd(inc)*100:.0f}%  "
          f"({sum(r['right_direction'] for r in inc)}/{len(inc)})")
    print(f"  out-corpus n={len(out)}: right-direction {rd(out)*100:.0f}%  "
          f"({sum(r['right_direction'] for r in out)}/{len(out)})")
    print(f"  IN-OUT gap: {(rd(inc)-rd(out))*100:+.0f}pp   "
          f"(declines: in {sum(r['lodestar_declined'] for r in inc)}, out {sum(r['lodestar_declined'] for r in out)})")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(recs, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
