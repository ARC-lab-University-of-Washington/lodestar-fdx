# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file run_flagship.py
# @brief ON-DEVICE flagship eval: replay the Apollo-13 O2-tank crew air-to-ground report one utterance at a time through the DEPLOYED lodestar_edge.diagnose (c.
#
# ite-or-drop), and record the TURN at which Lodestar's grounded diagnosis
# first covers the documented fault direction, plus cumulative on-device
# inference time. Compare to the human Mission-Control loop from the
# transcript. turns-to-diagnosis(lodestar) = first accumulation-turn coverage
# clears the bar (default 0.25) time-to-diagnosis(lodestar) = cumulative wall-
# clock (retrieval + generation) on THIS hardware
#
"""ON-DEVICE flagship eval: replay the Apollo-13 O2-tank crew air-to-ground report one utterance at
a time through the DEPLOYED lodestar_edge.diagnose (attribute-and-flag), and record the TURN at which
Lodestar's grounded diagnosis first covers the documented fault direction, plus cumulative on-device
inference time. Compare to the human Mission-Control loop from the transcript.

  turns-to-diagnosis(lodestar) = first accumulation-turn coverage clears the bar (default 0.25)
  time-to-diagnosis(lodestar)  = cumulative wall-clock (retrieval + generation) on THIS hardware
"""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.environ.get("LODESTAR_EDGE_DIR", os.path.dirname(HERE))
sys.path.insert(0, EDGE_ROOT)
from lodestar_edge.segment import load_corpus              # noqa: E402
from lodestar_edge.retriever import SalienceBM25Retriever  # noqa: E402
from lodestar_edge.diagnose import diagnose                # noqa: E402
from metrics import diagnosis_coverage, human_time_to_diagnosis, fmt_dur  # noqa: E402

DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
CORPUS = os.environ.get("LODESTAR_CORPUS", os.path.join(EDGE_ROOT, "lodestar_edge", "corpus.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("LODESTAR_EDGE_MODEL", "llama3.2:3b"))
    ap.add_argument("--base-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--cov-bar", type=float, default=0.25)
    ap.add_argument("--hardware", default=os.environ.get("LODESTAR_HW", "jetson-orin-nano"))
    ap.add_argument("--episode", default=os.path.join(DATA, "apollo13_o2_episode.json"))
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    ep = json.load(open(args.episode, encoding="utf-8"))
    anchor = ep["resolution_anchor"]
    hsec, _, _ = human_time_to_diagnosis(ep, "diagnostic_recognition")
    hturn = ep["human_baseline"]["turns_to_diagnosis"]["to_diagnostic_recognition"]

    retr = SalienceBM25Retriever(load_corpus(CORPUS))
    print(f"[flagship] {ep['episode_id']} ({ep['mission']}); model={args.model} k={args.k} "
          f"hw={args.hardware}", flush=True)
    print(f"HUMAN: characterized at crew turn {hturn}  (+{fmt_dur(hsec)} after report)\n")

    acc, cum_t, cum_calls = "", 0.0, 0
    first_turn = None
    char_time = None      # cumulative on-device time AT the convergence turn (the real time-to-diagnosis)
    traj = []
    for t, u in enumerate(ep["input_utterances"], 1):
        acc = (acc + " " + u["text"]).strip()
        t0 = time.perf_counter()
        res = diagnose(acc, retr, model=args.model, base_url=args.base_url, k=args.k, gate=False)
        dt = time.perf_counter() - t0
        cum_t += dt; cum_calls += res.get("n_points", 0) + res.get("n_dropped_uncited", 0)
        if res["declined"]:
            cov = 0.0
        else:
            cov = diagnosis_coverage(res["diagnosis"], anchor)["component_coverage"]
        characterized = (not res["declined"]) and cov >= args.cov_bar
        if characterized and first_turn is None:
            first_turn = t
            char_time = cum_t
        tag = "DECLINE" if res["declined"] else f"cov={cov:.2f}"
        print(f"  turn {t} [{u['get']}] {tag}{'  <CHARACTERIZED>' if characterized else ''}  "
              f"cum={cum_t:.1f}s", flush=True)
        traj.append({"turn": t, "get": u["get"], "declined": res["declined"], "coverage": cov,
                     "characterized": characterized, "cum_time_s": round(cum_t, 1),
                     "diagnosis": res["diagnosis"]})

    out = {"episode": ep["episode_id"], "hardware": args.hardware,
           "human_turns": hturn, "human_time_s": hsec,
           "lodestar_turns": first_turn,
           "lodestar_time_to_diagnosis_s": (round(char_time, 1) if char_time is not None else None),
           "lodestar_full_replay_s": round(cum_t, 1),
           "lodestar_internal_calls": cum_calls, "trajectory": traj}
    json.dump(out, open(os.path.join(RESULTS, "flagship_result.json"), "w", encoding="utf-8"), indent=1)
    print("\n" + "=" * 60)
    print(f"turns-to-diagnosis:  HUMAN {hturn}   LODESTAR {first_turn if first_turn else 'NOT REACHED'}")
    print(f"time-to-diagnosis:   HUMAN +{fmt_dur(hsec)}   LODESTAR "
          f"{fmt_dur(char_time) if char_time is not None else 'NOT REACHED'} ({args.hardware})")
    print("=" * 60)


if __name__ == "__main__":
    main()
