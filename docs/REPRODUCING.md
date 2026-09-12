# Reproducing the reported results

## Setup

Clone both repositories side by side.

```bash
git clone https://github.com/ARC-lab-University-of-Washington/lodestar-fdx
git clone https://github.com/ARC-lab-University-of-Washington/apollo-anomaly-atlas
cd lodestar-fdx && pip install -r requirements.txt
```

Set `LODESTAR_DATA` if they are elsewhere.

## Without a model or hardware

```bash
python scripts/verify_release.py
cd analysis && python3 score_subsystem.py
```

| Reported | From |
|---|---|
| 85% fault-direction accuracy on the 107 | `analysis/score_subsystem.py` over `baselines/decisions_146_k8.jsonl` |
| 255 / 146 / 107 / 39 split | `analysis/subsystem_match.is_in_corpus` |
| 3,370 → 3,148 corpus | `scripts/verify_release.py` |
| 13 turn pairs, LODESTAR 28/37 vs CAPCOM 37/37 | `analysis/build_lodestar_vs_capcom.py` |

These re-score stored run dumps rather than re-running generation. The Apollo 13
figures — 8 Mission Control turns, 13.5 min — are constants from the transcript
record in `eval/data/apollo13_o2_episode.json`.

## With a local model

Regenerating the dumps needs Ollama with `llama3.2:3b`. `eval/run_all.sh` does
this. Generation is not bit-reproducible across Ollama versions, quantizations or
context sizes.

## On the Jetson

Latency and median time to diagnosis, and mean 7.1 W / peak 10.6 W via
`eval/bench_latency.py`, need a Jetson Orin Nano Super 8 GB with `tegrastats`.
No power trace ships.

## Notes

- Set `PYTHONHASHSEED=0`. Retrieval is deterministic; `diagnose._dials` and
  `graph.chase` break ties by set iteration order.
- Use `subsystem_match.is_in_corpus` for the in-corpus split. Two older
  definitions exist in `run_stackup` and `exp_graph` and give different counts.
- `run_all.sh` runs `diagnose()` at k=6. The 85% came from `diagnose_chain()` at
  k=8, which is what the cite-score constants are calibrated for.
- The coverage AUC recomputes to 0.7339 from the shipped dump.
  `verify_release.py` prints it on every run.
- `LODESTAR_NUM_CTX` changes generation output. Default 8192; 4096 on the 8 GB
  Jetson.
