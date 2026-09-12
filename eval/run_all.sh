#!/usr/bin/env bash
# AI use statement: docs/AI_USE.md
# Run the full on-device eval batch, in order, logging each stage to results/.
# The stack-up is the long one (255 x k generation calls on the Orin Nano) and is resumable —
# if it is interrupted, just re-run this script and it picks up where it left off.
set -eu
cd "$(dirname "$0")"
mkdir -p results
PY="${PYTHON:-python3}"

echo "=== 1/5 flagship (multi-turn Apollo-13) ==="
$PY run_flagship.py      2>&1 | tee results/flagship.log

echo "=== 2/5 breadth stack-up (255 anomalies; resumable) ==="
$PY run_stackup.py       2>&1 | tee results/stackup.log

echo "=== 3/5 deterministic breadth scorer ==="
$PY distinctive_scorer.py 2>&1 | tee results/breadth.log

echo "=== 4/5 grounding-or-silence audit (64 scenarios) ==="
$PY run_grounding.py     2>&1 | tee results/grounding.log

echo "=== 5/5 on-device latency + power ==="
$PY bench_latency.py     2>&1 | tee results/latency.log

echo
echo "ALL DONE. Results JSON + logs in: $(pwd)/results/"
