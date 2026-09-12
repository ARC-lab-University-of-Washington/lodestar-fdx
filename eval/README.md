# Lodestar on-device eval (runs the DEPLOYED code)

Re-runs the paper's evaluation **on the Jetson**, against the **same `lodestar_edge` code that ships**
(salience + BM25, grounding-or-silence **attribute-and-flag** — an uncited candidate is kept, attributed to the source it was generated from, and flagged
emitted its `[S#]`; uncited points are dropped, never back-filled). So the numbers this produces are
real on-device measurements of the deployed system — no dev-box / research-harness gap.

## Layout
```
eval/
  run_flagship.py      # Apollo-13 O2 multi-turn replay -> turns + on-device time-to-diagnosis
  run_stackup.py       # 255-anomaly breadth (attribute-and-flag); crash-safe + resumable JSONL
  distinctive_scorer.py# official deterministic breadth metric (no model calls)
  run_grounding.py     # 64-scenario grounding audit (counts inferred vs cited leads)
  bench_latency.py     # on-device latency + tegrastats power (7-15 W envelope)
  metrics.py           # coverage + human-timing helpers (stdlib)
  run_all.sh           # runs all of the above, logs to results/
  data/                # apollo13_o2_episode.json, atlas_parsed.json, cross-mission scenarios + anchors
  results/             # (created on first run) *.json + *.log
```

## Prerequisites (on the Jetson)
- `lodestar_edge/` package + `lodestar_edge/corpus.json` present (default: `~/lodestar/lodestar_edge/`).
- Ollama running with `llama3.2:3b`.
- Python 3 with `requests`; `rank_bm25` (the vendored `~/lodestar/rank_bm25.py` is picked up because
  the eval adds `LODESTAR_EDGE_DIR` — default the eval's parent, i.e. `~/lodestar` — to `sys.path`).

This package is designed to sit at `~/lodestar/eval/` so its parent (`~/lodestar`) holds `lodestar_edge/`
and `rank_bm25.py`. If you put it elsewhere, set `LODESTAR_EDGE_DIR` to the dir that contains
`lodestar_edge/`.

## Run
```bash
cd ~/lodestar/eval
bash run_all.sh                       # full batch (the stack-up takes a while; it's resumable)
# or individually:
python3 run_flagship.py               # fast: turns + on-device time
python3 run_stackup.py --limit 5      # smoke test the breadth loop
python3 run_stackup.py                # full 255 (resume if interrupted)
python3 distinctive_scorer.py         # breadth % from the stored diagnoses
python3 run_grounding.py              # grounding-or-silence rates
python3 bench_latency.py              # latency + power
```

## Config (env or flags)
`LODESTAR_EDGE_MODEL` (default `llama3.2:3b`), `OLLAMA_URL` (`http://localhost:11434`),
`LODESTAR_EDGE_DIR` (default = eval's parent dir), `LODESTAR_CORPUS`, `LODESTAR_HW` (label for result
rows). All runners take `--k` (default 6), `--model`, `--base-url`.

## What changes vs the dev-box numbers
- **Grounding:** uncited leads are attributed to the source they came from and flagged `grounded: False`; `n_inferred_leads` counts them. The dev-box harness reported
  ~1.9% forged because its `generate_pointwise` back-filled the tag; the deployed code drops instead.
- **Breadth / flagship:** dropping uncited points can lower coverage vs the back-filled numbers.
  These runs produce the deployed figures, written to `results/`.
- **Latency / power:** measured on-device, not estimated.

## AI use statement

AI assistance (Anthropic Claude) was used during development of this project for code drafting, refactoring and documentation. The corpus metadata this project consumes is LLM-distilled from OCR text and is documented as such. All released content was reviewed and verified by the authors, who are solely responsible for its correctness.
