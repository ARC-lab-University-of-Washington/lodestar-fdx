# AI use statement: docs/AI_USE.md
##
# @file run_ingestion.py
# @brief Run this to ingest every PDF under PDF_DIR (recursively).
#
# Outputs (corpus.json, centrality_priors.json) are written next to THIS file
# — the project folder — NOT the current working directory. Point it at your
# PDFs with the LODESTAR_PDF_DIR env var (defaults to a `pdfs/` folder here).
# Checkpointed + resumable: corpus.json + an _ingested.json manifest are
# rewritten after EVERY document, and PDFs already in the manifest are skipped
# on re-run. So a crash (or Ctrl-C) mid-run loses at most the one document in
# flight, and re-running picks up where it left off. Documents are processed
# smallest-first, so the giant AOH scan is last and everything else is banked
# before it.
#
"""Run this to ingest every PDF under PDF_DIR (recursively).

Outputs (corpus.json, centrality_priors.json) are written next to THIS file
— the project folder — NOT the current working directory. Point it at your
PDFs with the LODESTAR_PDF_DIR env var (defaults to a `pdfs/` folder here).

Checkpointed + resumable: corpus.json + an _ingested.json manifest are rewritten
after EVERY document, and PDFs already in the manifest are skipped on re-run.
So a crash (or Ctrl-C) mid-run loses at most the one document in flight, and
re-running picks up where it left off. Documents are processed smallest-first,
so the giant AOH scan is last and everything else is banked before it.
"""
import os
import json
import time
from collections import Counter
from pathlib import Path
from ingest import ingest_pdf
from distill import EXTRACTION_FAILURES
from schema import Segment
from graph_router import SymptomGraph
from ranking import compute_centrality_priors, save_priors

HERE = Path(__file__).resolve().parent
PDF_DIR = Path(os.environ.get("LODESTAR_PDF_DIR", HERE / "pdfs"))
DOC_TYPE_BY_SUFFIX = {"mp": "malfunction_procedure", "fr": "flight_rule", "ar": "anomaly_report"}


def _doc_type(stem: str) -> str:
    s = stem.lower()
    if "anomaly" in s or "-report" in s:
        return "anomaly_report"
    if "rule" in s:
        return "flight_rule"
    return next((v for k, v in DOC_TYPE_BY_SUFFIX.items() if k in s), "malfunction_procedure")


def main():
    corpus_path = HERE / "corpus.json"
    priors_path = HERE / "centrality_priors.json"
    manifest_path = HERE / "_ingested.json"

    if not PDF_DIR.is_dir():
        print(f"[run_ingestion] PDF folder not found: {PDF_DIR}")
        print("  Create it (or set LODESTAR_PDF_DIR) and drop your PDFs in, then re-run.")
        return

    # Resume from any prior checkpoint.
    done = set(json.loads(manifest_path.read_text())) if manifest_path.exists() else set()
    all_segments = ([Segment(**d) for d in json.loads(corpus_path.read_text())]
                    if (corpus_path.exists() and done) else [])

    pdfs = sorted(PDF_DIR.rglob("*.pdf"), key=lambda p: p.stat().st_size)  # smallest first
    todo = [p for p in pdfs if str(p) not in done]
    print(f"{len(pdfs)} PDFs under {PDF_DIR.name}/  |  {len(done)} already done  |  {len(todo)} to ingest",
          flush=True)

    t_run = time.time()
    timings = []  # (doc, n_pages_or_segs, seconds) for the end-of-run benchmark table
    for n, pdf_path in enumerate(todo, 1):
        dt = _doc_type(pdf_path.stem)
        mb = pdf_path.stat().st_size / 1e6
        print(f"\n[{n}/{len(todo)}] {pdf_path.name}  ({mb:.0f} MB, {dt})", flush=True)
        t_doc = time.time()
        segs, _ = ingest_pdf(str(pdf_path), dt)
        el = time.time() - t_doc
        all_segments.extend(segs)
        done.add(str(pdf_path))
        # checkpoint after every document
        corpus_path.write_text(json.dumps([s.__dict__ for s in all_segments]))
        manifest_path.write_text(json.dumps(sorted(done)))
        ns = sum(1 for s in segs if s.subsystem)
        sy = sum(1 for s in segs if s.symptoms)
        timings.append((pdf_path.name, mb, len(segs), el))
        print(f"    +{len(segs)} segs ({ns} subsystem, {sy} symptoms) in {el:.0f}s "
              f"({len(segs)/max(el,0.1):.2f} seg/s) | corpus now {len(all_segments)} [checkpointed]", flush=True)

    # Priors (PageRank) computed once at the end over the full graph.
    graph = SymptomGraph().build(all_segments)
    save_priors(compute_centrality_priors(all_segments, graph), str(priors_path))

    n_sub = sum(1 for s in all_segments if s.subsystem)
    n_sym = sum(1 for s in all_segments if s.symptoms)
    confs = [s.ocr_confidence for s in all_segments if s.ocr_confidence is not None]
    print("\n" + "=" * 56, flush=True)
    print(f"DONE: {len(all_segments)} segments total", flush=True)
    total = time.time() - t_run
    print(f"  wall time this run: {total/60:.1f} min for {len(todo)} docs "
          f"({len(all_segments)/max(total,1):.2f} seg/s overall)")
    print("  per-document (slowest first):")
    for name, mb, ns_, el in sorted(timings, key=lambda x: -x[3]):
        print(f"    {el/60:6.1f} min  {mb:6.0f} MB  {ns_:5d} segs  {name}")
    print(f"  corpus -> {corpus_path}")
    print(f"  priors -> {priors_path}")
    print(f"  {n_sub}/{len(all_segments)} with subsystem, {n_sym}/{len(all_segments)} with symptoms")
    if confs:
        low = sum(1 for c in confs if c < 70)
        print(f"  mean OCR confidence {sum(confs)/len(confs):.1f}/100; {low} segments below 70")
    if EXTRACTION_FAILURES:
        reasons = Counter(v.split(":")[0] for v in EXTRACTION_FAILURES.values())
        print(f"  {len(EXTRACTION_FAILURES)} extraction failures: {dict(reasons)}")


if __name__ == "__main__":
    main()
