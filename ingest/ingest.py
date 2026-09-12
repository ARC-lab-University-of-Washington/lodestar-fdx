# AI use statement: docs/AI_USE.md
##
# @file ingest.py
# @brief End-to-end ingestion (§4.4): PDF -> raw segments -> per-chunk triple extraction -> merged fault knowledge graph + Segment objects with extracted (not .
#
# hand-tagged) subsystem/component/symptoms.
#
"""
End-to-end ingestion (§4.4): PDF -> raw segments -> per-chunk triple
extraction -> merged fault knowledge graph + Segment objects with
extracted (not hand-tagged) subsystem/component/symptoms.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pdf_ingest import extract_raw_segments
from distill import extract_triples, summarize_entity, EXTRACTION_FAILURES
from schema import Segment
from graph_router import SymptomGraph


def _extract_triples_safe(r: dict, base_url: str, model: str) -> list[dict]:
    """Defense in depth around extract_triples: it already catches its own
    request/JSON failures and returns [], but if anything unexpected still
    raises here, don't let one bad chunk kill the whole ThreadPoolExecutor.map
    (an uncaught exception in one worker previously propagated and crashed the
    entire ingestion run, wiping out every already-successful chunk with it)."""
    try:
        return extract_triples(r["raw_text"], base_url, model, chunk_id=r["id"])
    except Exception as e:
        EXTRACTION_FAILURES[r["id"]] = f"unexpected: {type(e).__name__}: {e}"
        print(f"[ingest] unexpected error extracting chunk {r['id']!r}: {type(e).__name__}: {e}")
        return []


def ingest_pdf(pdf_path: str, doc_type: str, base_url: str = "http://localhost:11434",
                model: str = os.environ.get("LODESTAR_GEN_MODEL", "llama3.1:8b"),
                max_workers: int = int(os.environ.get("LODESTAR_MAX_WORKERS", "1"))
                ) -> tuple[list[Segment], SymptomGraph]:
    raw = extract_raw_segments(pdf_path, doc_type)

    # 1. Per-chunk triple extraction, parallelized — Ollama serves concurrent
    #    requests fine (set OLLAMA_NUM_PARALLEL if you want more throughput).
    #    Each chunk is isolated (_extract_triples_safe) so one bad chunk can't
    #    take the rest of a 1000+ chunk ingestion run down with it.
    print(f"[ingest] extracting {len(raw)} chunks (workers={max_workers}) ...", flush=True)
    chunk_triples = {}
    _t0 = time.time()
    _done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        _fut2r = {ex.submit(_extract_triples_safe, r, base_url, model): r for r in raw}
        for _fut in as_completed(_fut2r):
            _r = _fut2r[_fut]
            chunk_triples[_r["id"]] = _fut.result()
            _done += 1
            if _done % 10 == 0 or _done == len(raw):
                _rate = _done / max(time.time() - _t0, 0.1)
                print(f"[ingest]   {_done}/{len(raw)} chunks extracted ({_rate:.2f}/s)", flush=True)

    n_empty = sum(1 for t in chunk_triples.values() if not t)
    n_failed = sum(1 for cid in chunk_triples if cid in EXTRACTION_FAILURES)
    print(f"[ingest] {pdf_path}: {len(raw)} chunks, {n_empty} returned zero triples "
          f"({n_failed} of those were hard failures — see EXTRACTION_FAILURES for reasons).")

    # 2. Entity description aggregation across chunks, before summarization
    entity_occurrences: dict[tuple[str, str], list[str]] = defaultdict(list)
    for seg_id, triples in chunk_triples.items():
        for t in triples:
            entity_occurrences[(t["subject"], t["subject_type"])].append(seg_id)
            entity_occurrences[(t["object"], t["object_type"])].append(seg_id)

    # 3. Build segments, populating subsystem/component/symptoms from extracted
    #    triples instead of manual tags.
    segments = []
    for r in raw:
        triples = chunk_triples[r["id"]]
        subsystem = next((t["object"] for t in triples if t["relation"] == "locatedIn"), None)
        components = [t["subject"] for t in triples if t["relation"] == "locatedIn"]
        symptoms = [t["subject"] for t in triples if t["relation"] == "indicates" and t["subject_type"] == "Symptom"]
        refs = [t["object"] for t in triples if t["relation"] in ("references", "precedes")]

        segments.append(Segment(
            id=r["id"],
            raw_text=r["raw_text"],
            signature=", ".join(sorted({t["subject"] for t in triples} | {t["object"] for t in triples})),
            subsystem=subsystem,
            component=components[0] if components else None,
            doc_type=doc_type,
            symptoms=symptoms,
            links_to=refs,
            ocr_confidence=r.get("ocr_confidence"),
            source_doc=os.path.basename(pdf_path),
        ))

    # 4. Merge into the symptom graph, then add the extracted relational
    #    edges triple-extraction found beyond the base subsystem/symptom taxonomy
    #    (causes, diagnosedBy, resolvedBy) — this is the graph-based part proper.
    graph = SymptomGraph().build(segments)
    for seg_id, triples in chunk_triples.items():
        for t in triples:
            if t["relation"] in ("causes", "diagnosedBy", "resolvedBy"):
                graph.g.add_node(t["subject"], kind=t["subject_type"].lower())
                graph.g.add_node(t["object"], kind=t["object_type"].lower())
                graph.g.add_edge(t["subject"], t["object"], kind=t["relation"])

    return segments, graph