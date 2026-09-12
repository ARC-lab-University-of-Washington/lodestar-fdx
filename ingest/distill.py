# AI use statement: docs/AI_USE.md
##
# @file distill.py
#
# The distillation model defaults to a local model. Remote use is opt-in via
# LODESTAR_DISTILL_MODEL. The released corpus was built before this default changed.
#
# @brief Ingestion distillation, GraphRAG/HybridRAG-style: 1.
#
# extract_triples() — per-chunk, schema-constrained LLM extraction (Ollama
# JSON mode) 2. summarize_entity() — merge an entity's descriptions across
# chunks into one summary This replaces free-text "20-term signature"
# distillation with checkable, schema-bound structured output — the "more
# principled" method you asked for.
#
"""
Ingestion distillation, GraphRAG/HybridRAG-style:
  1. extract_triples()   — per-chunk, schema-constrained LLM extraction (Ollama JSON mode)
  2. summarize_entity()  — merge an entity's descriptions across chunks into one summary
This replaces free-text "20-term signature" distillation with checkable,
schema-bound structured output — the "more principled" method you asked for.
"""
import os
import json
import time
import requests
from ontology import ENTITY_TYPES, RELATION_TYPES, TRIPLE_EXTRACTION_PROMPT

# Per-request HTTP timeout (s) for Ollama extraction calls. Raised from 120 and
# made configurable: under GPU batching, a dense page can legitimately take
# >120s, and a too-low timeout silently drops that chunk's metadata.
HTTP_TIMEOUT = int(os.environ.get("LODESTAR_HTTP_TIMEOUT", "300"))

# Per-chunk extraction failures, keyed by chunk id, populated by extract_triples()
# whenever it has to give up and return []. run_ingestion.py reads this at the
# end of a run so a failing ingest is *visible* instead of silently producing
# an empty-metadata corpus. {chunk_id: reason_string}
EXTRACTION_FAILURES: dict[str, str] = {}


def _post_with_retry(url, json_body, timeout, max_retries=5):
    """Retries on 429 (rate limit) AND on transient network errors
    (timeouts, connection resets/DNS blips) — a single flaky call used to
    kill the whole ingestion run since only 429 was retried before."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=json_body, timeout=timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
            time.sleep(2 ** attempt)  # 1, 2, 4, 8, 16s
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last_exc = requests.exceptions.HTTPError(f"{r.status_code} {r.text[:200]}")
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r
    # Exhausted retries — raise the last real failure so the caller can log it
    # (rather than a generic raise_for_status() on a response that may not exist).
    raise last_exc or RuntimeError(f"_post_with_retry: exhausted {max_retries} retries with no response")


def extract_triples(text: str, base_url: str = "http://localhost:11434",
                     model: str = os.environ.get("LODESTAR_DISTILL_MODEL", "llama3.2:3b"), chunk_id: str = "") -> list[dict]:
    prompt = TRIPLE_EXTRACTION_PROMPT.format(
        entity_types=", ".join(ENTITY_TYPES),
        relation_types=", ".join(RELATION_TYPES),
        text=text,
    )
    try:
        r = _post_with_retry(f"{base_url.rstrip('/')}/api/chat", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",   # Ollama structured-output mode — forces valid JSON
            "stream": False,
            # num_predict caps output length: a per-chunk triple extraction never
            # needs many hundreds of tokens, but dense/garbled OCR pages (e.g. the
            # G&N dictionary) can send the model into a repetition loop that runs to
            # the HTTP timeout. Capping truncates the loop (truncated JSON -> that one
            # chunk yields no triples, but its raw text stays retrievable) instead of
            # burning ~300s of GPU per bad chunk.
            "options": {"temperature": 0.0, "num_predict": int(os.environ.get("LODESTAR_NUM_PREDICT", "1024"))},
        }, timeout=HTTP_TIMEOUT)
    except requests.exceptions.RequestException as e:
        # HTTP/connection failure after retries — log which chunk and why,
        # return [] so ONE bad chunk doesn't crash a 1000+ chunk ingestion run.
        EXTRACTION_FAILURES[chunk_id] = f"http_error: {type(e).__name__}: {e}"
        print(f"[distill] extract_triples FAILED for chunk {chunk_id!r}: {type(e).__name__}: {e}")
        return []

    try:
        content = r.json()["message"]["content"]
    except (KeyError, ValueError) as e:
        EXTRACTION_FAILURES[chunk_id] = f"bad_response: {type(e).__name__}: {e}"
        print(f"[distill] extract_triples got malformed response for chunk {chunk_id!r}: {e}")
        return []

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        EXTRACTION_FAILURES[chunk_id] = f"invalid_json: {e}"
        print(f"[distill] extract_triples got non-JSON content for chunk {chunk_id!r}: {content[:200]!r}")
        return []

    triples = data.get("triples", [])
    # Drop anything outside the fixed ontology instead of silently keeping it —
    # a closed schema is only useful if it's actually enforced downstream.
    valid = [t for t in triples
             if t.get("subject_type") in ENTITY_TYPES
             and t.get("object_type") in ENTITY_TYPES
             and t.get("relation") in RELATION_TYPES]
    if not valid and triples:
        # Model returned triples, but every one fell outside the closed schema —
        # distinct from "model found nothing" and worth knowing about separately.
        EXTRACTION_FAILURES[chunk_id] = f"all_triples_rejected: {len(triples)} returned, 0 valid"
    return valid


SUMMARIZE_PROMPT = """Combine these descriptions of the same entity "{name}" ({etype})
into one concise (<=40 word) summary. Do not invent facts not present below.

DESCRIPTIONS:
{descriptions}
"""


def summarize_entity(name: str, etype: str, descriptions: list[str],
                      base_url: str = "http://localhost:11434", model: str = os.environ.get("LODESTAR_DISTILL_MODEL", "llama3.2:3b")) -> str:
    """Consolidate an entity's occurrences across chunks — mirrors GraphRAG's
    entity-summarization step, needed once the same component/symptom shows
    up in multiple procedures."""
    if len(descriptions) == 1:
        return descriptions[0]
    prompt = SUMMARIZE_PROMPT.format(name=name, etype=etype, descriptions="\n".join(f"- {d}" for d in descriptions))
    r = _post_with_retry(f"{base_url.rstrip('/')}/api/chat", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0},
    }, timeout=60)
    return r.json()["message"]["content"].strip()