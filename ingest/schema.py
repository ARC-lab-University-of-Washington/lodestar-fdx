# AI use statement: docs/AI_USE.md
##
# @file schema.py
# @brief Shared corpus schema for Lodestar retrieval (§4.4).
#
# A Segment is the atomic retrieval unit: one manual section / procedure step
# / malfunction-procedure entry. `signature` is the LLM-distilled ~20-term
# summary used for *finding*; `raw_text` is what citations resolve to (never
# the signature or a paraphrase).
#
"""
Shared corpus schema for Lodestar retrieval (§4.4).

A Segment is the atomic retrieval unit: one manual section / procedure
step / malfunction-procedure entry. `signature` is the LLM-distilled
~20-term summary used for *finding*; `raw_text` is what citations
resolve to (never the signature or a paraphrase).
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Segment:
    id: str                        # stable segment id, e.g. "MP-ECLSS-014.3"
    raw_text: str                  # verbatim source text (citation target)
    signature: str = ""            # LLM-distilled ~20-term summary (search target)
    subsystem: Optional[str] = None
    component: Optional[str] = None
    doc_type: str = ""             # "malfunction_procedure" | "flight_rule" | "drawing" | "anomaly_report"
    effectivity: Optional[str] = None   # revision/vehicle applicability tag
    symptoms: list = field(default_factory=list)   # symptom strings this segment is indexed under
    links_to: list = field(default_factory=list)   # ids of related segments (procedure steps, related checks)
    ocr_confidence: Optional[float] = None   # mean Tesseract word-confidence (0-100) this segment's
                                              # raw_text was transcribed at; None if source had a native
                                              # text layer / confidence wasn't tracked. Surface low values
                                              # at citation time rather than trusting OCR'd text silently.
    source_doc: Optional[str] = None          # basename of the PDF this segment came from — provenance,
                                              # and lets the corpus be filtered by document (e.g. drop
                                              # historical/narrative docs) without re-ingesting.

    def search_text(self) -> str:
        """Text handed to BM25/dense indexing — signature preferred, falls back to raw."""
        return self.signature or self.raw_text
