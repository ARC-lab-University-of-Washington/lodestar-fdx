# -*- coding: utf-8 -*-
# AI use statement: docs/AI_USE.md
##
# @file segment.py
# @brief The atomic retrieval and citation unit, plus a self-contained corpus loader.
#
# Minimal Segment: the atomic retrieval/citation unit. Self-contained loader for the
# LLM-distilled corpus.json — no dependency on the research tree.
#
import json
from dataclasses import dataclass, field


@dataclass
class Segment:
    """@brief One retrievable manual segment — the unit every citation resolves to.

    Every candidate cause LODESTAR returns is traceable to a Segment, and through its
    @ref source_doc and @ref id back to a page of the source manual.

    @var id              Stable segment identifier, e.g. `malfunction_procedure-p132`.
    @var raw_text        Verbatim OCR text of the page region.
    @var signature       LLM-distilled one-line summary; preferred over raw_text for indexing.
    @var subsystem       Vehicle subsystem label (OCR-derived, not hand-verified).
    @var source_doc      Filename of the source PDF this segment came from.
    @var symptoms        Annunciator / symptom strings extracted from the segment.
    @var ocr_confidence  Mean OCR confidence for the region, 0-100.
    @var component       Node label used to RESOLVE links_to into graph edges.
    @var links_to        Outgoing edges — referenced components or procedure steps.
    """
    id: str
    raw_text: str = ""
    signature: str = ""
    subsystem: str = ""
    source_doc: str = ""
    symptoms: list = field(default_factory=list)
    ocr_confidence: float = None
    component: str = ""            # the node label used to RESOLVE links_to into graph edges
    links_to: list = field(default_factory=list)   # outgoing edges (referenced components/steps)

    def search_text(self) -> str:
        """@brief Text this segment is indexed under.

        Index the distilled signature if present, else the raw OCR text.

        @return The string fed to the BM25 tokenizer for this segment.
        """
        return self.signature or self.raw_text

    def chain_text(self) -> str:
        """@brief Compact human/LLM-facing description of this node for a reasoning chain.

        @return Whitespace-collapsed label, truncated to 200 characters.
        """
        head = (self.component or self.signature or self.raw_text or "").strip()
        return " ".join(head.split())[:200]


def load_corpus(path: str) -> list:
    """@brief Load the distilled corpus from disk into Segment objects.

    The corpus is distributed separately, in the apollo-anomaly-atlas repository, so that
    exactly one copy of it exists. Point this at `corpus/corpus.json` there, or set the
    `LODESTAR_CORPUS` environment variable, which the eval harness reads.

    @param path Filesystem path to `corpus.json` (a JSON array of segment objects).
    @return list of @ref Segment, in file order. Missing fields become empty defaults.
    @exception FileNotFoundError if @p path does not exist.
    @exception json.JSONDecodeError if the file is not valid JSON.
    @see https://github.com/ARC-lab-University-of-Washington/apollo-anomaly-atlas
    """
    raw = json.load(open(path, encoding="utf-8"))
    segs = []
    for d in raw:
        segs.append(Segment(
            id=str(d.get("id")), raw_text=d.get("raw_text") or "",
            signature=d.get("signature") or "", subsystem=d.get("subsystem") or "",
            source_doc=d.get("source_doc") or "", symptoms=d.get("symptoms") or [],
            ocr_confidence=d.get("ocr_confidence"),
            component=d.get("component") or "", links_to=d.get("links_to") or [],
        ))
    return segs
