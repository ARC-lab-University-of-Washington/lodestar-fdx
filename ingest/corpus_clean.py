# AI use statement: docs/AI_USE.md
##
# @file corpus_clean.py
# @brief Retrieval-precision cleaning.
#
# The 844-page LM AOH is ~23% of the corpus, and its non-procedural
# boilerplate — illustration lists, tables of contents, near-empty date/figure
# headers, hatch/window/hinge reference pages — floods retrieval: an EPS 'main
# bus undervolt' query surfaces the forward-hatch page because they share
# words. Dropping this boilerplate from the RETRIEVAL set (not the archive)
# sharply raises precision without touching real procedure/checklist/rule
# content.
#
"""Retrieval-precision cleaning.

The 844-page LM AOH is ~23% of the corpus, and its non-procedural boilerplate — illustration
lists, tables of contents, near-empty date/figure headers, hatch/window/hinge reference pages —
floods retrieval: an EPS 'main bus undervolt' query surfaces the forward-hatch page because
they share words. Dropping this boilerplate from the RETRIEVAL set (not the archive) sharply
raises precision without touching real procedure/checklist/rule content.
"""
import re

_MARKERS = (
    "list of illustrations", "list of tables", "table of contents",
    "list of effective pages", "index of", "illustrations (cont",
)


def is_boilerplate(raw: str) -> bool:
    """True for non-procedural reference pages that should not be retrievable."""
    t = (raw or "").lower()
    words = re.findall(r"[a-z]{3,}", t)
    if len(words) < 6:                       # near-empty: pure tables / dates / numbers
        return True
    if any(m in t for m in _MARKERS):        # explicit front-matter lists
        return True
    if (t.count("figure") + t.count(" page ")) >= 3 and len(words) < 50:  # illustration/TOC lists
        return True
    # date/change-bar header pages with almost no prose (e.g. "Basic Date 2/6/70 Changed ...")
    if ("basic date" in t or "changed" in t) and len(words) < 12:
        return True
    return False


def clean_corpus(segments):
    """Return (kept_segments, n_dropped). Drops boilerplate for retrieval only."""
    kept = [s for s in segments if not is_boilerplate(s.raw_text)]
    return kept, len(segments) - len(kept)
