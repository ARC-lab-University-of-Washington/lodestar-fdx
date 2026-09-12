# AI use statement: docs/AI_USE.md
##
# @file __init__.py
# @brief Lodestar edge — self-contained, BM25-only fault-diagnosis assistant for offline deployment on the NVIDIA Jetson (via Ollama).
#
# No embedding model / torch required. Current design: BM25 finds entry
# manuals; the onboard-manual GRAPH is chased (links_to edges) into a
# reasoning chain-of-causes per manual; the LLM decides over the chains.
#
"""Lodestar edge — self-contained, BM25-only fault-diagnosis assistant for offline deployment on the
NVIDIA Jetson (via Ollama). No embedding model / torch required. Current design: BM25 finds entry
manuals; the onboard-manual GRAPH is chased (links_to edges) into a reasoning chain-of-causes per
manual; the LLM decides over the chains."""
from .segment import Segment, load_corpus
from .retriever import SalienceBM25Retriever, salience
from .graph import CorpusGraph
from .scope_gate import in_domain, out_of_domain, OUT_OF_DOMAIN_TERMS
from .diagnose import diagnose_chain, diagnose, DECLINE, DECLINE_DOMAIN

__all__ = ["Segment", "load_corpus", "SalienceBM25Retriever", "salience", "CorpusGraph",
           "in_domain", "out_of_domain", "OUT_OF_DOMAIN_TERMS",
           "diagnose_chain", "diagnose", "DECLINE", "DECLINE_DOMAIN"]
