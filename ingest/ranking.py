# AI use statement: docs/AI_USE.md
##
# @file ranking.py
# @brief Ranking without inference rounds.
#
"""
Ranking without inference rounds.

- Reciprocal Rank Fusion (RRF): combines graph/dense/BM25 rankings using
  only rank position, not raw scores — no LLM call, robust to the wildly
  different score scales of lexical vs. semantic vs. graph-proximity.
- Centrality prior: a per-segment importance score computed ONCE at
  ingest (graph PageRank), not per query. Segments central to many
  symptom/procedure chains get a small boost — this is where "cognitive
  load" belongs: paid once, amortized over every future query, never
  during the acute-regime clock.
"""
import json
from pathlib import Path
import networkx as nx
from schema import Segment

RRF_K = 60  # standard constant (Cormack et al.); dampens the impact of rank-1 outliers


def rrf_fuse(*ranked_lists: list[Segment], weights: list[float] = None) -> list[tuple[Segment, float]]:
    """Each arg is a ranked (best-first) list of Segments from one retriever.
    No scores needed — only position. Zero model calls."""
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    by_id: dict[str, Segment] = {}
    for ranked, w in zip(ranked_lists, weights):
        for rank, seg in enumerate(ranked):
            by_id[seg.id] = seg
            scores[seg.id] = scores.get(seg.id, 0.0) + w / (RRF_K + rank + 1)
    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [(by_id[i], scores[i]) for i in ranked_ids]


def compute_centrality_priors(segments: list[Segment], graph) -> dict[str, float]:
    """Ingest-time only. PageRank over the fault/symptom graph (graph_router.SymptomGraph.g),
    normalized to [0, 0.1] so it nudges ties rather than dominating retrieval relevance."""
    pr = nx.pagerank(graph.g.to_undirected() if graph.g.is_multigraph() else graph.g)
    proc_scores = {nid: v for nid, v in pr.items() if nid in {s.id for s in segments}}
    if not proc_scores:
        return {s.id: 0.0 for s in segments}
    lo, hi = min(proc_scores.values()), max(proc_scores.values())
    span = (hi - lo) or 1.0
    return {sid: 0.1 * (v - lo) / span for sid, v in proc_scores.items()}


def save_priors(priors: dict[str, float], path: str = "./centrality_priors.json"):
    Path(path).write_text(json.dumps(priors))


def load_priors(path: str = "./centrality_priors.json") -> dict[str, float]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}
