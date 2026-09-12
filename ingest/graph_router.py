# AI use statement: docs/AI_USE.md
##
# @file graph_router.py
# @brief Graph routing — §4.4 bullet 1.
#
"""
Graph routing — §4.4 bullet 1.

Builds an overlapping symptom -> subsystem -> component -> procedure
graph directly from the manuals' own structure (malfunction procedures
are already symptom-indexed). Routing *widens toward* a region of the
corpus; it never partitions/excludes, so causal chains that span
multiple subsystems stay reachable.
"""
import heapq
import math
import networkx as nx
from schema import Segment
from bm25_index import tokenize


class SymptomGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()

    def add_segment(self, seg: Segment):
        self.g.add_node(seg.id, kind="procedure", segment=seg)
        if seg.subsystem:
            self.g.add_node(seg.subsystem, kind="subsystem")
            self.g.add_edge(seg.subsystem, seg.id, kind="has_procedure")
        if seg.component:
            self.g.add_node(seg.component, kind="component")
            self.g.add_edge(seg.component, seg.id, kind="has_procedure")
            if seg.subsystem:
                self.g.add_edge(seg.subsystem, seg.component, kind="has_component")
        for symptom in seg.symptoms:
            snode = f"symptom::{symptom.lower()}"
            self.g.add_node(snode, kind="symptom", text=symptom)
            self.g.add_edge(snode, seg.id, kind="indicates")
            if seg.subsystem:
                self.g.add_edge(snode, seg.subsystem, kind="associated_with")
        for other_id in seg.links_to:
            self.g.add_edge(seg.id, other_id, kind="related_to")

    def build(self, segments: list[Segment]):
        for seg in segments:
            self.add_segment(seg)
        return self

    def _match_symptom_nodes(self, query: str) -> list[str]:
        """Token-overlap match from a free-text symptom report to indexed symptom nodes.
        Placeholder for a proper embedding/alias match — swap in without changing callers."""
        qtoks = set(tokenize(query))
        matches = []
        for n, d in self.g.nodes(data=True):
            if d.get("kind") != "symptom":
                continue
            overlap = len(qtoks & set(tokenize(d["text"])))
            if overlap:
                matches.append((n, overlap))
        matches.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in matches]

    EDGE_COST = {"related_to": -math.log(0.6)}  # everything else defaults to -log(0.8) below
    DEFAULT_COST = -math.log(0.8)

    def route(self, query: str, k: int = 20, max_expansions: int = 400,
              anomalous_subsystems: set[str] = None, exhaustive: bool = False) -> list[tuple[Segment, float]]:
        """A* from matched symptom nodes to procedure nodes. Cost is additive
        (-log of the old per-hop decay, so cheapest path = highest combined
        probability, same semantics as before). Heuristic: 0 for nodes inside
        a currently-anomalous subsystem (from telemetry.flag_anomalous_subsystems),
        a fixed penalty otherwise — this only REORDERS the frontier so the
        relevant region pops first; it never excludes a node from eventually
        being reached. Completeness (R8) is unaffected: with exhaustive=True
        the search runs until the frontier is empty, same reachable set as
        plain BFS, just visited in a better order. Default (exhaustive=False)
        stops early once `k` procedures are found — that's a pagination
        convenience for the acute regime (§4.2), not a correctness cut;
        call again with a larger `k` or exhaustive=True for full coverage
        (the deliberate regime's "next page")."""
        seeds = self._match_symptom_nodes(query)
        if not seeds:
            return []

        anomalous_subsystems = anomalous_subsystems or set()
        OFF_PATH_PENALTY = 1.0  # in cost units (~1 extra hop) — a tiebreaker, not a wall

        def heuristic(node: str) -> float:
            if not anomalous_subsystems:
                return 0.0
            return 0.0 if self._in_subsystem(node, anomalous_subsystems) else OFF_PATH_PENALTY

        best_cost: dict[str, float] = {s: 0.0 for s in seeds}
        procedure_scores: dict[str, float] = {}
        frontier = [(heuristic(s), s) for s in seeds]
        heapq.heapify(frontier)
        expansions = 0
        cap = math.inf if exhaustive else max_expansions

        while frontier and expansions < cap:
            _, node = heapq.heappop(frontier)
            expansions += 1
            g_node = best_cost[node]

            if self.g.nodes[node].get("kind") == "procedure":
                # accumulate: multiple paths to the same procedure reinforce it,
                # same as the old BFS's additive scoring.
                procedure_scores[node] = procedure_scores.get(node, 0.0) + math.exp(-g_node)
                if not exhaustive and len(procedure_scores) >= k:
                    break

            for _, nbr, edata in self.g.out_edges(node, data=True):
                step_cost = self.EDGE_COST.get(edata.get("kind"), self.DEFAULT_COST)
                g_nbr = g_node + step_cost
                if g_nbr < best_cost.get(nbr, math.inf):
                    best_cost[nbr] = g_nbr
                    heapq.heappush(frontier, (g_nbr + heuristic(nbr), nbr))

        ranked = sorted(procedure_scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.g.nodes[nid]["segment"], score) for nid, score in ranked[:k]]

    def _in_subsystem(self, node: str, subsystems: set[str]) -> bool:
        """True if `node`'s segment (procedure) or component belongs to one of
        the given subsystems — cheap attribute check, no traversal."""
        d = self.g.nodes[node]
        seg = d.get("segment")
        if seg is not None:
            return seg.subsystem in subsystems
        return False
