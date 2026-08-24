"""
bandit/hygiene.py — retraction & decay of false lessons (spec §10, C9).

Bounds graph growth AND removes confirmed false lessons. A node is a retraction
candidate when, in a context, its learned Q stays below `theta_retract` across
≥ `k_retract` exposures, OR its cited-failure rate exceeds
`cited_failure_retract_rate`. Demotion is staged and NEVER a hard delete (keep for
provenance/audit, §10):

    trusted  →  provisional  →  archived

`graph_quality_over_time` is reported as a first-class metric (§14.3): mean Q of
trusted nodes, count by tier, and #retractions — tracked across snapshots so the
paper can show the graph getting *cleaner*, not just bigger.

Design citation: forgetting/decay to remove poisoned experience — FadeMem
2601.18642, MemoryPoisoningDefense 2601.05504; never-hard-delete for audit
mirrors provenance (MemoryGraft 2512.16962).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import SolverConfig
from knowledge_graph.store import KnowledgeGraphStore
from bandit.bandit import TabularBandit

_TIER_ORDER = {"trusted": 2, "provisional": 1, "archived": 0}
_DEMOTE = {"trusted": "provisional", "provisional": "archived", "archived": "archived"}


@dataclass
class RetractionEvent:
    node_id: str
    context_id: str
    from_tier: str
    to_tier: str
    reason: str
    q: float
    n: float


class GraphHygiene:
    """Retraction/decay over the bandit cells + graph tiers."""

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        # cited-failure bookkeeping per node (across contexts)
        self._cited: Dict[str, int] = defaultdict(int)
        self._cited_fail: Dict[str, int] = defaultdict(int)

    def observe_citation_outcome(self, node_id: str, success: bool) -> None:
        """Call once per cited node per run to track cited-failure rate."""
        self._cited[node_id] += 1
        if not success:
            self._cited_fail[node_id] += 1

    def cited_failure_rate(self, node_id: str) -> float:
        c = self._cited.get(node_id, 0)
        return (self._cited_fail.get(node_id, 0) / c) if c else 0.0

    # ------------------------------------------------------------------
    def sweep(self, store: KnowledgeGraphStore, bandit: TabularBandit) -> List[RetractionEvent]:
        """Demote nodes that meet the retraction criteria. Returns the events."""
        cfg = self.config
        events: List[RetractionEvent] = []
        for context_id, nodes in bandit.cells.items():
            for node_id, cell in nodes.items():
                if node_id.startswith("run_"):
                    continue
                node = store.get_node(node_id)
                if node is None:
                    continue
                tier = node.attributes.get("tier", "trusted")
                if tier == "archived":
                    continue

                reason = None
                if cell.n >= cfg.k_retract and cell.Q < cfg.theta_retract:
                    reason = f"Q={cell.Q:.3f}<θ over n={cell.n:.0f} exposures"
                elif self.cited_failure_rate(node_id) >= cfg.cited_failure_retract_rate \
                        and self._cited.get(node_id, 0) >= cfg.k_retract:
                    reason = f"cited-failure rate={self.cited_failure_rate(node_id):.2f}"

                if reason:
                    new_tier = _DEMOTE[tier]
                    node.attributes["tier"] = new_tier
                    store.add_node(node, store.get_embedding(node_id))
                    events.append(RetractionEvent(
                        node_id=node_id, context_id=context_id, from_tier=tier,
                        to_tier=new_tier, reason=reason, q=cell.Q, n=cell.n))
        return events

    # ------------------------------------------------------------------
    def graph_quality(self, store: KnowledgeGraphStore,
                      bandit: Optional[TabularBandit] = None) -> Dict:
        """Graph-quality snapshot (§14.3): tier counts + mean trusted Q."""
        ids, _ = store.get_all_embeddings()
        by_tier = defaultdict(int)
        for nid in ids:
            node = store.get_node(nid)
            if node is None or node.attributes.get("is_run_anchor"):
                continue
            by_tier[node.attributes.get("tier", "trusted")] += 1
        mean_trusted_q = None
        if bandit is not None:
            qs = []
            for nodes in bandit.cells.values():
                for nid, cell in nodes.items():
                    node = store.get_node(nid)
                    if node and node.attributes.get("tier") == "trusted" and cell.n > 0:
                        qs.append(cell.Q)
            mean_trusted_q = (sum(qs) / len(qs)) if qs else None
        return {
            "tier_counts": dict(by_tier),
            "mean_trusted_Q": mean_trusted_q,
            "total_nodes": len(ids),
        }
