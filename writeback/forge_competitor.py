"""
writeback/forge_competitor.py - FORGE-style champion-broadcast memory (baseline BC, §14.2, C9).

FORGE (2605.16233) is the strongest external competitor: a self-evolving agent
memory that, per task family, keeps the BEST trajectory as a "champion" and
broadcasts it - adopted immediately and trusted, with no provisional gate. That is
precisely the design choice this project argues against (the birth-defect lever,
§11): an N=1 champion is trusted before its true success rate is known.

This class implements that competitor faithfully so the eval can contrast it
head-to-head with our GATED write-back (B5):
  - writes back only on SUCCESS,
  - keeps only the single fastest-success champion per category,
  - inserts it as tier=TRUSTED immediately (ungated broadcast), full weight,
  - replaces the champion whenever a faster success arrives.

If our gated system (B5) matches or beats BC while exposing fewer false lessons
(lower §14.3 exposure rate), that is the headline contribution: safe
self-consolidation without an oracle. If BC wins outright, that is an honest
negative the harness will surface.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from config import SolverConfig
from knowledge_graph.models import Node, NodeType
from knowledge_graph.store import KnowledgeGraphStore
from instrumentation.records import WriteBackRecord
from .writeback import MockTrajectoryDecomposer, TrajectoryDecomposer, node_id_for


class ForgeChampionWriteBack:
    """Champion-broadcast write-back (ungated, immediate-trust)."""

    def __init__(self, config: Optional[SolverConfig] = None,
                 decomposer: Optional[TrajectoryDecomposer] = None):
        self.config = config or SolverConfig()
        self.decomposer = decomposer or MockTrajectoryDecomposer()
        # category -> (best_steps, champion_node_id)
        self._champions: Dict[str, tuple] = {}

    def write_back(self, challenge: Dict, solve_result: Dict, run_id: str,
                   store: KnowledgeGraphStore, embedder) -> List[WriteBackRecord]:
        solved = bool(solve_result.get("flag"))
        if not solved:
            return []                       # FORGE broadcasts winners only
        steps = solve_result.get("rounds_used", 10**9)
        category = str(challenge.get("category", "unknown")).lower()

        best = self._champions.get(category)
        if best is not None and steps >= best[0]:
            return []                       # not a new champion -> nothing to broadcast

        # decompose; the champion summarizes the winning tools as ONE trusted node
        cands = self.decomposer.decompose(challenge, solve_result)
        tools = [c.node.attributes.get("canonical_name", "") for c in cands]
        text = f"CHAMPION {category} solve via {', '.join(t for t in tools if t)}"
        champ_id = node_id_for(NodeType.TECHNIQUE, f"forge::{category}")
        champ = Node(
            id=champ_id, type=NodeType.TECHNIQUE,
            attributes={
                "canonical_name": f"forge_champion_{category}",
                "invocation_signature": "|".join(t for t in tools if t) or category,
                "text": text, "category": category,
                "tier": "trusted",          # <-- ungated, immediate trust (the FORGE property)
                "source": "forge_champion",
                "applicability": {"categories": [category], "preconditions": [],
                                  "artifacts_required": [], "confidence": 1.0},
            },
        )
        store.add_node(champ, embedder.embed(text))
        self._champions[category] = (steps, champ_id)
        return [WriteBackRecord(
            run_id=run_id, new_node_id=champ_id, type=champ.type.value,
            derived_from_run=run_id, failure_derived=False,
            applicability=champ.attributes["applicability"],
            init_weight=self.config.optimistic_q0, tier="trusted")]

    def should_write_back(self, solved: bool) -> bool:
        return solved
