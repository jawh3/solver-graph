"""
writeback/writeback.py - trajectory decomposition + gated promotion (spec §9.4).

Two pieces:

1. TrajectoryDecomposer (mock+real): turn a finished run into typed nodes
   (§3.1 schema) with extracted `applicability` (§3.4) and a `derived_from`
   provenance edge. Mock = deterministic extraction from executed commands
   (the tools actually run); real = an LLM decomposition prompt (coordinated GPU).

2. WriteBackManager: inserts new nodes as `tier=provisional`, weight-capped, with
   provenance; tags failure-derived nodes; and runs the PROMOTION GATE.

Promotion gate (the birth-defect lever, §9.4): a provisional node is promoted to
`trusted` only after it is action-grounded-cited in ≥`k_promote` *successful* runs
spanning ≥`c_promote` *distinct* challenges. An N=1 lesson is therefore never
trusted prematurely. The "ungated" variant (B6) skips the gate so the eval can
show gating matters.

Mandatory comparison (§9.4): the full gated system (B5) must beat the
raw/episodic-only baseline (B3, no decomposition). If it does not, write-back is
unjustified (the Faulty-Memory null) - the eval harness flags this loudly.

Design citations: gated abstraction / episodic-only null - FaultyMemory
2605.12978, ExperienceFollowing 2505.16067; failure decomposition + champion
broadcast - FORGE 2605.16233; provenance against poisoned experience -
MemoryGraft 2512.16962, PoisonedRAG 2402.07867.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import SolverConfig, WritebackMode
from knowledge_graph.models import Edge, Node, NodeType
from knowledge_graph.store import KnowledgeGraphStore
from instrumentation.records import WriteBackRecord


_CMD_TOOL_RE = re.compile(r"^\s*([A-Za-z0-9_\-./]+)")
# tokens that are not real tools (shell noise) - never decompose these
_NON_TOOLS = {"echo", "cd", "ls", "cat", "true", "false", ":", "#"}


def node_id_for(node_type: NodeType, canonical_text: str) -> str:
    """Stable identity: sha1(type + canonical_text) (spec §3.3)."""
    blob = f"{node_type.value}\x00{canonical_text}".encode("utf-8")
    return f"{node_type.value}_{hashlib.sha1(blob).hexdigest()[:16]}"


@dataclass
class CandidateNode:
    node: Node
    text: str


class TrajectoryDecomposer(ABC):
    @abstractmethod
    def decompose(self, challenge: Dict, solve_result: Dict) -> List[CandidateNode]:
        ...


class MockTrajectoryDecomposer(TrajectoryDecomposer):
    """Deterministic, GPU-free decomposition from executed commands.

    Extracts the distinct tools actually invoked in the trajectory and emits one
    Tool node each, with `applicability` seeded from the challenge category. This
    mirrors what the real LLM decomposer should do but without a model, so the
    gated-promotion logic is fully testable offline.
    """

    def decompose(self, challenge: Dict, solve_result: Dict) -> List[CandidateNode]:
        category = str(challenge.get("category", "unknown")).lower()
        seen = set()
        out: List[CandidateNode] = []
        for rec in solve_result.get("trace", []) or []:
            for cmd in rec.get("commands", []) or []:
                m = _CMD_TOOL_RE.match(str(cmd))
                if not m:
                    continue
                tool = m.group(1).split("/")[-1]
                if tool in _NON_TOOLS or tool in seen or len(tool) < 2:
                    continue
                seen.add(tool)
                text = f"use {tool} for {category} challenges"
                node = Node(
                    id=node_id_for(NodeType.TOOL, text),
                    type=NodeType.TOOL,
                    attributes={
                        "canonical_name": tool,
                        "invocation_signature": tool,
                        "text": text,
                        "applicability": {
                            "categories": [category],
                            "preconditions": [],
                            "artifacts_required": [],
                            "confidence": 0.5,
                        },
                    },
                )
                out.append(CandidateNode(node=node, text=text))
        return out


class PromotionTracker:
    """Tracks provisional → trusted promotion state on node attributes (§9.4)."""

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()

    def register_citation(self, store: KnowledgeGraphStore, node_id: str,
                          challenge_id: str, success: bool, ungated: bool = False) -> bool:
        """Record that `node_id` was cited in a (successful?) run. Returns True if promoted now.

        Only successful, action-grounded-cited runs count toward promotion.
        """
        node = store.get_node(node_id)
        if node is None:
            return False
        if node.attributes.get("tier", "trusted") != "provisional":
            return False  # already trusted (or seed)

        if ungated:
            # B6: no gate - promote on first cited success (shows gating matters)
            if success:
                self._promote(store, node)
                return True
            return False

        if not success:
            return False
        promo = list(node.attributes.get("promo_challenges", []))
        if challenge_id not in promo:
            promo.append(challenge_id)
        cited_successes = int(node.attributes.get("promo_cited_successes", 0)) + 1
        node.attributes["promo_challenges"] = promo
        node.attributes["promo_cited_successes"] = cited_successes

        promoted = (cited_successes >= self.config.k_promote
                    and len(set(promo)) >= self.config.c_promote)
        if promoted:
            self._promote(store, node)
        else:
            # persist the updated counters
            store.add_node(node, store.get_embedding(node_id))
        return promoted

    def _promote(self, store: KnowledgeGraphStore, node: Node) -> None:
        node.attributes["tier"] = "trusted"
        store.add_node(node, store.get_embedding(node.id))


class WriteBackManager:
    """Inserts decomposed nodes (gated) and emits WriteBackRecords."""

    def __init__(self, config: Optional[SolverConfig] = None,
                 decomposer: Optional[TrajectoryDecomposer] = None):
        self.config = config or SolverConfig()
        self.decomposer = decomposer or MockTrajectoryDecomposer()
        self.promotion = PromotionTracker(self.config)

    def should_write_back(self, solved: bool) -> bool:
        mode = self.config.writeback_mode
        if mode is WritebackMode.OFF:
            return False
        if mode is WritebackMode.SUCCESSFUL:
            return solved
        return True  # BOTH

    def write_back(
        self,
        challenge: Dict,
        solve_result: Dict,
        run_id: str,
        store: KnowledgeGraphStore,
        embedder,
    ) -> List[WriteBackRecord]:
        """Decompose the trajectory and insert provisional nodes with provenance."""
        solved = bool(solve_result.get("flag"))
        if not self.should_write_back(solved):
            return []

        failure_derived = not solved
        records: List[WriteBackRecord] = []
        # provenance anchor: a run node so derived_from has a target
        run_node_id = f"run_{run_id}"
        if store.get_node(run_node_id) is None:
            store.add_node(
                Node(id=run_node_id, type=NodeType.CHALLENGE,
                     attributes={"text": f"run {run_id}", "source": "run_anchor",
                                 "is_run_anchor": True}),
                embedder.embed(f"run {run_id} {challenge.get('name','')}"),
            )

        for cand in self.decomposer.decompose(challenge, solve_result):
            existing = store.get_node(cand.node.id)
            if existing is not None and not existing.attributes.get("is_run_anchor"):
                # node already known; just add provenance, don't reset its tier
                self._add_provenance(store, cand.node.id, run_node_id)
                continue

            init_weight = self.config.optimistic_q0 if not failure_derived else 0.0
            cand.node.attributes.update({
                "tier": "provisional",
                "source": "writeback",
                "failure_derived": failure_derived,
                "init_weight": init_weight,
                "promo_challenges": [],
                "promo_cited_successes": 0,
            })
            store.add_node(cand.node, embedder.embed(cand.text))
            self._add_provenance(store, cand.node.id, run_node_id)

            records.append(WriteBackRecord(
                run_id=run_id, new_node_id=cand.node.id, type=cand.node.type.value,
                derived_from_run=run_id, failure_derived=failure_derived,
                applicability=cand.node.attributes.get("applicability", {}),
                init_weight=init_weight, tier="provisional",
            ))
        return records

    def _add_provenance(self, store: KnowledgeGraphStore, node_id: str, run_node_id: str) -> None:
        # derived_from(node -> run) provenance edge (§3.2)
        try:
            store.add_edge(Edge(source=node_id, target=run_node_id, weight=1.0,
                                attributes={"edge_type": "derived_from"}))
        except KeyError:
            pass
