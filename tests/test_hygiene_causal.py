"""Tests for C9: graph hygiene (retraction/decay) + LOO causal credit."""

import numpy as np

from config import SolverConfig
from knowledge_graph.store import KnowledgeGraphStore
from knowledge_graph.models import Node, NodeType
from retrieval.embedder import MockEmbedder
from bandit.bandit import TabularBandit
from bandit.hygiene import GraphHygiene
from pipeline.causal_credit import (
    loo_marginal_value, false_lesson_exposure_rate, identify_suspicious_nodes,
    mock_solve_fn,
)


def _store_with(node_id, tier="trusted"):
    store = KnowledgeGraphStore()
    emb = MockEmbedder(dim=16, seed=1)
    store.add_node(Node(id=node_id, type=NodeType.TOOL,
                        attributes={"canonical_name": node_id, "tier": tier, "text": node_id}),
                   emb.embed(node_id))
    return store


def test_persistently_bad_node_is_demoted_not_deleted():
    cfg = SolverConfig().with_overrides(theta_retract=-0.1, k_retract=5, use_baseline_b=False, alpha=0.5)
    store = _store_with("bad_tool")
    bandit = TabularBandit(cfg)
    # repeated failures drive Q below θ over many exposures
    for _ in range(10):
        bandit.update("ctx", "bad_tool", reward=-0.5, similarity=0.5, success=False)
    hy = GraphHygiene(cfg)
    events = hy.sweep(store, bandit)
    assert any(e.node_id == "bad_tool" for e in events)
    # demoted trusted -> provisional, NOT deleted
    assert store.get_node("bad_tool") is not None
    assert store.get_node("bad_tool").attributes["tier"] == "provisional"


def test_cited_failure_rate_triggers_retraction():
    cfg = SolverConfig().with_overrides(k_retract=4, cited_failure_retract_rate=0.6,
                                        theta_retract=-999)  # disable Q path
    store = _store_with("flaky")
    bandit = TabularBandit(cfg)
    hy = GraphHygiene(cfg)
    for _ in range(5):
        bandit.update("ctx", "flaky", reward=0.0, similarity=0.5, success=False)
        hy.observe_citation_outcome("flaky", success=False)
    events = hy.sweep(store, bandit)
    assert any(e.node_id == "flaky" for e in events)


def test_graph_quality_snapshot():
    cfg = SolverConfig()
    store = _store_with("a", tier="trusted")
    emb = MockEmbedder(dim=16, seed=1)
    store.add_node(Node(id="b", type=NodeType.TOOL, attributes={"tier": "provisional", "text": "b"}),
                   emb.embed("b"))
    bandit = TabularBandit(cfg)
    bandit.update("ctx", "a", reward=1.0, similarity=0.5, success=True)
    q = GraphHygiene(cfg).graph_quality(store, bandit)
    assert q["tier_counts"]["trusted"] == 1
    assert q["tier_counts"]["provisional"] == 1


def test_loo_detects_helpful_node_has_marginal_value():
    challenge = {"name": "rev1", "category": "rev",
                 "_world": {"solution_tools": ["r2"], "steps_base": 12, "p_solve_base": 0.3}}
    helpful = Node(id="t_r2", type=NodeType.TOOL,
                   attributes={"canonical_name": "r2", "invocation_signature": "r2"})
    mv = loo_marginal_value(challenge, [helpful], helpful,
                            solve_fn=mock_solve_fn(0), seeds=list(range(40)))
    # a genuinely helpful node should have positive marginal solve OR step savings
    assert mv.marginal_solve > 0 or mv.marginal_steps > 0
    assert not mv.is_false_lesson


def test_loo_flags_noncausal_node_as_false_lesson():
    # a distractor not in the solution should have ~0 marginal value -> false lesson
    challenge = {"name": "pwn1", "category": "pwn",
                 "_world": {"solution_tools": ["ropper"], "steps_base": 8, "p_solve_base": 0.5}}
    distractor = Node(id="t_x", type=NodeType.TOOL,
                      attributes={"canonical_name": "steghide", "invocation_signature": "steghide"})
    mv = loo_marginal_value(challenge, [distractor], distractor,
                            solve_fn=mock_solve_fn(100), seeds=list(range(40)),
                            false_lesson_eps=0.06)
    assert mv.is_false_lesson
    rate = false_lesson_exposure_rate([mv])
    assert rate["exposure_rate"] == 1.0


def test_identify_suspicious_high_q_nodes():
    cfg = SolverConfig().with_overrides(use_baseline_b=False, alpha=0.5)
    bandit = TabularBandit(cfg)
    for _ in range(6):
        bandit.update("c", "hot", reward=1.5, similarity=0.5, success=True)
        bandit.update("c", "cold", reward=0.0, similarity=0.5, success=False)
    suspects = identify_suspicious_nodes(bandit, q_high=0.5, min_n=3)
    assert "hot" in suspects
    assert "cold" not in suspects
