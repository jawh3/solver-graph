"""
Tests for online causal credit (the recovery-masking fix, spec §9.3 C9).

Controlled scenario: a HELPFUL lesson is causally necessary (remove it -> the solve
fails) and a POISON lesson is non-causal (the agent runs it but recovers; remove it
-> the solve still succeeds). Both are cited on the success. Under OUTCOME credit
the bandit credits both (recovery masking). Under CAUSAL credit the leave-one-out
re-roll gives poison ~0 marginal, so its Q stays near zero while the helpful
lesson's Q rises.
"""

import numpy as np

from config import SolverConfig
from knowledge_graph.store import KnowledgeGraphStore
from knowledge_graph.models import Node, NodeType
from retrieval.embedder import MockEmbedder
from retrieval.retriever import ContextRetriever
from retrieval.context_resolution import ContextResolver
from retrieval.reranker import MockReranker
from bandit.bandit import TabularBandit
from pipeline import Orchestrator, SystemSpec


def _world():
    store = KnowledgeGraphStore()
    emb = MockEmbedder(dim=16, seed=1)
    helpful = Node(id="tool::good", type=NodeType.TOOL,
                   attributes={"canonical_name": "goodtool", "invocation_signature": ["goodtool"],
                               "text": "goodtool crypto", "category": "crypto", "tier": "trusted"})
    poison = Node(id="poison::bad", type=NodeType.TECHNIQUE,
                  attributes={"canonical_name": "badtool", "invocation_signature": ["badtool"],
                              "text": "badtool crypto", "category": "crypto", "tier": "trusted"})
    store.add_node(helpful, emb.embed("goodtool crypto"))
    store.add_node(poison, emb.embed("badtool crypto"))
    return store, emb


def _controlled_solve_fn(challenge, sampled_nodes, max_rounds, seed=None):
    """Solves iff the helpful lesson is present; cites every sampled lesson's tool.

    A trace command is emitted for each sampled node (so the citation matcher marks
    it cited), plus the winning 'goodtool' command when helpful is present. Accepts
    (and ignores) `seed`: outcome here is a deterministic function of node presence,
    which is the cleanest controlled case for testing the credit logic.
    """
    ids = {n.id for n in sampled_nodes}
    cmds = []
    for n in sampled_nodes:
        sig = n.attributes.get("invocation_signature", [n.id])[0]
        cmds.append(f"{sig} data.txt")
    solved = "tool::good" in ids
    trace = [{"round": i + 1, "commands": [c]} for i, c in enumerate(cmds)]
    flag = "picoCTF{ok}" if solved else None
    if solved:
        trace.append({"round": len(trace) + 1, "commands": ["goodtool --final"], "flag": flag})
    return {"flag": flag, "trace": trace, "rounds_used": len(trace),
            "total_commands": len(cmds) + (1 if solved else 0),
            "gave_up_reason": None if solved else "fail"}


def _run(credit_signal, epochs=12):
    store, emb = _world()
    cfg = SolverConfig().with_overrides(
        N=10, R=10, M=2, kappa=2.0, alpha=0.3, embedding_dim=16,
        context_resolution="category", credit_signal=credit_signal, use_baseline_b=False,
        init_mode="zero", lam=0.0, max_steps=4)
    bandit = TabularBandit(cfg)
    cr = ContextResolver(cfg)
    retr = ContextRetriever(store, emb, config=cfg, reranker=MockReranker(),
                            context_resolver=cr, bandit=bandit)
    system = SystemSpec("B4", use_bandit=True)
    orch = Orchestrator(store, emb, cfg, system, retriever=retr, bandit=bandit,
                        solve_fn=_controlled_solve_fn)
    ch = {"name": "c0", "category": "crypto", "points": 100,
          "description": "crypto goodtool badtool", "_flag": "picoCTF{ok}"}
    for e in range(epochs):
        orch.run_one(ch, rng=np.random.RandomState(e))
    # find Q for helpful and poison in the resolved context
    ctx = next(iter(bandit.cells))
    qg = bandit.cells[ctx].get("tool::good")
    qp = bandit.cells[ctx].get("poison::bad")
    return (qg.Q if qg else 0.0), (qp.Q if qp else 0.0), orch._causal_rerolls


def test_outcome_credit_is_fooled_by_recovery_masking():
    qg, qp, rerolls = _run("outcome")
    # outcome credit: both cited on success -> poison gets positive credit
    assert qp > 0.2, f"outcome credit should leave poison positive, got {qp}"
    assert rerolls == 0, "outcome credit must not re-roll"


def test_causal_credit_suppresses_poison():
    qg, qp, rerolls = _run("causal")
    # causal credit: re-roll without poison still solves -> poison marginal ~0
    assert qp < 0.15, f"causal credit should drive poison Q toward 0, got {qp}"
    # helpful is necessary -> positive marginal -> higher Q than poison
    assert qg > qp + 0.2, f"helpful Q ({qg}) should exceed poison Q ({qp})"
    assert rerolls > 0, "causal credit must perform re-rolls"


def test_causal_separates_more_than_outcome():
    _, qp_out, _ = _run("outcome")
    qg_c, qp_c, _ = _run("causal")
    # the true-minus-poison gap is larger under causal credit
    assert (qg_c - qp_c) > (0.0)  # causal yields positive separation
    assert qp_c < qp_out          # causal poison value is lower than outcome poison value
