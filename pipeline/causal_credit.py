"""
pipeline/causal_credit.py — leave-one-out re-rollout causal credit (spec §9.3, §11, C9).

The oracle-free causal probe. Action-grounded citation (§9.3) tells us a lesson was
*used*; it does not prove the lesson was *causal*. To measure that without a human
oracle, we branch from a SHARED pre-lesson state and re-roll the solve WITH vs
WITHOUT the suspect node, crediting the **marginal** flag-reaching difference. A
node that is frequently credited yet has ~0 marginal value is a confirmed
false-lesson exposure (recovery masking). This populates the §14.3 false-lesson
exposure metric.

Gated to SUSPICIOUS nodes for cost (high Q AND high time-to-flag variance): those
are exactly where latent poison / recovery masking hide (§11). Re-rollouts are
expensive on the real path (each is a solve), so the suspicious-set gate keeps the
GPU budget bounded.

In mock mode the "shared pre-lesson state" is the fixed RNG seed; with/without the
node are two seeded mock solves whose only difference is the node's presence —
exactly the counterfactual. On the real path, the shared state is a checkpoint of
the agent/container just before the lesson would first be used.

Design citation: process-reward without human labels via re-rollout —
MathShepherd 2312.08935; marginal/segment credit — MemoryR2 2605.21768,
SegmentCredit 2605.27788, GraphCreditAssignment 2605.26684.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from config import SolverConfig
from knowledge_graph.models import Node
from bandit.bandit import TabularBandit


@dataclass
class MarginalValue:
    node_id: str
    challenge_id: str
    marginal_solve: float       # E[outcome | with] − E[outcome | without]
    marginal_steps: float       # E[steps | without] − E[steps | with]  (positive = node saves steps)
    n_rollouts: int
    is_false_lesson: bool       # credited but ~0 marginal causal value


def identify_suspicious_nodes(
    bandit: TabularBandit,
    q_high: float = 0.6,
    min_n: int = 3,
    time_to_flag_var: Optional[Dict[str, float]] = None,
    var_high: float = 4.0,
) -> List[str]:
    """Suspicious = high Q AND (high time-to-flag variance, if provided).

    `time_to_flag_var[node_id]` is the variance of steps across runs that cited
    the node (computed by the caller from RunRecords). When unavailable, the gate
    is high-Q-only (still a useful triage).
    """
    suspects = set()
    for nodes in bandit.cells.values():
        for nid, cell in nodes.items():
            if nid.startswith("run_") or cell.n < min_n:
                continue
            if cell.Q < q_high:
                continue
            if time_to_flag_var is None or time_to_flag_var.get(nid, 0.0) >= var_high:
                suspects.add(nid)
    return sorted(suspects)


def loo_marginal_value(
    challenge: Dict,
    sampled_nodes: List[Node],
    target_node: Node,
    solve_fn: Callable,
    seeds: List[int],
    false_lesson_eps: float = 0.05,
) -> MarginalValue:
    """Re-roll WITH vs WITHOUT `target_node`, averaged over `seeds`."""
    with_nodes = list(sampled_nodes)
    without_nodes = [n for n in sampled_nodes if n.id != target_node.id]

    o_with, o_without, s_with, s_without = [], [], [], []
    for seed in seeds:
        rw = solve_fn(challenge, with_nodes, seed)
        rwo = solve_fn(challenge, without_nodes, seed)
        o_with.append(1 if rw.get("flag") else 0)
        o_without.append(1 if rwo.get("flag") else 0)
        s_with.append(rw.get("rounds_used", 0))
        s_without.append(rwo.get("rounds_used", 0))

    marg_solve = float(np.mean(o_with) - np.mean(o_without))
    marg_steps = float(np.mean(s_without) - np.mean(s_with))
    is_false = abs(marg_solve) <= false_lesson_eps and marg_steps <= false_lesson_eps
    return MarginalValue(
        node_id=target_node.id, challenge_id=challenge.get("name", "?"),
        marginal_solve=marg_solve, marginal_steps=marg_steps,
        n_rollouts=len(seeds), is_false_lesson=is_false)


def false_lesson_exposure_rate(marginals: List[MarginalValue]) -> Dict:
    """§14.3 metric: fraction of credited nodes with ~0 marginal causal value."""
    if not marginals:
        return {"exposure_rate": 0.0, "n": 0, "mean_marginal_solve": 0.0}
    false = sum(1 for m in marginals if m.is_false_lesson)
    return {
        "exposure_rate": false / len(marginals),
        "n": len(marginals),
        "mean_marginal_solve": float(statistics.mean(m.marginal_solve for m in marginals)),
        "mean_marginal_steps": float(statistics.mean(m.marginal_steps for m in marginals)),
    }


def mock_solve_fn(seed_base: int = 0):
    """A solve_fn adapter over the mock solver for LOO in tests / offline analysis."""
    from solver.mock_solver import mock_solve_challenge

    def _fn(challenge: Dict, nodes: List[Node], seed: int) -> Dict:
        return mock_solve_challenge(challenge, sampled_nodes=nodes, seed=seed_base + seed)
    return _fn
