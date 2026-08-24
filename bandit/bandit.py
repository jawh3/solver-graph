"""
bandit/bandit.py — tabular cluster/category-conditioned value bandit (spec §6, §9.3).

This is the default bandit (C-1). Per `(context c, candidate v)` it stores `Q`
(estimated standardized reward) and `n` (effective count), sparsely. Three pieces:

  1. `score(c, v, prior)` — the sim-prior-blended sampling score (§6.2, C-2):
         score = (n*Q + kappa*prior) / (n + kappa)
     A new/unseen node (n=0) is scored by its similarity prior, NOT an arbitrary
     uniform weight — clean cold start (BUILD_CHECKLIST C2 acceptance (d)).

  2. `update(c, v, reward, distance_scale)` — the §9.3 credit rule:
         Q[c][v] += alpha * g(d_v) * ((R - b) - Q[c][v]); n[c][v] += 1
     applied ONLY to the credit-eligible set S = sampled ∩ cited (the caller
     enforces that; this class just updates whatever it is handed). `b` is the
     running mean reward in context c (variance reduction, §9.1).

  3. `sample_efficiency_budget(...)` — the §6.1 calc the build checklist (C6)
     requires recording BEFORE spending real GPU. If <5–10 updates/cell, the
     context granularity is too fine and will produce another null result.

Design citations:
  - Coarse contextual bandit + pseudo-count prior blend: PURPLE 2601.12078,
    LinUCB (Li et al. 1003.0146).
  - Distance-scaled / asymmetric credit to fight rich-get-richer: spec §9.3 A-04.
  - Optimistic init for forced exploration: JumpStartFalseStart 2604.02527.
  - Reward centering by baseline b: standard advantage/variance-reduction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from config import DistanceScaling, InitMode, SolverConfig


@dataclass
class BanditCell:
    """Q/n statistics for one (context, candidate) pair."""
    Q: float = 0.0
    n: float = 0.0


def sample_efficiency_budget(
    expected_solve_runs: int,
    M: int,
    num_context_nodes: int,
    avg_candidates_credited_per_run: float,
) -> Dict[str, float]:
    """The §6.1 budget check. Returns the expected updates/cell and a verdict.

    expected_updates_per_cell =
        (expected_solve_runs * avg_credited_per_run) / (num_context_nodes * mean_candidates_per_cell)

    The spec writes it as (runs*M)/(contexts*credited); but the quantity that
    actually accrues to a *cell* is the number of CREDITED candidate updates
    spread over context×candidate cells. We expose both the spec form and the
    credited form so the C6 dev log can record the honest number.
    """
    spec_form = (expected_solve_runs * M) / max(num_context_nodes * avg_candidates_credited_per_run, 1e-9)
    credited_total = expected_solve_runs * avg_candidates_credited_per_run
    # If candidates are shared across contexts, the per-cell number is lower;
    # spec_form is the optimistic upper bound used in §6.1.
    verdict = "OK" if spec_form >= 5 else ("MARGINAL" if spec_form >= 2 else "TOO_FINE")
    return {
        "expected_updates_per_cell": spec_form,
        "credited_updates_total": credited_total,
        "num_context_nodes": num_context_nodes,
        "verdict": verdict,
        "threshold": 5.0,
    }


class TabularBandit:
    """Sparse tabular value bandit keyed by (context_id, node_id)."""

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        # sparse dict-of-dicts: cells[context_id][node_id] = BanditCell
        self.cells: Dict[str, Dict[str, BanditCell]] = {}
        # running mean reward per context (baseline b, §9.1)
        self._b_sum: Dict[str, float] = {}
        self._b_count: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # cell access / init (§6.2 init modes)
    # ------------------------------------------------------------------
    def get_cell(self, context_id: str, node_id: str) -> BanditCell:
        ctx = self.cells.setdefault(context_id, {})
        if node_id not in ctx:
            ctx[node_id] = self._init_cell(node_id)
        return ctx[node_id]

    def _init_cell(self, node_id: str, seed_prior: Optional[float] = None) -> BanditCell:
        mode = self.config.init_mode
        if mode is InitMode.OPTIMISTIC:
            # optimistic value, n=0 so the prior still dominates until observed
            return BanditCell(Q=self.config.optimistic_q0, n=0.0)
        if mode is InitMode.SEED_WARM and seed_prior is not None:
            # warm start from a seed quality prior; alignment is logged elsewhere (§9.5)
            return BanditCell(Q=seed_prior, n=0.0)
        return BanditCell(Q=0.0, n=0.0)  # zero init

    def has_observations(self, context_id: str, node_id: str) -> bool:
        cell = self.cells.get(context_id, {}).get(node_id)
        return cell is not None and cell.n > 0

    # ------------------------------------------------------------------
    # scoring (§6.2, C-2 — keep similarity as a decaying prior)
    # ------------------------------------------------------------------
    def score(self, context_id: str, node_id: str, prior: float,
              provisional: bool = False) -> float:
        """Sim-prior-blended sampling score for (context, node).

        prior: similarity prior already mapped to the reward scale.
        score = (n*Q + kappa*prior) / (n + kappa)

        With n=0 (cold node) this collapses to `prior` — exactly the cold-start
        behavior the spec wants. Provisional write-back nodes are weight-capped
        (§9.4): their score is shrunk toward the prior so an unproven lesson can
        be sampled for exploration but cannot dominate.
        """
        cell = self.cells.get(context_id, {}).get(node_id)
        kappa = self.config.kappa
        if cell is None or cell.n == 0:
            q_contrib = self._init_value(node_id)  # init Q (e.g. optimistic) at n=0
            n = 0.0
        else:
            q_contrib = cell.Q
            n = cell.n
        blended = (n * q_contrib + kappa * prior) / (n + kappa)
        if provisional:
            cap = self.config.provisional_weight_cap
            # shrink the learned component toward the prior; cap its lift
            blended = prior + cap * (blended - prior)
        return blended

    def _init_value(self, node_id: str) -> float:
        if self.config.init_mode is InitMode.OPTIMISTIC:
            return self.config.optimistic_q0
        return 0.0

    # ------------------------------------------------------------------
    # baseline b (§9.1 variance reduction)
    # ------------------------------------------------------------------
    def baseline(self, context_id: str) -> float:
        if not self.config.use_baseline_b:
            return 0.0
        c = self._b_count.get(context_id, 0)
        if c == 0:
            return 0.0
        return self._b_sum[context_id] / c

    def observe_reward_for_baseline(self, context_id: str, reward: float) -> None:
        self._b_sum[context_id] = self._b_sum.get(context_id, 0.0) + reward
        self._b_count[context_id] = self._b_count.get(context_id, 0) + 1

    def set_q_table(self, table: Dict[str, Dict[str, float]], min_n: float = 1.0) -> None:
        """Overwrite Q from a batch estimate (the OBSERVATIONAL credit signal).

        `table[context_id][node_id] = q`. Sets n=min_n so the sampling blend
        (n*Q + κ*prior)/(n+κ) gives the learned value real weight while keeping the
        similarity prior influential for unseen nodes.
        """
        for ctx, nodes in table.items():
            cell_ctx = self.cells.setdefault(ctx, {})
            for nid, q in nodes.items():
                cell = cell_ctx.get(nid)
                if cell is None:
                    cell_ctx[nid] = BanditCell(Q=float(q), n=max(min_n, 1.0))
                else:
                    cell.Q = float(q); cell.n = max(cell.n, min_n)

    # ------------------------------------------------------------------
    # distance scaling g(d_v) (§9.3, A-04)
    # ------------------------------------------------------------------
    def distance_scale(self, similarity: float, success: bool) -> float:
        """g(d_v) where d_v = 1 - cosine_sim.

        symmetric:  scale by similarity for both success and failure
                    (high-sim citations credited/blamed more strongly).
        asymmetric: failure -> scale by similarity (blame near nodes more);
                    success -> scale by inverse similarity = d_v
                    (reward FAR nodes more, to fight rich-get-richer).
        Returned scale is clamped to [0.05, 1.0] so an update is never zeroed out.
        """
        sim = max(0.0, min(1.0, similarity))
        d = 1.0 - sim
        if self.config.distance_scaling is DistanceScaling.SYMMETRIC:
            scale = sim
        else:  # asymmetric
            scale = sim if not success else d
        return max(0.05, min(1.0, scale))

    # ------------------------------------------------------------------
    # the update rule (§9.3)
    # ------------------------------------------------------------------
    def update(
        self,
        context_id: str,
        node_id: str,
        reward: float,
        similarity: float,
        success: bool,
        subtract_baseline: bool = True,
    ) -> Tuple[float, float, float, float, float]:
        """Apply one credit update. Returns (pre_Q, post_Q, pre_n, post_n, scale).

        Q[c][v] += alpha * g(d_v) * ((reward - b) - Q[c][v]); n[c][v] += 1

        `subtract_baseline=False` for CAUSAL credit: the leave-one-out marginal is
        already an advantage (≈0 for a non-causal lesson), so centering it by the
        outcome baseline would wrongly push every lesson negative.
        """
        cell = self.get_cell(context_id, node_id)
        pre_Q, pre_n = cell.Q, cell.n
        b = self.baseline(context_id) if subtract_baseline else 0.0
        scale = self.distance_scale(similarity, success)
        target = reward - b
        cell.Q = cell.Q + self.config.alpha * scale * (target - cell.Q)
        cell.n = cell.n + 1.0
        return pre_Q, cell.Q, pre_n, cell.n, scale

    # ------------------------------------------------------------------
    # (de)serialization — stored alongside the graph, versioned per experiment (§4)
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_hash": self.config.hash(),
            "cells": {
                c: {v: {"Q": cell.Q, "n": cell.n} for v, cell in nodes.items()}
                for c, nodes in self.cells.items()
            },
            "b_sum": self._b_sum,
            "b_count": self._b_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Optional[SolverConfig] = None) -> "TabularBandit":
        b = cls(config=config)
        for c, nodes in data.get("cells", {}).items():
            b.cells[c] = {v: BanditCell(Q=cell["Q"], n=cell["n"]) for v, cell in nodes.items()}
        b._b_sum = dict(data.get("b_sum", {}))
        b._b_count = dict(data.get("b_count", {}))
        return b
