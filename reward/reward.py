"""
reward/reward.py — step-count reward with partial-pooled standardization (spec §9.1–9.2).

Reward (§9.1):
    o  ∈ {0,1}                 verified flag (the verifiable oracle)
    z_steps = (steps − μ_cell)/σ_cell      standardized within (category×difficulty)
    e  = −z_steps              efficiency (faster → higher)
    R  = o · (1 + λ·e)         λ = config.lam
        - fast success  (e>0) → R>1
        - slow success  (e<0) → R<1   ← the latent-poison down-weight
        - failure             → R=config.failure_reward (default 0)

Standardization (§9.2) uses PARTIAL POOLING so 1–3-run cells don't produce garbage
z-scores (the documented weak point that would make the bandit chase noise):
    μ_cell_hat = (n_cell·mean_cell + τ0·μ_global) / (n_cell + τ0)
    var_cell_hat analogous (shrink toward global var)
τ0 = config.tau0. This is a James-Stein / empirical-Bayes shrinkage; with n_cell≫τ0
the cell speaks for itself, with n_cell≈0 it borrows the global distribution.

Design citations: step-count efficiency / verifiable≠correct — SpuriousRewards
2601.11061, LLMsGamingVerifiers 2604.15149. Partial pooling for sparse cells —
standard hierarchical-Bayes shrinkage (ErrorBarsToEvals 2411.00640 motivates the
variance discipline).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from config import SolverConfig


@dataclass
class RewardBreakdown:
    outcome: int
    steps: int
    cell: str
    mu_cell: float
    sigma_cell: float
    z_steps: float
    efficiency_e: float
    R_final: float
    n_cell: int


class RewardStandardizer:
    """Accumulates step observations per (category×difficulty) and computes R.

    Stateful: feed it every run's (category, difficulty, steps) via `observe`,
    then `reward(...)` uses the current pooled stats. In an eval, observe the
    whole dataset's step distribution first (or update online) — both are
    supported. Serializable so an experiment's standardization is reproducible.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        # per-cell sufficient statistics
        self._n: Dict[str, int] = defaultdict(int)
        self._sum: Dict[str, float] = defaultdict(float)
        self._sumsq: Dict[str, float] = defaultdict(float)
        # global sufficient statistics
        self._gn = 0
        self._gsum = 0.0
        self._gsumsq = 0.0

    @staticmethod
    def cell_key(category: Optional[str], difficulty: Optional[str]) -> str:
        c = (str(category) if category else "unknown").strip().lower()
        d = (str(difficulty) if difficulty else "unknown").strip().lower()
        return f"{c}×{d}"

    def observe(self, category: Optional[str], difficulty: Optional[str], steps: float) -> None:
        key = self.cell_key(category, difficulty)
        self._n[key] += 1
        self._sum[key] += steps
        self._sumsq[key] += steps * steps
        self._gn += 1
        self._gsum += steps
        self._gsumsq += steps * steps

    # ------------------------------------------------------------------
    def _global_mean_var(self) -> Tuple[float, float]:
        if self._gn == 0:
            return 0.0, 1.0
        mean = self._gsum / self._gn
        var = max(self._gsumsq / self._gn - mean * mean, 0.0)
        if var <= 0:
            var = 1.0
        return mean, var

    def pooled_stats(self, category: Optional[str], difficulty: Optional[str]) -> Tuple[float, float, int]:
        """Partial-pooled (μ̂, σ̂, n_cell) for a cell (§9.2)."""
        key = self.cell_key(category, difficulty)
        g_mean, g_var = self._global_mean_var()
        n = self._n.get(key, 0)
        tau0 = self.config.tau0
        if n == 0:
            return g_mean, math.sqrt(g_var), 0
        cell_mean = self._sum[key] / n
        cell_var = max(self._sumsq[key] / n - cell_mean * cell_mean, 0.0)
        mu_hat = (n * cell_mean + tau0 * g_mean) / (n + tau0)
        var_hat = (n * cell_var + tau0 * g_var) / (n + tau0)
        sigma_hat = math.sqrt(max(var_hat, 1e-9))
        if sigma_hat <= 1e-6:
            sigma_hat = math.sqrt(g_var)
        return mu_hat, sigma_hat, n

    # ------------------------------------------------------------------
    def reward(self, outcome: int, steps: float,
               category: Optional[str], difficulty: Optional[str]) -> RewardBreakdown:
        """Compute R = o·(1+λe) with partial-pooled standardization."""
        cfg = self.config
        mu, sigma, n = self.pooled_stats(category, difficulty)
        z = (steps - mu) / sigma if sigma > 0 else 0.0
        e = -z
        if outcome:
            R = 1.0 + cfg.lam * e
        else:
            R = cfg.failure_reward
        return RewardBreakdown(
            outcome=int(outcome), steps=int(steps),
            cell=self.cell_key(category, difficulty),
            mu_cell=mu, sigma_cell=sigma, z_steps=z,
            efficiency_e=e, R_final=R, n_cell=n,
        )

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "n": dict(self._n), "sum": dict(self._sum), "sumsq": dict(self._sumsq),
            "gn": self._gn, "gsum": self._gsum, "gsumsq": self._gsumsq,
        }

    @classmethod
    def from_dict(cls, data: dict, config: Optional[SolverConfig] = None) -> "RewardStandardizer":
        s = cls(config)
        s._n = defaultdict(int, data.get("n", {}))
        s._sum = defaultdict(float, data.get("sum", {}))
        s._sumsq = defaultdict(float, data.get("sumsq", {}))
        s._gn = data.get("gn", 0)
        s._gsum = data.get("gsum", 0.0)
        s._gsumsq = data.get("gsumsq", 0.0)
        return s


def compute_reward(outcome: int, steps: float, category: Optional[str],
                   difficulty: Optional[str], standardizer: RewardStandardizer) -> RewardBreakdown:
    """Convenience wrapper around a fitted standardizer."""
    return standardizer.reward(outcome, steps, category, difficulty)
