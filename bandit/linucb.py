"""
bandit/linucb.py - optional parametric contextual bandit (spec §6.3, [ABLATION]).

Hybrid LinUCB (Li et al., "A Contextual-Bandit Approach to Personalized News
Article Recommendation", 1003.0146). Context feature for arm (candidate node) v
given query q:

    x = [query_emb ⊕ candidate_emb ⊕ (query_emb * candidate_emb)]   # interactions

Maintains A = I + Σ x xᵀ and b = Σ x r. Ridge estimate θ̂ = A⁻¹ b. Selection
score (UCB):

    p(v) = θ̂ᵀx + alpha_ucb * sqrt(xᵀ A⁻¹ x)

This generalizes smoothly across nodes (no sparsity cliff), which is the reason
it is the spec's "likely target" - but it is built AFTER the tabular default
works and is a planned *comparison*, not the first cut (§6.3). The tabular bandit
remains the load-bearing default.

For OPE compatibility the same sequential-softmax sampling wrapper is reused
(retrieval/retriever.py) over LinUCB scores; this class only supplies the score.
"""

from __future__ import annotations

import numpy as np

from config import SolverConfig


def _features(query_emb: np.ndarray, cand_emb: np.ndarray) -> np.ndarray:
    """Hybrid feature vector [q ⊕ v ⊕ q*v]. L2-normalized inputs assumed."""
    q = np.asarray(query_emb, dtype=np.float64).ravel()
    v = np.asarray(cand_emb, dtype=np.float64).ravel()
    inter = q * v
    return np.concatenate([q, v, inter])


class LinUCBBandit:
    """Hybrid LinUCB over candidate-node arms."""

    def __init__(self, dim: int, config: SolverConfig | None = None,
                 alpha_ucb: float = 1.0):
        self.config = config or SolverConfig()
        self.feat_dim = 3 * dim          # q ⊕ v ⊕ q*v
        self.alpha_ucb = alpha_ucb
        self.A = np.identity(self.feat_dim)
        self.b = np.zeros(self.feat_dim)
        self._A_inv = np.identity(self.feat_dim)   # cached inverse
        self._dirty = False

    def _refresh_inv(self) -> None:
        if self._dirty:
            self._A_inv = np.linalg.inv(self.A)
            self._dirty = False

    def score(self, query_emb: np.ndarray, cand_emb: np.ndarray) -> float:
        """UCB score for one (query, candidate)."""
        self._refresh_inv()
        x = _features(query_emb, cand_emb)
        theta = self._A_inv @ self.b
        mean = float(theta @ x)
        bonus = self.alpha_ucb * float(np.sqrt(max(x @ self._A_inv @ x, 0.0)))
        return mean + bonus

    def update(self, query_emb: np.ndarray, cand_emb: np.ndarray, reward: float) -> None:
        """Online ridge update A += x xᵀ, b += r x."""
        x = _features(query_emb, cand_emb)
        self.A += np.outer(x, x)
        self.b += reward * x
        self._dirty = True

    def to_dict(self) -> dict:
        return {"feat_dim": self.feat_dim, "alpha_ucb": self.alpha_ucb,
                "A": self.A.tolist(), "b": self.b.tolist()}

    @classmethod
    def from_dict(cls, data: dict, config: SolverConfig | None = None) -> "LinUCBBandit":
        dim = data["feat_dim"] // 3
        obj = cls(dim=dim, config=config, alpha_ucb=data.get("alpha_ucb", 1.0))
        obj.A = np.array(data["A"])
        obj.b = np.array(data["b"])
        obj._dirty = True
        return obj
