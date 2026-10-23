"""
retrieval/sampling.py - sequential-without-replacement softmax sampling + propensities.

This is the propensity core (BUILD_CHECKLIST C2, MANDATORY/non-reconstructable).
The pitfall the checklist calls out: a plain per-item softmax is the WRONG
propensity and biases OPE. The correct object is the probability of the realized
*ordered* draw under sampling-without-replacement (a Plackett-Luce model):

    p_t = softmax(score/τ) over the candidates still remaining at draw t
    P(ordered set | c) = Π_t p_t,   set_logprob = Σ_t log p_t

Both the logging policy and any counterfactual policy π′ score the SAME ordered
draw, so the per-decision importance ratio P_{π'}/P_{π_log} is valid for IPS/DR
(§13). This module is reused verbatim by `ope/ope.py` to recompute P under π′,
which is why it is factored out of the retriever.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def softmax_temperature(scores: np.ndarray, tau: float) -> np.ndarray:
    """Numerically-stable softmax of scores/τ."""
    tau = max(tau, 1e-6)
    z = scores / tau
    z = z - np.max(z)
    e = np.exp(z)
    s = e.sum()
    if s <= 0:
        return np.ones_like(scores) / len(scores)
    return e / s


def sample_without_replacement(
    scores: Sequence[float],
    m: int,
    tau: float,
    rng: np.random.RandomState,
) -> Tuple[List[int], List[float], float]:
    """Sample m indices without replacement via sequential softmax.

    Returns:
        order:        the realized draw indices, in draw order
        per_draw_p:   conditional probability of each realized draw (aligned to order)
        set_logprob:  Σ log(per_draw_p) - log prob of the realized ordered draw
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    m = min(m, n)
    remaining = list(range(n))
    order: List[int] = []
    per_draw_p: List[float] = []
    set_logprob = 0.0

    for _ in range(m):
        sub = scores[remaining]
        probs = softmax_temperature(sub, tau)
        choice_local = int(rng.choice(len(remaining), p=probs))
        chosen = remaining[choice_local]
        p = float(probs[choice_local])
        order.append(chosen)
        per_draw_p.append(p)
        set_logprob += np.log(max(p, 1e-300))
        remaining.pop(choice_local)

    return order, per_draw_p, set_logprob


def stratified_design_marginals(n: int, k: int, m: int):
    """Exact inclusion marginals of the stratified exploration design (review 2026-09-04 #3).

    Design: each of the k on-target items is flipped in independently w.p. 1/2; if more than m
    are flipped in, a uniform m-subset of them is kept; the remaining m - kept slots are a uniform
    subset of the n - k other items. Returns (pi_on_target, pi_other), exact for every (n, k, m)
    including k >= m and small pools (the closed form (m - k/2)/(n - k) was wrong there)."""
    from math import comb
    m = min(m, n)
    if n <= 0:
        return 0.0, 0.0
    pi_ot = 0.0; e_need = 0.0
    for j in range(k + 1):                       # j = number of on-target items flipped in
        pj = comb(k, j) * 0.5 ** k
        kept = min(j, m)
        if k:
            pi_ot += pj * (kept / k)             # P(item i flipped) * P(kept | j flipped) = kept/k
        e_need += pj * min(m - kept, n - k)
    pi_rest = (e_need / (n - k)) if n > k else 0.0
    return pi_ot, pi_rest


def ordered_set_logprob(
    scores: Sequence[float],
    order: Sequence[int],
    tau: float,
) -> float:
    """Recompute Σ log p_t for a GIVEN ordered draw under a policy's scores/τ.

    Used by OPE: given the logged draw `order` and a counterfactual policy's
    per-candidate `scores` (recomputed for new κ/Q) and `tau`, this yields
    log P_{π'}(ordered set | c) WITHOUT re-running the solver.
    """
    scores = np.asarray(scores, dtype=np.float64)
    remaining = list(range(len(scores)))
    logp = 0.0
    for idx in order:
        if idx not in remaining:
            # policy could not have drawn this (e.g. node filtered out) -> -inf
            return float("-inf")
        sub = scores[remaining]
        probs = softmax_temperature(sub, tau)
        local = remaining.index(idx)
        logp += np.log(max(float(probs[local]), 1e-300))
        remaining.pop(local)
    return logp
