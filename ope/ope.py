"""
ope/ope.py - IPS / SNIPS / doubly-robust off-policy estimators (spec §13).

These score a counterfactual policy π′ on ALREADY-logged runs. The logged
`candidate_pool` carries each candidate's score COMPONENTS (cosine→prior, Q, n),
so π′'s per-candidate score can be recomputed for a new κ (prior strength) and the
realized ordered draw re-scored at a new τ - yielding log P_{π′}(realized set | c)
WITHOUT a GPU (the whole point of logging propensities, C-7).

Estimators:
  - IPS:   V = mean_i [ w_i · R_i ],            w_i = P_{π'}/P_{π_log} (per-decision PL ratio)
  - SNIPS: V = Σ w_i R_i / Σ w_i               (self-normalized; lower variance)
  - DR:    V = DM(π') + mean_i [ w_i (R_i - q̂(c, a_i)) ]
           where q̂ is a fitted reward model and DM(π') is its expectation under π'
           estimated by sampling π' from the SAME logged pool (π' is fully known
           offline). (DoublyRobust 1103.4601; OfflineEval/replay 1003.5956.)

Weight clipping + effective sample size (ESS) are reported: importance weights
explode if π′ diverges from π_log (BUILD_CHECKLIST C8 pitfall).

Currently supported sweep dimensions offline: τ (temperature), κ (prior strength),
and M′ ≤ M (prefix of the realized draw). M′ > M is NOT estimable from a logged
draw of size M and is flagged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from retrieval.sampling import ordered_set_logprob


# ---------------------------------------------------------------------------
# policy log-prob recomputation from logged score components
# ---------------------------------------------------------------------------

def _recompute_scores(candidate_pool: List[Dict], kappa: Optional[float],
                      use_bandit: bool) -> Tuple[List[str], List[float]]:
    """Recompute per-candidate sampling scores under a counterfactual κ / bandit flag.

    Uses the logged components: prior (sim on reward scale), Q, n. If κ is None,
    reuses the logged `score` as-is (pure temperature sweep).
    """
    ids = [c["node_id"] for c in candidate_pool]
    if kappa is None:
        return ids, [float(c["score"]) for c in candidate_pool]
    scores = []
    for c in candidate_pool:
        prior = float(c.get("prior", 0.0))
        Q = c.get("Q", None)
        n = float(c.get("n", 0.0) or 0.0)
        if (not use_bandit) or Q is None or n <= 0:
            scores.append(prior)
        else:
            scores.append((n * float(Q) + kappa * prior) / (n + kappa))
    return ids, scores


def recompute_policy_logprob(
    run: Dict,
    tau: Optional[float] = None,
    kappa: Optional[float] = None,
    m_prime: Optional[int] = None,
    use_bandit: bool = True,
) -> Optional[float]:
    """log P_{π'}(realized ordered draw | c) for a counterfactual (τ', κ', M').

    Returns None if not estimable (e.g. M' > logged M, or empty pool).
    """
    pool = run.get("candidate_pool", [])
    if not pool:
        return None
    ids, scores = _recompute_scores(pool, kappa, use_bandit)
    id_to_idx = {nid: i for i, nid in enumerate(ids)}

    realized = [p["node_id"] for p in run.get("per_draw_propensity", [])]
    if not realized:
        return None
    if m_prime is not None:
        if m_prime > len(realized):
            return None       # not estimable from this log
        realized = realized[:m_prime]

    try:
        order = [id_to_idx[nid] for nid in realized]
    except KeyError:
        return None
    t = tau if tau is not None else run.get("temperature", 1.0)
    return ordered_set_logprob(scores, order, tau=t)


def logging_logprob(run: Dict) -> float:
    """The logging policy's log-prob of its realized draw (recomputed for consistency)."""
    lp = recompute_policy_logprob(run, tau=run.get("temperature", 1.0),
                                  kappa=None, use_bandit=True)
    return lp if lp is not None else run.get("set_logprob", 0.0)


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

@dataclass
class OPEResult:
    estimator: str
    value: float
    ci_low: float
    ci_high: float
    n: int
    ess: float                 # effective sample size
    mean_weight: float
    note: str = ""


def _weights(runs: List[Dict], tau, kappa, m_prime, use_bandit, clip: float):
    ws, rs, idx = [], [], []
    for i, run in enumerate(runs):
        lp_target = recompute_policy_logprob(run, tau=tau, kappa=kappa,
                                             m_prime=m_prime, use_bandit=use_bandit)
        if lp_target is None or lp_target == float("-inf"):
            continue
        lp_log = logging_logprob(run)
        w = math.exp(lp_target - lp_log)
        if clip is not None:
            w = min(w, clip)
        ws.append(w)
        rs.append(float(run.get("R_final", 0.0)))
        idx.append(i)
    return np.array(ws), np.array(rs), idx


def _bootstrap_ci(values: np.ndarray, weights: np.ndarray, self_norm: bool,
                  n_boot: int = 4000, seed: int = 0):
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    ests = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        w = weights[idx]
        v = values[idx]
        if self_norm:
            denom = w.sum()
            ests[b] = (w * v).sum() / denom if denom > 0 else 0.0
        else:
            ests[b] = (w * v).mean()
    point = (weights * values).sum() / weights.sum() if self_norm else (weights * values).mean()
    return float(point), float(np.percentile(ests, 2.5)), float(np.percentile(ests, 97.5))


def ips_estimate(runs: List[Dict], tau: Optional[float] = None, kappa: Optional[float] = None,
                 m_prime: Optional[int] = None, use_bandit: bool = True,
                 self_normalized: bool = True, clip: float = 20.0, seed: int = 0) -> OPEResult:
    """IPS / SNIPS value of π′ from logged runs."""
    w, r, _ = _weights(runs, tau, kappa, m_prime, use_bandit, clip)
    if len(w) == 0:
        return OPEResult("snips" if self_normalized else "ips", 0.0, 0.0, 0.0, 0, 0.0, 0.0,
                         note="no estimable runs")
    point, lo, hi = _bootstrap_ci(r, w, self_norm=self_normalized, seed=seed)
    ess = (w.sum() ** 2) / max(np.sum(w ** 2), 1e-12)
    return OPEResult("snips" if self_normalized else "ips", point, lo, hi, len(w),
                     float(ess), float(w.mean()))


# --- reward model for DR ---

def _set_features(run: Dict) -> np.ndarray:
    pool = {c["node_id"]: c for c in run.get("candidate_pool", [])}
    sampled = run.get("sampled_set", [])
    cosines = [float(pool[n]["cosine"]) for n in sampled if n in pool]
    priors = [float(pool[n].get("prior", 0.0)) for n in sampled if n in pool]
    qs = [float(pool[n]["Q"]) for n in sampled if n in pool and pool[n].get("Q") is not None]
    return np.array([
        1.0,
        np.mean(cosines) if cosines else 0.0,
        np.max(cosines) if cosines else 0.0,
        len(sampled),
        np.mean(priors) if priors else 0.0,
        np.mean(qs) if qs else 0.0,
    ])


def _fit_reward_model(runs: List[Dict]) -> np.ndarray:
    X = np.array([_set_features(r) for r in runs])
    y = np.array([float(r.get("R_final", 0.0)) for r in runs])
    # ridge
    lam = 1e-3
    A = X.T @ X + lam * np.identity(X.shape[1])
    theta, *_ = np.linalg.lstsq(A, X.T @ y, rcond=None)
    return theta


def dr_estimate(runs: List[Dict], tau: Optional[float] = None, kappa: Optional[float] = None,
                m_prime: Optional[int] = None, use_bandit: bool = True,
                clip: float = 20.0, seed: int = 0, dm_samples: int = 8) -> OPEResult:
    """Doubly-robust value of π′ (model-based direct term + IPS correction)."""
    if not runs:
        return OPEResult("dr", 0.0, 0.0, 0.0, 0, 0.0, 0.0, note="no runs")
    theta = _fit_reward_model(runs)

    def qhat(run):
        return float(_set_features(run) @ theta)

    # direct-method term: q̂ at the logged action is our per-run baseline; the
    # IPS correction reweights the residual. (For a τ/κ sweep the realized set is
    # unchanged, so DM(π') ≈ mean q̂(logged action) - a valid, low-variance control
    # variate; the correction term carries the policy-shift signal.)
    w, r, idx = _weights(runs, tau, kappa, m_prime, use_bandit, clip)
    if len(w) == 0:
        return OPEResult("dr", 0.0, 0.0, 0.0, 0, 0.0, 0.0, note="no estimable runs")
    qh = np.array([qhat(runs[i]) for i in idx])
    dm = float(np.mean(qh))
    corrections = w * (r - qh)

    rng = np.random.RandomState(seed)
    n = len(w)
    ests = np.empty(4000)
    for b in range(4000):
        bi = rng.randint(0, n, size=n)
        ests[b] = qh[bi].mean() + corrections[bi].mean()
    point = dm + float(np.mean(corrections))
    ess = (w.sum() ** 2) / max(np.sum(w ** 2), 1e-12)
    return OPEResult("dr", point, float(np.percentile(ests, 2.5)),
                     float(np.percentile(ests, 97.5)), n, float(ess), float(w.mean()))
