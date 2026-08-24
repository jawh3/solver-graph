"""
evaluation/stats.py — paired bootstrap CIs + power analysis (spec §14.4).

The prior result was non-significant; this module is the discipline that prevents
a repeat. Three tools:

  1. paired_bootstrap_ci: the headline estimator. Compares two systems on the SAME
     challenges (paired / within-challenge, common random seeds) and bootstraps
     the CI of the mean paired difference. Paired design slashes variance vs
     between-subjects (PairedBootstrap 2511.19794; ErrorBarsToEvals 2411.00640).

  2. mcnemar_exact: for the binary solve/no-solve paired outcome, the exact
     McNemar test (discordant pairs) — a complement to the bootstrap for the
     solve-rate lift.

  3. power_analysis: BEFORE spending GPU, how many challenges×seeds are needed to
     detect a target effect (default 10pp) at the observed variance? (§14.4
     pre-registered power analysis.)

Everything is seeded; two runs with the same seed give identical CIs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PairedResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    n_pairs: int
    p_value_sign: float        # two-sided bootstrap p that mean_diff == 0
    significant: bool


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedResult:
    """Bootstrap CI of mean(a − b) over paired observations (a,b aligned).

    `a` = treatment per-challenge metric (e.g. solved∈{0,1} or steps), `b` =
    control, same challenges in the same order. Returns the paired difference CI.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert len(a) == len(b), "paired arrays must align"
    diffs = a - b
    n = len(diffs)
    if n == 0:
        return PairedResult(0.0, 0.0, 0.0, 0, 1.0, False)
    rng = np.random.RandomState(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot[i] = diffs[idx].mean()
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    mean_diff = float(diffs.mean())
    # two-sided bootstrap p: 2 * min(frac<=0, frac>=0)
    frac_le = float(np.mean(boot <= 0))
    frac_ge = float(np.mean(boot >= 0))
    p = min(1.0, 2 * min(frac_le, frac_ge))
    significant = not (lo <= 0 <= hi)
    return PairedResult(mean_diff, lo, hi, n, p, significant)


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> Dict[str, float]:
    """Exact McNemar test on paired binary outcomes (a,b ∈ {0,1}).

    Counts discordant pairs: b01 = a solved & b didn't, b10 = b solved & a didn't.
    Exact binomial two-sided p under H0: discordants split 50/50.
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    b01 = int(np.sum((a == 1) & (b == 0)))
    b10 = int(np.sum((a == 0) & (b == 1)))
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p_value": 1.0}
    k = min(b01, b10)
    # two-sided exact binomial
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(n, i) * (0.5 ** n)
    p = min(1.0, 2 * p)
    return {"b01": float(b01), "b10": float(b10), "p_value": float(p)}


def power_analysis(
    baseline_rate: float,
    target_lift: float = 0.10,
    paired_sd: Optional[float] = None,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, float]:
    """Required #challenges to detect `target_lift` at `power` (§14.4).

    If `paired_sd` (sd of per-challenge paired differences) is known from a pilot,
    uses it directly (paired t-style). Otherwise approximates the paired SD for a
    binary outcome assuming moderate discordance, which is conservative.

    Returns required n (challenges); multiply by #seeds for total runs.
    """
    # normal approximation z-values
    z_alpha = _z(1 - alpha / 2)
    z_beta = _z(power)
    if paired_sd is None:
        # binary paired difference variance upper bound:
        # treat treatment rate = baseline + lift; discordance drives variance.
        p1 = min(max(baseline_rate + target_lift, 0.0), 1.0)
        p0 = min(max(baseline_rate, 0.0), 1.0)
        # variance of per-pair difference ~ p1(1-p1)+p0(1-p0) (independent upper bound)
        var = p1 * (1 - p1) + p0 * (1 - p0)
        paired_sd = math.sqrt(max(var, 1e-6))
    n = ((z_alpha + z_beta) ** 2 * paired_sd ** 2) / max(target_lift ** 2, 1e-9)
    return {
        "required_challenges": math.ceil(n),
        "assumed_paired_sd": paired_sd,
        "target_lift": target_lift,
        "alpha": alpha, "power": power,
    }


def _z(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation; no scipy dependency)."""
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
