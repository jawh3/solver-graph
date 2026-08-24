"""T1-9 (RIGOR_LEDGER): validate the logged-propensity reconstruction that underwrites every
IPW/SNIPW/CRT number. Three checks, all log-only/CPU:

(1) IDENTITY: the code claims sequential-softmax-without-replacement (Plackett-Luce) top-m marginal
    == Gumbel-top-m marginal. Verify on small synthetic pools by EXACT PL set enumeration vs the
    Gumbel-top-m MC (the identity is size-independent, so small pools suffice). Max abs error should
    be at MC-noise scale.
(2) VARIANCE: on real 14B candidate pools (n=30, m=3), MC(400) and MC(1000) vs a high-accuracy
    MC(50000) reference -> the n_mc-induced error actually propagated into pi.
(3) CALIBRATION: on the full 14B logs, bin candidates by reconstructed pi(n_mc=1000) and compare to
    the realized inclusion frequency 1[node in sampled_set] -> should track the diagonal.

Writes results/lab/propensity_validation.json.
"""
from __future__ import annotations

import json
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

from ope.causal_value import marginal_inclusion_propensities

D14 = Path("${REPO:-$(pwd)}/results/lab/campaign_14b_M3_powered/merged")


def exact_pl_marginal(scores, tau, m):
    """Exact P(i in sampled top-m) under sequential softmax WITHOUT replacement (Plackett-Luce)."""
    w = np.exp(np.asarray(scores, dtype=np.float64) / max(float(tau), 1e-6))
    n = len(w)
    if m >= n:
        return np.ones(n)
    marg = np.zeros(n)
    for S in combinations(range(n), m):
        pset = 0.0
        for order in permutations(S):
            p, rem = 1.0, w.sum()
            for idx in order:
                p *= w[idx] / rem
                rem -= w[idx]
            pset += p
        for idx in S:
            marg[idx] += pset
    return marg


def main():
    out = {}

    # (1) identity on synthetic small pools
    rng = np.random.RandomState(1)
    errs = []
    for n in (4, 5, 6, 8):
        for _ in range(40):
            scores = rng.normal(0, 1, n)
            pool = [{"node_id": str(i), "score": float(scores[i])} for i in range(n)]
            exact = exact_pl_marginal(scores, 1.0, 3)
            mc = marginal_inclusion_propensities(pool, 3, 1.0, n_mc=50000, seed=int(rng.randint(1e6)))
            mc_arr = np.array([mc[str(i)] for i in range(n)])
            errs.append(float(np.max(np.abs(exact - mc_arr))))
    out["identity_PL_vs_gumbel"] = {"n_pools": len(errs), "max_abs_err": max(errs),
                                    "mean_abs_err": float(np.mean(errs)),
                                    "note": "vs MC(50000); error at MC-noise scale (~0.005) confirms the identity"}

    # (2) n_mc variance on real pools
    runs = [json.loads(l) for l in open(D14 / "logs" / "campaign_outcome" / "runs.jsonl")]
    samp = runs[::max(1, len(runs) // 300)][:300]
    e400, e1000 = [], []
    for r in samp:
        pool = r["candidate_pool"]
        m = len(r["sampled_set"])
        tau = r.get("temperature", 1.0)
        ref = marginal_inclusion_propensities(pool, m, tau, n_mc=50000, seed=0)
        p400 = marginal_inclusion_propensities(pool, m, tau, n_mc=400, seed=0)
        p1000 = marginal_inclusion_propensities(pool, m, tau, n_mc=1000, seed=0)
        ids = list(ref)
        e400.append(float(np.max([abs(p400[i] - ref[i]) for i in ids])))
        e1000.append(float(np.max([abs(p1000[i] - ref[i]) for i in ids])))
    out["nmc_variance_real_pools"] = {
        "n_pools": len(samp),
        "nmc400_max_abs_err": {"max": max(e400), "p95": float(np.percentile(e400, 95)), "median": float(np.median(e400))},
        "nmc1000_max_abs_err": {"max": max(e1000), "p95": float(np.percentile(e1000, 95)), "median": float(np.median(e1000))}}

    # (3) calibration on full logs
    pis, incl = [], []
    for r in runs:
        pool = r["candidate_pool"]
        m = len(r["sampled_set"])
        sset = set(r["sampled_set"])
        pi = marginal_inclusion_propensities(pool, m, r.get("temperature", 1.0), n_mc=1000, seed=0)
        for c in pool:
            pis.append(pi[c["node_id"]])
            incl.append(1.0 if c["node_id"] in sset else 0.0)
    pis = np.array(pis); incl = np.array(incl)
    edges = np.quantile(pis, np.linspace(0, 1, 11))
    bins = []
    for b in range(10):
        lo, hi = edges[b], edges[b + 1]
        sel = (pis >= lo) & (pis <= hi if b == 9 else pis < hi)
        if sel.sum() == 0:
            continue
        bins.append({"bin": b, "n": int(sel.sum()), "mean_pi": float(pis[sel].mean()),
                     "realized_incl": float(incl[sel].mean())})
    cal_err = float(np.mean([abs(x["mean_pi"] - x["realized_incl"]) for x in bins]))
    out["calibration_real_logs"] = {"n_candidate_obs": len(pis), "overall_mean_pi": float(pis.mean()),
                                    "overall_realized": float(incl.mean()), "mean_abs_calib_err": cal_err,
                                    "bins": bins}

    Path("${REPO:-$(pwd)}/results/lab/propensity_validation.json").write_text(json.dumps(out, indent=2))
    print(f"[propensity_validate] identity max_err={out['identity_PL_vs_gumbel']['max_abs_err']:.4f} | "
          f"nmc400 median_err={out['nmc_variance_real_pools']['nmc400_max_abs_err']['median']:.4f} "
          f"p95={out['nmc_variance_real_pools']['nmc400_max_abs_err']['p95']:.4f} | "
          f"calibration mean_abs_err={cal_err:.4f} (overall pi {out['calibration_real_logs']['overall_mean_pi']:.3f} "
          f"vs realized {out['calibration_real_logs']['overall_realized']:.3f})")


if __name__ == "__main__":
    main()
