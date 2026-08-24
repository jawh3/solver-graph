"""
ope/sweep.py — offline policy sweep from logged runs (spec §13, §16 C8).

Consumes JSONL RunRecords (the logging policy's runs) and ranks counterfactual
policies π′ over a grid of (τ, κ, M′) using IPS/SNIPS + DR with bootstrap CIs —
BEFORE committing any real-run sweep on the scarce GPU. The sweep runner emits a
ranked table; the top candidates are the only ones worth spending a coordinated
vLLM event on.

Sanity check baked in (BUILD_CHECKLIST C8 acceptance): the OPE estimate of the
LOGGING policy itself (τ=logged, κ=None) should ≈ its measured mean reward.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np

from instrumentation.jsonl_logger import load_runs, iter_jsonl
from .ope import ips_estimate, dr_estimate, OPEResult


def measured_value(runs: List[Dict]) -> float:
    """The on-policy mean reward of the logged runs (the OPE sanity target)."""
    if not runs:
        return 0.0
    return float(np.mean([float(r.get("R_final", 0.0)) for r in runs]))


def sanity_check(runs: List[Dict], tol: float = 0.15) -> Dict:
    """OPE of the logging policy ≈ its measured value (DR should be close)."""
    measured = measured_value(runs)
    dr = dr_estimate(runs, tau=None, kappa=None)        # logging policy
    snips = ips_estimate(runs, tau=None, kappa=None, self_normalized=True)
    return {
        "measured": measured,
        "dr_estimate": dr.value,
        "snips_estimate": snips.value,
        "dr_abs_error": abs(dr.value - measured),
        "passes": abs(dr.value - measured) <= tol,
    }


def offline_sweep(
    runs: List[Dict],
    taus: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    kappas: Sequence[Optional[float]] = (1.0, 5.0, 10.0),
    m_primes: Sequence[Optional[int]] = (None,),
    use_bandit: bool = True,
    estimator: str = "dr",
    seed: int = 0,
) -> List[Dict]:
    """Grid OPE over (τ, κ, M′). Returns rows sorted by estimated value desc."""
    rows: List[Dict] = []
    for tau, kappa, mp in itertools.product(taus, kappas, m_primes):
        if estimator == "dr":
            res = dr_estimate(runs, tau=tau, kappa=kappa, m_prime=mp,
                              use_bandit=use_bandit, seed=seed)
        else:
            res = ips_estimate(runs, tau=tau, kappa=kappa, m_prime=mp,
                               use_bandit=use_bandit, self_normalized=True, seed=seed)
        rows.append({
            "tau": tau, "kappa": kappa, "m_prime": mp,
            "value": res.value, "ci_low": res.ci_low, "ci_high": res.ci_high,
            "ess": res.ess, "mean_weight": res.mean_weight, "n": res.n,
            "estimator": res.estimator, "note": res.note,
        })
    rows.sort(key=lambda r: r["value"], reverse=True)
    return rows


def offline_sweep_from_log(experiment_id: str, root: str = "results/logs", **kwargs) -> Dict:
    """Load an experiment's RunRecords and run the sweep + sanity check."""
    runs = load_runs(experiment_id, root=root)
    return {
        "experiment_id": experiment_id,
        "n_runs": len(runs),
        "sanity": sanity_check(runs),
        "sweep": offline_sweep(runs, **kwargs),
    }


def _main():
    """CLI: offline-sweep every experiment under a logs root (no GPU)."""
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Offline OPE sweep over logged runs (§13)")
    ap.add_argument("--logs", default="results/logs",
                    help="Root holding <experiment_id>/runs.jsonl dirs.")
    ap.add_argument("--experiment", default=None,
                    help="Single experiment id; default = every dir under --logs.")
    ap.add_argument("--estimator", choices=["dr", "snips"], default="dr")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.logs)
    exp_ids = ([args.experiment] if args.experiment
               else [p.name for p in root.iterdir() if (p / "runs.jsonl").exists()])
    report = {}
    for eid in exp_ids:
        res = offline_sweep_from_log(eid, root=str(root), estimator=args.estimator)
        report[eid] = res
        s = res["sanity"]
        print(f"\n=== {eid}  (n={res['n_runs']}) ===")
        print(f"  OPE sanity: measured={s['measured']:.3f} DR={s['dr_estimate']:.3f} "
              f"|err|={s['dr_abs_error']:.3f} {'PASS' if s['passes'] else 'FAIL'}")
        print(f"  {'tau':>5} {'kappa':>6} {'value':>8} {'95% CI':>20} {'ESS':>7}")
        for row in res["sweep"][:8]:
            ci = f"[{row['ci_low']:.3f},{row['ci_high']:.3f}]"
            print(f"  {row['tau']:>5} {str(row['kappa']):>6} {row['value']:>8.3f} "
                  f"{ci:>20} {row['ess']:>7.1f}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nWrote sweep report to {args.out}")


if __name__ == "__main__":
    _main()
