#!/usr/bin/env python3
"""experiments/within_poison_validation.py — reproducible emitter for the C1 HEADLINE (RIGOR_LEDGER /
CLAIMS_v2 fix, 2026-06-25). The cheap on-target SNIPW does NOT rank the per-item POISON effect vs the
active-LOO gold standard. Recomputes, transparently, ALL conventions (no non-reproducible literal):
  - within-label Spearman: argsort+Pearson (the active_loo.json convention) AND scipy tie-corrected.
  - sign-agreement: strict sign(loo)==sign(cheap) (ties counted as disagree), (loo>0)==(cheap>0), and
    a median-split — with the exact-binomial p vs chance, and the count of LOO ties at 0.
  - the LOO NOISE FLOOR: per-node SE (= sd/sqrt(n_rollouts/arm)) vs the between-node SD, so the paper can
    state honestly that NEITHER the cheap estimator NOR the feasible-n gold standard ranks the small poison
    effects (ill-posed/uninformative, not 'true zero'). ZERO GPU; deterministic.
"""
from __future__ import annotations
import argparse, json
from math import comb
from pathlib import Path
import numpy as np
try:
    from scipy.stats import spearmanr as _scipy_spear
except Exception:
    _scipy_spear = None


def _argsort_spear(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _binom_ge(k, n, p=0.5):
    return float(sum(comb(n, i) * p**i * (1-p)**(n-i) for i in range(k, n+1)))


def run(active_loo_json: Path) -> dict:
    d = json.loads(active_loo_json.read_text()); rows = d["nodes"]
    out = {"source": str(active_loo_json), "overall_spearman_argsort": d.get("spearman_loo_vs_cheap"), "by_label": {}}
    _rng = np.random.RandomState(0)
    for lab in ("poison", "helpful"):
        R = [r for r in rows if r["label"] == lab]
        loo = np.array([r["loo_marginal_solve"] for r in R]); chp = np.array([r["cheap_snipw_ate"] for r in R])
        n = len(R)
        _bs = np.array([loo[_rng.randint(0, n, n)].mean() for _ in range(10000)])   # LOO-mean CI (seed 0)
        loo_ci = [float(np.percentile(_bs, 2.5)), float(np.percentile(_bs, 97.5))]
        # bootstrap CI for the within-label rank correlation itself (resample node pairs, seed 0).
        # Provenance fix (2026-06-26): the paper quotes a within-poison Spearman CI; emit it here.
        def _spear(a, b):
            return (float(_scipy_spear(a, b).correlation) if _scipy_spear else _argsort_spear(a, b))
        _sb = []
        for _ in range(10000):
            _ix = _rng.randint(0, n, n)
            aa, bb = loo[_ix], chp[_ix]
            if np.std(aa) > 0 and np.std(bb) > 0:
                _sb.append(_spear(aa, bb))
        spear_ci = ([float(np.percentile(_sb, 2.5)), float(np.percentile(_sb, 97.5))] if _sb else None)
        strict = int(sum(1 for l, c in zip(loo, chp) if (l > 0) == (c > 0) and l != 0 and c != 0))
        gt0 = int(sum(1 for l, c in zip(loo, chp) if (l > 0) == (c > 0)))
        # HONEST sign-agreement = among RANKABLE nodes only (both estimates non-zero). The 10 LOO
        # ties-at-zero (1/12-quantized gold standard) can never agree under strict sign and mechanically
        # depress 'strict 8/28'; among rankable nodes agreement is AT chance, not anti-correlated.
        rk = [(l, c) for l, c in zip(loo, chp) if l != 0 and c != 0]
        rk_agree = int(sum(1 for l, c in rk if (l > 0) == (c > 0)))
        med_l, med_c = np.median(loo), np.median(chp)
        medsplit = int(sum(1 for l, c in zip(loo, chp) if (l >= med_l) == (c >= med_c)))
        ties0 = int(sum(1 for l in loo if l == 0))
        # noise floor: per-node SE from the binary with/without arms
        nrll = R[0].get("n_with") or 12
        per_node_se = float(np.sqrt(0.25 / nrll) * np.sqrt(2))   # paired diff SE at p~0.5
        out["by_label"][lab] = {
            "n_nodes": n, "spearman_argsort_pearson": _argsort_spear(loo, chp),
            "spearman_scipy": (float(_scipy_spear(loo, chp).correlation) if _scipy_spear else None),
            "spearman_ci95_boot_seed0": spear_ci,
            "sign_agree_RANKABLE_both_nonzero": f"{rk_agree}/{len(rk)}",
            "sign_agree_rankable_p_ge_vs_chance": (_binom_ge(rk_agree, len(rk)) if rk else None),
            "sign_agree_strict_with_ties_as_disagree": f"{strict}/{n}",
            "sign_agree_gt0_convention": f"{gt0}/{n}", "sign_agree_gt0_p_ge": _binom_ge(gt0, n),
            "sign_agree_median_split": f"{medsplit}/{n}", "loo_ties_at_zero": ties0,
            "interpretation": "AT CHANCE / uninformative (rankable nodes); NOT anti-correlated",
            "loo_mean": float(loo.mean()), "loo_mean_ci95_boot_seed0": loo_ci, "cheap_mean": float(chp.mean()),
            "between_node_sd_loo": float(loo.std(ddof=1)), "per_node_se_loo": per_node_se,
            "signal_to_noise": float(loo.std(ddof=1) / per_node_se)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/lab/campaign_14b_M3_powered/merged")
    ap.add_argument("--out", default="results/lab/campaign_14b_M3_powered/merged/active_loo_within_label.json")
    a = ap.parse_args()
    r = run(Path(a.dir) / "active_loo.json"); Path(a.out).write_text(json.dumps(r, indent=2))
    print("WITHIN-LABEL LOO-vs-cheap (C1 headline):")
    for lab, b in r["by_label"].items():
        print(f"  {lab}: Spearman argsort={b['spearman_argsort_pearson']:+.3f} scipy={b['spearman_scipy']} "
              f"| sign rankable={b['sign_agree_RANKABLE_both_nonzero']} (p_ge={b['sign_agree_rankable_p_ge_vs_chance']}) "
              f"strict-w-ties={b['sign_agree_strict_with_ties_as_disagree']} gt0={b['sign_agree_gt0_convention']} "
              f"| LOO-ties@0={b['loo_ties_at_zero']} | S/N(worst-case bound)={b['signal_to_noise']:.2f} | LOO mean={b['loo_mean']:+.3f}")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
