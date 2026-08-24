#!/usr/bin/env python3
"""experiments/crypto_interference_analysis.py — reproducible re-analysis of the crypto constructed-context
interference (replaces transcript numbers with a committed script; RIGOR_LEDGER REDIRECT 2026-06-25).

From results/lab/constructed_loo_crypto.json (forced contexts none/helpful/poison/helpful+poison/
helpful+malignant_*), computes, on ALL families and on the pre-treatment-gated in-band subset (helpful-alone
solve in [0.20,0.90]), family-cluster bootstrap:
  Delta_interf   = solve({helpful,poison}) - solve({helpful})        (the 'interference')
  Delta_alone    = solve({poison})         - solve({none})           (poison alone)
  DiD            = Delta_interf - Delta_alone                        (context-dep vs uniform-harm discriminator)
  budget bracket = Delta_interf - (solve({helpful,exhaust}) - solve({helpful}))   (poison vs pure budget-burn)
HONEST CAVEATS baked into the report: (1) the source rollouts had the POISON recipe rendered FIRST in 100% of
contexts (assemble_context id-sort primacy confound, fixed 2026-06-25) -> Delta_interf is aliased with order;
(2) base/none floors at ~0.04 so Delta_alone is floor-bounded. ZERO GPU; deterministic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _boot(vals, nb=10000, seed=0):
    a = np.array(vals, dtype=float); rng = np.random.RandomState(seed)
    bs = np.array([a[rng.randint(0, len(a), len(a))].mean() for _ in range(nb)])
    return {"mean": float(a.mean()), "ci_lo": float(np.percentile(bs, 2.5)),
            "ci_hi": float(np.percentile(bs, 97.5)),
            "p_two_sided": float(2 * min((bs >= 0).mean(), (bs <= 0).mean())), "n": len(a)}


def run(src: Path) -> dict:
    pf = json.loads(src.read_text())["per_family"]
    need = ("none", "helpful", "poison", "helpful_poison")
    fams = [v for v in pf.values() if all(v.get(c) is not None for c in need)]
    out = {"source": str(src), "note": "rollouts were POISON-FIRST (primacy confound, fixed 2026-06-25); "
           "base floors ~0.04 so Delta_alone is floor-bounded", "by_subset": {}}
    for tag, S in [("all", fams), ("in_band_0.20_0.90", [v for v in fams if 0.20 <= v["helpful"] <= 0.90])]:
        d = {"n_families": len(S), "base_none_mean": float(np.mean([v["none"] for v in S])),
             "helpful_mean": float(np.mean([v["helpful"] for v in S])),
             "delta_interf": _boot([v["helpful_poison"] - v["helpful"] for v in S]),
             "delta_alone": _boot([v["poison"] - v["none"] for v in S]),
             "DiD": _boot([(v["helpful_poison"] - v["helpful"]) - (v["poison"] - v["none"]) for v in S])}
        ex = [v for v in S if v.get("helpful_malignant_exhaust") is not None]
        if ex:
            d["budget_bracket_poison_minus_exhaust"] = _boot(
                [(v["helpful_poison"] - v["helpful"]) - (v["helpful_malignant_exhaust"] - v["helpful"]) for v in ex])
            d["exhaust_in_context_mean"] = float(np.mean([v["helpful_malignant_exhaust"] - v["helpful"] for v in ex]))
        out["by_subset"][tag] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/lab/constructed_loo_crypto.json")
    ap.add_argument("--out", default="results/lab/crypto_interference_DiD.json")
    args = ap.parse_args()
    r = run(Path(args.src)); Path(args.out).write_text(json.dumps(r, indent=2))
    for tag, d in r["by_subset"].items():
        print(f"[{tag}] n_fam={d['n_families']} base={d['base_none_mean']:.3f} helpful={d['helpful_mean']:.3f}")
        for k in ("delta_interf", "delta_alone", "DiD", "budget_bracket_poison_minus_exhaust"):
            b = d.get(k)
            if b:
                print(f"   {k:34s} {b['mean']:+.3f} [{b['ci_lo']:+.3f},{b['ci_hi']:+.3f}] p={b['p_two_sided']:.3f} (n={b['n']})")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
