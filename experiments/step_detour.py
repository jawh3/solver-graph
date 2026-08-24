"""T0-5b (RIGOR_LEDGER): direct step-detour measurement on solved runs.

Tests the theorem's Assumption-3 channel DIRECTLY (instead of inferring it from the
reward/outcome ATE ratio, which is a scale identity):
  arm1 : solved runs with on-target poison cited AND on-target helpful NOT cited
  arm0 : solved runs with NO on-target label cited (no-recipe)
  detour = family-weighted mean steps(arm1) - steps(arm0)   (w_f = n_arm1_f)
Positive => a real treated-side step detour; ~0/negative => the reward residue is
displacement of fast helpful solves, not a detour. Also late-solve fractions
(steps >= max_steps - 1) per arm, for the censoring check.

Writes <merged>/step_detour.json.   Usage: python -m experiments.step_detour --dir <merged>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments._loglib import load_runs_with_flags, stratified_boot


def _detour(runs):
    by_fam: dict[str, dict] = {}
    for r in runs:
        if not r["outcome"] or r["steps"] is None:
            continue
        d = by_fam.setdefault(r["family"], {"a1": [], "a0": []})
        if r["PC"] and not r["HC"]:
            d["a1"].append(r["steps"])
        elif not r["PC"] and not r["HC"]:
            d["a0"].append(r["steps"])
    num = den = 0.0
    for d in by_fam.values():
        if not d["a1"] or not d["a0"]:
            continue
        w = len(d["a1"])
        num += w * (sum(d["a1"]) / len(d["a1"]) - sum(d["a0"]) / len(d["a0"]))
        den += w
    return (num / den) if den else None


def _late_fracs(runs):
    f1 = [0, 0]
    f0 = [0, 0]
    for r in runs:
        if not r["outcome"] or r["steps"] is None or r["max_steps"] is None:
            continue
        late = int(r["steps"] >= r["max_steps"] - 1)
        if r["PC"] and not r["HC"]:
            f1[0] += late; f1[1] += 1
        elif not r["PC"] and not r["HC"]:
            f0[0] += late; f0[1] += 1
    return {"poison_pure_late": (f1[0] / f1[1]) if f1[1] else None, "n_poison_pure_solved": f1[1],
            "no_recipe_late": (f0[0] / f0[1]) if f0[1] else None, "n_no_recipe_solved": f0[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    merged = Path(args.dir)
    runs = load_runs_with_flags(merged)

    est = _detour(runs)
    lo, hi = stratified_boot(runs, _detour, n_boot=args.n_boot, seed=args.seed)
    out = {"detour_steps_weighted": est, "ci95": [lo, hi], **_late_fracs(runs)}
    (merged / "step_detour.json").write_text(json.dumps(out, indent=2))
    print(f"[step_detour] {merged}: detour={est if est is None else f'{est:+.3f}'} steps "
          f"CI[{lo},{hi}]; late {out['poison_pure_late']} vs {out['no_recipe_late']}")


if __name__ == "__main__":
    main()
