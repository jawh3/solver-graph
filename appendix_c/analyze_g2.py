#!/usr/bin/env python3
"""appendix_c/analyze_g2.py — paired analysis of the step probe.

Answers the question G2 turns on: does following a latent-poison recipe cost the agent
STEPS? Everything is paired within challenge instance (same instance, same solver seed,
only the injected lesson differs), using the pre-existing statistics layer
(`evaluation/stats.py`) so the discipline matches the prior work:

  * paired bootstrap CIs on the step and solve-rate deltas
  * exact McNemar on the paired binary solve outcomes

Reported per contrast against `helpful` and against `none`. A null step delta is a
real result and is labelled as such — it means the reward channel `R = o(1+λe)` has no
signal to rank latent poison with, which is the deeper blind spot.

    python -m appendix_c.analyze_g2 --dir appendix_c/results/g2_step_probe
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.stats import paired_bootstrap_ci, mcnemar_exact
from appendix_c.provenance import write_result


def load_rows(d: str):
    rows = []
    for f in sorted(glob.glob(str(Path(d) / "rows_shard*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pair_up(rows, cond_a: str, cond_b: str, field: str):
    """Return (a_values, b_values, challenge_ids) over challenges present in BOTH."""
    by = defaultdict(dict)
    for r in rows:
        if r.get("error"):
            continue
        by[r["challenge"]][r["condition"]] = r
    a, b, ids = [], [], []
    for ch, d in sorted(by.items()):
        if cond_a in d and cond_b in d:
            va, vb = d[cond_a].get(field), d[cond_b].get(field)
            if va is None or vb is None:
                continue
            a.append(float(va)); b.append(float(vb)); ids.append(ch)
    return a, b, ids


def contrast(rows, cond_a, cond_b):
    """cond_b - cond_a, paired. `a` is the reference (usually `helpful`)."""
    out = {"reference": cond_a, "arm": cond_b}
    for field in ("rounds_used", "total_commands"):
        a, b, ids = pair_up(rows, cond_a, cond_b, field)
        if not a:
            out[field] = {"n_pairs": 0}
            continue
        # stats.paired_bootstrap_ci(x, y) is mean(x − y); we report arm − reference,
        # so the ARM goes first.
        res = paired_bootstrap_ci(b, a, n_boot=10000, alpha=0.05, seed=0)
        out[field] = {
            "n_pairs": len(a),
            f"mean_{cond_a}": float(np.mean(a)),
            f"mean_{cond_b}": float(np.mean(b)),
            "delta": float(np.mean(b) - np.mean(a)),
            "ci_low": res.ci_low, "ci_high": res.ci_high,
            "excludes_zero": bool(res.ci_low > 0 or res.ci_high < 0),
        }
    sa, sb, ids = pair_up(rows, cond_a, cond_b, "solved")
    if sa:
        res = paired_bootstrap_ci(sb, sa, n_boot=10000, alpha=0.05, seed=0)
        mc = mcnemar_exact([int(x) for x in sa], [int(x) for x in sb])
        out["solved"] = {
            "n_pairs": len(sa),
            f"rate_{cond_a}": float(np.mean(sa)),
            f"rate_{cond_b}": float(np.mean(sb)),
            "delta": float(np.mean(sb) - np.mean(sa)),
            "ci_low": res.ci_low, "ci_high": res.ci_high,
            "excludes_zero": bool(res.ci_low > 0 or res.ci_high < 0),
            "mcnemar": mc,
        }
    else:
        out["solved"] = {"n_pairs": 0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="appendix_c/results/g2_step_probe")
    ap.add_argument("--out", default="g2_step_contrast")
    ap.add_argument("--arms", default="latent_detour,latent_search,sibling_poison,none")
    ap.add_argument("--reference", default="helpful")
    args = ap.parse_args()

    rows = load_rows(args.dir)
    if not rows:
        print(f"no rows in {args.dir}"); return

    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    marginal = {c: {"n": len(v),
                    "solve_rate": float(np.mean([x["solved"] for x in v])),
                    "mean_rounds": float(np.mean([x["rounds_used"] or 0 for x in v])),
                    "mean_commands": float(np.mean([x["total_commands"] or 0 for x in v])),
                    "wrong_flag_rate": float(np.mean([x.get("returned_flag_but_wrong", 0) for x in v]))}
                for c, v in sorted(by_cond.items())}

    contrasts = [contrast(rows, args.reference, arm)
                 for arm in [a.strip() for a in args.arms.split(",")]
                 if arm != args.reference]

    payload = {"n_rows": len(rows), "reference": args.reference,
               "domain": rows[0].get("domain"), "model": rows[0].get("model"),
               "max_rounds": rows[0].get("max_rounds"),
               "marginal": marginal, "contrasts": contrasts}
    path = write_result(args.out, payload, script="appendix_c/analyze_g2.py",
                        seeds=sorted({r["seed"] for r in rows})[:20],
                        model=rows[0].get("model", "unknown"),
                        extra={"source_dir": args.dir})

    print(f"\nG2 STEP PROBE — {len(rows)} runs, model={payload['model']}, "
          f"budget={payload['max_rounds']} rounds")
    print(f"\n{'condition':<16} {'n':>4} {'solve':>7} {'rounds':>8} {'cmds':>7} {'wrongflag':>10}")
    for c, m in marginal.items():
        print(f"{c:<16} {m['n']:>4} {m['solve_rate']:>7.3f} {m['mean_rounds']:>8.2f} "
              f"{m['mean_commands']:>7.2f} {m['wrong_flag_rate']:>10.3f}")

    print(f"\nPAIRED CONTRASTS (arm − {args.reference}); CI = 95% paired bootstrap")
    for c in contrasts:
        arm = c["arm"]
        r = c.get("rounds_used", {})
        s = c.get("solved", {})
        if r.get("n_pairs"):
            star = "*" if r["excludes_zero"] else " "
            print(f"  {arm:<16} Δrounds = {r['delta']:+.2f} "
                  f"[{r['ci_low']:+.2f},{r['ci_high']:+.2f}]{star}  (n={r['n_pairs']})")
        if s.get("n_pairs"):
            star = "*" if s["excludes_zero"] else " "
            print(f"  {'':<16} Δsolve  = {s['delta']:+.3f} "
                  f"[{s['ci_low']:+.3f},{s['ci_high']:+.3f}]{star}  "
                  f"McNemar p={s['mcnemar'].get('p_value', float('nan')):.4f}")
    print("\n* = 95% CI excludes zero")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
