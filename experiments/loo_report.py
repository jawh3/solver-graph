#!/usr/bin/env python3
"""experiments/loo_report.py - the plan: the per-family leave-one-out table with its per-family MDE and the
outer-alphabet heterogeneity of displacement (review 2026-09-08 F7/F14). Accepts one or more constructed_loo JSONs
(e.g. the 09-07 smoke-config run and the campaign-config re-run) and prints them side by side.
Usage: python -m experiments.loo_report --families results/lab/phase2_14b/families.json --loo a.json[,b.json] --out docs/...md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def mde(n, p=0.5, alpha=1.96, power=0.84):
    """Normal-approximation minimal detectable difference of two proportions with n solves per cell."""
    return (alpha + power) * np.sqrt(2 * p * (1 - p) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True); ap.add_argument("--loo", required=True); ap.add_argument("--out", default=None)
    args = ap.parse_args()
    fam = json.load(open(args.families)); dose = fam.get("dose", {})
    loos = [(p, json.load(open(p))) for p in args.loo.split(",")]
    L = ["# Leave-one-out factorial (P2): per-family table, MDE, outer-alphabet heterogeneity", ""]
    for path, d in loos:
        pr = d.get("params", {}); n = int(pr.get("n_inst", 8)) * int(pr.get("seeds", 1))
        cfg = f"max_output_tokens {pr.get('max_output_tokens', '384 (smoke-era default)')}, context_window {pr.get('context_window', '6144')}, trim {pr.get('trim_fraction', '0.6')}"
        L += [f"## `{path}` - {d['n_families']} families × {n} solves per cell; solver {cfg}", "",
              f"Per-family MDE (two-sided α 0.05, 80% power, p ≈ 0.5, n = {n} per cell): **±{mde(n):.2f}**; label-level MDE over "
              f"{d['n_families']} families (n = {n * d['n_families']} per cell): ±{mde(n * d['n_families']):.3f}. Per-family cells are descriptive.", ""]
        pf = d["per_family"]; cells = sorted({c for v in pf.values() for c in v})
        base = ["none", "helpful", "poison", "helpful_poison"]
        extra = [c for c in ("helpful_filler1", "helpful_filler2", "poison_cross", "helpful_poison_cross") if c in cells]
        mal = [c for c in cells if c.startswith("helpful_malignant_")]
        head = "| family | outer | first wrong step / class | " + " | ".join(base + extra) + " | displacement (H+P - H) | " + " | ".join(m.replace("helpful_malignant_", "mal:") for m in mal) + " |"
        L += [head, "|" + "---|" * (head.count("|") - 1)]
        by_outer = defaultdict(list)
        for f in sorted(pf, key=lambda x: (x.split("_")[0], x)):
            v = pf[f]; o = f.split("_")[0]; ds = dose.get(f, {})
            disp = v.get("helpful_poison", np.nan) - v.get("helpful", np.nan); by_outer[o].append(disp)
            L.append(f"| {f} | {o} | {ds.get('first_wrong_decode_step', '?')} / {ds.get('wrong_class', '?')} | " +
                     " | ".join(f"{v.get(c, float('nan')):.3f}" for c in base + extra) + f" | {disp:+.3f} | " +
                     " | ".join(f"{v.get(m, float('nan')):.3f}" for m in mal) + " |")
        L += ["", "| outer alphabet | families | mean displacement | negative / zero / positive |", "|---|---|---|---|"]
        for o, xs in sorted(by_outer.items()):
            xs = np.array(xs); L.append(f"| {o} | {len(xs)} | {xs.mean():+.3f} | {(xs < 0).sum()} / {(xs == 0).sum()} / {(xs > 0).sum()} |")
        L += ["", "Label-level contrasts (family-cluster bootstrap, from the JSON):", ""]
        for k, c in d.get("contrasts", {}).items():
            if c:
                L.append(f"- {k}: {c['delta']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] (p {c.get('p_two_sided', float('nan')):.3f})")
        L.append("")
    out = args.out or "LOO_REPORT.md"; Path(out).write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
