#!/usr/bin/env python3
"""experiments/phase2_ladder.py - the Phase-2 held-out ladder from progress.json eval metrics.
Review 2026-09-08 F2: the first ladder table was produced by an uncommitted script; this is its committed,
auditable form. Unit = the FAMILY: paired family-mean differences, family-cluster bootstrap CI, family-level
sign-permutation p; plus the run-level exact McNemar (secondary; ignores clustering) and the D8 nested 12-family
subset (first three selected per alphabet in selection order).
Usage: python -m experiments.phase2_ladder --root results/lab/phase2_14b [--ref B2det] [--out docs/...md]
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

CONTRASTS = [("deployed_alphabet_tau01", "B2det"), ("deployed_alphabet_tau01", "deployed_global_tau01"),
             ("deployed_family_tau01", "deployed_alphabet_tau01"), ("deployed_alphabet_tau01", "outcome_legacy_alphabet"),
             ("deployed_alphabet_tau01", "deployed_alphabet_tau02"), ("deployed_cluster_tau01", "deployed_alphabet_tau01"),
             ("s0_lure_alphabet", "deployed_alphabet_tau01"), ("s1_cross_outer_alphabet", "deployed_alphabet_tau01"),
             ("deployed_global_tau01", "B2det"), ("deployed_obs_alphabet", "B2det"), ("deployed_causal_alphabet", "deployed_alphabet_tau01")]


def eval_metrics(root, arm):
    p = json.load(open(f"{root}/arms/{arm}/shard_00/progress.json")); c = list(p["conditions"].values())[0]
    if not c.get("done"):
        return None
    return {(m["name"], m["seed"]): (int(m["solved"]), m["family"]) for m in c["metrics"] if m["phase"] == "eval"}


def fam_diff(E, a, b, keys, n_boot=4000, n_perm=20000, seed=0):
    fams = defaultdict(list)
    for k in keys:
        fams[E[a][k][1]].append(E[a][k][0] - E[b][k][0])
    fm = np.array([np.mean(v) for v in fams.values()]); rng = np.random.RandomState(seed)
    bs = [np.mean(fm[rng.randint(0, len(fm), len(fm))]) for _ in range(n_boot)]
    flips = rng.choice([-1, 1], size=(n_perm, len(fm)))
    p_sign = float(np.mean(np.abs((flips * fm).mean(axis=1)) >= abs(fm.mean()) - 1e-12))
    # run-level exact McNemar on discordant pairs (secondary; treats seeds as independent, ignores clustering)
    a_only = sum(1 for k in keys if E[a][k][0] > E[b][k][0]); b_only = sum(1 for k in keys if E[a][k][0] < E[b][k][0])
    p_mc = float(binomtest(a_only, a_only + b_only, 0.5).pvalue) if a_only + b_only else 1.0
    return {"diff": float(fm.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "up": int((fm > 0).sum()), "down": int((fm < 0).sum()), "n_families": len(fm), "p_sign": p_sign,
            "mcnemar": {"a_only": a_only, "b_only": b_only, "p": p_mc}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--ref", default="B2det"); ap.add_argument("--out", default=None)
    args = ap.parse_args(); root = args.root
    arms = sorted(a.rstrip("/").split("/")[-1] for a in glob.glob(f"{root}/arms/*/"))
    E = {a: eval_metrics(root, a) for a in arms}; E = {a: v for a, v in E.items() if v}
    keys = sorted(set.intersection(*[set(v) for v in E.values()]))
    fams_sel = json.load(open(f"{root}/families.json"))["families"]
    nested12 = [f for o in ("b32", "b64", "b85", "hex") for f in [x for x in fams_sel if x.startswith(o + "_")][:3]]
    keys12 = [k for k in keys if E[args.ref][k][1] in set(nested12)]
    out = {"n_items": len(keys), "n_families": len({E[args.ref][k][1] for k in keys}), "nested12": nested12, "arms": {}, "contrasts": {}, "contrasts_nested12": {}}
    lines = [f"# Phase-2 held-out ladder - {root} - {len(E)} complete arms", "",
             f"Paired held-out items: {len(keys)} = {out['n_families']} families × 27 items × 2 LLM seeds; unit = family; "
             f"family-cluster bootstrap 95% CI (4,000 reps), family-level sign-permutation p (20,000), run-level exact McNemar (secondary).", "",
             f"| arm | solve | vs {args.ref}: family-mean diff [CI] | up/down | p_sign | McNemar (a/b, p) |", "|---|---|---|---|---|---|"]
    for a in arms:
        if a not in E:
            continue
        rate = float(np.mean([E[a][k][0] for k in keys])); out["arms"][a] = {"solve": rate}
        if a == args.ref:
            lines.append(f"| {a} | {rate:.3f} | reference | | | |"); continue
        d = fam_diff(E, a, args.ref, keys); out["arms"][a]["vs_ref"] = d
        lines.append(f"| {a} | {rate:.3f} | {d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] | {d['up']}/{d['down']} | {d['p_sign']:.4f} | {d['mcnemar']['a_only']}/{d['mcnemar']['b_only']}, {d['mcnemar']['p']:.1e} |")
    lines += ["", "## Contrasts - 22 families and a nested 12-family subset (the first three families per alphabet)", "",
              "| contrast | 22 families: diff [CI], up/down, p_sign | nested 12: diff [CI], up/down, p_sign |", "|---|---|---|"]
    for a, b in CONTRASTS:
        if a in E and b in E:
            d = fam_diff(E, a, b, keys); d12 = fam_diff(E, a, b, keys12, seed=1)
            out["contrasts"][f"{a} - {b}"] = d; out["contrasts_nested12"][f"{a} - {b}"] = d12
            lines.append(f"| {a} - {b} | {d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}], {d['up']}/{d['down']}, p={d['p_sign']:.4f} | "
                         f"{d12['diff']:+.3f} [{d12['ci'][0]:+.3f}, {d12['ci'][1]:+.3f}], {d12['up']}/{d12['down']}, p={d12['p_sign']:.4f} |")
    lines += ["", "Multiplicity: the four context arms form one ordering (global < alphabet ≤ family); the "
              "poison kinds are a dose-response, not separate tests; other rows are descriptive. Error-flagged runs "
              "(`llm_bad_request`) are counted as failures in every arm, as in progress.json."]
    md = "\n".join(lines) + "\n"; outp = args.out or f"{root}/LADDER.md"
    Path(outp).write_text(md); Path(outp).with_suffix(".json").write_text(json.dumps(out, indent=1)); print(md)


if __name__ == "__main__":
    main()
