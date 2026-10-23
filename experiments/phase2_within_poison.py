#!/usr/bin/env python3
"""experiments/phase2_within_poison.py - the paper's headline statistic (C1: within-poison rank agreement between the
observational estimate and the leave-one-out gold standard) recomputed on the Phase-2 instrument: observational = the
per-family exploration-slice contrast pooled over the arms with independent draws (P vs random filler beside the helpful
lesson); gold = the per-family LOO cell difference [H, P] - [H] (P vs empty slot) at the campaign solver configuration.
Usage: python -m experiments.phase2_within_poison --root results/lab/phase2_14b --arms a,b,c --loo results/lab/phase2_14b/constructed_loo_campaigncfg.json --out docs/...md
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import importlib as _il
hard = _il.import_module(os.environ.get("CTF_DOMAIN", "benchmark.hard_ctf"))
from retrieval.embedder import MockEmbedder
from experiments.phase2_estimands import load, rows_for


def agreement(fams, o, g, seed=0, n_boot=4000):
    x = np.array([o[f] for f in fams]); y = np.array([g[f] for f in fams]); rho = float(spearmanr(x, y).correlation)
    rng = np.random.RandomState(seed); bs = []
    for _ in range(n_boot):
        i = rng.randint(0, len(fams), len(fams))
        if len(set(i.tolist())) > 2:
            bs.append(spearmanr(x[i], y[i]).correlation)
    rank = [f for f in fams if abs(g[f]) > 1e-9 and abs(o[f]) > 1e-9]
    agree = sum(1 for f in rank if np.sign(o[f]) == np.sign(g[f]))
    return {"rho": rho, "rho_ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "n": len(fams),
            "signs_agree": agree, "rankable": len(rank), "obs_mean": float(x.mean()), "gold_mean": float(y.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--arms", required=True); ap.add_argument("--loo", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    fams = open(f"{args.root}/families.txt").read().strip().split(",")
    _, t = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=fams)
    hid = {f: n for n, (l, f) in t.items() if l == "helpful"}; pid = {f: n for n, (l, f) in t.items() if l == "poison"}
    rows = []
    for a in args.arms.split(","):
        rows += rows_for(load(args.root, a), hid, pid)
    by = defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    loo = json.load(open(args.loo))["per_family"]
    cell = lambda rs, H, P: (np.mean([r[3] for r in rs if r[1] == H and r[2] == P]) if any(r[1] == H and r[2] == P for r in rs) else np.nan)
    obs_disp = {f: cell(by[f], 1, 1) - cell(by[f], 1, 0) for f in fams}
    obs_help = {f: np.mean([r[3] for r in by[f] if r[1] == 1]) - np.mean([r[3] for r in by[f] if r[1] == 0]) for f in fams}
    gold_disp = {f: loo[f]["helpful_poison"] - loo[f]["helpful"] for f in fams}
    gold_help = {f: loo[f]["helpful"] - loo[f]["none"] for f in fams}
    P = agreement(fams, obs_disp, gold_disp, 0); H = agreement(fams, obs_help, gold_help, 1)
    L = ["# Within-label rank agreement, observational vs leave-one-out gold standard - Phase-2 instrument", "",
         f"Observational: per-family exploration-slice contrasts pooled over {args.arms} ({len(rows)} exploration runs); gold: per-family LOO",
         f"cells from `{args.loo}` (8 fresh instances per cell). Family-cluster bootstrap CIs (4,000).", "",
         "| label | estimand pair | Spearman ρ [CI] | signs agree / rankable | obs mean | gold mean |", "|---|---|---|---|---|---|",
         f"| poison | E[Y∣H,P]-E[Y∣H,¬P] (P vs random filler) vs [H,P]-[H] (P vs empty slot) | {P['rho']:+.2f} [{P['rho_ci'][0]:+.2f}, {P['rho_ci'][1]:+.2f}] | {P['signs_agree']}/{P['rankable']} | {P['obs_mean']:+.3f} | {P['gold_mean']:+.3f} |",
         f"| helpful | E[Y∣H]-E[Y∣¬H] vs [H]-[none] | {H['rho']:+.2f} [{H['rho_ci'][0]:+.2f}, {H['rho_ci'][1]:+.2f}] | {H['signs_agree']}/{H['rankable']} | {H['obs_mean']:+.3f} | {H['gold_mean']:+.3f} |",
         "", "| family | obs poison displacement | gold poison displacement | obs helpful | gold helpful |", "|---|---|---|---|---|"]
    for f in fams:
        L.append(f"| {f} | {obs_disp[f]:+.3f} | {gold_disp[f]:+.3f} | {obs_help[f]:+.3f} | {gold_help[f]:+.3f} |")
    L += ["", "Reading: the helpful lesson's per-family value agrees in sign in every family and correlates across the two designs; the poison's",
          "does not - and the two poison estimands differ in SIGN on average (positive against the random filler the deployed system would",
          "otherwise place, negative against an empty slot). The per-item non-rankability is an estimand mismatch, not a noise floor."]
    out = args.out or f"{args.root}/WITHIN_POISON.md"; Path(out).write_text("\n".join(L) + "\n")
    Path(out).with_suffix(".json").write_text(json.dumps({"poison": P, "helpful": H, "obs_disp": obs_disp, "gold_disp": gold_disp, "obs_help": obs_help, "gold_help": gold_help}, indent=1))
    print("\n".join(L[:9]))


if __name__ == "__main__":
    main()
