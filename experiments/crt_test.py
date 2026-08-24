#!/usr/bin/env python3
"""experiments/crt_test.py — exact Conditional Randomization Test per lesson (elimination layer).

Our retrieval is a KNOWN randomized treatment: each lesson i is included with a logged propensity
pi_i. That is the ideal ("model-X with known X-law") regime for a CRT — an EXACT, assumption-light
test of H0_i: lesson i's inclusion is conditionally independent of the outcome, needing NO outcome
model and not relying on IPW asymptotics or positivity. Statistic S_i = |sum_run (T_i - pi_i) * Y|;
null is built by resampling T_i' ~ Bernoulli(pi_i) (the known law). p_i = (1 + #{S' >= S_obs})/(1+B).

If even this exact test finds no poison signal (uniform-ish p), the null is not an estimator artifact.
Reports per-label CRT results, ON-TARGET (own-family) and MARGINAL.  ZERO GPU.
  python -m experiments.crt_test --dir results/lab/<campaign>/merged
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import os as _os, importlib as _il
hard = _il.import_module(_os.environ.get("CTF_DOMAIN", "benchmark.hard_ctf"))  # T2-7 domain switch
from instrumentation.jsonl_logger import load_runs
from ope.causal_value import _collect_obs
from retrieval.embedder import MockEmbedder

_fam_of = lambda cid: cid.rsplit("_", 1)[0]


def _crt_p(rows, rng, B: int = 4000) -> float:
    R = np.array([r[0] for r in rows], dtype=np.float64)
    T = np.array([r[1] for r in rows], dtype=np.float64)
    pi = np.array([r[2] for r in rows], dtype=np.float64)
    if len(R) < 10 or T.sum() < 3:
        return float("nan")
    s_obs = abs(float(np.sum((T - pi) * R)))
    draws = (rng.random((B, len(pi))) < pi).astype(np.float64)        # T' ~ Bernoulli(pi_i)
    s_null = np.abs((draws - pi) @ R)
    return float((1 + np.sum(s_null >= s_obs)) / (1 + B))


def run(merged_dir: Path, cond: str = "outcome", key: str = "R_final",
        clip: float = 1e-6, n_mc: int = 1000) -> dict:
    """clip: propensity clip for the CRT null. The IPW pipeline uses 0.05 as a positivity
    guard, but the EXACT CRT must resample T' from the TRUE logged law -> clip~0 (1e-6 to
    avoid degenerate pi=0/1). n_mc: propensity-reconstruction MC draws (raised from the
    pipeline default 400 to cut reconstruction noise in the known X-law)."""
    summ = json.loads((merged_dir / "summary.json").read_text())
    families = summ.get("families") or json.loads((merged_dir / "progress.json").read_text()).get("families")
    _, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=families, malignant_arms=list(hard.MALIGNANT_ARMS), distractor_arm=True)
    runs = load_runs(f"campaign_{cond}", root=str(merged_dir / "logs"))
    rng = np.random.RandomState(0)

    # MARGINAL: one obs pass over all runs
    marg_obs = _collect_obs(runs, clip=clip, n_mc=n_mc, response_key=key)
    marg_p = {v: _crt_p(rows, rng) for v, rows in marg_obs.items() if v in truth}

    # ON-TARGET: per family, only that family's own lessons
    by_fam = defaultdict(list)
    for r in runs:
        by_fam[_fam_of(r["challenge_id"])].append(r)
    on_p = {}
    for f, fr in by_fam.items():
        obs = _collect_obs(fr, clip=clip, n_mc=n_mc, response_key=key)
        for v, rows in obs.items():
            if truth.get(v, ("?", "?"))[1] == f:
                on_p[v] = _crt_p(rows, rng)

    def by_label(pmap):
        agg = defaultdict(list)
        for v, p in pmap.items():
            if v in truth and not np.isnan(p):
                agg[truth[v][0]].append(p)
        return {lab: {"n": len(ps), "frac_sig_p05": float(np.mean([p < 0.05 for p in ps])),
                      "median_p": float(np.median(ps)), "min_p": float(np.min(ps))}
                for lab, ps in agg.items()}

    return {"dir": str(merged_dir), "key": key,
            "marginal": by_label(marg_p), "on_target": by_label(on_p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/lab/campaign_7b35_M3_rep/merged")
    ap.add_argument("--cond", default="outcome")
    ap.add_argument("--key", default="R_final")
    ap.add_argument("--out", default="crt_test.json",
                    help="output filename inside --dir (use crt_test_outcome.json for --key outcome "
                         "so the R_final artifact is not clobbered)")
    ap.add_argument("--clip", type=float, default=1e-6,
                    help="propensity clip for the null (CRT exact => ~0; pass 0.05 to reproduce the "
                         "old IPW-clipped behavior)")
    ap.add_argument("--n-mc", type=int, default=1000, help="propensity reconstruction MC draws")
    args = ap.parse_args()
    r = run(Path(args.dir), cond=args.cond, key=args.key, clip=args.clip, n_mc=args.n_mc)
    r["clip"] = args.clip
    r["n_mc"] = args.n_mc
    print(f"CRT (exact, known-propensity) — {r['dir']} (key={r['key']})")
    for scope in ("marginal", "on_target"):
        print(f"  [{scope}]")
        extra = sorted(l for l in r[scope] if l.startswith("malignant"))
        for lab in ("helpful", "poison", *extra):
            d = r[scope].get(lab)
            if d:
                print(f"    {lab:16s}: frac p<.05 = {d['frac_sig_p05']:.2f}  median p = {d['median_p']:.3f}  min p = {d['min_p']:.3f}  (n={d['n']})")
    (Path(args.dir) / args.out).write_text(json.dumps(r, indent=2))
    print(f"  wrote {args.dir}/{args.out}")


if __name__ == "__main__":
    main()
