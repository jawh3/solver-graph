#!/usr/bin/env python3
"""experiments/learning_curve.py - pre-specified estimator for the the Phase-2 plan learning predictions
(review 2026-09-05 F-9).

From a campaign's runs.jsonl (one learner per condition), for the DEPLOY stream of the TRAIN phase
(eval runs are frozen; exploration runs are design draws), by training tercile:
  * empirical inclusion of the on-target helpful / poison lesson in the sampled M-set
  * mean logged deploy propensity pi_deploy(on-target helpful / poison) - the noise-free version,
    recomputed from the logged candidate scores at tau_deploy (ope.causal_value)
  * chance = M / pool
Test statistic: last-tercile minus first-tercile difference, family-cluster bootstrap CI (families
are the clusters; runs within a family share instances and lessons).

Usage: python -m experiments.learning_curve --root <campaign root with shard_*/logs> [--cond deployed]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

import importlib as _il
hard = _il.import_module(os.environ.get("CTF_DOMAIN", "benchmark.hard_ctf"))
from ope.causal_value import marginal_inclusion_propensities
from retrieval.embedder import MockEmbedder

_fam_of = lambda cid: cid.rsplit("_", 1)[0]


def load_runs(root: Path, cond: str):
    runs = []
    for f in sorted(root.glob(f"shard_*/logs/campaign_{cond}/runs.jsonl")) or sorted(root.glob(f"logs/campaign_{cond}/runs.jsonl")):
        runs += [json.loads(l) for l in open(f) if l.strip()]
    return runs


def curve(runs, hid, pid, n_mc=300, n_boot=2000, seed=0, terciles=3):
    """Returns dict with per-tercile stats and the first-vs-last contrast with cluster-bootstrap CIs."""
    # Review 2026-09-08 F1: `RunRecord.seed` is the CONFIG seed (constant), so the old seed-floor filter counted
    # the frozen eval phase as training. Eval runs are frozen (explore_enabled False, epsilon 0); train
    # deploy-stream runs have explore_enabled True and are not exploration draws. New logs also carry `phase`.
    def _is_train(r):
        ph = r.get("phase")
        if ph:
            return ph == "train"
        pol = r.get("policy") or {}
        return bool(pol.get("explore_enabled", True)) and not pol.get("deterministic")
    train = [r for r in runs if _is_train(r) and not r.get("exploration")
             and not (r.get("policy") or {}).get("deterministic") and r.get("candidate_pool")]
    train.sort(key=lambda r: r.get("ts", ""))
    if not train:
        return {"n_train_deploy_runs": 0}
    rows = []          # (tercile, family, incl_h, incl_p, pi_h, pi_p, chance)
    for k, r in enumerate(train):
        f = _fam_of(r["challenge_id"]); pool = r["candidate_pool"]; m = len(r["sampled_set"])
        tau = float((r.get("policy") or {}).get("tau_deploy", r.get("temperature", 1.0)) or 1.0)
        pis = marginal_inclusion_propensities(pool, m, max(tau, 1e-3), n_mc=n_mc, seed=seed + k)
        h, p_ = hid.get(f), pid.get(f)
        rows.append((min(terciles - 1, k * terciles // len(train)), f,
                     float(h in set(r["sampled_set"])), float(p_ in set(r["sampled_set"])),
                     float(pis.get(h, 0.0)) if h else np.nan, float(pis.get(p_, 0.0)) if p_ else np.nan,
                     m / max(len(pool), 1)))
    out = {"n_train_deploy_runs": len(train), "terciles": []}
    for t in range(terciles):
        sub = [x for x in rows if x[0] == t]
        if not sub:
            continue
        out["terciles"].append({"tercile": t, "n": len(sub),
                                "incl_helpful": float(np.mean([x[2] for x in sub])),
                                "incl_poison": float(np.mean([x[3] for x in sub])),
                                "pi_helpful": float(np.nanmean([x[4] for x in sub])),
                                "pi_poison": float(np.nanmean([x[5] for x in sub])),
                                "chance": float(np.mean([x[6] for x in sub]))})
    # first-vs-last contrast, family-cluster bootstrap
    fams = sorted({x[1] for x in rows}); rng = np.random.RandomState(seed)
    by_fam = {f: [x for x in rows if x[1] == f] for f in fams}

    def contrast(sample_rows, col):
        first = [x[col] for x in sample_rows if x[0] == 0]; last = [x[col] for x in sample_rows if x[0] == terciles - 1]
        if not first or not last:
            return np.nan
        return float(np.nanmean(last) - np.nanmean(first))
    out["contrast"] = {}
    for name, col in (("pi_helpful", 4), ("pi_poison", 5), ("incl_helpful", 2), ("incl_poison", 3)):
        pt = contrast(rows, col); bs = []
        for _ in range(n_boot):
            samp = [x for f in rng.choice(fams, len(fams), replace=True) for x in by_fam[f]]
            bs.append(contrast(samp, col))
        bs = np.array([b for b in bs if not np.isnan(b)])
        lo, hi = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if len(bs) else (np.nan, np.nan)
        out["contrast"][name] = {"last_minus_first": pt, "ci_lo": lo, "ci_hi": hi,
                                 "excl0": bool(len(bs) and (lo > 0 or hi < 0)), "n_families": len(fams)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cond", default="deployed")
    ap.add_argument("--n-mc", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    runs = load_runs(root, args.cond)
    fams = sorted({_fam_of(r["challenge_id"]) for r in runs})
    _, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=fams)
    hid = {f: n for n, (l, f) in truth.items() if l == "helpful"}
    pid = {f: n for n, (l, f) in truth.items() if l == "poison"}
    res = curve(runs, hid, pid, n_mc=args.n_mc)
    res["cond"] = args.cond; res["contexts"] = sorted({str(r.get("context_node_id")) for r in runs})
    print(f"[learning_curve] {args.cond}: {res.get('n_train_deploy_runs', 0)} deploy-stream train runs; contexts {res['contexts'][:6]}")
    for t in res.get("terciles", []):
        print(f"  tercile {t['tercile']}: n={t['n']} incl helpful {t['incl_helpful']:.3f} poison {t['incl_poison']:.3f} | "
              f"pi helpful {t['pi_helpful']:.3f} poison {t['pi_poison']:.3f} | chance {t['chance']:.3f}")
    for k, c in res.get("contrast", {}).items():
        print(f"  {k:13s} last-first {c['last_minus_first']:+.3f} CI[{c['ci_lo']:+.3f},{c['ci_hi']:+.3f}] excl0={c['excl0']}")
    out = args.out or (root / f"learning_curve_{args.cond}.json")
    Path(out).write_text(json.dumps(res, indent=2)); print(f"  wrote {out}")


if __name__ == "__main__":
    main()
