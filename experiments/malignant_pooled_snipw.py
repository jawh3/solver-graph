#!/usr/bin/env python3
"""experiments/malignant_pooled_snipw.py — T1-1: the run-level family-clustered on-target SNIPW.

The label-level (lesson-resampled, equal-weight) on-target SNIPW is heterogeneity-limited: its SE is
almost entirely BETWEEN-family variance (sd~0.063/sqrt(34) ~= the whole label-level SE), so adding runs
per family does NOT shrink it (within-lesson noise is already ~0). The bottleneck is that ~32% of
families have no solve headroom (base~0), where a floor-violating lesson can't lower solve and noise
pushes the per-lesson ATE slightly positive (the hex_gzip-class anomaly, T1-6).

The appropriate, more powerful estimator for "average per-RUN causal harm of including the malignant
lesson" is the RUN-LEVEL pooled SNIPW: per family compute the within-family SNIPW ATE (observational_
lesson_values, on-target), pool across families weighted by n_treat (down-weighting low-n no-headroom
families), and get a FAMILY-CLUSTER bootstrap CI (the cluster-robust inference adopted in T1-4). Report
the pre-registered ONE-SIDED p (PREREG b51a184: malignant ATE < 0) plus two-sided + the equal-weight
lesson-level for contrast. ZERO GPU; deterministic (seeded).

  python -m experiments.malignant_pooled_snipw --dir results/lab/campaign_14b_malig/merged
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
from ope.causal_value import observational_lesson_values
from retrieval.embedder import MockEmbedder

_fam = lambda cid: cid.rsplit("_", 1)[0]


def _per_family_ates(merged_dir: Path, cond: str, key: str):
    summ = json.loads((merged_dir / "summary.json").read_text())
    fams = summ.get("families") or json.loads((merged_dir / "progress.json").read_text()).get("families")
    _, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=fams,
                                     malignant_arms=list(hard.MALIGNANT_ARMS), distractor_arm=True)
    runs = load_runs(f"campaign_{cond}", root=str(merged_dir / "logs"))
    by_fam = defaultdict(list)
    for r in runs:
        by_fam[_fam(r["challenge_id"])].append(r)
    per = defaultdict(list)        # label -> [(family, ate, n_treat)]
    for f, fr in by_fam.items():
        vals = observational_lesson_values(fr, truth=truth, n_boot=0, response_key=key)
        for nid, d in vals.items():
            if nid not in truth or d.get("ate") is None:
                continue
            lab, lfam = truth[nid]
            if lfam != f:           # ON-TARGET only
                continue
            per[lab].append((f, float(d["ate"]), float(d.get("n_treat") or 1)))
    return per


def _boot(rows, weighted: bool, nboot: int = 8000, seed: int = 0):
    rng = np.random.RandomState(seed)
    ate = np.array([r[1] for r in rows]); n = np.array([r[2] for r in rows])
    w = n if weighted else np.ones_like(n)
    pt = float(np.sum(w * ate) / np.sum(w))
    bs = np.empty(nboot)
    for b in range(nboot):
        idx = rng.randint(0, len(rows), size=len(rows))
        bs[b] = np.sum(w[idx] * ate[idx]) / np.sum(w[idx])
    lo, hi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
    p2 = float(2 * min((bs >= 0).mean(), (bs <= 0).mean()))
    p1 = float((bs >= 0).mean() if pt < 0 else (bs <= 0).mean())   # one-sided toward observed sign
    return {"ate": pt, "ci_lo": lo, "ci_hi": hi, "p_one_sided": p1, "p_two_sided": p2,
            "n_families": len(rows), "excl0_two_sided": (lo < 0 and hi < 0) or (lo > 0 and hi > 0)}


def run(merged_dirs, cond: str = "outcome", key: str = "outcome") -> dict:
    if isinstance(merged_dirs, (str, Path)):
        merged_dirs = [merged_dirs]
    per = defaultdict(list)
    for md in merged_dirs:
        for lab, rows in _per_family_ates(Path(md), cond, key).items():
            per[lab].extend(rows)        # pool families across campaigns (each family appears once per campaign)
    labels = ["poison", "helpful", "malignant_destroy", "malignant_exhaust", "malignant_wrongflag",
              "distractor_ontarget"]
    out = {"dirs": [str(m) for m in merged_dirs], "cond": cond, "key": key, "by_label": {}}
    for lab in labels:
        rows = per.get(lab)
        if not rows:
            continue
        out["by_label"][lab] = {
            "equal_weight_lesson_level": _boot(rows, weighted=False),
            "n_treat_weighted_run_level": _boot(rows, weighted=True)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/lab/campaign_14b_malig/merged")
    ap.add_argument("--dirs", default=None, help="comma list of merged dirs to POOL families across "
                    "(overrides --dir; e.g. campaign_14b_malig/merged,campaign_14b_malig_expand/merged)")
    ap.add_argument("--cond", default="outcome")
    ap.add_argument("--key", default="outcome", help="outcome (binary) or R_final (effort-weighted)")
    ap.add_argument("--out", default="malignant_pooled_snipw.json")
    args = ap.parse_args()
    dirs = [d.strip() for d in args.dirs.split(",")] if args.dirs else [args.dir]
    r = run(dirs, cond=args.cond, key=args.key)
    # write the artifact into the FIRST dir (or cwd) — keep single-dir behavior unchanged
    write_dir = Path(dirs[0])
    print(f"RUN-LEVEL family-clustered on-target SNIPW — {r['dirs']} (key={r['key']})")
    for lab, d in r["by_label"].items():
        for est in ("equal_weight_lesson_level", "n_treat_weighted_run_level"):
            b = d[est]
            flag = "  <-- p1<.05" if (b["p_one_sided"] < 0.05 and b["ate"] < 0) else ""
            print(f"  {lab:20s} [{est:26s}] ATE={b['ate']:+.4f} "
                  f"CI[{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}] p1={b['p_one_sided']:.4f} "
                  f"p2={b['p_two_sided']:.4f} (n_fam={b['n_families']}){flag}")
    (write_dir / args.out).write_text(json.dumps(r, indent=2))
    print(f"  wrote {write_dir}/{args.out}")


if __name__ == "__main__":
    main()
