#!/usr/bin/env python3
"""experiments/empirical_null.py — negative-control / empirical-null calibration (OHDSI/Schuemie-style).

An apparent per-lesson signal can be a pipeline artifact (our name-leak was one). The guard: run
PRESUMED-INERT lessons through the IDENTICAL estimator and fit an empirical null; a true signal
sticks out beyond the null band, an artifact sits inside it. Our negative controls are OFF-TARGET
lesson placements: a lesson on a challenge from a DIFFERENT family is causally irrelevant, so its
ATE there should be ~0. We fit the null from the pooled off-target per-lesson ATEs, then calibrate
the ON-TARGET helpful/poison estimates against it.

Headline expectation: helpful on-target sits FAR outside the null (real signal); poison on-target
sits INSIDE the null (calibrated null — undetectable, not merely unconfirmed). ZERO GPU.

  python -m experiments.empirical_null --dir results/lab/<campaign>/merged
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import benchmark.hard_ctf as hard
from instrumentation.jsonl_logger import load_runs
from ope.causal_value import observational_lesson_values
from retrieval.embedder import MockEmbedder

_fam_of = lambda cid: cid.rsplit("_", 1)[0]


def calibrate(merged_dir: Path, cond: str = "outcome", key: str = "R_final") -> dict:
    summ = json.loads((merged_dir / "summary.json").read_text())
    families = summ.get("families") or json.loads((merged_dir / "progress.json").read_text()).get("families")
    _, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=families)
    runs = load_runs(f"campaign_{cond}", root=str(merged_dir / "logs"))
    by_fam = defaultdict(list)
    for r in runs:
        by_fam[_fam_of(r["challenge_id"])].append(r)

    # ON-TARGET per-lesson ATEs (own family) — the candidate signal
    on = defaultdict(list)
    for f, fr in by_fam.items():
        v = observational_lesson_values(fr, truth=truth, response_key=key, n_boot=0)
        for nid, d in v.items():
            if truth.get(nid, ("?", "?"))[1] == f and d["ate"] is not None:
                on[truth[nid][0]].append(d["ate"])

    # NEGATIVE CONTROLS = off-target placements (lesson on OTHER families' runs -> irrelevant -> ~0).
    # Pool every lesson's off-target ATE as the empirical null.
    null_ates = []
    fams = sorted(by_fam)
    for nid, (lab, fam) in truth.items():
        off_runs = [r for r in runs if _fam_of(r["challenge_id"]) != fam]
        if len(off_runs) < 50:
            continue
        v = observational_lesson_values(off_runs, truth=truth, response_key=key, n_boot=0)
        d = v.get(nid)
        if d and d["ate"] is not None:
            null_ates.append(d["ate"])
    null_ates = np.array(null_ates, dtype=np.float64)
    mu = float(np.mean(null_ates)); sigma = float(np.std(null_ates) + 1e-9)
    lo, hi = float(np.percentile(null_ates, 2.5)), float(np.percentile(null_ates, 97.5))

    out = {"dir": str(merged_dir), "key": key, "n_controls": int(len(null_ates)),
           "null_mu": mu, "null_sigma": sigma, "null_95_band": [lo, hi], "labels": {}}
    from math import erfc, sqrt
    for lab in ("helpful", "poison"):
        xs = np.array(on.get(lab, []), dtype=np.float64)
        if len(xs) == 0:
            continue
        m = float(np.mean(xs))
        z = (m - mu) / sigma
        p_cal = float(erfc(abs(z) / sqrt(2)))           # calibrated two-sided p vs empirical null
        out["labels"][lab] = {"on_target_mean_ate": m, "n": int(len(xs)),
                              "calibrated_z": z, "calibrated_p": p_cal,
                              "outside_null_band": bool(m < lo or m > hi)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/lab/campaign_7b35_M3_rep/merged")
    ap.add_argument("--cond", default="outcome")
    ap.add_argument("--key", default="R_final")
    args = ap.parse_args()
    r = calibrate(Path(args.dir), cond=args.cond, key=args.key)
    print(f"EMPIRICAL-NULL CALIBRATION — {r['dir']} (key={r['key']})")
    print(f"  negative controls (off-target placements): n={r['n_controls']}, "
          f"null mu={r['null_mu']:+.4f} sigma={r['null_sigma']:.4f} 95%band=[{r['null_95_band'][0]:+.4f},{r['null_95_band'][1]:+.4f}]")
    for lab, d in r["labels"].items():
        verdict = "OUTSIDE null -> REAL signal" if d["outside_null_band"] else "INSIDE null -> calibrated NULL"
        print(f"  on-target {lab:8s}: ATE={d['on_target_mean_ate']:+.4f}  z={d['calibrated_z']:+.2f}  p_cal={d['calibrated_p']:.4f}  [{verdict}]")
    (Path(args.dir) / "empirical_null.json").write_text(json.dumps(r, indent=2))
    print(f"  wrote {args.dir}/empirical_null.json")


if __name__ == "__main__":
    main()
