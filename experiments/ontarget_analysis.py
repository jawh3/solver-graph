#!/usr/bin/env python3
"""experiments/ontarget_analysis.py — ON-TARGET (family-conditioned) per-lesson causal credit.

The MARGINAL per-lesson ATE averages a lesson's effect over ALL runs where it was retrieved —
but a lesson can only help/harm on challenges from ITS OWN family. Stochastic retrieval puts a
lesson on-target only a small fraction of draws, so the marginal estimand DILUTES the signal to
~0. Conditioning on the challenge family (a PRE-treatment variable — no collider bias) isolates
the effect where the lesson is applicable: the per-lesson causal value the credit gate actually needs.

Pre-registered 2026-06-03 as the refined primary estimand for H2 (see docs/PREREG.md addendum).
Emits BOTH marginal and on-target by-label ATEs (solve + effort-weighted reward) with bootstrap CIs.
ZERO new compute — re-analysis of logged runs.

  python -m experiments.ontarget_analysis --dir results/lab/<campaign>/merged --cond outcome
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
from ope.causal_value import observational_lesson_values, summarize_by_label_ci
from retrieval.embedder import MockEmbedder

_fam_of = lambda cid: cid.rsplit("_", 1)[0]   # "b64_rot13_rev_05" -> "b64_rot13_rev"


def analyze(merged_dir: Path, cond: str = "outcome", n_boot: int = 3000) -> dict:
    summ = json.loads((merged_dir / "summary.json").read_text())
    families = summ.get("families") or json.loads(
        (merged_dir / "progress.json").read_text()).get("families")
    _, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=families, malignant_arms=list(hard.MALIGNANT_ARMS), distractor_arm=True)
    runs = load_runs(f"campaign_{cond}", root=str(merged_dir / "logs"))
    if not runs:
        raise SystemExit(f"no runs for campaign_{cond} under {merged_dir}/logs")

    # descriptive: solve rate when an in-context poison is on- vs off-target
    on, off = [], []
    for r in runs:
        f = _fam_of(r["challenge_id"]); ss = set(r.get("sampled_set", []))
        pois = [nid for nid in ss if truth.get(nid, ("?",))[0] == "poison"]
        if not pois:
            continue
        (on if any(truth[nid][1] == f for nid in pois) else off).append(int(r["outcome"]))

    out = {"dir": str(merged_dir), "cond": cond, "n_runs": len(runs),
           "descriptive_poison_in_context": {
               "on_target_solve": (float(np.mean(on)) if on else None), "on_target_n": len(on),
               "off_target_solve": (float(np.mean(off)) if off else None), "off_target_n": len(off)},
           "by_label": {}}

    by_fam = defaultdict(list)
    for r in runs:
        by_fam[_fam_of(r["challenge_id"])].append(r)

    for key in ("outcome", "R_final"):
        marg = observational_lesson_values(runs, truth=truth, response_key=key, n_boot=0)
        ont = {}
        for f, fr in by_fam.items():
            v = observational_lesson_values(fr, truth=truth, response_key=key, n_boot=0)
            for nid, d in v.items():
                if truth.get(nid, ("?", "?"))[1] == f:   # keep on-target lessons of this family
                    ont[nid] = d
        out["by_label"][key] = {
            "marginal": summarize_by_label_ci(marg, n_boot=n_boot, seed=0),
            "on_target": summarize_by_label_ci(ont, n_boot=n_boot, seed=0)}
    return out


def _fmt(d):
    if not d:
        return "n/a"
    return f"ATE={d['mean']:+.4f} CI[{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}] p={d['p']:.3f} n={d['n']} excl0={d['excl0']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/lab/campaign_7b35_M3/merged")
    ap.add_argument("--cond", default="outcome")
    ap.add_argument("--n-boot", type=int, default=3000)
    args = ap.parse_args()
    res = analyze(Path(args.dir), cond=args.cond, n_boot=args.n_boot)

    lines = [f"# ON-TARGET (family-conditioned) causal credit — {res['dir']}  cond={res['cond']}  n_runs={res['n_runs']}", ""]
    dsc = res["descriptive_poison_in_context"]
    lines.append(f"poison in-context solve: on-target={dsc['on_target_solve']} (n={dsc['on_target_n']}) "
                 f"vs off-target={dsc['off_target_solve']} (n={dsc['off_target_n']})")
    for key in ("outcome", "R_final"):
        lines.append(f"\n## response={key}")
        extra = sorted(l for l in res['by_label'][key]['marginal'] if l.startswith("malignant") or l=="distractor_ontarget")
        for lab in ("helpful", "poison", *extra):
            lines.append(f"  MARGINAL  {lab:16s} {_fmt(res['by_label'][key]['marginal'].get(lab))}")
            lines.append(f"  ON-TARGET {lab:16s} {_fmt(res['by_label'][key]['on_target'].get(lab))}")
    md = "\n".join(lines)
    print(md)
    (Path(args.dir) / "ontarget_analysis.json").write_text(json.dumps(res, indent=2))
    (Path(args.dir) / "ontarget_analysis.md").write_text(md + "\n")
    print(f"\n[ontarget] wrote {args.dir}/ontarget_analysis.{{json,md}}")


if __name__ == "__main__":
    main()
