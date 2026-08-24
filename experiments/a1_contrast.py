"""T0-5d (RIGOR_LEDGER): family-stratified A1 floor contrast.

Tests Assumption 1 (base-rate floor) DIRECTLY, within family, excluding displacement:
  arm1 (poison-pure) : runs where the on-target POISON was cited AND the on-target helpful was NOT
  arm0 (no-recipe)   : runs where NEITHER on-target label was cited  (defines b_f)
  contrast = sum_f w_f * [ mean(outcome|arm1,f) - mean(outcome|arm0,f) ],  w_f = n_arm1_f / sum n_arm1
Negative & CI-excluding-0 => genuine sub-floor effect (A1 violation). Also the ITT
(sampled-instead-of-cited) version. Within-family stratified bootstrap CI.

Writes <merged>/a1_contrast.json.   Usage: python -m experiments.a1_contrast --dir <merged>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments._loglib import load_runs_with_flags, stratified_boot


def _contrast(runs, cited=True):
    hc, pc = ("HC", "PC") if cited else ("HS", "PS")
    by_fam: dict[str, dict] = {}
    for r in runs:
        d = by_fam.setdefault(r["family"], {"a1": [], "a0": []})
        if r[pc] and not r[hc]:
            d["a1"].append(r["outcome"])
        elif not r[pc] and not r[hc]:
            d["a0"].append(r["outcome"])
    num = den = 0.0
    fam_rows = {}
    for f, d in by_fam.items():
        if not d["a1"] or not d["a0"]:
            continue
        m1 = sum(d["a1"]) / len(d["a1"])
        m0 = sum(d["a0"]) / len(d["a0"])
        w = len(d["a1"])
        num += w * (m1 - m0)
        den += w
        fam_rows[f] = {"poison_pure_solve": round(m1, 4), "base_b_f": round(m0, 4),
                       "diff": round(m1 - m0, 4), "n_arm1": len(d["a1"]), "n_arm0": len(d["a0"])}
    if den == 0:
        return None, fam_rows
    return num / den, fam_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    merged = Path(args.dir)
    runs = load_runs_with_flags(merged)

    out = {"n_runs": len(runs)}
    for kind, cited in (("cited", True), ("sampled_ITT", False)):
        est, fam_rows = _contrast(runs, cited)
        if est is None:
            out[kind] = None
            continue
        lo, hi = stratified_boot(
            runs, lambda x, c=cited: _contrast(x, c)[0], n_boot=args.n_boot, seed=args.seed)
        out[kind] = {"contrast_weighted": est, "ci95": [lo, hi],
                     "n_families": len(fam_rows)}
        if cited:
            out["per_family_cited"] = fam_rows
    (merged / "a1_contrast.json").write_text(json.dumps(out, indent=2))
    c = out["cited"]
    print(f"[a1_contrast] {merged}: poison-pure - b_f = {c['contrast_weighted']:+.4f} "
          f"CI[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] (ITT {out['sampled_ITT']['contrast_weighted']:+.4f})")


if __name__ == "__main__":
    main()
