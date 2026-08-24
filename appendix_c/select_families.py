#!/usr/bin/env python3
"""appendix_c/select_families.py — pick in-band families from the calibration probe.

The prior work documents models being "harness-FLOORED" on this corpus. A family the
solver never solves (floor) or always solves (ceiling) carries no information about a
lesson in either direction, so campaigns run on it burn GPU for nothing.

The selection rule is fixed in DECISIONS.md D-04 *before* the probe ran, and the probe
contains ONLY `none` and `helpful` conditions — it never sees a latent-poison arm — so
selection cannot bias the arms under test.

    keep family iff  helpful_rate > none_rate  and  none_rate <= none_max
                     and helpful_rate >= helpful_min

    python -m appendix_c.select_families --dir appendix_c/results/calib_tok768
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

from appendix_c.provenance import write_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="appendix_c/results/calib_tok768")
    ap.add_argument("--out", default="inband_selection_crypto")
    ap.add_argument("--none-max", type=float, default=0.5,
                    help="drop families the solver already solves unaided this often")
    ap.add_argument("--helpful-min", type=float, default=0.5,
                    help="drop families the helpful lesson cannot rescue (floored)")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(str(Path(args.dir) / "rows_shard*.jsonl"))):
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print(f"no calibration rows in {args.dir}"); return

    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("error"):
            continue
        agg[r["family"]][r["condition"]].append(r)

    per_family, kept = [], []
    for fam, conds in sorted(agg.items()):
        none = [x["solved"] for x in conds.get("none", [])]
        helpful = [x["solved"] for x in conds.get("helpful", [])]
        if not none or not helpful:
            continue
        nr, hr = float(np.mean(none)), float(np.mean(helpful))
        keep = (hr > nr) and (nr <= args.none_max) and (hr >= args.helpful_min)
        reason = ("in_band" if keep else
                  "no_lesson_effect" if hr <= nr else
                  "ceiling" if nr > args.none_max else "floored")
        rec = {"family": fam, "none_rate": nr, "helpful_rate": hr,
               "n_none": len(none), "n_helpful": len(helpful),
               "lift": hr - nr, "keep": keep, "reason": reason,
               "mean_rounds_helpful": float(np.mean([x["rounds_used"] or 0
                                                     for x in conds["helpful"]]))}
        per_family.append(rec)
        if keep:
            kept.append(fam)

    payload = {
        "rule": {"none_max": args.none_max, "helpful_min": args.helpful_min,
                 "text": "keep iff helpful_rate > none_rate and none_rate <= none_max "
                         "and helpful_rate >= helpful_min (fixed in DECISIONS.md D-04 "
                         "before the probe ran; probe contains no latent condition)"},
        "source_dir": args.dir, "n_families_probed": len(per_family),
        "n_kept": len(kept), "in_band_families": kept,
        "model": rows[0].get("model"), "per_family": per_family,
    }
    path = write_result(args.out, payload, script="appendix_c/select_families.py",
                        seeds=sorted({r["seed"] for r in rows})[:20],
                        model=rows[0].get("model", "unknown"))

    print(f"\nFAMILY SELECTION — {len(per_family)} probed, {len(kept)} in band")
    print(f"{'family':<18} {'none':>6} {'helpful':>8} {'lift':>6}  {'':<4} reason")
    for r in sorted(per_family, key=lambda x: -x["lift"]):
        print(f"{r['family']:<18} {r['none_rate']:>6.2f} {r['helpful_rate']:>8.2f} "
              f"{r['lift']:>+6.2f}  {'KEEP' if r['keep'] else '    '} {r['reason']}")
    print(f"\nin-band: {','.join(kept)}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
