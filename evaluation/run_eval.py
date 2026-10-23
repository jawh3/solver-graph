#!/usr/bin/env python3
"""
evaluation/run_eval.py - the §16 "definition of done" entry point.

    python -m evaluation.run_eval --systems B0,B1,B2,B3,B4,B5,B6 \
        --datasets seed,heldout,families --mock --out results/eval

Runs the baseline ladder over the dataset partitions and writes paper-ready JSON
tables (per-system solve rate, paired lift over B0 with bootstrap CIs, median
step-to-flag, recovery-masking rate, write-back justification, power analysis).

`--mock` (default) runs the GPU-free synthetic world so the WHOLE pipeline is
exercised offline. The real path (no `--mock`) is launched on the GPU machine via
`batch_test/` once a seed graph + vLLM are available; the harness code is
identical, only the graph_factory and solve_fn change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import SolverConfig
from evaluation.baselines import get_baselines, FULL_LADDER
from evaluation.harness import EvalHarness
from evaluation.synthetic import build_world
from evaluation.datasets import make_synthetic_partitions


def main():
    ap = argparse.ArgumentParser(description="CTF solver evaluation harness (§14)")
    ap.add_argument("--systems", default="B0,B1,B2,B3,B4,B5,B6")
    ap.add_argument("--datasets", default="seed,heldout,families")
    ap.add_argument("--mock", action="store_true", default=True,
                    help="Run the GPU-free synthetic world (default).")
    ap.add_argument("--n-challenges", type=int, default=48)
    ap.add_argument("--train-epochs", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--world-seed", type=int, default=4)
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args()

    cfg = SolverConfig().with_overrides(
        N=60, R=30, M=8, n_clusters=12, embedding_dim=64,
        context_resolution="category", seed=1)
    system_ids = [s.strip() for s in args.systems.split(",") if s.strip()]
    dataset_ids = [d.strip() for d in args.datasets.split(",") if d.strip()]
    systems = get_baselines(system_ids)

    # Build the world once to derive partitions; rebuild fresh per system via factory.
    base_world = build_world(n_challenges=args.n_challenges, dim=64, seed=args.world_seed)
    partitions = make_synthetic_partitions(base_world)

    def graph_factory():
        w = build_world(n_challenges=args.n_challenges, dim=64, seed=args.world_seed)
        return w.store, w.embedder

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    harness = EvalHarness(cfg, results_root=str(out_dir / "logs"), base_seed=12345)

    summary = {"config_hash": cfg.hash(), "datasets": {}}
    for ds in dataset_ids:
        if ds not in partitions:
            print(f"  (skipping unknown dataset {ds})")
            continue
        challenges = partitions[ds]
        print(f"\n=== dataset={ds}  ({len(challenges)} challenges) ===")
        result = harness.evaluate(systems, challenges, graph_factory,
                                  dataset_name=ds, train_epochs=args.train_epochs,
                                  n_seeds=args.n_seeds)
        result.pop("_reports", None)   # drop in-memory handles before serializing
        summary["datasets"][ds] = result
        _print_table(ds, result)

    out_file = out_dir / "results_table.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote paper-ready table to {out_file}")


def _print_table(ds: str, result: dict):
    print(f"  {'system':<6} {'solve':>7} {'clean':>7} {'med_steps':>10} {'rec_mask':>9} "
          f"{'lift/B0':>9} {'95% CI':>18} {'sig':>4}")
    lifts = result.get("paired_lift_over_B0", {})
    for sid, row in result["systems"].items():
        lift = lifts.get(sid)
        if lift:
            ci = f"[{lift['ci_low']:+.3f},{lift['ci_high']:+.3f}]"
            sig = "*" if lift["significant"] else ""
            liftv = f"{lift['lift']:+.3f}"
        else:
            ci, sig, liftv = "-", "", "-"
        ms = row["median_steps_to_flag"]
        print(f"  {sid:<6} {row['solve_rate']:>7.3f} {row['solve_rate_excl_non_exploit']:>7.3f} "
              f"{(ms if ms is not None else 0):>10.1f} {row['recovery_masking_rate']:>9.3f} "
              f"{liftv:>9} {ci:>18} {sig:>4}")
    wb = result.get("writeback_justification")
    if wb:
        print(f"  write-back: {wb['note']}")


if __name__ == "__main__":
    main()
