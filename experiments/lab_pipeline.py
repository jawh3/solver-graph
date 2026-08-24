#!/usr/bin/env python3
"""
experiments/lab_pipeline.py — ONE detached, resumable process that runs the whole causal
campaign end to end: launch sharded run_campaign -> wait for every shard -> merge -> analyze.

This exists because chaining stages via background "waiters" proved fragile (a waiter can die
and never fire, leaving the GPUs idle). Here a single long-lived process owns the whole chain;
if it dies, re-running it resumes (run_campaign shards are checkpointed; done shards are skipped).

Run detached:
  setsid nohup python -m experiments.lab_pipeline --families-file results/lab/probe_7b/selection_merged.json \
    --out results/lab/campaign_7b --vllm-urls http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1 \
    --shards 11 --n-per-family 12 --epochs 5 --eval-seeds 5 --slack 2 > logs/pipeline_7b.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd, **kw):
    print(f"[pipeline] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, **kw)


def _all_shards_done(root: Path):
    man = root / "manifest.json"
    if not man.exists():
        return False, 0, 0
    shards = json.loads(man.read_text())["shards"]
    done = 0
    for s in shards:
        pp = Path(s["out"]) / "progress.json"
        if pp.exists():
            try:
                d = json.loads(pp.read_text())
                if d.get("conditions") and all(c.get("done") for c in d["conditions"].values()):
                    done += 1
            except Exception:
                pass
    return done == len(shards) and len(shards) > 0, done, len(shards)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families-file", default="results/lab/probe_7b/selection_merged.json")
    ap.add_argument("--families", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vllm-urls", default=None)
    ap.add_argument("--shards", type=int, default=11)
    ap.add_argument("--n-per-family", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--slack", type=int, default=2)
    ap.add_argument("--conditions", default="B0,B2,outcome,observational")
    ap.add_argument("--n-clusters", type=int, default=4)
    ap.add_argument("--M", type=int, default=None, help="lessons sampled into context")
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--world-seed", type=int, default=None,
                    help="base world-seed (use a DISTINCT value for a fresh-data replication run)")
    ap.add_argument("--malignant-arms", default=None,
                    help="passthrough: comma subset of destroy,exhaust,wrongflag (T1-1 positive control)")
    ap.add_argument("--leak-names", action="store_true", help="passthrough: reveal family name (T2-2 A/B)")
    ap.add_argument("--distractor-arm", action="store_true", help="passthrough: on-target distractor (T2-3)")
    ap.add_argument("--cond", default="outcome", help="headline condition for lab_analysis")
    ap.add_argument("--replicate", action="store_true",
                    help="each shard = full family set, distinct seed (correct causal pooling)")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--deadline-hours", type=float, default=20.0,
                    help="hard cap: on timeout, merge+analyze whatever completed and exit "
                         "(never loop forever on a hung shard).")
    args = ap.parse_args()
    root = Path(args.out)
    py = sys.executable

    # 1) launch (or resume) shards
    camp_cmd = [py, "-m", "experiments.lab_campaign", "--out", args.out,
                "--shards", str(args.shards), "--n-per-family", str(args.n_per_family),
                "--epochs", str(args.epochs), "--eval-seeds", str(args.eval_seeds),
                "--slack", str(args.slack), "--conditions", args.conditions,
                "--n-clusters", str(args.n_clusters)]
    if args.families:
        camp_cmd += ["--families", args.families]
    else:
        camp_cmd += ["--families-file", args.families_file]
    if args.vllm_urls:
        camp_cmd += ["--vllm-urls", args.vllm_urls]
    if args.replicate:
        camp_cmd += ["--replicate"]
    if args.M is not None:
        camp_cmd += ["--M", str(args.M)]
    if args.kappa is not None:
        camp_cmd += ["--kappa", str(args.kappa)]
    if args.world_seed is not None:
        camp_cmd += ["--world-seed", str(args.world_seed)]
    if args.malignant_arms:
        camp_cmd += ["--malignant-arms", args.malignant_arms]
    if getattr(args, "leak_names", False):
        camp_cmd += ["--leak-names"]
    if getattr(args, "distractor_arm", False):
        camp_cmd += ["--distractor-arm"]
    _run(camp_cmd, check=True)

    # 2) wait for all shards (poll progress; resilient to a crashed shard by re-launching
    # ONLY the dead ones — lab_campaign skips done+running shards). Hard deadline so a hung
    # shard can never block the overnight run: on timeout we merge+analyze what completed.
    t0 = time.time(); stale = 0; deadline = args.deadline_hours * 3600
    while True:
        ok, done, total = _all_shards_done(root)
        elapsed = int(time.time() - t0)
        print(f"[pipeline] shards done {done}/{total}  (elapsed {elapsed}s)", flush=True)
        if ok:
            break
        if elapsed > deadline:
            print(f"[pipeline] DEADLINE {args.deadline_hours}h hit with {done}/{total} shards "
                  f"done; proceeding to merge+analyze on completed data.", flush=True)
            break
        stale += 1
        if stale % 10 == 0:        # restart dead-and-not-done shards (idempotent; resumes)
            _run(camp_cmd)
        time.sleep(args.poll)

    # 3) merge + 4) analyze
    _run([py, "-m", "experiments.lab_merge", "--root", args.out, "--conditions", args.conditions], check=True)
    _run([py, "-m", "experiments.lab_analysis", "--dir", str(root / "merged"), "--cond", args.cond], check=False)  # non-fatal: an analysis hiccup (e.g. outcome-only run, no B0 baseline) must not lose ontarget
    # on-target (family-conditioned) estimand — the pre-registered refined primary for H2 (2026-06-03).
    # Non-fatal: a failure here must not lose the completed run's headline analysis.
    _run([py, "-m", "experiments.ontarget_analysis", "--dir", str(root / "merged"), "--cond", args.cond], check=False)
    print(f"[pipeline] DONE. results in {root/'merged'}/lab_analysis.md (+ ontarget_analysis.md)", flush=True)


if __name__ == "__main__":
    main()
