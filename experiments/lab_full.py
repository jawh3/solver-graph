#!/usr/bin/env python3
"""
experiments/lab_full.py — ONE detached process that runs a whole model's study end to end:
  probe (label-blind in-band + teeth selection) -> campaign (replicate) -> merge -> analyze.

This is the robust unattended entry point (no fragile background waiters between stages). The
model must already be served at --vllm-urls. Resumable: re-running skips the probe if its
selection.json exists and resumes the campaign via lab_pipeline.

  setsid nohup env VLLM_URL=http://127.0.0.1:8000/v1 MODEL_NAME=Qwen/Qwen3.5-27B-FP8 \
    python -m experiments.lab_full --tag 27b --vllm-urls http://127.0.0.1:8000/v1 \
    --probe-seeds 3 --shards 12 --n-per-family 6 --epochs 4 --eval-seeds 4 --slack 2 \
    > logs/full_27b.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd, env=None):
    print(f"[full] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="results/lab/<tag>_* dirs")
    ap.add_argument("--vllm-urls", required=True, help="comma list; probe uses the first")
    ap.add_argument("--probe-seeds", type=int, default=3)
    ap.add_argument("--probe-inst", type=int, default=1)
    ap.add_argument("--probe-workers", type=int, default=24)
    ap.add_argument("--difficulty", default=None, help="probe family band filter (default all 192)")
    ap.add_argument("--shards", type=int, default=12)
    ap.add_argument("--n-per-family", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--eval-seeds", type=int, default=4)
    ap.add_argument("--slack", type=int, default=2)
    ap.add_argument("--conditions", default="B0,B2,outcome,observational")
    ap.add_argument("--n-clusters", type=int, default=8)
    ap.add_argument("--cond", default="outcome")
    ap.add_argument("--deadline-hours", type=float, default=18.0)
    args = ap.parse_args()

    urls = [u.strip() for u in args.vllm_urls.split(",")]
    probe_dir = Path(f"results/lab/probe_{args.tag}")
    sel = probe_dir / "selection.json"
    py = sys.executable

    # 1) PROBE (single process on the first endpoint) — skip if already selected
    if not sel.exists():
        import os
        penv = dict(os.environ, VLLM_URL=urls[0], PYTHONPATH=str(Path.cwd()))
        pcmd = [py, "-m", "experiments.lab_probe", "--out", str(probe_dir),
                "--n-inst", str(args.probe_inst), "--seeds", str(args.probe_seeds),
                "--workers", str(args.probe_workers), "--slack", str(args.slack)]
        if args.difficulty:
            pcmd += ["--difficulty", args.difficulty]
        _run(pcmd, env=penv)
    selj = json.loads(sel.read_text())
    print(f"[full] in-band families: {selj['n_in_band']} (teeth {selj['teeth_pass']})", flush=True)
    if selj["n_in_band"] < 2:
        print("[full] FATAL: <2 in-band families — nothing to run.", flush=True)
        return

    # 2) CAMPAIGN -> MERGE -> ANALYZE via the resilient pipeline (replicate; pools logs)
    pl = [py, "-m", "experiments.lab_pipeline", "--families-file", str(sel),
          "--out", f"results/lab/campaign_{args.tag}", "--vllm-urls", args.vllm_urls,
          "--replicate", "--shards", str(args.shards), "--n-per-family", str(args.n_per_family),
          "--epochs", str(args.epochs), "--eval-seeds", str(args.eval_seeds),
          "--slack", str(args.slack), "--conditions", args.conditions,
          "--n-clusters", str(args.n_clusters), "--cond", args.cond,
          "--deadline-hours", str(args.deadline_hours)]
    _run(pl)
    print(f"[full] DONE tag={args.tag}. analysis: results/lab/campaign_{args.tag}/merged/lab_analysis.md", flush=True)


if __name__ == "__main__":
    main()
