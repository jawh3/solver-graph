#!/usr/bin/env python3
"""
experiments/lab_sweep.py — run a sequence of full campaigns (one per M value) to characterize
how retrieval concentration (M) affects the H1 banking / H2 causal-flip / H3 promotion gap.
Tests the dilution hypothesis: the pilot's strong flip used M=6 of 12 nodes; at 11 families
(26 nodes) M=6 dilutes each lesson. Smaller M should concentrate the per-lesson causal signal.

Single detached, resumable process: optionally wait for a prerequisite run, then run
lab_pipeline (campaign->merge->analyze) for each M sequentially. Re-running skips M-dirs whose
analysis already exists. Each result -> results/lab/campaign_<tag>_M<m>/merged/lab_analysis.md.

  setsid nohup env OMP_NUM_THREADS=2 ... python -m experiments.lab_sweep --tag 7b \
    --wait-for results/lab/campaign_7b/merged/lab_analysis.md --m-values 4,3 \
    --vllm-urls http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1 \
    --families-file results/lab/probe_7b/selection_merged.json > logs/sweep_7b.log 2>&1 &
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="7b")
    ap.add_argument("--m-values", default="4,3")
    ap.add_argument("--wait-for", default=None, help="path to wait for before starting (prior run's analysis)")
    ap.add_argument("--families-file", default="results/lab/probe_7b/selection_merged.json")
    ap.add_argument("--vllm-urls", default="http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1")
    ap.add_argument("--shards", type=int, default=10)
    ap.add_argument("--n-per-family", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--slack", type=int, default=2)
    ap.add_argument("--n-clusters", type=int, default=4)
    ap.add_argument("--deadline-hours", type=float, default=10.0)
    args = ap.parse_args()
    py = sys.executable

    if args.wait_for:
        print(f"[sweep] waiting for prerequisite {args.wait_for} ...", flush=True)
        while not Path(args.wait_for).exists():
            time.sleep(120)
        print(f"[sweep] prerequisite ready at {time.strftime('%H:%M:%S')}", flush=True)

    for m in [int(x) for x in args.m_values.split(",")]:
        out = f"results/lab/campaign_{args.tag}_M{m}"
        if (Path(out) / "merged" / "lab_analysis.md").exists():
            print(f"[sweep] M={m} already done ({out}) — skipping", flush=True)
            continue
        print(f"[sweep] === M={m} -> {out} === ({time.strftime('%H:%M:%S')})", flush=True)
        cmd = [py, "-m", "experiments.lab_pipeline", "--families-file", args.families_file,
               "--out", out, "--vllm-urls", args.vllm_urls, "--replicate",
               "--shards", str(args.shards), "--n-per-family", str(args.n_per_family),
               "--epochs", str(args.epochs), "--eval-seeds", str(args.eval_seeds),
               "--slack", str(args.slack), "--n-clusters", str(args.n_clusters),
               "--M", str(m), "--cond", "outcome", "--deadline-hours", str(args.deadline_hours)]
        subprocess.run(cmd)
        print(f"[sweep] M={m} done ({time.strftime('%H:%M:%S')})", flush=True)
    print("[sweep] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
