#!/usr/bin/env python3
"""
experiments/lab_campaign.py — launch a SHARDED parallel campaign.

run_campaign is sequential within a process (online-bandit conditions must be). To use the
vLLM server's batching we shard the in-band families into G groups and launch G run_campaign
processes concurrently (each detached, resumable). Each shard owns disjoint families, so the
per-lesson causal estimates pool cleanly across shards (lessons are unique per family); solve
rates pool across all eval items. Merge with experiments/lab_merge.py, analyze with lab_analysis.

Usage:
  python -m experiments.lab_campaign --families-file a probe artifact not retained \
      --out results/lab/campaign_7b --shards 24 --n-per-family 10 --epochs 5 --eval-seeds 5 \
      --slack 2 --conditions B0,B2,outcome,observational --n-clusters 4
Re-run to (re)launch only shards that are not yet done (each shard's run_campaign resumes).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _load_families(args):
    if args.families:
        return [f.strip() for f in args.families.split(",")]
    p = Path(args.families_file)
    d = json.loads(p.read_text())
    fams = d.get("in_band_families") or d.get("families")
    if fams and isinstance(fams[0], dict):
        fams = [f["family"] for f in fams]
    return list(fams)


def _shard(items, g):
    groups = [[] for _ in range(g)]
    for i, it in enumerate(items):
        groups[i % g].append(it)        # round-robin for balance
    return [grp for grp in groups if grp]


def _shard_done(out):
    prog = Path(out) / "progress.json"
    if not prog.exists():
        return False
    try:
        d = json.loads(prog.read_text())
        return all(c.get("done") for c in d.get("conditions", {}).values()) and d.get("conditions")
    except Exception:
        return False


def _shard_running(out):
    """True if a run_campaign process for this shard out-dir is alive (avoids launching a
    duplicate of a still-running/hung shard — re-launch is then idempotent)."""
    r = subprocess.run(["pgrep", "-f", f"run_campaign --out {out} "],
                       capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families-file", default="a probe artifact not retained")
    ap.add_argument("--families", default=None, help="explicit comma list (overrides file)")
    ap.add_argument("--out", default="results/lab/campaign")
    ap.add_argument("--shards", type=int, default=24)
    ap.add_argument("--n-per-family", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--slack", type=int, default=2)
    ap.add_argument("--conditions", default="B0,B2,outcome,observational")
    ap.add_argument("--n-clusters", type=int, default=4)
    ap.add_argument("--M", type=int, default=None, help="lessons sampled into context (small M = less dilution)")
    ap.add_argument("--kappa", type=float, default=None, help="similarity-prior pseudo-count")
    ap.add_argument("--world-seed", type=int, default=424242)
    ap.add_argument("--malignant-arms", default=None,
                    help="passthrough to run_campaign: comma subset of destroy,exhaust,wrongflag (T1-1)")
    ap.add_argument("--leak-names", action="store_true",
                    help="passthrough to run_campaign: reveal family name in filename/desc (T2-2 A/B)")
    ap.add_argument("--distractor-arm", action="store_true", help="passthrough: on-target distractor (T2-3)")
    ap.add_argument("--replicate", action="store_true",
                    help="CORRECT mode for causal estimation: every shard runs the FULL family "
                         "set (so the candidate pool >> M and lesson inclusion is genuinely "
                         "stochastic) with a DISTINCT world-seed (different fresh challenges). "
                         "Pool all shard logs. Default (off) partitions families — only valid "
                         "when each shard still has many families.")
    ap.add_argument("--vllm-urls", default=None,
                    help="comma list of solver endpoints to round-robin shards across "
                         "(e.g. http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1). "
                         "Default: inherit VLLM_URL for all shards.")
    args = ap.parse_args()
    urls = [u.strip() for u in args.vllm_urls.split(",")] if args.vllm_urls else None

    fams = _load_families(args)
    if args.replicate:
        # every shard = ALL families, distinct world-seed (different fresh challenges).
        shard_specs = [(fams, args.world_seed + 1000 * i) for i in range(args.shards)]
    else:
        shard_specs = [(grp, args.world_seed) for grp in _shard(fams, args.shards)]
    root = Path(args.out); (root / "logs").mkdir(parents=True, exist_ok=True)
    manifest = {"out": str(root), "n_families": len(fams), "n_shards": len(shard_specs),
                "mode": "replicate" if args.replicate else "partition",
                "conditions": args.conditions, "slack": args.slack,
                "n_per_family": args.n_per_family, "epochs": args.epochs,
                "eval_seeds": args.eval_seeds, "shards": []}
    launched = 0
    for i, (shard_fams, wseed) in enumerate(shard_specs):
        sout = root / f"shard_{i:02d}"
        rec = {"shard": i, "out": str(sout), "families": shard_fams, "n": len(shard_fams),
               "world_seed": wseed}
        if _shard_done(sout):
            rec["status"] = "done"; manifest["shards"].append(rec); continue
        if _shard_running(sout):                       # idempotent: don't duplicate a live shard
            rec["status"] = "running"; manifest["shards"].append(rec); continue
        logf = root / "logs" / f"shard_{i:02d}.log"
        cmd = [sys.executable, "-m", "experiments.run_campaign",
               "--out", str(sout), "--families", ",".join(shard_fams),
               "--conditions", args.conditions, "--n-per-family", str(args.n_per_family),
               "--epochs", str(args.epochs), "--eval-seeds", str(args.eval_seeds),
               "--slack", str(args.slack), "--n-clusters", str(args.n_clusters),
               "--world-seed", str(wseed)]
        if args.M is not None:
            cmd += ["--M", str(args.M)]
        if args.kappa is not None:
            cmd += ["--kappa", str(args.kappa)]
        if getattr(args, "malignant_arms", None):
            cmd += ["--malignant-arms", args.malignant_arms]
        if getattr(args, "leak_names", False):
            cmd += ["--leak-names"]
        if getattr(args, "distractor_arm", False):
            cmd += ["--distractor-arm"]
        env = dict(os.environ, PYTHONPATH=str(Path.cwd()))
        if urls:
            env["VLLM_URL"] = urls[i % len(urls)]      # round-robin shards across servers
            rec["vllm_url"] = env["VLLM_URL"]
        with open(logf, "a") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True, env=env)
        rec["status"] = "launched"; rec["pid"] = p.pid; rec["log"] = str(logf)
        manifest["shards"].append(rec); launched += 1
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[lab_campaign] {len(fams)} families ({manifest['mode']}) -> {len(shard_specs)} shards; "
          f"launched {launched}, {len(shard_specs)-launched} already done. manifest: {root/'manifest.json'}")
    print(f"[lab_campaign] monitor: tail -f {root}/logs/shard_*.log ; progress via shard_*/progress.json")


if __name__ == "__main__":
    main()
