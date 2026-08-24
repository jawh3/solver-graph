#!/usr/bin/env python3
"""
experiments/lab_merge.py — merge a sharded campaign into one analyzable dir.

Each shard ran run_campaign over disjoint families. This concatenates per-condition run
logs, unions the bandit cells (node ids are globally unique per family, so there is no real
collision), and merges progress eval/train metrics + the family list, producing <root>/merged
which `experiments.analyze_campaign --dir` and `experiments.lab_analysis --dir` consume directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _concat_jsonl(srcs, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(dst, "w") as out:
        for s in srcs:
            if not s.exists():
                continue
            with open(s) as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="campaign root containing shard_* and manifest.json")
    ap.add_argument("--conditions", default="B0,B2,outcome,observational")
    args = ap.parse_args()
    root = Path(args.root)
    shards = sorted(root.glob("shard_*"))
    conds = [c.strip() for c in args.conditions.split(",")]
    merged = root / "merged"
    (merged / "logs").mkdir(parents=True, exist_ok=True)
    (merged / "state").mkdir(parents=True, exist_ok=True)

    # 1) concat per-condition run/step/citation/update logs
    for cond in conds:
        for kind in ("runs", "steps", "citations", "updates"):
            srcs = [s / "logs" / f"campaign_{cond}" / f"{kind}.jsonl" for s in shards]
            n = _concat_jsonl(srcs, merged / "logs" / f"campaign_{cond}" / f"{kind}.jsonl")
            if kind == "runs":
                print(f"  {cond}: merged {n} runs from {len(shards)} shards")

    # 2) merge bandit cells per learning condition. NAMESPACE each shard's context ids
    # (shardK::ctx) so cells never collide across replicate shards — then the per-node mean
    # Q (over all shard-contexts) computed downstream POOLS the learned Q across shards,
    # rather than one shard overwriting another (which would discard most of the data).
    for cond in conds:
        cells = {}
        base = None
        for si, s in enumerate(shards):
            bp = s / "state" / cond / "bandit.json"
            if not bp.exists():
                continue
            d = json.loads(bp.read_text())
            base = base or d
            for ctx, nodes in d.get("cells", {}).items():
                cells[f"s{si:02d}::{ctx}"] = nodes
        if base is not None:
            base["cells"] = cells
            (merged / "state" / cond).mkdir(parents=True, exist_ok=True)
            (merged / "state" / cond / "bandit.json").write_text(json.dumps(base))

    # 3) merge progress metrics + family union
    families, conditions = set(), {}
    for s in shards:
        pp = s / "progress.json"
        if not pp.exists():
            continue
        d = json.loads(pp.read_text())
        for name, cp in d.get("conditions", {}).items():
            agg = conditions.setdefault(name, {"metrics": [], "train_solves": 0,
                                               "causal_rerolls": 0, "done": True})
            agg["metrics"].extend(cp.get("metrics", []))
            agg["train_solves"] += cp.get("train_solves", 0)
            agg["causal_rerolls"] += cp.get("causal_rerolls", 0)
            agg["done"] = agg["done"] and cp.get("done", False)
            for m in cp.get("metrics", []):
                families.add(m.get("family"))
    (merged / "progress.json").write_text(json.dumps({"conditions": conditions}, indent=2))
    families.discard(None)
    summ = {"families": sorted(families), "n_shards": len(shards)}
    (merged / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"[merge] {len(shards)} shards -> {merged}  ({len(families)} families, "
          f"conditions={list(conditions)})")
    print(f"[merge] analyze:  python -m experiments.lab_analysis --dir {merged} --cond outcome")


if __name__ == "__main__":
    main()
