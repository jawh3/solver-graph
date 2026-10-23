#!/usr/bin/env python3
"""
batch_test/run_batch.py - the coordinated REAL-GPU evaluation (the merge-time run).

The run on the GPU server after merging. It runs the SAME
`evaluation.EvalHarness` ladder as the mock smoke test, but with the real Qwen
solver, the real vLLM embedder, and the real cross-encoder reranker. Because every
run logs propensities (§12), the parameter sweep happens OFFLINE afterward
(`python -m ope.sweep`) with NO extra GPU.

    python -m batch_test.run_batch \
        --challenges data/challenges.jsonl \
        --graph data/seed_graph/kg \
        --systems B0,B1,B2,B3,B4,B5,B6,BC \
        --vllm-url http://localhost:8000/v1 \
        --embed-url http://localhost:8001/v1/embeddings \
        --out results/real_eval

Challenge JSONL schema (one object per line):
    {"name","category","points"|"difficulty","description",
     "hints":[...], "code_blocks":[...], "local_files":[...],
     "dataset":"seed"|"heldout"|"families", "contamination_flag":bool}

Partitioning: challenges are grouped by their `dataset` field; HELD-OUT-NOVEL +
TRANSFORMED-FAMILIES carry the headline numbers (§14.1). Lift is over B0.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from config import SolverConfig
from evaluation.baselines import get_baselines
from evaluation.harness import EvalHarness


def load_challenges(path: str) -> List[Dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def partition(challenges: List[Dict]) -> Dict[str, List[Dict]]:
    parts = defaultdict(list)
    for ch in challenges:
        parts[ch.get("dataset", "seed")].append(ch)
    return dict(parts)


def main():
    ap = argparse.ArgumentParser(description="Coordinated real-GPU eval (§14)")
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--graph", required=True, help="Seed graph base path (.json/.npy pair)")
    ap.add_argument("--systems", default="B0,B1,B2,B3,B4,B5,B6,BC")
    ap.add_argument("--vllm-url", default=os.getenv("VLLM_URL", "http://localhost:8000/v1"))
    ap.add_argument("--embed-url", default=os.getenv("EMBED_URL", "http://localhost:8001/v1/embeddings"))
    ap.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "Qwen/Qwen3.5-27B-FP8"))
    ap.add_argument("--train-epochs", type=int, default=3)
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--out", default="results/real_eval")
    args = ap.parse_args()

    # real backends (heavy imports happen here, not at module load)
    from batch_test.real_adapters import real_embedder, real_reranker, real_solve_fn, make_real_factory

    cfg = SolverConfig()  # spec defaults; sweep happens offline via OPE
    embedder = real_embedder(args.embed_url, args.embed_model)
    graph_factory = make_real_factory(args.graph, embedder)

    challenges = load_challenges(args.challenges)
    parts = partition(challenges)
    systems = get_baselines([s.strip() for s in args.systems.split(",")])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    harness = EvalHarness(
        cfg, results_root=str(out_dir / "logs"), base_seed=12345,
        reranker_factory=lambda: real_reranker(), solve_fn=real_solve_fn(),
    )

    summary = {"config_hash": cfg.hash(), "model": args.embed_model, "datasets": {}}
    for ds, ds_challenges in parts.items():
        print(f"\n=== REAL dataset={ds} ({len(ds_challenges)} challenges) ===")
        result = harness.evaluate(systems, ds_challenges, graph_factory,
                                  dataset_name=ds, train_epochs=args.train_epochs,
                                  n_seeds=args.n_seeds)
        result.pop("_reports", None)
        summary["datasets"][ds] = result

    out_file = out_dir / "results_table.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote real-eval table to {out_file}")
    print("Next: offline sweep with NO extra GPU →  python -m ope.sweep "
          f"(reads {out_dir/'logs'})")


if __name__ == "__main__":
    main()
