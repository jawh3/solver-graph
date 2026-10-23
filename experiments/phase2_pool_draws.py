#!/usr/bin/env python3
"""experiments/phase2_pool_draws.py - a held-out contrast pooled over independent draws (Phase 2, P2-15).

Each draw is an (arm, twin) pair run with its own sampling stream; the pooled contrast stacks the paired held-out items of
all draws and takes the family-mean difference, with the same family bootstrap and sign-permutation test as
phase2_ladder.py. Default: causal (leave-one-out) credit vs outcome credit at 32B, three draws.

    python experiments/phase2_pool_draws.py --root results/lab/phase2_32b
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_ladder import eval_metrics, fam_diff  # noqa: E402

DRAWS = [("deployed_causal_alphabet", "deployed_alphabet_tau01"),
         ("deployed_causal_alphabet_rep1", "deployed_alphabet_tau01_rep1"),
         ("deployed_causal_alphabet_rep2", "deployed_alphabet_tau01_rep2")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    E = {"a": {}, "b": {}}
    for d, (a, b) in enumerate(DRAWS):
        ea, eb = eval_metrics(args.root, a), eval_metrics(args.root, b)
        for k in set(ea) & set(eb):
            E["a"][(d,) + k] = ea[k]
            E["b"][(d,) + k] = eb[k]
    keys = sorted(E["a"])
    r = fam_diff(E, "a", "b", keys)
    print(json.dumps({"draws": DRAWS, "paired_items": len(keys)} | r, indent=1))


if __name__ == "__main__":
    main()
