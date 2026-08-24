#!/usr/bin/env python3
"""appendix_c/g2_step_probe.py — the paired forced-context step probe.

THE decisive cheap measurement for G2 and G3. `R = o(1 + λe)` can only rank latent
poison if following the longer recipe actually costs the agent STEPS. That is an
empirical question about the agent, not about the recipe: the corpus's own decode layers
"collapse into one piped round" (`pipeline/orchestrator.py:225`), and an agent that
compresses a three-step detour into one piped command pays nothing for it.

Design — paired, within-challenge, forced context (M=1, no retrieval stochasticity, so
the lesson is the only thing that varies):

    none            no lesson in context                       (the base rate)
    helpful         the family's correct recipe
    latent_detour   correct flag, route lengthened by rot13∘rot13
    latent_search   correct flag, reached by exhaustive trial
    sibling_poison  the prior work's benign-false arm          (reference)

Every condition sees the SAME challenge instance under the SAME solver seed, so
differences are within-challenge and pair exactly. Budget is deliberately GENEROUS here
(the point is to measure the step cost, not to censor it at the cap); G3 runs the tight
budget where that cost converts into failures.

Sharded across solver endpoints by process (`solver.solver` binds its OpenAI client at
import from VLLM_URL, so one endpoint per process).

    python -m appendix_c.g2_step_probe --shard 0 --n-shards 2 --n-per-family 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmark.hard_ctf as hard
import benchmark.sql_domain as sqld
from knowledge_graph.models import Node, NodeType
from appendix_c import latent_poison as lp

CONDITIONS = ("none", "helpful", "latent_detour", "latent_search", "sibling_poison")


def _node(text, sig, name_seed, category="crypto") -> Node:
    """A lesson node shaped exactly like the seed graph's (so context rendering,
    citation matching and credit treat it identically)."""
    import hashlib
    return Node(id=f"probe::{hashlib.sha1(name_seed.encode()).hexdigest()[:10]}",
                type=NodeType.TECHNIQUE,
                attributes={
                    "canonical_name": f"m{hashlib.sha1(text.encode()).hexdigest()[:4]}",
                    "invocation_signature": list(sig), "keywords": list(sig),
                    "text": text, "category": category, "tier": "trusted", "source": "seed",
                    "applicability": {"categories": [category], "confidence": 0.9},
                })


def build_condition_nodes(domain: str, fam_name: str):
    """Map condition name -> [Node] (or [] for the no-lesson base rate)."""
    out = {"none": []}
    if domain == "crypto":
        by_name = {f.name: f for f in hard.GRID_FAMILIES}
        by_name.update({f.name: f for f in hard.FAMILIES})
        fam = by_name[fam_name]
        from experiments.execution_verifier import _sibling_map
        out["helpful"] = [_node(fam.true_recipe, fam.true_sig, f"h{fam_name}")]
        sib = _sibling_map().get(fam_name)
        if sib is not None and sib.name != fam_name:
            out["sibling_poison"] = [_node(fam.poison_recipe or sib.true_recipe,
                                           fam.poison_sig or sib.true_sig, f"p{fam_name}")]
        for les in lp.crypto_latent_lessons([fam_name]).get(fam_name, []):
            out[f"latent_{les.arm}"] = [_node(les.recipe_text, les.signature,
                                              f"l{les.arm}{fam_name}")]
    else:
        by_name = {f.name: f for f in sqld.GRID_FAMILIES}
        fam = by_name[fam_name]
        htext, hsig = sqld._recipe_text(fam)
        out["helpful"] = [_node(htext, hsig, f"h{fam_name}", "forensics")]
        ptext, psig = sqld._recipe_text(sqld._sibling(fam))
        out["sibling_poison"] = [_node(ptext, psig, f"p{fam_name}", "forensics")]
        for les in lp.sql_latent_lessons([fam_name]).get(fam_name, []):
            out[f"latent_{les.arm}"] = [_node(les.recipe_text, les.signature,
                                              f"l{les.arm}{fam_name}", "forensics")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="crypto", choices=["crypto", "sql"])
    ap.add_argument("--out", default="appendix_c/results/g2_step_probe")
    ap.add_argument("--n-families", type=int, default=12)
    ap.add_argument("--n-per-family", type=int, default=4)
    ap.add_argument("--max-rounds", type=int, default=8,
                    help="GENEROUS on purpose: censoring at the cap would hide the step cost")
    ap.add_argument("--world-seed", type=int, default=424242)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--max-output-tokens", type=int, default=384,
                    help="raise if the solver truncates mid-JSON (parse failures burn rounds)")
    ap.add_argument("--families", default=None, help="explicit comma list (overrides selection)")
    ap.add_argument("--save-commands", action="store_true",
                    help="record executed commands per run (mechanism analysis)")
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(",")]
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    jsonl = outdir / f"rows_shard{args.shard:02d}.jsonl"

    mod = hard if args.domain == "crypto" else sqld
    if args.families:
        fams = [f.strip() for f in args.families.split(",") if f.strip()]
    elif args.domain == "crypto":     # depth-1: both latent arms are defined there
        fams = [f for f in hard.grid_family_names(seed=args.world_seed)
                if len(hard._inner_tags_of(f)) == 1][:args.n_families]
    else:
        fams = sqld.grid_family_names(n=args.n_families, seed=args.world_seed)

    corpus = mod.generate_corpus(outdir / "corpus", n_per_family=args.n_per_family,
                                 seed=args.world_seed, families=fams)
    corpus = [c for i, c in enumerate(corpus) if i % args.n_shards == args.shard]

    from batch_test.real_adapters import real_solve_fn
    solve = real_solve_fn(max_output_tokens=args.max_output_tokens, context_window=6144)

    done = set()
    if jsonl.exists():                                  # resumable
        for line in jsonl.read_text().splitlines():
            try:
                r = json.loads(line); done.add((r["challenge"], r["condition"]))
            except Exception:
                pass

    t_start = time.time()
    n_run = 0
    with open(jsonl, "a") as fh:
        for ch in corpus:
            nodes_by_cond = build_condition_nodes(args.domain, ch["_family"])
            for cond in conds:
                if cond not in nodes_by_cond or (ch["name"], cond) in done:
                    continue
                # SAME seed across conditions -> paired within challenge
                seed = abs(hash((ch["name"], args.world_seed))) % (2**31 - 1)
                t0 = time.time()
                try:
                    res = solve(ch, nodes_by_cond[cond], args.max_rounds, seed=seed)
                    err = None
                except Exception as e:
                    res, err = {}, f"{type(e).__name__}: {e}"
                    traceback.print_exc()
                flag = res.get("flag")
                correct = bool(flag) and mod.check_flag(ch, flag)
                row = {
                    "domain": args.domain, "challenge": ch["name"], "family": ch["_family"],
                    "condition": cond, "solved": int(correct),
                    "returned_flag_but_wrong": int(bool(flag) and not correct),
                    "rounds_used": res.get("rounds_used"),
                    "total_commands": res.get("total_commands"),
                    "gave_up_reason": res.get("gave_up_reason"),
                    "max_rounds": args.max_rounds, "seed": seed,
                    "wall_s": round(time.time() - t0, 2),
                    "solution_length": ch.get("_solution_length"),
                    "error": err, "shard": args.shard,
                    # executed commands: lets the analysis ask whether the agent actually
                    # FOLLOWED the longer route or silently collapsed it (§G2 mechanism)
                    "commands": ([str(c) for rec in (res.get("trace") or [])
                                  for c in (rec.get("commands") or [])]
                                 if args.save_commands else None),
                    "model": os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
                    "vllm_url": os.getenv("VLLM_URL", ""),
                }
                fh.write(json.dumps(row) + "\n"); fh.flush()
                n_run += 1
                print(f"[{args.shard}] {ch['name']:<28} {cond:<15} solved={correct} "
                      f"rounds={row['rounds_used']} cmds={row['total_commands']} "
                      f"({row['wall_s']}s)", flush=True)

    print(f"[shard {args.shard}] {n_run} runs in {time.time()-t_start:.0f}s -> {jsonl}")


if __name__ == "__main__":
    main()
