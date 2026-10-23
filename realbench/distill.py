#!/usr/bin/env python3
"""realbench/distill.py - distill one lesson per Cybench task from its reference solution + write-up with the served model.

    export MODEL_NAME=... VLLM_URL=http://127.0.0.1:8010/v1
    python -m realbench.distill --out results/realbench_mem/<model>/lessons.jsonl --parallel 4
    python -m realbench.distill --out ... --review          # print every stored lesson as the solver would see it

Resumable (tasks already in the file are skipped). `--dry-run` prints the source material sizes per task without a model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from realbench import cybench as cb
from realbench.lessons import LessonPool, distill_all, flag_prefixes, gather_sources, scrub_flags


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("CYBENCH_ROOT", "<path>"))
    ap.add_argument("--cyber-zero-root", default=os.environ.get("CYBER_ZERO_CYBENCH_ROOT", "<path>"))
    ap.add_argument("--tasks", default="all", help="all | comma-separated task ids / suffixes / display names")
    ap.add_argument("--out", required=True); ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--max-attempts", type=int, default=3); ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--review", action="store_true")
    from realbench.run_cybench import add_bench_args, load_bench
    add_bench_args(ap)
    args = ap.parse_args(argv)
    bench, tasks = load_bench(args)
    sources_fn = getattr(bench, "gather_sources", gather_sources)   # nyuctf ships its own; cybench uses lessons.gather_sources
    if not tasks:
        sys.exit("no tasks selected")
    if args.review:
        from solver.context_assembly import render_node
        pool = LessonPool.load(args.out)
        for n in pool.nodes:
            print(f"--- {n.attributes['_display_name']} ({n.attributes['category']}; sig {n.attributes['invocation_signature']})\n{render_node(n)}\n")
        print(f"{len(pool.nodes)} lessons"); return 0
    if args.dry_run:
        known = [f for t in tasks for f in t.accepted_flags]; pre = flag_prefixes(tasks)
        for t in tasks:
            src = sources_fn(t); n_scrub = sum(scrub_flags(v, known, pre)[1] for v in src.values())
            print(f"{t.display_name} {t.category:9s} sources={ {k: len(v) for k, v in src.items()} } flags_scrubbed={n_scrub}")
        print(f"{len(tasks)} tasks; flag prefixes {pre}"); return 0
    from openai import OpenAI
    model = os.environ.get("MODEL_NAME") or sys.exit("MODEL_NAME not set")
    client = OpenAI(base_url=os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1"), api_key="not-needed", timeout=1800.0)
    summ = distill_all(tasks, client, model, args.out, parallel=args.parallel, max_attempts=args.max_attempts,
                       max_tokens=args.max_tokens, temperature=args.temperature, sources_fn=sources_fn)
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "failed"}), flush=True)
    return 0 if summ["n_failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
