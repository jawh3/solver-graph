#!/usr/bin/env python3
"""realbench/nyu_screen_analysis.py - the NYU CTF Bench screening estimands (docs/phase3/NYU_PLAN.md): per model, solve
rate over the usable tasks with a Wilson 95% interval, by category, by kind (file-only / service), by competition year,
and with the flag-in-handout tasks removed; across models, the union of solved tasks and how many models solve each.

    python -m realbench.nyu_screen_analysis --root results/realbench_nyu --name screen --out docs/phase3/NYU_SCREEN_RESULTS_<date>.md

One seed per task: per-task outcomes are not stable estimates; the run is a screen, not a comparison of arms.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from realbench import nyuctf as ny
from realbench.run_cybench import wilson


def _rate(recs):
    n = len(recs); k = sum(int(r["solved"]) for r in recs)
    lo, hi = wilson(k, n)[1:3] if n else (0.0, 0.0)          # wilson() returns (p, lo, hi)
    return k, n, (k / n if n else 0.0), lo, hi


def _cell(recs):
    k, n, p, lo, hi = _rate(recs)
    return f"{k}/{n} = {p:.3f} [{lo:.3f}, {hi:.3f}]" if n else "-"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/realbench_nyu"); ap.add_argument("--name", default="screen")
    ap.add_argument("--split", default="test"); ap.add_argument("--nyu-root", default=str(ny.DEFAULT_NYU_ROOT))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    tasks = {t.id: t for t in ny.enumerate_tasks(Path(args.nyu_root), split=args.split, strict=False)}
    models = {}
    for f in sorted(glob.glob(f"{args.root}/*/{args.name}.jsonl")):
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if not recs:
            continue
        model = (recs[0].get("extra") or {}).get("model") or Path(f).parent.name
        models[model] = recs
    if not models:
        raise SystemExit(f"no {args.name}.jsonl under {args.root}")
    usable = sorted({r["task_id"] for recs in models.values() for r in recs})
    L = [f"# NYU CTF Bench {args.split} screen - {len(usable)} usable tasks, one seed, no memory ({len(models)} models)", "",
         "Solve rate = exact flag match over the usable set; brackets are Wilson 95% intervals. One seed per task, so",
         "per-task outcomes are not stable estimates (docs/phase3/NYU_PLAN.md, *Estimands*). Records:",
         f"`{args.root}/<model>/{args.name}.jsonl`.", "",
         "## Per model", "",
         "| model | records | errors | solve rate [95%] | without flag-in-handout tasks | file-only | service | mean rounds | mean s/solve |",
         "|---|---|---|---|---|---|---|---|---|"]
    order = sorted(models, key=lambda m: -_rate(models[m])[2])
    for m in order:
        recs = models[m]; errs = sum(1 for r in recs if r.get("error"))
        clean = [r for r in recs if not r.get("flag_leaks")]
        fo = [r for r in recs if not r["needs_service"]]; sv = [r for r in recs if r["needs_service"]]
        L.append(f"| {m} | {len(recs)} | {errs} | {_cell(recs)} | {_cell(clean)} | {_cell(fo)} | {_cell(sv)} | "
                 f"{sum(r['rounds_used'] for r in recs) / len(recs):.1f} | {sum(r['wall_s'] for r in recs) / len(recs):.0f} |")
    cats = sorted({r["category"] for recs in models.values() for r in recs})
    L += ["", "## By category (solves/tasks = rate [95%])", "", "| model | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for m in order:
        by = defaultdict(list)
        for r in models[m]:
            by[r["category"]].append(r)
        L.append(f"| {m} | " + " | ".join(_cell(by[c]) for c in cats) + " |")
    years = sorted({tasks[r["task_id"]].year for recs in models.values() for r in recs if r["task_id"] in tasks})
    L += ["", "## By competition year", "", "| model | " + " | ".join(str(y) for y in years) + " |", "|---|" + "---|" * len(years)]
    for m in order:
        by = defaultdict(list)
        for r in models[m]:
            if r["task_id"] in tasks:
                by[tasks[r["task_id"]].year].append(r)
        L.append(f"| {m} | " + " | ".join(_cell(by[y]) for y in years) + " |")
    L += ["", "## Docker-socket service tasks vs the rest (run policy: included)", "", "| model | docker.sock services | other services | file-only |", "|---|---|---|---|"]
    for m in order:
        ds = [r for r in models[m] if r["needs_service"] and tasks.get(r["task_id"]) and tasks[r["task_id"]].docker_sock]
        os_ = [r for r in models[m] if r["needs_service"] and not (tasks.get(r["task_id"]) and tasks[r["task_id"]].docker_sock)]
        fo = [r for r in models[m] if not r["needs_service"]]
        L.append(f"| {m} | {_cell(ds)} | {_cell(os_)} | {_cell(fo)} |")
    solved_by = defaultdict(set)
    for m, recs in models.items():
        for r in recs:
            if r["solved"]:
                solved_by[r["task_id"]].add(m)
    union = sorted(solved_by); k_hist = Counter(len(v) for v in solved_by.values())
    L += ["", "## Across models", "",
          f"- Tasks solved by at least one model: **{len(union)} of {len(usable)}** ({len(union) / len(usable):.3f}).",
          "- Tasks solved by exactly k models: " + ", ".join(f"k={k}: {k_hist[k]}" for k in sorted(k_hist)) + ".",
          f"- Solved by every model: {sum(1 for v in solved_by.values() if len(v) == len(models))}.", "",
          "| task | category | kind | year | solved by |", "|---|---|---|---|---|"]
    for tid in union:
        t = tasks.get(tid); dn = next(r["display_name"] for recs in models.values() for r in recs if r["task_id"] == tid)
        L.append(f"| {dn} | {t.category if t else '?'} | {'service' if (t and t.needs_service) else 'file-only'} | {t.year if t else '?'} | "
                 f"{len(solved_by[tid])}: {', '.join(sorted(solved_by[tid]))} |")
    L += ["", "## Pairwise agreement on solved sets (Jaccard)", "", "| | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    sets = {m: {r["task_id"] for r in models[m] if r["solved"]} for m in models}
    for a in order:
        row = []
        for b in order:
            u = sets[a] | sets[b]; row.append(f"{(len(sets[a] & sets[b]) / len(u)) if u else 0:.2f}")
        L.append(f"| {a} | " + " | ".join(row) + " |")
    L += ["", "## Not claimed", "", "Nothing about memory or learning; nothing about tasks outside the usable set; no comparison to",
          "published NYU CTF Bench numbers (different scaffold, budgets, container image; 23 tasks not run)."]
    md = "\n".join(L) + "\n"
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
