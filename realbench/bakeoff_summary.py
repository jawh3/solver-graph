#!/usr/bin/env python3
"""realbench/bakeoff_summary.py - one table over every results/realbench/<model>/bakeoff.summary.json (+ per-task grid)."""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results/realbench"); ap.add_argument("--out", default=None)
    ap.add_argument("--name", default="bakeoff", help="record file stem under each <root>/<model>/ (bakeoff | screen | ...)")
    ap.add_argument("--title", default="Cybench bake-off - file-only tasks, per model")
    args = ap.parse_args()
    rows, grid, tasks = [], defaultdict(dict), set()
    for f in sorted(glob.glob(f"{args.root}/*/{args.name}.summary.json")):
        s = json.load(open(f)); model = s.get("model") or Path(f).parent.name
        rows.append((model, s))
        recs = [json.loads(l) for l in open(Path(f).with_suffix("").with_suffix(".jsonl")) if l.strip()] if Path(f).parent.joinpath(f"{args.name}.jsonl").exists() else []
        for r in recs:
            tasks.add((r["category"], r["display_name"])); g = grid[model].setdefault(r["display_name"], [0, 0]); g[0] += 1; g[1] += int(r["solved"])
    L = [f"# {args.title}", "",
         "| model | solve rate [Wilson 95%] | runs | tasks solved ≥1 / tasks | errors | brace-only | mean rounds | mean s/solve | hours |", "|---|---|---|---|---|---|---|---|---|"]
    for model, s in sorted(rows, key=lambda x: -x[1]["solve_rate"]):
        L.append(f"| {model} | {s['solve_rate']:.3f} [{s['wilson95'][0]:.3f}, {s['wilson95'][1]:.3f}] | {s['runs']} | {s['tasks_solved_at_least_once']}/{s['tasks']} | {s['errors']} | {s['brace_only']} | {s['mean_rounds']:.1f} | {s['mean_wall_s']:.0f} | {s.get('elapsed_s', 0) / 3600:.1f} |")
    if grid:
        models = [m for m, _ in sorted(rows, key=lambda x: -x[1]["solve_rate"])]
        L += ["", "## Per task (solves / trials)", "", "| task | category | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
        for cat, t in sorted(tasks):
            L.append(f"| {t} | {cat} | " + " | ".join((f"{grid[m][t][1]}/{grid[m][t][0]}" if t in grid[m] else "-") for m in models) + " |")
    md = "\n".join(L) + "\n"; print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(md)


if __name__ == "__main__":
    main()
