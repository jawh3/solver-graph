#!/usr/bin/env python3
"""appendix_c/analyze_mechanism.py — why is the detour arm invisible?

G2 shows the route-lengthened (`detour`) recipe costs the agent no measurable steps. Two
different mechanisms would produce that null, and they mean different things:

  (a) the agent IGNORES the prescribed padding entirely (never runs it), so the recipe's
      extra length is fiction — the agent silently recovers the correct method; or
  (b) the agent RUNS the padding but folds it into commands it was issuing anyway (a
      shell pipeline), so the work happens without costing a round.

Both make a step-count reward blind, but (a) means the payload never actually steers the
agent, while (b) means it does steer it and the cost is simply not charged in rounds.

This reads the executed commands recorded by `g2_step_probe --save-commands` and reports,
for each condition, how often a padding operation actually appears in the trace and how
many operations are packed per issued command.

    python -m appendix_c.analyze_mechanism --dir appendix_c/results/g2_mechanism
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from appendix_c.provenance import write_result

# the padding the `detour` arm prescribes (default pad = rot13)
PAD_PATTERNS = [r"\brot13\b", r"\btr\s+['\"]?A-Za-z", r"codecs\.encode\([^)]*rot", r"\brot_13\b"]
PAD_RE = re.compile("|".join(PAD_PATTERNS), re.I)
# operations packed into one issued command (pipes / chained shell operators)
PACK_RE = re.compile(r"\||&&|;")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="appendix_c/results/g2_mechanism")
    ap.add_argument("--out", default="g2_mechanism")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(str(Path(args.dir) / "rows_shard*.jsonl"))):
        for line in open(f):
            if line.strip():
                rows.append(json.loads(line))
    rows = [r for r in rows if r.get("commands") is not None]
    if not rows:
        print(f"no command-carrying rows in {args.dir}")
        return

    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)

    summary = {}
    for cond, rs in sorted(by.items()):
        pad_runs, packs, ncmds = [], [], []
        for r in rs:
            cmds = r["commands"] or []
            pad_runs.append(any(PAD_RE.search(c) for c in cmds))
            ncmds.append(len(cmds))
            for c in cmds:
                packs.append(len(PACK_RE.findall(c)) + 1)
        summary[cond] = {
            "n_runs": len(rs),
            "pad_executed_rate": float(np.mean(pad_runs)) if pad_runs else None,
            "mean_commands": float(np.mean(ncmds)) if ncmds else None,
            "mean_ops_per_command": float(np.mean(packs)) if packs else None,
            "max_ops_per_command": int(np.max(packs)) if packs else None,
            "solve_rate": float(np.mean([r["solved"] for r in rs])),
        }

    payload = {"n_rows": len(rows), "by_condition": summary,
               "pad_patterns": PAD_PATTERNS,
               "model": rows[0].get("model")}
    path = write_result(args.out, payload, script="appendix_c/analyze_mechanism.py",
                        seeds=sorted({r["seed"] for r in rows})[:20],
                        model=rows[0].get("model", "unknown"),
                        extra={"source_dir": args.dir})

    print(f"\nG2 MECHANISM — {len(rows)} runs with recorded commands")
    print(f"{'condition':<16} {'n':>4} {'pad executed':>13} {'cmds/run':>9} "
          f"{'ops/cmd':>8} {'solve':>7}")
    for cond, s in summary.items():
        print(f"{cond:<16} {s['n_runs']:>4} {s['pad_executed_rate']:>13.3f} "
              f"{s['mean_commands']:>9.2f} {s['mean_ops_per_command']:>8.2f} "
              f"{s['solve_rate']:>7.3f}")
    print(f"wrote {path}")

    # emit the report fragment directly, so the prose cannot drift from the numbers
    det = summary.get("latent_detour")
    hlp = summary.get("helpful")
    if det and hlp:
        tex = Path(__file__).resolve().parent / "report" / "g2_mechanism.tex"
        rate = det["pad_executed_rate"]
        verdict = ("the agent largely does \\emph{not} execute the prescribed padding at "
                   "all --- the recipe's extra length never becomes work"
                   if rate < 0.34 else
                   "the agent does execute the prescribed padding, but folds it into "
                   "commands it was issuing anyway, so the work is never charged as a round"
                   if rate > 0.66 else
                   "the agent executes the prescribed padding in only some runs, so the "
                   "null is a mixture of ignoring the padding and absorbing it")
        tex.write_text(
            "\\paragraph{Which mechanism produces the null?} Inspecting the executed "
            f"commands of {det['n_runs']} detour runs against {hlp['n_runs']} matched "
            "helpful runs: a padding operation appears in "
            f"{rate:.3f} of detour traces (versus {hlp['pad_executed_rate']:.3f} of "
            "helpful traces), with "
            f"{det['mean_ops_per_command']:.2f} shell operations packed into the average "
            f"issued command (helpful: {hlp['mean_ops_per_command']:.2f}, maximum "
            f"observed {det['max_ops_per_command']}). So {verdict}.\n\n"
            "\\paragraph{The extra work is real, but it lands where the metric is not "
            "looking.} This matters for what a defender could do instead. The efficiency "
            "term $e$ standardizes \\emph{rounds} --- a deliberate choice in the "
            "pre-existing design, which rejected wall-clock as irreproducible on a shared "
            "machine. The detour's cost does not land in rounds; it lands inside them, as "
            "extra commands and longer pipelines. Counting issued commands instead is the "
            "obvious repair, and it points the right way but does not rescue the signal "
            "at this sample size: paired against the correct recipe, the detour arm costs "
            "\\GtwoDetourDCmds{} commands with a CI of [\\GtwoDetourDCmdsLo{}, "
            "\\GtwoDetourDCmdsHi{}] that still contains zero, against "
            "\\GtwoDetourDRounds{} rounds [\\GtwoDetourDRoundsLo{}, "
            "\\GtwoDetourDRoundsHi{}]. The honest conclusion is that \\emph{neither} "
            "available effort metric separates this payload here --- rounds because the "
            "work is absorbed within them, commands because the effect is too small "
            "relative to the noise at $n=\\GtwoDetourNpairs{}$ pairs. A command-counting "
            "defense is not refuted by this experiment; it is simply not supported by it, "
            "and would need substantially more power to test.\n")
        print(f"wrote {tex}")


if __name__ == "__main__":
    main()
