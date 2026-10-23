#!/usr/bin/env python3
"""experiments/compute_report.py - test-time compute for a Phase-2 campaign root (user request 2026-09-07;
spec rule B4: report compute so a win is never confused with more compute).

Per arm, from the run records (every run logs `tokens` = prompt+completion from the server's usage,
`wall_clock_s`, `steps` = LLM rounds, `total_commands`): runs, total tokens, tokens/run, LLM rounds/run,
solver wall time, solves and tokens per solve. Campaign totals, plus the vLLM servers' cumulative counters
(prompt vs generation tokens, requests by finish reason) from live `/metrics` endpoints or saved snapshots
(`metrics_*.prom` written by phase2_health.sh).

Usage: python -m experiments.compute_report --root results/lab/phase2_14b [--metrics-urls http://127.0.0.1:8002/metrics,...]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path


def arm_table(root: Path):
    rows = {}
    for f in sorted(glob.glob(f"{root}/arms/*/shard_00/logs/campaign_*/runs.jsonl")):
        arm = f.split("/arms/")[1].split("/")[0]
        t = rows.setdefault(arm, {"runs": 0, "tokens": 0, "wall_s": 0.0, "rounds": 0, "commands": 0, "solves": 0,
                                  "reroll_solves": 0, "reroll_tokens": 0, "reroll_wall_s": 0.0})
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            t["runs"] += 1; t["tokens"] += int(r.get("tokens") or 0); t["wall_s"] += float(r.get("wall_clock_s") or 0)
            t["rounds"] += int(r.get("steps") or 0); t["commands"] += int(r.get("total_commands") or 0); t["solves"] += int(bool(r.get("outcome")))
            t["reroll_solves"] += int(r.get("reroll_solves") or 0); t["reroll_tokens"] += int(r.get("reroll_tokens") or 0)
            t["reroll_wall_s"] += float(r.get("reroll_wall_s") or 0)
    for t in rows.values():
        n = max(t["runs"], 1)
        t["tokens_all"] = t["tokens"] + t["reroll_tokens"]          # main solves + causal re-rolls
        t["tokens_per_run"] = t["tokens_all"] / n; t["rounds_per_run"] = t["rounds"] / n
        t["tokens_per_solve"] = (t["tokens_all"] / t["solves"]) if t["solves"] else None
        t["wall_h"] = (t["wall_s"] + t["reroll_wall_s"]) / 3600
    return rows


def parse_metrics(text: str):
    out = defaultdict(float)
    for line in text.splitlines():
        m = re.match(r'^vllm:(prompt_tokens_total|generation_tokens_total)\{.*\} ([0-9.e+]+)$', line)
        if m:
            out[m.group(1)] += float(m.group(2))
        m = re.match(r'^vllm:request_success_total\{.*finished_reason="([a-z]+)".*\} ([0-9.e+]+)$', line)
        if m:
            out[f"requests_{m.group(1)}"] += float(m.group(2))
    return dict(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--metrics-urls", default="")
    ap.add_argument("--snapshots", default="", help="glob of saved /metrics snapshots (fallback when servers are down)")
    args = ap.parse_args()
    root = Path(args.root)
    rows = arm_table(root)
    tot = {k: sum(t[k] for t in rows.values()) for k in ("runs", "tokens", "wall_s", "rounds", "commands", "solves", "reroll_solves", "reroll_tokens", "reroll_wall_s", "tokens_all")}
    server = {}
    for u in [u for u in args.metrics_urls.split(",") if u]:
        try:
            server[u] = parse_metrics(urllib.request.urlopen(u, timeout=10).read().decode())
        except Exception as e:  # noqa
            server[u] = {"error": str(e)}
    for f in (glob.glob(args.snapshots) if args.snapshots else []):
        server[f] = parse_metrics(open(f).read())
    lines = [f"# Test-time compute - {root}", "", "| arm | runs | solves | tokens incl. re-rolls (M) | tokens/run (k) | rounds/run | re-roll solves | tokens/solve (k) | solver wall (h) |", "|---|---|---|---|---|---|---|---|---|"]
    for a, t in sorted(rows.items()):
        tps = f"{t['tokens_per_solve']/1000:.0f}" if t["tokens_per_solve"] else "-"
        lines.append(f"| {a} | {t['runs']} | {t['solves']} | {t['tokens_all']/1e6:.1f} | {t['tokens_per_run']/1000:.1f} | {t['rounds_per_run']:.2f} | {t['reroll_solves']} | {tps} | {t['wall_h']:.1f} |")
    lines += ["", f"**Totals:** {tot['runs']} runs (+ {tot['reroll_solves']} causal re-roll solves), {tot['solves']} solves, {tot['tokens_all']/1e6:.1f}M tokens "
              f"(prompt+completion as billed by the server, re-rolls included; {tot['reroll_tokens']/1e6:.1f}M of them re-rolls), "
              f"{tot['rounds']} LLM rounds, {tot['commands']} shell commands, {(tot['wall_s']+tot['reroll_wall_s'])/3600:.1f} solver-hours (sum of per-run wall clock; runs overlap across arms)."]
    if server:
        lines += ["", "## Server counters (cumulative since each server started; include probe/smoke traffic on 8002)"]
        for u, m in server.items():
            if "error" in m:
                lines.append(f"- {u}: unavailable ({m['error']})"); continue
            p, g = m.get("prompt_tokens_total", 0), m.get("generation_tokens_total", 0)
            req = {k[9:]: int(v) for k, v in m.items() if k.startswith("requests_")}
            lines.append(f"- {u}: prompt {p/1e6:.1f}M, generation {g/1e6:.1f}M tokens; requests by finish reason {req}"
                         + (f"; truncated {req.get('length',0)/max(sum(req.values()),1):.1%}" if req else ""))
    out = root / "COMPUTE_REPORT.md"; out.write_text("\n".join(lines) + "\n")
    (root / "compute_report.json").write_text(json.dumps({"arms": rows, "totals": tot, "servers": server}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
