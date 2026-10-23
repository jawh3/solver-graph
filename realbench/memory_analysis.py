#!/usr/bin/env python3
"""realbench/memory_analysis.py - the memory campaign's pre-specified estimands.

Pairs the OFF records (`results/realbench_32k/<slug>/bakeoff.jsonl`) with the ON records
(`results/realbench_mem/<slug>/on_static.jsonl`) by (task, trial) and reports, per model and pooled:

  * paired risk difference ON - OFF with a TASK-cluster bootstrap CI (the task is the unit; trials of a task are not
    independent), the Wilson interval per condition, discordant pairs, and a sign test over tasks;
  * the same by category, by contamination split (competition before / after the 2023-12 cutoff), and - descriptive,
    post hoc - by whether a same-category lesson was retrieved and whether any injected lesson was cited (executed against);
  * pooled over models: the mean of the per-model paired differences, task-cluster bootstrap keeping every model of a task.

    python -m realbench.memory_analysis --root-off results/realbench_32k --root-on results/realbench_mem --out docs/phase3/MEMORY_CAMPAIGN_<date>.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def load(path: Path) -> Dict[Tuple[str, int], dict]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            continue
        out[(r["task_id"], int((r.get("extra") or {}).get("trial", 0)))] = r
    return out


def pair(off: Dict[Tuple[str, int], dict], on: Dict[Tuple[str, int], dict]) -> List[dict]:
    rows = []
    for key in sorted(set(off) & set(on)):
        a, b = off[key], on[key]
        mem = (b.get("extra") or {}).get("memory") or {}
        rows.append({"task_id": key[0], "trial": key[1], "category": a["category"], "off": int(bool(a["solved"])), "on": int(bool(b["solved"])),
                     "same_cat_lesson": a["category"] in (mem.get("lesson_categories") or []), "cited": bool(mem.get("cited")),
                     "rounds_off": a.get("rounds_used", 0), "rounds_on": b.get("rounds_used", 0)})
    return rows


def cluster_bootstrap(rows: List[dict], n_boot: int = 4000, seed: int = 0, key: str = "task_id") -> Tuple[float, float, float]:
    """Paired RD = mean(on - off); resample clusters (tasks) with replacement, all rows of a cluster move together."""
    if not rows:
        return (float("nan"), float("nan"), float("nan"))
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r["on"] - r["off"])
    clusters = [np.asarray(v, dtype=float) for v in by.values()]
    rd = float(np.mean(np.concatenate(clusters)))
    rng = np.random.RandomState(seed); reps = []
    idx = np.arange(len(clusters))
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        reps.append(float(np.mean(np.concatenate([clusters[i] for i in pick]))))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return (rd, float(lo), float(hi))


def sign_test(rows: List[dict]) -> Tuple[int, int, float]:
    """Over TASKS: sign of (mean ON - mean OFF); two-sided binomial p on the non-ties."""
    by = defaultdict(list)
    for r in rows:
        by[r["task_id"]].append(r["on"] - r["off"])
    d = [float(np.mean(v)) for v in by.values()]
    pos = sum(1 for x in d if x > 0); neg = sum(1 for x in d if x < 0); n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg); p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return pos, neg, min(1.0, p)


def cell(rows: List[dict], seed: int = 0) -> dict:
    n = len(rows); k_off = sum(r["off"] for r in rows); k_on = sum(r["on"] for r in rows)
    rd, lo, hi = cluster_bootstrap(rows, seed=seed); pos, neg, p = sign_test(rows)
    return {"pairs": n, "tasks": len({r["task_id"] for r in rows}), "off": k_off, "on": k_on,
            "off_rate": wilson(k_off, n), "on_rate": wilson(k_on, n), "rd": rd, "rd_ci": [lo, hi],
            "on_only": sum(1 for r in rows if r["on"] and not r["off"]), "off_only": sum(1 for r in rows if r["off"] and not r["on"]),
            "tasks_pos": pos, "tasks_neg": neg, "sign_p": p,
            "mean_rounds_off": (sum(r["rounds_off"] for r in rows) / n) if n else 0.0, "mean_rounds_on": (sum(r["rounds_on"] for r in rows) / n) if n else 0.0}


def analyse_model(rows: List[dict], pre_cutoff: Optional[Dict[str, bool]] = None) -> dict:
    out = {"all": cell(rows), "by_category": {}, "by_contamination": {}, "by_same_cat_lesson": {}, "by_cited": {}}
    for c in sorted({r["category"] for r in rows}):
        out["by_category"][c] = cell([r for r in rows if r["category"] == c])
    if pre_cutoff:
        for label, val in (("pre_cutoff", True), ("post_cutoff", False)):
            sub = [r for r in rows if pre_cutoff.get(r["task_id"]) is val]
            if sub:
                out["by_contamination"][label] = cell(sub)
    for label, val in (("same_category_lesson", True), ("no_same_category_lesson", False)):
        sub = [r for r in rows if r["same_cat_lesson"] is val]
        if sub:
            out["by_same_cat_lesson"][label] = cell(sub)
    for label, val in (("cited", True), ("not_cited", False)):
        sub = [r for r in rows if r["cited"] is val]
        if sub:
            out["by_cited"][label] = cell(sub)
    return out


def fmt_rate(w) -> str:
    return f"{w[0]:.3f} [{w[1]:.3f}, {w[2]:.3f}]"


def fmt_rd(c: dict) -> str:
    return f"{c['rd']:+.3f} [{c['rd_ci'][0]:+.3f}, {c['rd_ci'][1]:+.3f}]"


def markdown(per_model: Dict[str, dict], pooled: Optional[dict], meta: dict) -> str:
    L = [f"# Memory campaign on Cybench - ON-static vs OFF ({meta.get('date', '')})", "",
         f"Paired by (task, trial); OFF = `{meta.get('root_off')}`, ON = `{meta.get('root_on')}`. RD = paired risk difference ON - OFF, "
         "task-cluster bootstrap 95% CI (4,000 resamples of tasks). Sign test over tasks (mean over trials), two-sided.", "",
         "| model | pairs (tasks) | OFF solves | ON solves | OFF rate | ON rate | RD ON-OFF | ON-only / OFF-only | tasks +/- | sign p | rounds OFF→ON |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, a in per_model.items():
        c = a["all"]
        L.append(f"| {m} | {c['pairs']} ({c['tasks']}) | {c['off']} | {c['on']} | {fmt_rate(c['off_rate'])} | {fmt_rate(c['on_rate'])} | {fmt_rd(c)} | "
                 f"{c['on_only']} / {c['off_only']} | {c['tasks_pos']}/{c['tasks_neg']} | {c['sign_p']:.3f} | {c['mean_rounds_off']:.1f}→{c['mean_rounds_on']:.1f} |")
    if pooled:
        L.append(f"| **pooled (mean of models)** | {pooled['pairs']} | {pooled['off']} | {pooled['on']} | | | {fmt_rd(pooled)} | | | | |")
    L += ["", "## By category (RD ON-OFF, pairs)", ""]
    cats = sorted({c for a in per_model.values() for c in a["by_category"]})
    L.append("| model | " + " | ".join(cats) + " |"); L.append("|---|" + "---|" * len(cats))
    for m, a in per_model.items():
        L.append(f"| {m} | " + " | ".join((f"{fmt_rd(a['by_category'][c])} (n={a['by_category'][c]['pairs']}, {a['by_category'][c]['off']}→{a['by_category'][c]['on']})" if c in a["by_category"] else "-") for c in cats) + " |")
    for title, key in (("By contamination split (competition before / after 2023-12)", "by_contamination"),
                       ("By whether a same-category lesson was retrieved (descriptive)", "by_same_cat_lesson"),
                       ("By whether an injected lesson was cited, i.e. executed against (post hoc, descriptive)", "by_cited")):
        L += ["", f"## {title}", "", "| model | subgroup | pairs | OFF→ON | RD |", "|---|---|---|---|---|"]
        for m, a in per_model.items():
            for sub, c in a[key].items():
                L.append(f"| {m} | {sub} | {c['pairs']} | {c['off']}→{c['on']} | {fmt_rd(c)} |")
    L += ["", "Reading rule (pre-specified): a model's memory effect is a gain only when the RD interval excludes 0; intervals that "
          "cover 0 are 'no detectable effect at this size', not 'no effect'. Subgroups are descriptive.", ""]
    return "\n".join(L)


def markdown_multi(off_root: Path, on_root: Path, off_name: str, on_names: List[str], slugs: List[str], meta: dict) -> Tuple[str, dict]:
    """3-way (or N-way) paired table: OFF, and each ON condition's RD vs OFF; plus, when both static and learned are present,
    the direct learned-minus-static paired difference (the 'does learning beat static injection' contrast)."""
    labels = {"on_static.jsonl": "ON-static", "on_learned.jsonl": "ON-learned"}
    L = [f"# Memory campaign on Cybench - OFF / " + " / ".join(labels.get(n, n) for n in on_names) + f" ({meta['date']})", "",
         f"Paired by (task, trial). OFF = `{off_root}`. RD = paired risk difference vs OFF, task-cluster bootstrap 95% CI. "
         "ON-learned trains the Phase-2 bandit on held-out seeds (2000-2001) and is scored greedily on the OFF/static seeds (1000-1002).", "",
         "| model | OFF | " + " | ".join(f"{labels.get(n,n)} (RD vs OFF)" for n in on_names) + " | learned - static |",
         "|---|---|" + "---|" * (len(on_names) + 1)]
    js: dict = {}
    for s in slugs:
        off = load(off_root / s / off_name)
        conds = {n: load(on_root / s / n) for n in on_names}
        row = [s, f"{sum(r['solved'] for r in off.values())}/{len(off)}" if off else "-"]
        js[s] = {"off_solves": sum(r["solved"] for r in off.values()), "off_pairs": len(off)}
        for n in on_names:
            rows = pair(off, conds[n])
            if not rows:
                row.append("-"); continue
            c = cell(rows)
            row.append(f"{c['on']}/{c['pairs']} ({fmt_rd(c)})")
            js[s][labels.get(n, n)] = {"solves": c["on"], "pairs": c["pairs"], "rd": c["rd"], "rd_ci": c["rd_ci"], "sign_p": c["sign_p"]}
        if "on_static.jsonl" in on_names and "on_learned.jsonl" in on_names:
            st, ln = conds["on_static.jsonl"], conds["on_learned.jsonl"]
            keys = sorted(set(st) & set(ln))
            rows = [{"task_id": k[0], "trial": k[1], "category": st[k]["category"], "off": int(bool(st[k]["solved"])),
                     "on": int(bool(ln[k]["solved"]))} for k in keys]
            if rows:
                rd, lo, hi = cluster_bootstrap(rows)
                row.append(f"{rd:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                js[s]["learned_minus_static"] = {"rd": rd, "rd_ci": [lo, hi], "pairs": len(rows)}
            else:
                row.append("-")
        else:
            row.append("-")
        L.append("| " + " | ".join(row) + " |")
    L += ["", "Reading rule (pre-specified): a condition beats OFF only when its RD interval excludes 0; the learned-static "
          "column is the direct test of whether closed-loop learning improves on static top-M injection. Per-model detail: run "
          "`memory_analysis` with a single --on-name for the full category / contamination / citation breakdown.", ""]
    return "\n".join(L), js


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-off", default="results/realbench_32k"); ap.add_argument("--root-on", default="results/realbench_mem")
    ap.add_argument("--off-name", default="bakeoff.jsonl"); ap.add_argument("--on-name", default="on_static.jsonl")
    ap.add_argument("--on-names", default="", help="comma list of ON conditions (e.g. on_static.jsonl,on_learned.jsonl) for the combined OFF/static/learned table")
    ap.add_argument("--models", default="", help="comma-separated slugs (default: every slug present under both roots)")
    ap.add_argument("--cybench-root", default=os.environ.get("CYBENCH_ROOT", "<path>"))
    ap.add_argument("--cyber-zero-root", default=os.environ.get("CYBER_ZERO_CYBENCH_ROOT", "<path>"))
    ap.add_argument("--no-contamination", action="store_true", help="skip the contamination split (no Cybench checkout available)")
    ap.add_argument("--out", default=""); ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    roff, ron = Path(args.root_off), Path(args.root_on)
    on_names = [n for n in args.on_names.split(",") if n]
    if on_names:                                     # combined OFF / static / learned table
        import datetime
        slugs = [s for s in args.models.split(",") if s] or sorted(
            p.name for p in ron.iterdir() if any((p / n).exists() for n in on_names) and (roff / p.name / args.off_name).exists())
        if not slugs:
            sys.exit("no model has OFF records and at least one ON condition")
        meta = {"date": datetime.date.today().isoformat()}
        md, js = markdown_multi(roff, ron, args.off_name, on_names, slugs, meta)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(md)
        if args.json:
            Path(args.json).write_text(json.dumps(js, indent=1, default=float))
        print(md); return 0
    slugs = [s for s in args.models.split(",") if s] or sorted(p.name for p in ron.iterdir() if (p / args.on_name).exists() and (roff / p.name / args.off_name).exists())
    if not slugs:
        sys.exit("no model has both OFF and ON records")
    pre_cutoff = None
    if not args.no_contamination:
        try:
            from realbench import cybench as cb
            tasks = cb.enumerate_tasks(Path(args.cybench_root), cyber_zero_root=(Path(args.cyber_zero_root) if args.cyber_zero_root else None), strict=False)
            pre_cutoff = {t.id: (not t.post_cutoff()) if t.post_cutoff() is not None else None for t in tasks}
        except Exception as e:  # noqa
            print(f"[memory_analysis] contamination split skipped: {e}", file=sys.stderr)
    per_model = {}; all_rows = []
    for s in slugs:
        rows = pair(load(roff / s / args.off_name), load(ron / s / args.on_name))
        if not rows:
            print(f"[memory_analysis] {s}: no paired records", file=sys.stderr); continue
        per_model[s] = analyse_model(rows, pre_cutoff)
        for r in rows:
            all_rows.append(dict(r, model=s))
    pooled = None
    if len(per_model) > 1:
        # pooled: task-cluster bootstrap over rows of every model (a task's rows across models move together)
        rd, lo, hi = cluster_bootstrap(all_rows, seed=1)
        pooled = {"pairs": len(all_rows), "off": sum(r["off"] for r in all_rows), "on": sum(r["on"] for r in all_rows), "rd": rd, "rd_ci": [lo, hi]}
    import datetime
    meta = {"date": datetime.date.today().isoformat(), "root_off": str(roff), "root_on": str(ron)}
    md = markdown(per_model, pooled, meta)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(md)
    if args.json:
        Path(args.json).write_text(json.dumps({"per_model": per_model, "pooled": pooled, "meta": meta}, indent=1, default=float))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
