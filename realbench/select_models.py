#!/usr/bin/env python3
"""realbench/select_models.py - apply the pre-declared bake-off selection rule (PHASE3_PLAN.md, incl. the 2026-09-15
cross-family amendment) to the summaries under a results root. Prints the chosen model ids, one per line (or JSON).
Rule: qualify = solve rate ≥ 0.10 and Wilson lower bound ≥ 0.04 and errors < 10%. Set = best qualifying Qwen + the
Qwen2.5-32B continuity rung + the two best qualifying non-Qwen models of different origins (if fewer than two
qualify, the best two non-Qwen anyway, flagged). Flagged reference runs (e.g. Gemma-2 at 8k) are eligible but marked.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

ORIGIN = [("qwen", "Alibaba"), ("glm", "Zhipu"), ("gpt-oss", "OpenAI"), ("gemma", "Google"), ("mistral", "Mistral"),
          ("seed-oss", "ByteDance"), ("llama", "Meta"), ("yi-", "01.AI"), ("internlm", "Shanghai AI Lab"), ("olmo", "AI2")]


def origin(model: str) -> str:
    m = model.lower()
    for k, o in ORIGIN:
        if k in m:
            return o
    return "unknown"


def qualifies(s: dict) -> bool:
    return s["solve_rate"] >= 0.10 and s["wilson95"][0] >= 0.04 and s["errors"] / max(s["runs"], 1) < 0.10


def select(root: str):
    rows = []
    for f in sorted(glob.glob(f"{root}/*/bakeoff.summary.json")):
        s = json.load(open(f)); m = s.get("model") or os.path.basename(os.path.dirname(f))
        if s.get("degenerate"):
            continue
        rows.append({"model": m, "origin": origin(m), "rate": s["solve_rate"], "lo": s["wilson95"][0], "s_per_run": s["mean_wall_s"], "qualifies": qualifies(s)})
    qwen = [r for r in rows if r["origin"] == "Alibaba" and "qwen2.5" not in r["model"].lower()]
    best_qwen = max([r for r in qwen if r["qualifies"]] or qwen, key=lambda r: (r["rate"], -r["s_per_run"]), default=None)
    cont = next((r for r in rows if "qwen2.5-32b" in r["model"].lower()), None)
    non = sorted([r for r in rows if r["origin"] != "Alibaba"], key=lambda r: (-r["rate"], r["s_per_run"]))
    chosen_non, origins, flagged = [], set(), []
    for r in [x for x in non if x["qualifies"]] + [x for x in non if not x["qualifies"]]:
        if len(chosen_non) == 2:
            break
        if r["origin"] in origins:
            continue
        chosen_non.append(r); origins.add(r["origin"])
        if not r["qualifies"]:
            flagged.append(r["model"])
    chosen = [r for r in [best_qwen, cont] if r] + chosen_non
    return {"rows": rows, "chosen": [r["model"] for r in chosen], "flagged_below_bar": flagged,
            "detail": [{k: r[k] for k in ("model", "origin", "rate", "lo", "qualifies")} for r in chosen]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results/realbench"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args(); res = select(a.root)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        for m in res["chosen"]:
            print(m)


if __name__ == "__main__":
    main()
