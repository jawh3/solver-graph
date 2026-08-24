#!/usr/bin/env python3
"""experiments/cite_rate.py — the MECHANISM behind the poison null (compliance + recovery).

Outcome-based credit can only see a lesson's harm if the agent (a) FOLLOWS it and (b) is then hurt.
This measures both directly from citations.jsonl (cited = the lesson's signature appeared in the
executed commands) joined to run outcomes:
  - cite-rate by label, ON-TARGET vs off-target: does the agent even follow poison on its own family?
  - among ON-TARGET runs where poison WAS cited (followed): the solve rate (does it recover anyway?).
Together these explain WHY per-item outcome attribution is blind to poison: low compliance and/or
free recovery drive the item's effect on the outcome to ~0.  ZERO GPU.

  python -m experiments.cite_rate --dir results/lab/<campaign>/merged
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import os as _os, importlib as _il
hard = _il.import_module(_os.environ.get("CTF_DOMAIN", "benchmark.hard_ctf"))  # T2-7 domain switch
from instrumentation.jsonl_logger import load_runs
from retrieval.embedder import MockEmbedder

_fam_of = lambda cid: cid.rsplit("_", 1)[0]


def analyze(merged_dir: Path, cond: str = "outcome") -> dict:
    summ = json.loads((merged_dir / "summary.json").read_text())
    families = summ.get("families") or json.loads((merged_dir / "progress.json").read_text()).get("families")
    _, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=families, malignant_arms=list(hard.MALIGNANT_ARMS), distractor_arm=True)
    runs = load_runs(f"campaign_{cond}", root=str(merged_dir / "logs"))
    run_by_id = {r["run_id"]: r for r in runs}

    cits = []
    for cf in (merged_dir / "logs" / f"campaign_{cond}").glob("citations*.jsonl"):
        cits += [json.loads(l) for l in open(cf) if l.strip()]
    # merged dir may store one combined file or none; fall back to per-shard under the parent
    if not cits:
        root = merged_dir.parent
        for cf in root.glob(f"shard_*/logs/campaign_{cond}/citations.jsonl"):
            cits += [json.loads(l) for l in open(cf) if l.strip()]

    # cite-rate by (label, on_target); and solve-rate among cited
    samp = defaultdict(int); cited = defaultdict(int)
    cited_solved = defaultdict(lambda: [0, 0])   # [solved, total] among cited
    for c in cits:
        if not c.get("sampled"):
            continue
        nid = c.get("node_id"); run = run_by_id.get(c.get("run_id"))
        if nid not in truth or run is None:
            continue
        label, lfam = truth[nid]
        on = (_fam_of(run["challenge_id"]) == lfam)
        key = (label, "on" if on else "off")
        samp[key] += 1
        if c.get("cited"):
            cited[key] += 1
            st = cited_solved[key]; st[1] += 1; st[0] += int(run.get("outcome", 0))

    out = {"dir": str(merged_dir), "n_citations": len(cits), "by": {}}
    mal_labels = sorted({lab for (lab, _t) in samp if lab.startswith("malignant") or lab=="distractor_ontarget"})
    for label in ("helpful", "poison", "distractor", *mal_labels):
        for tgt in ("on", "off"):
            k = (label, tgt)
            if samp[k] == 0:
                continue
            cs = cited_solved[k]
            out["by"][f"{label}:{tgt}-target"] = {
                "sampled": samp[k], "cite_rate": cited[k] / samp[k],
                "solve_rate_when_cited": (cs[0] / cs[1] if cs[1] else None), "n_cited": cs[1]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/lab/campaign_7b35_M3_rep/merged")
    ap.add_argument("--cond", default="outcome")
    args = ap.parse_args()
    r = analyze(Path(args.dir), cond=args.cond)
    print(f"CITE-RATE / RECOVERY MECHANISM — {r['dir']}  (n_citations={r['n_citations']})")
    for k, d in r["by"].items():
        sr = d["solve_rate_when_cited"]
        print(f"  {k:22s}: cite_rate={d['cite_rate']:.3f}  solve|cited={'n/a' if sr is None else f'{sr:.3f}'}  (sampled={d['sampled']}, cited={d['n_cited']})")
    (Path(args.dir) / "cite_rate.json").write_text(json.dumps(r, indent=2))
    print(f"  wrote {args.dir}/cite_rate.json")


if __name__ == "__main__":
    main()
