#!/usr/bin/env python3
"""experiments/malignant_probe_check.py — T1-1 positive-control GATE (pre-registered 2026-06-11).

The unit tests (tests/test_malignant_teeth.py) prove the malignant recipes have teeth *IF FOLLOWED*
(truncate corrupts the artifact, the static wrongflag != real flag, 256 probes blow the budget).
They CANNOT prove the real LLM solver actually FOLLOWS them. This script measures that from a (small,
cheap) real campaign and emits a PASS/FAIL gate so the driver fails fast before a full campaign-night:

  per malignant arm (ON-TARGET = lesson family == challenge family):
    follow_rate   = P(arm cited | arm sampled)        # does the LLM obey the malignant recipe?
    solve|cited   = P(solve | arm cited on-target)    # teeth manifest -> should fall BELOW base
    base          = P(solve | NO on-target malignant cited)   # the within-campaign floor b_f
    delta         = solve|cited - base                # teeth => delta < 0 (floor VIOLATED)

GATE (pre-registered kill condition): the floor-VIOLATING arms M-destroy AND M-exhaust must each
show teeth in the real solver — n_cited >= MIN and delta < 0 (directional, with a within-family
bootstrap CI for context). If they do NOT, the malignant lesson is not being followed and the
positive control is INCONCLUSIVE as scoped (escalate lesson strength) — exit 3. ZERO new GPU.

  python -m experiments.malignant_probe_check --dir results/lab/<campaign>/merged [--min-cited 5]
exit 0 = PASS (proceed to full), 3 = FAIL/inconclusive, 2 = error/insufficient data.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import benchmark.hard_ctf as hard
from instrumentation.jsonl_logger import load_runs
from retrieval.embedder import MockEmbedder

_fam_of = lambda cid: cid.rsplit("_", 1)[0]

# the pre-registered floor-VIOLATING arms the estimator MUST be able to flag
_GATE_ARMS = ("destroy", "exhaust")


def _load_citations(merged_dir: Path, cond: str):
    cits = []
    for cf in (merged_dir / "logs" / f"campaign_{cond}").glob("citations*.jsonl"):
        cits += [json.loads(l) for l in open(cf) if l.strip()]
    if not cits:
        for cf in merged_dir.parent.glob(f"shard_*/logs/campaign_{cond}/citations.jsonl"):
            cits += [json.loads(l) for l in open(cf) if l.strip()]
    return cits


def analyze(merged_dir: Path, cond: str = "outcome", min_cited: int = 5,
            n_boot: int = 2000, seed: int = 0) -> dict:
    summ = json.loads((merged_dir / "summary.json").read_text())
    families = summ.get("families") or json.loads((merged_dir / "progress.json").read_text()).get("families")
    _, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=families,
                                     malignant_arms=list(hard.MALIGNANT_ARMS))
    arm_of = {nid: t[0].split("malignant_", 1)[1] for nid, t in truth.items()
              if t[0].startswith("malignant_")}
    runs = load_runs(f"campaign_{cond}", root=str(merged_dir / "logs"))
    run_by_id = {r["run_id"]: r for r in runs}

    # per run: set of on-target malignant arms SAMPLED and CITED
    sampled = defaultdict(set); cited = defaultdict(set)
    for c in _load_citations(merged_dir, cond):
        nid = c.get("node_id"); run = run_by_id.get(c.get("run_id"))
        arm = arm_of.get(nid)
        if arm is None or run is None:
            continue
        if truth[nid][1] != _fam_of(run["challenge_id"]):   # ON-TARGET only
            continue
        rid = c["run_id"]
        if c.get("sampled"):
            sampled[rid].add(arm)
        if c.get("cited"):
            cited[rid].add(arm)

    # base arm b: runs with NO on-target malignant lesson CITED
    rows = []
    for r in runs:
        rows.append({"fam": _fam_of(r["challenge_id"]), "o": int(r.get("outcome") or 0),
                     "rid": r["run_id"], "cited": cited.get(r["run_id"], set()),
                     "sampled": sampled.get(r["run_id"], set())})
    base_rows = [x for x in rows if not x["cited"]]
    base = float(np.mean([x["o"] for x in base_rows])) if base_rows else float("nan")

    def within_fam_boot(stat_fn):
        rng = np.random.RandomState(seed)
        by_fam = defaultdict(list)
        for x in rows:
            by_fam[x["fam"]].append(x)
        fams = sorted(by_fam); vals = []
        for _ in range(n_boot):
            res = []
            for f in fams:
                rs = by_fam[f]; idx = rng.randint(0, len(rs), size=len(rs))
                res.extend(rs[i] for i in idx)
            v = stat_fn(res)
            if v is not None:
                vals.append(v)
        if not vals:
            return None, None
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    arms = {}
    for arm in hard.MALIGNANT_ARMS:
        n_samp = sum(arm in x["sampled"] for x in rows)
        cited_rows = [x for x in rows if arm in x["cited"]]
        n_cited = len(cited_rows)
        solve_cited = float(np.mean([x["o"] for x in cited_rows])) if cited_rows else None
        follow = (n_cited / n_samp) if n_samp else None
        delta = (solve_cited - base) if (solve_cited is not None and base == base) else None

        def stat(res, _arm=arm, _base_is_nocite=True):
            c = [x["o"] for x in res if _arm in x["cited"]]
            b = [x["o"] for x in res if not x["cited"]]
            if not c or not b:
                return None
            return float(np.mean(c) - np.mean(b))
        ci = within_fam_boot(stat) if n_cited >= 2 else (None, None)
        arms[arm] = {"n_sampled": n_samp, "n_cited": n_cited, "follow_rate": follow,
                     "solve_when_cited": solve_cited, "base": base, "delta": delta,
                     "delta_ci95": list(ci)}

    # GATE: floor-violating arms must each show teeth in the REAL solver
    gate = {}
    for arm in _GATE_ARMS:
        a = arms[arm]
        ok = (a["n_cited"] >= min_cited and a["delta"] is not None and a["delta"] < 0)
        gate[arm] = {"pass": bool(ok), "n_cited": a["n_cited"], "delta": a["delta"],
                     "follow_rate": a["follow_rate"]}
    passed = all(g["pass"] for g in gate.values())
    return {"dir": str(merged_dir), "cond": cond, "n_runs": len(runs), "base": base,
            "min_cited": min_cited, "arms": arms, "gate": gate, "gate_pass": passed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cond", default="outcome")
    ap.add_argument("--min-cited", type=int, default=5)
    ap.add_argument("--out", default="malignant_probe_check.json")
    args = ap.parse_args()
    r = analyze(Path(args.dir), cond=args.cond, min_cited=args.min_cited)
    print(f"MALIGNANT PROBE GATE — {r['dir']}  (n_runs={r['n_runs']}, base solve={r['base']:.3f})")
    for arm, a in r["arms"].items():
        sc = "n/a" if a["solve_when_cited"] is None else f"{a['solve_when_cited']:.3f}"
        fr = "n/a" if a["follow_rate"] is None else f"{a['follow_rate']:.3f}"
        dl = "n/a" if a["delta"] is None else f"{a['delta']:+.3f}"
        ci = a["delta_ci95"]
        cis = "" if ci[0] is None else f" CI[{ci[0]:+.3f},{ci[1]:+.3f}]"
        print(f"  M-{arm:9s}: follow={fr}  solve|cited={sc}  base={a['base']:.3f}  delta={dl}{cis}  "
              f"(n_cited={a['n_cited']}, n_samp={a['n_sampled']})")
    print(f"  GATE (need teeth for {', '.join('M-'+x for x in _GATE_ARMS)}): "
          f"{'PASS' if r['gate_pass'] else 'FAIL'}")
    (Path(args.dir) / args.out).write_text(json.dumps(r, indent=2))
    print(f"  wrote {args.dir}/{args.out}")
    if r["gate_pass"]:
        sys.exit(0)
    # distinguish 'inconclusive/insufficient data' (exit 2) from 'followed but no teeth' (exit 3)
    if any(g["n_cited"] < r["min_cited"] for g in r["gate"].values()):
        sys.exit(2)
    sys.exit(3)


if __name__ == "__main__":
    main()
