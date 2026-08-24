#!/usr/bin/env python3
"""experiments/follow_poison_freecheck.py — FREE observational check (RIGOR_LEDGER REDIRECT 2026-06-25):
does EXECUTING the poison's distinctive WRONG-INNER method predict failure, on the existing crypto logs?

The campaign citation matcher marks a lesson "cited" if its signature appears in executed commands — but
~90% of poison citations match the SHARED OUTER transport (e.g. 'base64 ~ base64'), which the correct recipe
ALSO uses. So we recompute "followed the wrong method" from the DISTINCTIVE-INNER tokens only:
  distinctive(family) = set(poison_sig) - set(true_sig)   (the sibling's wrong inner transform tokens)
and scan the run's EXECUTED commands (steps.jsonl) for any distinctive token. Among on-target-poison-sampled
runs (and the helpful-co-present subset), compare solve rate + total_commands for executed-wrong-inner vs not,
family-cluster bootstrap. This is OBSERVATIONAL/CONDITIONAL (a post-hoc split, not randomized) — report as such.
Expected: a 14B harm signal, 32B weaker/null. ZERO GPU; deterministic.

  python -m experiments.follow_poison_freecheck --dirs results/lab/campaign_14b_M3_powered/merged,... --label 14B
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import benchmark.hard_ctf as hard
from instrumentation.jsonl_logger import load_runs
from retrieval.embedder import MockEmbedder

_fam = lambda cid: cid.rsplit("_", 1)[0]
_GENERIC = {"python3", "python", "bash", "sh", "-c", "echo", "cat", "|", "base64", "xxd", "tr"}


def _distinctive(merged_dir: Path):
    summ = json.loads((merged_dir / "summary.json").read_text())
    fams = summ.get("families") or json.loads((merged_dir / "progress.json").read_text()).get("families")
    store, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=fams)
    true_sig, pois_sig = {}, {}
    for nid, (lab, f) in truth.items():
        sig = set(s.lower() for s in (store.get_node(nid).attributes.get("invocation_signature") or []))
        if lab == "helpful":
            true_sig[f] = sig
        elif lab == "poison":
            pois_sig[f] = sig
    # distinctive inner tokens = poison sig minus the family's own sig minus generic shell tokens,
    # keep tokens of length>=3 (drop noise); a family with empty distinctive set is EXCLUDED.
    dist = {}
    for f in pois_sig:
        d = {t for t in (pois_sig[f] - true_sig.get(f, set()) - _GENERIC) if len(t) >= 3}
        if d:
            dist[f] = d
    pid_of = {f: nid for nid, (lab, f) in truth.items() if lab == "poison"}
    hid_of = {f: nid for nid, (lab, f) in truth.items() if lab == "helpful"}
    return dist, pid_of, hid_of


def _commands_by_run(merged_dir: Path, cond: str):
    cmds = defaultdict(list)
    for sf in list((merged_dir / "logs" / f"campaign_{cond}").glob("steps*.jsonl")) + \
              list(merged_dir.parent.glob(f"shard_*/logs/campaign_{cond}/steps.jsonl")):
        for line in open(sf):
            try:
                s = json.loads(line); c = s.get("command")
                if c:
                    cmds[s["run_id"]].append(c.lower())
            except Exception:
                pass
    return {r: " ".join(v) for r, v in cmds.items()}


def analyze(merged_dirs, cond="outcome", label="14B"):
    rows = []   # (family, exec_distinctive, co_present, outcome, total_commands)
    for md in merged_dirs:
        md = Path(md)
        dist, pid_of, hid_of = _distinctive(md)
        cmds = _commands_by_run(md, cond)
        for r in load_runs(f"campaign_{cond}", root=str(md / "logs")):
            f = _fam(r["challenge_id"])
            if f not in dist or f not in pid_of:
                continue
            sset = set(r.get("sampled_set") or [])
            if pid_of[f] not in sset:            # require poison on-target SAMPLED
                continue
            cmd = cmds.get(r["run_id"], "")
            exec_d = any(re.search(r"\b" + re.escape(t) + r"\b", cmd) for t in dist[f])
            rows.append((f, exec_d, hid_of.get(f) in sset, int(r.get("outcome") or 0),
                         r.get("total_commands") or 0))
    return _report(rows, label)


def _famclust_diff(rows, idx_flag, idx_val, nb=10000, seed=0):
    by = defaultdict(lambda: {True: [], False: []})
    for r in rows:
        by[r[0]][r[idx_flag]].append(r[idx_val])
    fams = [f for f in by if by[f][True] and by[f][False]]
    if len(fams) < 3:
        return None
    per = {f: (np.mean(by[f][True]) - np.mean(by[f][False])) for f in fams}
    a = np.array([per[f] for f in fams]); rng = np.random.RandomState(seed)
    bs = np.array([a[rng.randint(0, len(a), len(a))].mean() for _ in range(nb)])
    return {"mean": float(a.mean()), "ci_lo": float(np.percentile(bs, 2.5)),
            "ci_hi": float(np.percentile(bs, 97.5)),
            "p": float(2 * min((bs >= 0).mean(), (bs <= 0).mean())), "n_families": len(fams)}


def _report(rows, label):
    def desc(S):
        ed = [r for r in S if r[1]]; ne = [r for r in S if not r[1]]
        return {"n": len(S), "n_exec_distinctive": len(ed), "n_not": len(ne),
                "solve_exec": (float(np.mean([r[3] for r in ed])) if ed else None),
                "solve_not": (float(np.mean([r[3] for r in ne])) if ne else None),
                "cmds_exec": (float(np.mean([r[4] for r in ed])) if ed else None),
                "cmds_not": (float(np.mean([r[4] for r in ne])) if ne else None),
                "solve_diff_famcluster": _famclust_diff(S, 1, 3),
                "cmds_diff_famcluster": _famclust_diff(S, 1, 4)}
    return {"label": label, "n_rows": len(rows),
            "poison_sampled": desc(rows),
            "co_present_helpful": desc([r for r in rows if r[2]])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="comma list of merged dirs (same model)")
    ap.add_argument("--cond", default="outcome")
    ap.add_argument("--label", default="14B")
    ap.add_argument("--out", default="results/lab/follow_poison_freecheck.json")
    args = ap.parse_args()
    r = analyze([d.strip() for d in args.dirs.split(",")], cond=args.cond, label=args.label)
    print(f"FOLLOW-POISON ({r['label']}) n_rows={r['n_rows']} — exec distinctive-wrong-inner vs not:")
    for scope in ("poison_sampled", "co_present_helpful"):
        d = r[scope]; sd = d["solve_diff_famcluster"]; cd = d["cmds_diff_famcluster"]
        print(f"  [{scope}] n={d['n']} (exec={d['n_exec_distinctive']}, not={d['n_not']})")
        print(f"     solve: exec={d['solve_exec']} vs not={d['solve_not']}  "
              f"Δ(famcluster)={sd['mean']:+.3f} [{sd['ci_lo']:+.3f},{sd['ci_hi']:+.3f}] p={sd['p']:.4f} (n_fam={sd['n_families']})" if sd else "     solve: n/a")
        if cd:
            print(f"     cmds:  exec={d['cmds_exec']:.1f} vs not={d['cmds_not']:.1f}  Δ={cd['mean']:+.1f} [{cd['ci_lo']:+.1f},{cd['ci_hi']:+.1f}] p={cd['p']:.4f}")
    Path(args.out).write_text(json.dumps(r, indent=2))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
