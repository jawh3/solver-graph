#!/usr/bin/env python3
"""
experiments/lab_probe.py — calibration probe for the scaled run (real solver, concurrent).

For each candidate family, solve fresh instances under FORCED retrieval sets:
  none      (no lesson)         -> base solve rate
  helpful   (true recipe)       -> in-band check: keep families with helpful solve in [lo,hi]
  poison    (wrong recipe only) -> TEETH check: poison-only should solve < helpful-only
  both      (helpful+poison)    -> displacement/distraction check
Emits a probe artifact not retained with per-family rates and a label-BLIND in-band family
selection (decided on solve rate, before any ATE/causal analysis). Concurrent via threads
(each solve is network-bound on vLLM; every (family,set,seed) gets an isolated working dir).
Resumable: completed families are skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import os as _os, importlib as _il
hard = _il.import_module(_os.environ.get("CTF_DOMAIN", "benchmark.hard_ctf"))  # T2-7 domain switch
from retrieval.embedder import MockEmbedder
try:
    from batch_test.real_adapters import real_solve_fn
except ModuleNotFoundError:  # not shipped in the public release
    def real_solve_fn(*a, **k):
        raise RuntimeError(
            "real-GPU backend adapters (batch_test/) are not included in "
            "this release; supply your own solve_fn")
from ope.causal_value import _stable_seed

_LOCK = threading.Lock()


def _nodes_for(name):
    """Helpful + poison Node objects for a family (from the seed graph)."""
    store, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=[name])
    helpful = poison = None
    for nid, (lab, fam) in truth.items():
        if fam != name:
            continue
        node = store.get_node(nid)
        if lab == "helpful":
            helpful = node
        elif lab == "poison":
            poison = node
    return helpful, poison


def _isolate(challenge, tag, seed, workroot):
    """Copy the challenge artifact into a unique dir so concurrent solves never collide."""
    d = Path(workroot) / f"{challenge['name']}_{tag}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    ch = json.loads(json.dumps(challenge))   # deep copy (plain dict)
    lf = ch["local_files"][0]
    dst = d / Path(lf["local_path"]).name
    shutil.copy(lf["local_path"], dst)
    lf["local_path"] = str(dst)
    return ch


def probe_family(name, n_inst, seeds, slack, solve, workroot, corpus_root):
    helpful, poison = _nodes_for(name)
    sl = hard._BY_NAME[name].solution_length
    budget = sl + slack
    corpus = hard.generate_corpus(Path(corpus_root) / name, n_per_family=n_inst,
                                  seed=_stable_seed(name) % 10000, families=[name])
    sets = {"none": [], "helpful": [helpful], "poison": [poison], "both": [helpful, poison]}
    rec = {s: [] for s in sets}
    for ch in corpus:
        for s, nodes in sets.items():
            for sd in range(seeds):
                cj = _isolate(ch, s, sd, workroot)
                res = solve(cj, [n for n in nodes if n is not None], budget, seed=sd)
                rec[s].append(1 if hard.check_flag(ch, res.get("flag")) else 0)
    out = {"family": name, "solution_length": sl, "budget": budget,
           "difficulty": hard._BY_NAME[name].difficulty,
           "n": len(corpus) * seeds,
           "solve": {s: float(np.mean(v)) if v else None for s, v in rec.items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", default=None, help="comma band filter, e.g. easy,medium")
    ap.add_argument("--max-families", type=int, default=None)
    ap.add_argument("--families", default=None, help="explicit comma family list (overrides grid selection)")
    ap.add_argument("--shard", default=None, help="k/N — process only families[k::N] (split across servers)")
    ap.add_argument("--n-inst", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--slack", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    # in-band = headroom (base solve low) + lesson enables solving (not floor) + real lift.
    # NOT an upper cap on helpful: a family with none=0, helpful=1.0 is MAXIMAL clean signal,
    # not a problematic ceiling (the bad ceiling is none~1, where the lesson is useless).
    ap.add_argument("--max-none", type=float, default=0.5, help="exclude if base solve > this (no headroom)")
    ap.add_argument("--min-helpful", type=float, default=0.4, help="exclude if helpful solve < this (floor)")
    ap.add_argument("--min-lift", type=float, default=0.3, help="exclude if helpful-none < this (lesson must matter)")
    ap.add_argument("--out", default="results/lab/probe")
    ap.add_argument("--max-output-tokens", type=int, default=512)
    args = ap.parse_args()

    diff = [d.strip() for d in args.difficulty.split(",")] if args.difficulty else None
    if args.families:
        fams = [f.strip() for f in args.families.split(",")]
    else:
        fams = hard.grid_family_names(difficulty=diff, n=args.max_families, seed=1)
    if args.shard:
        k, N = (int(x) for x in args.shard.split("/"))
        fams = fams[k::N]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "probe.json"
    done = {}
    if res_path.exists():
        done = {r["family"]: r for r in json.loads(res_path.read_text()).get("families", [])}
    todo = [f for f in fams if f not in done]
    print(f"[probe] {len(fams)} candidate families, {len(done)} done, {len(todo)} to run; "
          f"budget=len+{args.slack}; workers={args.workers}", flush=True)

    solve = real_solve_fn(max_output_tokens=args.max_output_tokens, context_window=6144)
    workroot = out_dir / "work"; corpus_root = out_dir / "corpus"
    results = list(done.values())
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(probe_family, f, args.n_inst, args.seeds, args.slack,
                          solve, workroot, corpus_root): f for f in todo}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa
                r = {"family": f, "error": str(e)}
            results.append(r)
            with _LOCK:
                payload = {"families": results, "slack": args.slack,
                           "criterion": {"max_none": args.max_none, "min_helpful": args.min_helpful,
                                         "min_lift": args.min_lift}}
                res_path.write_text(json.dumps(payload, indent=2))
            sv = r.get("solve", {})
            print(f"  {f:22s} none={sv.get('none')} helpful={sv.get('helpful')} "
                  f"poison={sv.get('poison')} both={sv.get('both')}", flush=True)

    # label-BLIND in-band selection (on solve rates only) + teeth flag
    inband, teeth = [], []
    for r in results:
        sv = r.get("solve") or {}
        h, nn, po = sv.get("helpful"), sv.get("none"), sv.get("poison")
        if h is None or nn is None:
            continue
        if (nn <= args.max_none) and (h >= args.min_helpful) and (h - nn >= args.min_lift):
            inband.append(r["family"])
            if po is not None and po < h:      # teeth: poison-only solves less than helpful-only
                teeth.append(r["family"])
    sel = {"in_band_families": sorted(inband), "n_in_band": len(inband),
           "teeth_pass": len(teeth), "teeth_families": sorted(teeth),
           "criterion": {"max_none": args.max_none, "min_helpful": args.min_helpful,
                         "min_lift": args.min_lift}}
    (out_dir / "selection.json").write_text(json.dumps(sel, indent=2))
    print(f"\n[probe] in-band families: {len(inband)} (teeth pass {len(teeth)}/{len(inband)})")
    print(f"[probe] wrote {res_path} and {out_dir/'selection.json'}")


if __name__ == "__main__":
    main()
