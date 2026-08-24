#!/usr/bin/env python3
"""experiments/malignant_direct_loo.py — gold-standard DIRECT harm of malignant lessons (T1-1/T1-12).

The realized-context active-LOO reads malignant marginal ~0 because malignant lessons DISPLACE helpful
(only ~14/670 malignant citations co-occur with helpful), so in their realized contexts there is no
would-be-success to destroy. To isolate DIRECT destruction we CONSTRUCT a solvable context and toggle
the malignant lesson: on FRESH instances of each family, solve with {helpful} vs {helpful, malignant_arm}.
If adding the malignant lesson collapses a would-be solve, that is gold-standard direct harm (the proper
positive control). Pooled per arm with a family-cluster bootstrap. REQUIRES a served solver. Concurrent.

  MODEL_NAME=... VLLM_URL=... python -m experiments.malignant_direct_loo --families-file <selection.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import os as _os, importlib as _il
hard = _il.import_module(_os.environ.get("CTF_DOMAIN", "benchmark.hard_ctf"))  # T2-7 domain switch
from retrieval.embedder import MockEmbedder


def _isolate(challenge, tag, workroot):
    d = Path(workroot) / tag; d.mkdir(parents=True, exist_ok=True)
    ch = json.loads(json.dumps(challenge))
    lf = ch["local_files"][0]
    dst = d / Path(lf["local_path"]).name
    shutil.copy(lf["local_path"], dst); lf["local_path"] = str(dst)
    return ch


def _nodes(store, truth, fam):
    h = m = {}
    out = {"helpful": None, "malignant_destroy": None, "malignant_exhaust": None, "malignant_wrongflag": None}
    for nid, (lab, lfam) in truth.items():
        if lfam == fam and lab in out:
            out[lab] = store.get_node(nid)
    return out


def run(families, n_inst, seeds, slack, workers, workroot, out_path):
    try:
        from batch_test.real_adapters import real_solve_fn
    except ModuleNotFoundError:  # not shipped in the public release
        def real_solve_fn(*a, **k):
            raise RuntimeError(
                "real-GPU backend adapters (batch_test/) are not included in "
                "this release; supply your own solve_fn")
    base = real_solve_fn(max_output_tokens=384, context_window=6144)
    def solve(ch, nodes, budget, seed):
        return 1 if hard.check_flag(ch, base(ch, nodes, budget, seed=seed).get("flag")) else 0

    store, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=families,
                                         malignant_arms=list(hard.MALIGNANT_ARMS))
    arms = ["destroy", "exhaust", "wrongflag"]
    jobs = []   # (family, arm, cond, nodes, ch_iso, budget, seed)
    for f in families:
        nd = _nodes(store, truth, f)
        if nd["helpful"] is None:
            continue
        budget = hard._BY_NAME[f].solution_length + slack
        seed0 = int(hashlib.sha1(f.encode()).hexdigest()[:6], 16) % 9000 + 1000
        corpus = hard.generate_corpus(workroot / "corpus" / f, n_per_family=n_inst, seed=seed0, families=[f])
        for ii, ch in enumerate(corpus):
            for sd in range(seeds):
                # baseline solvable context: helpful alone
                jobs.append((f, "_base", "H", [nd["helpful"]], _isolate(ch, f"{f}_{ii}_{sd}_H", workroot), budget, sd))
                for arm in arms:
                    mn = nd.get(f"malignant_{arm}")
                    if mn is None:
                        continue
                    jobs.append((f, arm, "HM", [nd["helpful"], mn],
                                 _isolate(ch, f"{f}_{ii}_{sd}_{arm}", workroot), budget, sd))
    print(f"[direct_loo] {len(families)} families, {len(jobs)} solve jobs (n_inst={n_inst}, seeds={seeds})")

    # accumulate per family: solve(H) and solve(H+arm)
    perf = defaultdict(lambda: {"H": [], "destroy": [], "exhaust": [], "wrongflag": []})
    def do(j):
        f, arm, cond, nodes, ch_iso, budget, sd = j
        return f, ("H" if cond == "H" else arm), solve(ch_iso, nodes, budget, sd)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(do, j) for j in jobs]):
            f, key, o = fut.result(); perf[f][key].append(o); done += 1
            if done % 100 == 0:
                print(f"  ...{done}/{len(jobs)}")

    # per-family solve rates, then pooled direct harm per arm with family-cluster bootstrap
    fam_rates = {f: {k: (float(np.mean(v)) if v else None) for k, v in d.items()} for f, d in perf.items()}
    def boot(pairs, nboot=5000, seed=0):           # pairs = [(solveH, solveHM)] per family
        rng = np.random.RandomState(seed)
        H = np.array([p[0] for p in pairs]); HM = np.array([p[1] for p in pairs])
        pt = float(np.mean(HM) - np.mean(H))
        bs = np.empty(nboot)
        for b in range(nboot):
            idx = rng.randint(0, len(pairs), len(pairs)); bs[b] = np.mean(HM[idx]) - np.mean(H[idx])
        lo, hi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
        p1 = float((bs >= 0).mean() if pt < 0 else (bs <= 0).mean())
        return {"direct_harm": pt, "ci_lo": lo, "ci_hi": hi, "p_one_sided": p1, "n_families": len(pairs)}
    res = {"by_arm": {}, "n_families": len(fam_rates), "params": {"n_inst": n_inst, "seeds": seeds}}
    for arm in arms:
        pairs = [(r["H"], r[arm]) for r in fam_rates.values() if r["H"] is not None and r[arm] is not None]
        if len(pairs) >= 3:
            b = boot(pairs)
            b["mean_solve_helpful"] = float(np.mean([p[0] for p in pairs]))
            b["mean_solve_helpful_plus_malignant"] = float(np.mean([p[1] for p in pairs]))
            res["by_arm"][arm] = b
    Path(out_path).write_text(json.dumps({**res, "per_family": fam_rates}, indent=2))
    print(f"[direct_loo] CONSTRUCTED-context direct harm (solve(H+M) - solve(H)):")
    for arm, b in res["by_arm"].items():
        print(f"  M-{arm:9s}: solve(H)={b['mean_solve_helpful']:.3f} -> solve(H+M)={b['mean_solve_helpful_plus_malignant']:.3f}  "
              f"direct harm={b['direct_harm']:+.3f} CI[{b['ci_lo']:+.3f},{b['ci_hi']:+.3f}] p1={b['p_one_sided']:.4f} (n_fam={b['n_families']})")
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families-file", default="results/lab/probe_14b/selection.json")
    ap.add_argument("--n-inst", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--slack", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--workroot", default="results/lab/direct_loo_work")
    ap.add_argument("--out", default="results/lab/malignant_direct_loo.json")
    args = ap.parse_args()
    d = json.loads(Path(args.families_file).read_text())
    fams = d.get("in_band_families") or d.get("families") or []
    if fams and isinstance(fams[0], dict):
        fams = [f["family"] for f in fams]
    run(fams, args.n_inst, args.seeds, args.slack, args.workers, Path(args.workroot), args.out)


if __name__ == "__main__":
    main()
