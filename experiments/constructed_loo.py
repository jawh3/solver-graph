#!/usr/bin/env python3
"""experiments/constructed_loo.py — domain-agnostic CONSTRUCTED-context gold-standard battery (boundary).

The paper's whole point is that OBSERVATIONAL per-item SNIPW is unreliable for poison. So the boundary
claim ("poison is recoverable/null in crypto, committed/harmful in SQL") must be shown by INTERVENTION,
identically in both domains. On FRESH solvable instances of each family we re-roll the real solver under
constructed contexts and measure solve:
  none=[]            helpful=[H]            poison=[P]            helpful+poison=[H,P]
  (+ helpful+malignant_arm=[H,M] where a malignant lesson exists — crypto only)
and report family-cluster-bootstrapped contrasts:
  helpful_lift          = solve(helpful) − solve(none)            (should be >0)
  poison_alone          = solve(poison)  − solve(none)            (sub-floor: poison vs nothing)
  poison_in_context     = solve(H,P)     − solve(helpful)         (KEY: ~0 if recoverable, <0 if committed)
  malignant_in_context  = solve(H,M)     − solve(helpful)         (crypto control: <0, destroys)
Run with CTF_DOMAIN=benchmark.hard_ctf and CTF_DOMAIN=benchmark.sql_domain for the symmetric comparison.
REQUIRES a served solver. Concurrent, isolated workdirs, deterministic corpus seeds.
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
hard = _il.import_module(_os.environ.get("CTF_DOMAIN", "benchmark.hard_ctf"))   # domain switch
from retrieval.embedder import MockEmbedder


def _isolate(challenge, tag, workroot):
    d = Path(workroot) / tag; d.mkdir(parents=True, exist_ok=True)
    ch = json.loads(json.dumps(challenge))
    lf = ch["local_files"][0]
    dst = d / Path(lf["local_path"]).name
    shutil.copy(lf["local_path"], dst); lf["local_path"] = str(dst)
    return ch


def _fam_nodes(store, truth, fam):
    out = {}
    for nid, (lab, lfam) in truth.items():
        if lfam == fam:
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

    mal = list(getattr(hard, "MALIGNANT_ARMS", ()))
    store, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=families, malignant_arms=mal)
    mal_labels = [f"malignant_{a}" for a in mal]

    jobs = []   # (family, condition, nodes, ch_iso, budget, seed)
    for f in families:
        nd = _fam_nodes(store, truth, f)
        H, P = nd.get("helpful"), nd.get("poison")
        if H is None or P is None:
            continue
        budget = hard._BY_NAME[f].solution_length + slack
        seed0 = int(hashlib.sha1(f.encode()).hexdigest()[:6], 16) % 9000 + 1000
        corpus = hard.generate_corpus(workroot / "corpus" / f, n_per_family=n_inst, seed=seed0, families=[f])
        conds = {"none": [], "helpful": [H], "poison": [P], "helpful_poison": [H, P]}
        for ml in mal_labels:
            if nd.get(ml) is not None:
                conds[f"helpful_{ml}"] = [H, nd[ml]]
        for ii, ch in enumerate(corpus):
            for sd in range(seeds):
                for cond, nodes in conds.items():
                    jobs.append((f, cond, nodes, _isolate(ch, f"{f}_{ii}_{sd}_{cond}", workroot), budget, sd))
    print(f"[constructed_loo] domain={hard.__name__} {len(families)} families, {len(jobs)} solve jobs")

    perf = defaultdict(lambda: defaultdict(list))   # family -> cond -> [outcomes]
    def do(j):
        f, cond, nodes, ch_iso, budget, sd = j
        return f, cond, solve(ch_iso, nodes, budget, sd)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(do, j) for j in jobs]):
            f, cond, o = fut.result(); perf[f][cond].append(o); done += 1
            if done % 100 == 0:
                print(f"  ...{done}/{len(jobs)}")

    fam_rate = {f: {c: (float(np.mean(v)) if v else None) for c, v in d.items()} for f, d in perf.items()}

    def boot(contrast, nboot=5000, seed=0):
        """contrast(fam_rate_dict)->(a,b) per family; returns mean(a-b) + family-cluster CI."""
        rng = np.random.RandomState(seed)
        pairs = []
        for f, r in fam_rate.items():
            ab = contrast(r)
            if ab is not None and ab[0] is not None and ab[1] is not None:
                pairs.append(ab)
        if len(pairs) < 3:
            return None
        A = np.array([p[0] for p in pairs]); B = np.array([p[1] for p in pairs])
        pt = float(np.mean(A) - np.mean(B))
        bs = np.empty(nboot)
        for b in range(nboot):
            idx = rng.randint(0, len(pairs), len(pairs)); bs[b] = np.mean(A[idx]) - np.mean(B[idx])
        lo, hi = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
        p2 = float(2 * min((bs >= 0).mean(), (bs <= 0).mean()))
        return {"delta": pt, "ci_lo": lo, "ci_hi": hi, "p_two_sided": p2, "n_families": len(pairs),
                "mean_a": float(np.mean(A)), "mean_b": float(np.mean(B)), "excl0": (lo < 0 and hi < 0) or (lo > 0 and hi > 0)}

    contrasts = {
        "helpful_lift":      lambda r: (r.get("helpful"), r.get("none")),
        "poison_alone":      lambda r: (r.get("poison"), r.get("none")),
        "poison_in_context": lambda r: (r.get("helpful_poison"), r.get("helpful")),
    }
    for ml in mal_labels:
        contrasts[f"{ml}_in_context"] = (lambda r, _ml=ml: (r.get(f"helpful_{_ml}"), r.get("helpful")))

    res = {"domain": hard.__name__, "n_families": len(fam_rate),
           "params": {"n_inst": n_inst, "seeds": seeds, "slack": slack},
           "contrasts": {k: boot(fn) for k, fn in contrasts.items()},
           "per_family": fam_rate}
    Path(out_path).write_text(json.dumps(res, indent=2))
    print(f"[constructed_loo] {hard.__name__} (n_fam={res['n_families']}):")
    for k, b in res["contrasts"].items():
        if b:
            print(f"  {k:26s}: {b['mean_a']:.3f} vs {b['mean_b']:.3f}  delta={b['delta']:+.3f} "
                  f"CI[{b['ci_lo']:+.3f},{b['ci_hi']:+.3f}] p={b['p_two_sided']:.4f} excl0={b['excl0']} (n_fam={b['n_families']})")
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families-file", default="results/lab/probe_14b/selection.json")
    ap.add_argument("--n-inst", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--slack", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--workroot", default="results/lab/constructed_loo_work")
    ap.add_argument("--out", default="results/lab/constructed_loo.json")
    args = ap.parse_args()
    d = json.loads(Path(args.families_file).read_text())
    fams = d.get("in_band_families") or d.get("families") or []
    if fams and isinstance(fams[0], dict):
        fams = [f["family"] for f in fams]
    run(fams, args.n_inst, args.seeds, args.slack, args.workers, Path(args.workroot), args.out)


if __name__ == "__main__":
    main()
