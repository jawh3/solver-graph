#!/usr/bin/env python3
"""experiments/active_loo.py — T1-12 / Task#10: GOLD-STANDARD validation of the cheap SNIPW estimator.

Everything in the paper rests on the cheap, log-only on-target SNIPW per-lesson ATE. This validates it
against the ACTIVE leave-one-out ground truth: for a sample of on-target lessons, re-roll the real solver
WITH vs WITHOUT the lesson (loo_marginal_value, pipeline/causal_credit.py) on FRESH instances of the
lesson's family, using REALIZED campaign contexts (the M-sets the lesson actually appeared in), and
compare the ground-truth marginal_solve to the cheap SNIPW ATE: Spearman rho + per-label agreement +
false-lesson exposure. If they agree, the cheap estimator is validated; the poison-null is then a real
property, not an estimator artifact. REQUIRES a served solver (real_solve_fn). Concurrent (network-bound).

  MODEL_NAME=... VLLM_URL=... python -m experiments.active_loo --dir results/lab/campaign_14b_M3_powered/merged
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import os as _os, importlib as _il
hard = _il.import_module(_os.environ.get("CTF_DOMAIN", "benchmark.hard_ctf"))  # T2-7 domain switch
from instrumentation.jsonl_logger import load_runs
from ope.causal_value import observational_lesson_values
from retrieval.embedder import MockEmbedder

_fam = lambda cid: cid.rsplit("_", 1)[0]


def _isolate(challenge, tag, workroot):
    d = Path(workroot) / tag
    d.mkdir(parents=True, exist_ok=True)
    ch = json.loads(json.dumps(challenge))
    lf = ch["local_files"][0]
    dst = d / Path(lf["local_path"]).name
    shutil.copy(lf["local_path"], dst)
    lf["local_path"] = str(dst)
    return ch


def run(merged_dir: Path, n_inst: int, seeds: int, contexts: int, max_nodes: int,
        slack: int, workers: int, workroot: Path, out_name: str):
    try:
        from batch_test.real_adapters import real_solve_fn
    except ModuleNotFoundError:  # not shipped in the public release
        def real_solve_fn(*a, **k):
            raise RuntimeError(
                "real-GPU backend adapters (batch_test/) are not included in "
                "this release; supply your own solve_fn")
    base = real_solve_fn(max_output_tokens=384, context_window=6144)

    def solve(ch, nodes, budget, seed):
        res = base(ch, nodes, budget, seed=seed)
        return 1 if hard.check_flag(ch, res.get("flag")) else 0

    summ = json.loads((merged_dir / "summary.json").read_text())
    fams = summ.get("families") or json.loads((merged_dir / "progress.json").read_text()).get("families")
    mal = list(hard.MALIGNANT_ARMS)
    store, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=fams, malignant_arms=mal)
    runs = load_runs("campaign_outcome", root=str(merged_dir / "logs"))

    # cheap SNIPW on-target ATE per node (binary outcome)
    by_fam = defaultdict(list)
    for r in runs:
        by_fam[_fam(r["challenge_id"])].append(r)
    cheap, realized_ctx = {}, defaultdict(Counter)
    for f, fr in by_fam.items():
        vals = observational_lesson_values(fr, truth=truth, n_boot=0, response_key="outcome")
        for nid, d in vals.items():
            if nid in truth and truth[nid][1] == f and d.get("ate") is not None:
                cheap[nid] = d["ate"]
        for r in fr:                                 # realized on-target contexts per node
            sset = tuple(r.get("sampled_set") or [])
            for nid in sset:
                if nid in truth and truth[nid][1] == f:
                    realized_ctx[nid][sset] += 1

    # choose target nodes: on-target lessons with a cheap ATE and >=1 realized context, capped
    targets = [v for v in cheap if realized_ctx.get(v)]
    targets.sort(key=lambda v: -sum(realized_ctx[v].values()))    # best-sampled first
    targets = targets[:max_nodes]

    # build the job list: per node, top-`contexts` realized M-sets x fresh instances x seeds x {with,without}
    corpus_cache = {}
    def fresh_corpus(f):
        if f not in corpus_cache:
            seed = int(hashlib.sha1(f.encode()).hexdigest()[:6], 16) % 9000 + 1000   # deterministic
            corpus_cache[f] = hard.generate_corpus(workroot / "corpus" / f, n_per_family=n_inst,
                                                   seed=seed, families=[f])
        return corpus_cache[f]

    jobs = []          # (node, arm, nodes_list, challenge_iso, budget, seed)
    for ti, v in enumerate(targets):
        f = truth[v][1]
        budget = hard._BY_NAME[f].solution_length + slack
        ctxs = [list(c) for c, _ in realized_ctx[v].most_common(contexts)]
        for ci, ctx in enumerate(ctxs):
            with_nodes = [store.get_node(i) for i in ctx if store.get_node(i) is not None]
            if not any(n.id == v for n in with_nodes):
                with_nodes.append(store.get_node(v))
            without_nodes = [n for n in with_nodes if n.id != v]
            for ii, ch in enumerate(fresh_corpus(f)):
                for sd in range(seeds):
                    tag = f"n{ti}_{ci}_{ii}_{sd}"          # unique per (target,ctx,inst,seed)
                    jobs.append((v, "with", with_nodes, _isolate(ch, tag + "_w", workroot), ch, budget, sd))
                    jobs.append((v, "without", without_nodes, _isolate(ch, tag + "_o", workroot), ch, budget, sd))

    print(f"[active_loo] {len(targets)} target nodes, {len(jobs)} solve jobs "
          f"(contexts={contexts}, n_inst={n_inst}, seeds={seeds})")
    acc = defaultdict(lambda: {"with": [], "without": []})
    def do(job):
        v, arm, nodes, ch_iso, ch, budget, sd = job
        return v, arm, solve(ch_iso, nodes, budget, sd)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(do, j) for j in jobs]):
            v, arm, o = fut.result()
            acc[v][arm].append(o)
            done += 1
            if done % 100 == 0:
                print(f"  ...{done}/{len(jobs)} solves")

    rows = []
    for v in targets:
        w, wo = acc[v]["with"], acc[v]["without"]
        if not w or not wo:
            continue
        marg = float(np.mean(w) - np.mean(wo))
        rows.append({"node": v, "label": truth[v][0], "family": truth[v][1],
                     "loo_marginal_solve": marg, "cheap_snipw_ate": float(cheap[v]),
                     "n_with": len(w), "n_without": len(wo)})

    loo = np.array([r["loo_marginal_solve"] for r in rows])
    chp = np.array([r["cheap_snipw_ate"] for r in rows])
    def spearman(a, b):
        if len(a) < 3:
            return None
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    bylab = defaultdict(lambda: {"loo": [], "cheap": []})
    for r in rows:
        base_lab = "malignant" if r["label"].startswith("malignant") else r["label"]
        bylab[base_lab]["loo"].append(r["loo_marginal_solve"]); bylab[base_lab]["cheap"].append(r["cheap_snipw_ate"])
    res = {"dir": str(merged_dir), "n_nodes": len(rows),
           "spearman_loo_vs_cheap": spearman(loo, chp),
           "pearson_loo_vs_cheap": (float(np.corrcoef(loo, chp)[0, 1]) if len(rows) > 2 else None),
           "by_label": {k: {"n": len(v["loo"]), "mean_loo_marginal": float(np.mean(v["loo"])),
                            "mean_cheap_ate": float(np.mean(v["cheap"]))} for k, v in bylab.items()},
           "params": {"n_inst": n_inst, "seeds": seeds, "contexts": contexts, "slack": slack},
           "nodes": rows}
    (merged_dir / out_name).write_text(json.dumps(res, indent=2))
    print(f"[active_loo] Spearman(LOO, cheap SNIPW) = {res['spearman_loo_vs_cheap']}  (n={len(rows)} nodes)")
    for k, v in res["by_label"].items():
        print(f"  {k:10s}: mean LOO marginal={v['mean_loo_marginal']:+.4f}  mean cheap ATE={v['mean_cheap_ate']:+.4f}  (n={v['n']})")
    print(f"  wrote {merged_dir/out_name}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/lab/campaign_14b_M3_powered/merged")
    ap.add_argument("--n-inst", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--contexts", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=60)
    ap.add_argument("--slack", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--workroot", default="results/lab/active_loo_work")
    ap.add_argument("--out", default="active_loo.json")
    args = ap.parse_args()
    run(Path(args.dir), args.n_inst, args.seeds, args.contexts, args.max_nodes,
        args.slack, args.workers, Path(args.workroot), args.out)


if __name__ == "__main__":
    main()
