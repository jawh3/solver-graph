#!/usr/bin/env python3
"""
experiments/run_pilot.py — the real laptop-pilot experiment (the paper's data).

Runs the REAL small LLM (local_llm/server.py) + real embeddings over the
controlled micro-CTF corpus, producing the results the paper reports. Every flag
is freshly random (zero contamination) and every poison lesson is ground-truth
labeled, so false-lesson exposure is measured exactly, oracle-free.

Phases (each writes its result to results/pilot/ as it finishes, so partial data
survives an interruption):
  A  Retrieval value     — B0 (no graph) vs B2 (static retrieval), paired, multi-seed.
  B  Credit/reward ablation — stream the corpus under {cited_only|all_m} × {step|outcome}
                              reward; track Q(true) vs Q(poison) per epoch.
  C  Write-back gating   — gated (B5) vs ungated (B6) streaming.
  D  LOO false-lesson exposure — re-roll with/without each poison lesson; marginal value.

Usage:
  VLLM_URL=http://127.0.0.1:8000/v1 MODEL_NAME=Qwen2.5-7B-Instruct \
  python -m experiments.run_pilot --n-per-family 4 --epochs 2 --seeds 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import SolverConfig
from knowledge_graph.store import KnowledgeGraphStore
from retrieval.embedder import SentenceTransformerEmbedder
from retrieval.retriever import ContextRetriever
from retrieval.context_resolution import ContextResolver
from retrieval.reranker import MockReranker
from bandit.bandit import TabularBandit, sample_efficiency_budget
from reward.reward import RewardStandardizer
from writeback.writeback import WriteBackManager
from pipeline import Orchestrator, SystemSpec
from instrumentation.jsonl_logger import ExperimentLogger
from evaluation.stats import paired_bootstrap_ci, mcnemar_exact
try:
    from batch_test.real_adapters import real_solve_fn
except ModuleNotFoundError:  # not shipped in the public release
    def real_solve_fn(*a, **k):
        raise RuntimeError(
            "real-GPU backend adapters (batch_test/) are not included in "
            "this release; supply your own solve_fn")
from benchmark.micro_ctf import generate_corpus, build_seed_graph, check_flag, CATEGORY_OF

RESULTS = Path("results/pilot")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# real solve_fn wrapped so the orchestrator's outcome == ground-truth flag check
# laptop-pilot speed caps: short JSON tool replies need few tokens; capping
# max_output_tokens and the context window bounds per-round generation+prefill so
# the experiment finishes in ~1.5h on a single 8GB GPU (documented in the paper).
SOLVE_MAX_TOKENS = 256
SOLVE_CTX = 4096
SOLVE_ROUNDS = 4


def make_checked_solver(corpus_by_name):
    base = real_solve_fn(max_output_tokens=SOLVE_MAX_TOKENS, context_window=SOLVE_CTX)

    def _solve(challenge, sampled_nodes, max_rounds):
        res = base(challenge, sampled_nodes, max_rounds)
        # override flag to None unless it EXACTLY matches ground truth (no partial credit)
        if res.get("flag") and not check_flag(challenge, res["flag"]):
            res = dict(res); res["flag"] = None; res["gave_up_reason"] = "wrong_flag"
        return res
    return _solve


def build_config(seed, **over):
    base = dict(embedding_dim=384, context_resolution="category",
                N=30, R=25, M=8, kappa=5.0, tau=1.0, tau_min=0.3,
                alpha=0.15, lam=0.5, max_steps=SOLVE_ROUNDS, seed=seed, n_clusters=3)
    base.update(over)
    return SolverConfig().with_overrides(**base)


def fresh_graph(embedder, families):
    store, truth = build_seed_graph(embedder, families=families)
    return store, truth


def make_orchestrator(cfg, system, store, embedder, truth, exp_id, solve_fn):
    bandit = TabularBandit(cfg) if system.use_bandit else None
    cr = ContextResolver(cfg)  # category mode: no fit needed
    retr = ContextRetriever(store, embedder, config=cfg, reranker=MockReranker(),
                            context_resolver=cr, bandit=bandit)
    logger = ExperimentLogger(exp_id, root=str(RESULTS / "logs"))
    orch = Orchestrator(store, embedder, cfg, system, retriever=retr, bandit=bandit,
                        standardizer=RewardStandardizer(cfg),
                        writeback=WriteBackManager(cfg), logger=logger, solve_fn=solve_fn)
    orch.solver_model = os.getenv("MODEL_NAME", "local")
    return orch, bandit, logger


def q_by_truth(bandit, truth):
    """Aggregate learned Q by ground-truth label (helpful/poison), per context."""
    agg = defaultdict(list)
    for ctx, nodes in bandit.cells.items():
        for nid, cell in nodes.items():
            if cell.n > 0 and nid in truth:
                agg[truth[nid][0]].append(cell.Q)
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in agg.items()}


# ===========================================================================
def phase_A_retrieval_value(corpus, embedder, families, seeds, solve_fn):
    log("PHASE A — retrieval value (B0 vs B2)")
    per = {"B0": {}, "B2": {}}
    for sysid, system in [("B0", SystemSpec("B0", use_graph=False)),
                          ("B2", SystemSpec("B2", use_rerank=True, use_mmr=True, use_bandit=False))]:
        for s in seeds:
            cfg = build_config(seed=s)
            store, truth = fresh_graph(embedder, families)
            orch, _, _ = make_orchestrator(cfg, system, store, embedder, truth,
                                           f"A_{sysid}_s{s}", solve_fn)
            for i, ch in enumerate(corpus):
                run = orch.run_one(ch, experiment_id=f"A_{sysid}",
                                   rng=np.random.RandomState(9000 + 100 * s + i))
                per[sysid][f"{ch['name']}#s{s}"] = {"solved": run.outcome, "steps": run.steps,
                                                    "family": ch["_family"]}
            log(f"  {sysid} seed {s}: solve={np.mean([v['solved'] for k,v in per[sysid].items() if k.endswith(f's{s}')]):.2f}")
    keys = sorted(set(per["B0"]) & set(per["B2"]))
    a = [per["B2"][k]["solved"] for k in keys]
    b = [per["B0"][k]["solved"] for k in keys]
    pr = paired_bootstrap_ci(a, b, seed=1)
    out = {"per_run": per,
           "B0_solve": float(np.mean([v["solved"] for v in per["B0"].values()])),
           "B2_solve": float(np.mean([v["solved"] for v in per["B2"].values()])),
           "paired_lift": pr.mean_diff, "ci": [pr.ci_low, pr.ci_high],
           "significant": pr.significant, "mcnemar_p": mcnemar_exact(a, b)["p_value"],
           "n_pairs": pr.n_pairs}
    (RESULTS / "phaseA.json").write_text(json.dumps(out, indent=2))
    log(f"  -> B0={out['B0_solve']:.2f} B2={out['B2_solve']:.2f} lift={pr.mean_diff:+.3f} "
        f"CI[{pr.ci_low:+.2f},{pr.ci_high:+.2f}] sig={pr.significant}")
    return out


def phase_B_credit_reward(corpus, embedder, families, epochs, solve_fn, seeds, phaseA):
    log("PHASE B — credit/reward ablation (bandit learns true vs poison)")
    configs = {
        "cited_step":    dict(credit_mode="cited_only", lam=0.5),   # default (full method)
        "allM_step":     dict(credit_mode="all_m",      lam=0.5),   # recovery-masking ablation
        "cited_outcome": dict(credit_mode="cited_only", lam=0.0),   # latent-poison ablation (no step reward)
    }
    results = {}
    for name, over in configs.items():
        cfg = build_config(seed=1, **over)
        store, truth = fresh_graph(embedder, families)
        system = SystemSpec("B4", use_rerank=True, use_mmr=True, use_bandit=True)
        orch, bandit, logger = make_orchestrator(cfg, system, store, embedder, truth,
                                                 f"B_{name}", solve_fn)
        traj = []
        # per-epoch solve rate (the learning curve)
        epoch_solve = []
        for ep in range(epochs):
            order = list(range(len(corpus)))
            np.random.RandomState(1234 + ep).shuffle(order)
            ep_out = []
            for j, i in enumerate(order):
                r = orch.run_one(corpus[i], experiment_id=f"B_{name}",
                                 rng=np.random.RandomState(5000 + 1000 * ep + j))
                ep_out.append(r.outcome)
            epoch_solve.append(float(np.mean(ep_out)))
            traj.append({"epoch": ep + 1, "q_by_truth": q_by_truth(bandit, truth)})
            log(f"  {name} epoch {ep+1}: solve={epoch_solve[-1]:.2f} Q={traj[-1]['q_by_truth']}")
        # recovery-masking rate from citation logs (on successes)
        runs = logger.read("run"); cites = logger.read("citation")
        cited_by_run = defaultdict(lambda: [0, 0])
        for c in cites:
            cited_by_run[c["run_id"]][0] += 1
            cited_by_run[c["run_id"]][1] += int(c["cited"])
        rm = [(t - cc) / t for r in runs if r["outcome"]
              for (t, cc) in [cited_by_run.get(r["run_id"], (len(r["sampled_set"]), 0))] if t > 0]
        # poison citation rate: how often a poison node was action-grounded-cited
        poison_ids = {nid for nid, (lab, _) in truth.items() if lab == "poison"}
        pois_cited = [int(c["cited"]) for c in cites if c["node_id"] in poison_ids]
        entry = {
            "trajectory": traj,
            "epoch_solve": epoch_solve,
            "final_q_by_truth": traj[-1]["q_by_truth"] if traj else {},
            "recovery_masking_rate": float(np.mean(rm)) if rm else 0.0,
            "poison_citation_rate": float(np.mean(pois_cited)) if pois_cited else 0.0,
            "n_runs": len(runs),
            "solve_rate": float(np.mean([r["outcome"] for r in runs])),
        }
        # for the FULL method, run a frozen eval (alpha=0) paired vs B0/B2 (headline)
        if name == "cited_step":
            bandit.config = cfg.with_overrides(alpha=0.0)   # freeze learning
            per = {}
            for s in seeds:
                for i, ch in enumerate(corpus):
                    r = orch.run_one(ch, experiment_id=f"B_{name}_eval",
                                     rng=np.random.RandomState(9000 + 100 * s + i))
                    per[f"{ch['name']}#s{s}"] = {"solved": r.outcome, "steps": r.steps}
            entry["eval_per_run"] = per
            entry["B4_solve"] = float(np.mean([v["solved"] for v in per.values()]))
            for base in ("B0", "B2"):
                keys = sorted(set(per) & set(phaseA["per_run"][base]))
                a = [per[k]["solved"] for k in keys]
                b = [phaseA["per_run"][base][k]["solved"] for k in keys]
                pr = paired_bootstrap_ci(a, b, seed=1)
                entry[f"lift_vs_{base}"] = {"lift": pr.mean_diff, "ci": [pr.ci_low, pr.ci_high],
                                            "significant": pr.significant, "n": pr.n_pairs,
                                            "mcnemar_p": mcnemar_exact(a, b)["p_value"]}
            log(f"  cited_step FROZEN EVAL: B4={entry['B4_solve']:.2f} "
                f"vs B0 {entry['lift_vs_B0']['lift']:+.2f} vs B2 {entry['lift_vs_B2']['lift']:+.2f}")
        results[name] = entry
    (RESULTS / "phaseB.json").write_text(json.dumps(results, indent=2))
    log("  -> phase B done")
    return results


def phase_C_writeback(corpus, embedder, families, epochs, solve_fn):
    log("PHASE C — write-back gating (gated B5 vs ungated B6)")
    results = {}
    for name, gated in [("B5_gated", True), ("B6_ungated", False)]:
        cfg = build_config(seed=2)
        store, truth = fresh_graph(embedder, families)
        system = SystemSpec(name, use_rerank=True, use_mmr=True, use_bandit=True,
                            use_writeback=True, writeback_gated=gated)
        orch, bandit, logger = make_orchestrator(cfg, system, store, embedder, truth,
                                                 f"C_{name}", solve_fn)
        for ep in range(epochs):
            for j, ch in enumerate(corpus):
                orch.run_one(ch, experiment_id=f"C_{name}",
                             rng=np.random.RandomState(7000 + 1000 * ep + j))
        # tier census of write-back nodes
        ids, _ = store.get_all_embeddings()
        tiers = defaultdict(int)
        wb_nodes = 0
        for nid in ids:
            n = store.get_node(nid)
            if n and n.attributes.get("source") == "writeback":
                tiers[n.attributes.get("tier", "provisional")] += 1
                wb_nodes += 1
        runs = logger.read("run")
        results[name] = {
            "writeback_nodes": wb_nodes, "tiers": dict(tiers),
            "solve_rate": float(np.mean([r["outcome"] for r in runs])),
            "n_runs": len(runs),
        }
        log(f"  {name}: wb_nodes={wb_nodes} tiers={dict(tiers)} solve={results[name]['solve_rate']:.2f}")
    (RESULTS / "phaseC.json").write_text(json.dumps(results, indent=2))
    return results


def phase_D_loo(corpus, embedder, families, solve_fn, n_fam=5, n_chal=2, seeds=(0, 1)):
    log("PHASE D — LOO false-lesson exposure (oracle-free causal probe)")
    from knowledge_graph.models import Node, NodeType
    store, truth = fresh_graph(embedder, families)
    # gather poison nodes and a true node per family
    poison_nodes = {}
    for nid, (lab, fam) in truth.items():
        if lab == "poison":
            poison_nodes[fam] = store.get_node(nid)
    # sample families that have a poison node
    fams = [f for f in poison_nodes if any(c["_family"] == f for c in corpus)][:n_fam]
    marginals = []
    for fam in fams:
        chals = [c for c in corpus if c["_family"] == fam][:n_chal]
        poison = poison_nodes[fam]
        # a TRUE node for the family (the correct tool) as the positive control
        true_node = None
        for nid, (lab, f2) in truth.items():
            if lab == "helpful" and f2 == fam:
                true_node = store.get_node(nid); break
        for ch in chals:
            for node, kind in [(poison, "poison")] + ([(true_node, "true")] if true_node else []):
                o_with, o_without, s_with, s_without = [], [], [], []
                for sd in seeds:
                    rw = solve_fn(ch, [node], SOLVE_ROUNDS)
                    rwo = solve_fn(ch, [], SOLVE_ROUNDS)
                    o_with.append(int(bool(rw["flag"]))); o_without.append(int(bool(rwo["flag"])))
                    s_with.append(rw["rounds_used"]); s_without.append(rwo["rounds_used"])
                marginals.append({
                    "family": fam, "challenge": ch["name"], "node_kind": kind,
                    "marginal_solve": float(np.mean(o_with) - np.mean(o_without)),
                    "marginal_steps": float(np.mean(s_without) - np.mean(s_with)),
                    "n": len(seeds),
                })
        log(f"  {fam}: {[m for m in marginals if m['family']==fam]}")
    # exposure: poison nodes whose marginal causal value ~ 0
    pois = [m for m in marginals if m["node_kind"] == "poison"]
    tru = [m for m in marginals if m["node_kind"] == "true"]
    eps = 0.05
    exposure = np.mean([abs(m["marginal_solve"]) <= eps and m["marginal_steps"] <= eps for m in pois]) if pois else 0.0
    out = {"marginals": marginals,
           "poison_mean_marginal_solve": float(np.mean([m["marginal_solve"] for m in pois])) if pois else 0.0,
           "true_mean_marginal_solve": float(np.mean([m["marginal_solve"] for m in tru])) if tru else 0.0,
           "false_lesson_exposure_rate": float(exposure)}
    (RESULTS / "phaseD.json").write_text(json.dumps(out, indent=2))
    log(f"  -> poison Δsolve={out['poison_mean_marginal_solve']:+.2f} "
        f"true Δsolve={out['true_mean_marginal_solve']:+.2f} exposure={exposure:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-family", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--phases", default="ABCD")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    log(f"loading embedder (all-MiniLM, CPU to spare VRAM for the 7B)")
    embedder = SentenceTransformerEmbedder(device="cpu")
    families = list(CATEGORY_OF.keys())

    log(f"generating corpus: {args.n_per_family}/family")
    corpus = generate_corpus(RESULTS / "corpus", n_per_family=args.n_per_family, seed=20260531)
    corpus_by_name = {c["name"]: c for c in corpus}
    solve_fn = make_checked_solver(corpus_by_name)

    # §6.1 budget for the record
    budget = sample_efficiency_budget(
        expected_solve_runs=len(corpus) * args.epochs * 3, M=8,
        num_context_nodes=3, avg_candidates_credited_per_run=2)
    meta = {"model": os.getenv("MODEL_NAME"), "n_challenges": len(corpus),
            "families": families, "n_per_family": args.n_per_family,
            "epochs": args.epochs, "seeds": args.seeds,
            "categories": {c: CATEGORY_OF.get(c) for c in families},
            "budget_6_1": budget}
    (RESULTS / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"corpus={len(corpus)} challenges; §6.1 updates/cell={budget['expected_updates_per_cell']:.0f} ({budget['verdict']})")

    seeds = list(range(args.seeds))
    t0 = time.time()
    phaseA = None
    if "A" in args.phases:
        phaseA = phase_A_retrieval_value(corpus, embedder, families, seeds, solve_fn)
    if "B" in args.phases:
        if phaseA is None:   # allow standalone B by loading A's result
            phaseA = json.loads((RESULTS / "phaseA.json").read_text())
        phase_B_credit_reward(corpus, embedder, families, args.epochs, solve_fn, seeds, phaseA)
    if "C" in args.phases:
        phase_C_writeback(corpus, embedder, families, args.epochs, solve_fn)
    if "D" in args.phases:
        phase_D_loo(corpus, embedder, families, solve_fn)
    log(f"ALL PHASES DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
