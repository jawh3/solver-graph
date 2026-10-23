"""
Online observational bandit (the deployable solution): recompute Q from logged
sampling each epoch via IPW (no per-run updates, no re-rolls) and verify the bandit
learns to value helpful lessons above poison - the separation outcome credit cannot.
"""

import numpy as np

from config import SolverConfig
from retrieval.retriever import ContextRetriever
from retrieval.context_resolution import ContextResolver
from retrieval.reranker import MockReranker
from bandit.bandit import TabularBandit
from pipeline import Orchestrator, SystemSpec
from instrumentation.jsonl_logger import ExperimentLogger
from evaluation.synthetic import build_world
from ope.causal_value import observational_q_table


def _run(credit_signal, tmp_path, epochs=6):
    world = build_world(n_challenges=36, dim=64, seed=5)
    cfg = SolverConfig().with_overrides(N=40, R=24, M=6, kappa=2.0, alpha=0.2,
                                        embedding_dim=64, context_resolution="category",
                                        credit_signal=credit_signal, seed=1, tau0=4.0)
    bandit = TabularBandit(cfg)
    cr = ContextResolver(cfg)
    retr = ContextRetriever(world.store, world.embedder, config=cfg, reranker=MockReranker(),
                            context_resolver=cr, bandit=bandit)
    logger = ExperimentLogger(f"onl_{credit_signal}", root=tmp_path)
    orch = Orchestrator(world.store, world.embedder, cfg, SystemSpec("B4", use_bandit=True),
                        retriever=retr, bandit=bandit, logger=logger)
    truth = {}
    for ids, lab in [(sum(world.helpful_node_ids.values(), []), "helpful"),
                     (world.poison_node_ids, "poison"),
                     (world.distractor_node_ids, "distractor")]:
        for nid in ids:
            truth[nid] = (lab, "x")
    for ep in range(epochs):
        for i, ch in enumerate(world.challenges):
            orch.run_one(ch, rng=np.random.RandomState(1000 * ep + i))
        if credit_signal == "observational":
            runs = logger.read("run")
            bandit.set_q_table(observational_q_table(runs, tau0=cfg.tau0))
    # mean Q by label
    from collections import defaultdict
    agg = defaultdict(list)
    for ctx, nodes in bandit.cells.items():
        for nid, cell in nodes.items():
            if nid in truth and cell.n > 0:
                agg[truth[nid][0]].append(cell.Q)
    return {k: float(np.mean(v)) for k, v in agg.items() if v}


def test_online_observational_separates_where_outcome_fails(tmp_path):
    out = _run("outcome", tmp_path)
    obs = _run("observational", tmp_path)
    # outcome credit: helpful ~ poison (fooled)
    out_gap = out.get("helpful", 0) - out.get("poison", 0)
    # observational credit: helpful clearly above poison (and poison driven low/negative)
    obs_gap = obs.get("helpful", 0) - obs.get("poison", 0)
    assert obs_gap > out_gap, (out, obs)
    assert obs.get("poison", 1.0) < obs.get("helpful", 0.0), obs
    # poison should be driven near or below zero by the causal estimate
    assert obs.get("poison", 1.0) < 0.15, obs
