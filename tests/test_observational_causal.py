"""
Validate observational causal credit (ope/causal_value.py): recover the true>poison
ranking from logged sampling data ALONE - no re-rolls - using the propensity that the
stochastic retrieval already logs.
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
from ope.causal_value import observational_lesson_values, summarize_by_label, marginal_inclusion_propensities


def test_marginal_inclusion_propensity_sums_to_M():
    pool = [{"node_id": f"n{i}", "score": s} for i, s in enumerate([2.0, 1.0, 0.5, 0.2, 0.0])]
    pi = marginal_inclusion_propensities(pool, m=2, tau=1.0, n_mc=2000, seed=1)
    # sum of inclusion probs ≈ M (each draw includes exactly one node)
    assert abs(sum(pi.values()) - 2.0) < 0.1
    # higher score -> higher inclusion prob
    assert pi["n0"] > pi["n4"]


def test_observational_recovers_true_over_poison(tmp_path):
    # synthetic world: helpful tools raise reward, poison nodes route the long way.
    world = build_world(n_challenges=36, dim=64, seed=5, poison_per_category=1)
    cfg = SolverConfig().with_overrides(N=40, R=24, M=6, kappa=2.0, alpha=0.2,
                                        embedding_dim=64, context_resolution="category",
                                        credit_signal="outcome", seed=1)
    bandit = TabularBandit(cfg)
    cr = ContextResolver(cfg)
    retr = ContextRetriever(world.store, world.embedder, config=cfg, reranker=MockReranker(),
                            context_resolver=cr, bandit=bandit)
    logger = ExperimentLogger("obs_test", root=tmp_path)
    orch = Orchestrator(world.store, world.embedder, cfg,
                        SystemSpec("B4", use_bandit=True), retriever=retr, bandit=bandit,
                        logger=logger)
    # run several epochs (outcome credit - the 'fooled' bandit)
    for ep in range(6):
        for i, ch in enumerate(world.challenges):
            orch.run_one(ch, rng=np.random.RandomState(1000 * ep + i))

    runs = logger.read("run")
    # ground-truth labels from the synthetic world
    truth = {}
    for ids, lab in [(sum(world.helpful_node_ids.values(), []), "helpful"),
                     (world.poison_node_ids, "poison"),
                     (world.distractor_node_ids, "distractor")]:
        for nid in ids:
            truth[nid] = (lab, "x")

    values = observational_lesson_values(runs, truth=truth)
    summ = summarize_by_label(values)
    # the observational estimator should rank helpful ABOVE poison (the separation the
    # outcome bandit's Q could not achieve), using only logged sampling propensities.
    assert "helpful" in summ and "poison" in summ
    assert summ["helpful"]["mean_ate"] > summ["poison"]["mean_ate"], summ
