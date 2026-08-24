"""Tests for C8: OPE (IPS/SNIPS/DR), recompute logprob, offline sweep."""

import numpy as np

from config import SolverConfig
from retrieval.retriever import ContextRetriever
from retrieval.context_resolution import ContextResolver
from retrieval.reranker import MockReranker
from bandit.bandit import TabularBandit
from pipeline import Orchestrator, SystemSpec
from instrumentation.jsonl_logger import ExperimentLogger
from evaluation.synthetic import build_world
from ope.ope import (
    ips_estimate, dr_estimate, recompute_policy_logprob, logging_logprob,
)
from ope.sweep import offline_sweep, sanity_check, measured_value


def _collect_runs(tmp_path, n_challenges=24, epochs=4, system_id="B4"):
    cfg = SolverConfig().with_overrides(N=40, R=20, M=6, n_clusters=8,
                                        embedding_dim=64, context_resolution="category",
                                        seed=1, kappa=5.0, tau=1.0)
    world = build_world(n_challenges=n_challenges, dim=64, seed=4)
    bandit = TabularBandit(cfg)
    cr = ContextResolver(cfg).fit(world.store)
    retr = ContextRetriever(world.store, world.embedder, config=cfg,
                            reranker=MockReranker(), context_resolver=cr, bandit=bandit)
    logger = ExperimentLogger(f"ope_{system_id}", root=tmp_path)
    system = SystemSpec(system_id, use_bandit=True)
    orch = Orchestrator(world.store, world.embedder, cfg, system,
                        retriever=retr, bandit=bandit, logger=logger)
    for ep in range(epochs):
        for i, ch in enumerate(world.challenges):
            orch.run_one(ch, experiment_id=f"ope_{system_id}",
                         rng=np.random.RandomState(1000 * ep + i))
    return logger.read("run")


def test_logging_logprob_matches_logged_set_logprob(tmp_path):
    runs = _collect_runs(tmp_path)
    run = next(r for r in runs if r["sampled_set"])
    lp = logging_logprob(run)
    assert abs(lp - run["set_logprob"]) < 1e-6


def test_recompute_logprob_changes_with_temperature(tmp_path):
    runs = _collect_runs(tmp_path)
    run = next(r for r in runs if len(r["sampled_set"]) >= 3)
    lp_hot = recompute_policy_logprob(run, tau=2.0, kappa=None)
    lp_cold = recompute_policy_logprob(run, tau=0.3, kappa=None)
    assert lp_hot != lp_cold


def test_ope_sanity_logging_policy_recovers_measured(tmp_path):
    # Acceptance: DR estimate of the logging policy ≈ its measured mean reward.
    runs = _collect_runs(tmp_path, n_challenges=36, epochs=5)
    measured = measured_value(runs)
    snips = ips_estimate(runs, tau=None, kappa=None, self_normalized=True)
    dr = dr_estimate(runs, tau=None, kappa=None)
    # logging policy has weight 1 everywhere -> both should equal measured closely
    assert abs(snips.value - measured) < 1e-6
    assert abs(dr.value - measured) < 0.05
    s = sanity_check(runs)
    assert s["passes"]


def test_ess_drops_for_divergent_policy(tmp_path):
    runs = _collect_runs(tmp_path, n_challenges=36, epochs=5)
    on_policy = ips_estimate(runs, tau=None, kappa=None)        # weights ≈ 1
    off_policy = ips_estimate(runs, tau=0.3, kappa=20.0)        # very different π'
    assert on_policy.ess >= off_policy.ess


def test_offline_sweep_ranks_and_runs(tmp_path):
    runs = _collect_runs(tmp_path, n_challenges=36, epochs=5)
    rows = offline_sweep(runs, taus=(0.5, 1.0, 2.0), kappas=(1.0, 5.0), estimator="dr")
    assert len(rows) == 6
    # sorted descending by value
    assert rows == sorted(rows, key=lambda r: r["value"], reverse=True)
    assert all("ess" in r for r in rows)


def test_m_prime_greater_than_m_not_estimable(tmp_path):
    runs = _collect_runs(tmp_path)
    run = next(r for r in runs if r["sampled_set"])
    m = len(run["sampled_set"])
    assert recompute_policy_logprob(run, m_prime=m + 5) is None
    assert recompute_policy_logprob(run, m_prime=max(1, m - 1)) is not None
