"""Tests for the tabular bandit (spec §6, §9.3)."""

import numpy as np

from config import SolverConfig, InitMode, DistanceScaling
from bandit import TabularBandit, LinUCBBandit, sample_efficiency_budget


def test_cold_node_scored_by_prior_not_uniform():
    # zero-init bandit: an unobserved node's score == its similarity prior
    cfg = SolverConfig().with_overrides(init_mode="zero")
    b = TabularBandit(cfg)
    assert abs(b.score("c0", "v_new", prior=0.8) - 0.8) < 1e-9
    assert abs(b.score("c0", "v_other", prior=0.2) - 0.2) < 1e-9


def test_bandit_upweights_good_downweights_bad():
    cfg = SolverConfig().with_overrides(init_mode="zero", distance_scaling="symmetric",
                                        use_baseline_b=False, alpha=0.3)
    b = TabularBandit(cfg)
    ctx = "ctx::cluster::1"
    # good lesson repeatedly earns high reward; bad earns low
    for _ in range(30):
        b.update(ctx, "good", reward=1.4, similarity=0.9, success=True)
        b.update(ctx, "bad", reward=0.0, similarity=0.9, success=False)
    good = b.cells[ctx]["good"].Q
    bad = b.cells[ctx]["bad"].Q
    assert good > 0.8, good
    assert bad < 0.2, bad
    # once warm, the blended score should rank good above bad even at equal prior
    s_good = b.score(ctx, "good", prior=0.5)
    s_bad = b.score(ctx, "bad", prior=0.5)
    assert s_good > s_bad


def test_pseudocount_blend_decays_prior_with_observations():
    cfg = SolverConfig().with_overrides(init_mode="zero", kappa=5.0, use_baseline_b=False, alpha=0.5)
    b = TabularBandit(cfg)
    ctx = "c"
    # with few obs, prior dominates; with many, Q dominates
    b.update(ctx, "v", reward=1.5, similarity=0.5, success=True)
    score_few = b.score(ctx, "v", prior=0.0)   # n=1 -> (1*Q + 5*0)/6, small
    for _ in range(50):
        b.update(ctx, "v", reward=1.5, similarity=0.5, success=True)
    score_many = b.score(ctx, "v", prior=0.0)  # n large -> ~Q
    assert score_many > score_few


def test_asymmetric_distance_scaling():
    cfg = SolverConfig().with_overrides(distance_scaling="asymmetric")
    b = TabularBandit(cfg)
    # failure: blame near nodes more (scale ~ sim)
    assert b.distance_scale(0.9, success=False) > b.distance_scale(0.2, success=False)
    # success: reward far nodes more (scale ~ 1-sim)
    assert b.distance_scale(0.2, success=True) > b.distance_scale(0.9, success=True)


def test_provisional_nodes_are_weight_capped():
    cfg = SolverConfig().with_overrides(init_mode="zero", provisional_weight_cap=0.3,
                                        use_baseline_b=False, alpha=0.5)
    b = TabularBandit(cfg)
    ctx = "c"
    for _ in range(50):
        b.update(ctx, "prov", reward=1.5, similarity=0.5, success=True)
    trusted_score = b.score(ctx, "prov", prior=0.1, provisional=False)
    prov_score = b.score(ctx, "prov", prior=0.1, provisional=True)
    # provisional lift over the prior is capped, so it scores lower than trusted
    assert prov_score < trusted_score


def test_baseline_b_tracks_mean_reward():
    b = TabularBandit(SolverConfig())
    for r in [1.0, 0.0, 1.0, 0.0]:
        b.observe_reward_for_baseline("c", r)
    assert abs(b.baseline("c") - 0.5) < 1e-9


def test_sample_efficiency_budget_flags_too_fine():
    # per-challenge (thousands of contexts) -> too fine
    fine = sample_efficiency_budget(expected_solve_runs=2000, M=8,
                                    num_context_nodes=2000, avg_candidates_credited_per_run=3)
    assert fine["verdict"] in ("TOO_FINE", "MARGINAL")
    # cluster(32) -> OK
    coarse = sample_efficiency_budget(expected_solve_runs=2000, M=8,
                                      num_context_nodes=32, avg_candidates_credited_per_run=3)
    assert coarse["verdict"] == "OK"


def test_bandit_serialization_roundtrip():
    b = TabularBandit(SolverConfig())
    b.update("c", "v", reward=1.0, similarity=0.5, success=True)
    b.observe_reward_for_baseline("c", 1.0)
    d = b.to_dict()
    b2 = TabularBandit.from_dict(d, config=SolverConfig())
    assert b2.cells["c"]["v"].n == 1.0
    assert b2.baseline("c") == b.baseline("c")


def test_linucb_prefers_rewarded_arm():
    rng = np.random.RandomState(0)
    dim = 8
    lin = LinUCBBandit(dim=dim, alpha_ucb=0.0)  # no exploration bonus -> pure mean
    q = np.ones(dim) / np.sqrt(dim)
    good = rng.randn(dim); good /= np.linalg.norm(good)
    bad = rng.randn(dim); bad /= np.linalg.norm(bad)
    for _ in range(50):
        lin.update(q, good, reward=1.5)
        lin.update(q, bad, reward=0.0)
    assert lin.score(q, good) > lin.score(q, bad)
