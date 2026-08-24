"""Tests for C5: reward, partial-pooled standardization, grader + calibration."""

import math

from config import SolverConfig
from reward import RewardStandardizer, DifficultyGrader, MockDifficultyGrader, GraderCalibration
from reward.grader import difficulty_from_ground_truth


def test_fast_success_beats_slow_success():
    cfg = SolverConfig().with_overrides(lam=0.5)
    s = RewardStandardizer(cfg)
    for steps in [8, 9, 10, 11, 12]:        # cell mean ~10
        s.observe("pwn", "easy", steps)
    fast = s.reward(1, steps=6, category="pwn", difficulty="easy")
    slow = s.reward(1, steps=14, category="pwn", difficulty="easy")
    assert fast.R_final > 1.0 > slow.R_final     # fast >1, slow <1
    assert fast.R_final > slow.R_final


def test_failure_reward_default_zero():
    s = RewardStandardizer(SolverConfig())
    s.observe("web", "easy", 10)
    r = s.reward(0, steps=5, category="web", difficulty="easy")
    assert r.R_final == 0.0


def test_partial_pooling_shrinks_sparse_cells_to_global():
    cfg = SolverConfig().with_overrides(tau0=5.0)
    s = RewardStandardizer(cfg)
    # lots of global data centered ~10
    for c in ["a", "b", "c"]:
        for steps in range(5, 16):
            s.observe(c, "easy", steps)
    # a brand-new sparse cell with a single extreme observation
    s.observe("rare", "hard", 50)
    mu, sigma, n = s.pooled_stats("rare", "hard")
    g_mean, _ = s._global_mean_var()
    # with n=1 and tau0=5, mu should be pulled strongly toward global (~10), not 50
    assert n == 1
    assert mu < 20, mu
    assert abs(mu - g_mean) < abs(50 - g_mean)


def test_unseen_cell_uses_global():
    s = RewardStandardizer(SolverConfig())
    for steps in range(5, 16):
        s.observe("x", "easy", steps)
    mu, sigma, n = s.pooled_stats("never", "seen")
    assert n == 0
    g_mean, g_var = s._global_mean_var()
    assert abs(mu - g_mean) < 1e-9


def test_ground_truth_difficulty_from_points():
    assert difficulty_from_ground_truth({"points": 50}) == "easy"
    assert difficulty_from_ground_truth({"points": 150}) == "medium"
    assert difficulty_from_ground_truth({"points": 350}) == "hard"
    assert difficulty_from_ground_truth({"points": 500}) == "expert"
    assert difficulty_from_ground_truth({"difficulty": "Hard"}) == "hard"
    assert difficulty_from_ground_truth({}) is None


def test_grader_prefers_ground_truth_over_estimate():
    g = MockDifficultyGrader()
    res = g.grade({"points": 50, "description": "heap kernel rop format string lattice " * 20})
    assert res.source == "ground_truth"
    assert res.difficulty == "easy"   # GT wins even though text looks hard
    # no ground truth -> grader estimate, flagged as such, non-Qwen model name
    res2 = g.grade({"description": "heap rop format string lattice obfuscation " * 10,
                    "files": [1, 2, 3]})
    assert res2.source == "grader"
    assert res2.grader_model == "mock-heuristic-grader"


def test_grader_calibration_monotonic():
    cal = GraderCalibration()
    # honest grader: harder buckets solved less
    for _ in range(10):
        cal.observe("easy", solved=True)
    for i in range(10):
        cal.observe("medium", solved=i < 6)
    for i in range(10):
        cal.observe("hard", solved=i < 2)
    rep = cal.report()
    assert rep["monotonic_agreement"] == 1.0
    assert rep["solve_rate_by_bucket"]["easy"] > rep["solve_rate_by_bucket"]["hard"]
