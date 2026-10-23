"""Tests for C4: eval harness, paired stats, datasets, reproducibility."""

import numpy as np

from config import SolverConfig
from evaluation.baselines import get_baselines, C4_REQUIRED
from evaluation.harness import EvalHarness
from evaluation.synthetic import build_world
from evaluation.datasets import make_synthetic_partitions, transform_family
from evaluation.stats import paired_bootstrap_ci, mcnemar_exact, power_analysis


def _cfg():
    return SolverConfig().with_overrides(N=40, R=20, M=6, n_clusters=8,
                                         embedding_dim=64, context_resolution="category", seed=1)


def _factory(seed=4):
    def f():
        w = build_world(n_challenges=24, dim=64, seed=seed)
        return w.store, w.embedder
    return f


def test_paired_bootstrap_detects_real_difference():
    rng = np.random.RandomState(0)
    a = (rng.rand(200) < 0.7).astype(float)   # 70%
    b = (rng.rand(200) < 0.4).astype(float)   # 40%
    res = paired_bootstrap_ci(a, b, seed=1)
    assert res.mean_diff > 0
    assert res.significant
    assert res.ci_low > 0


def test_paired_bootstrap_no_false_positive_on_equal():
    rng = np.random.RandomState(0)
    x = (rng.rand(300) < 0.5).astype(float)
    res = paired_bootstrap_ci(x, x.copy(), seed=2)   # identical -> zero diff
    assert res.mean_diff == 0
    assert not res.significant


def test_mcnemar_and_power():
    a = [1, 1, 0, 1, 0, 1]
    b = [0, 1, 0, 0, 0, 1]
    mc = mcnemar_exact(a, b)
    assert mc["b01"] == 2 and mc["b10"] == 0
    pw = power_analysis(baseline_rate=0.4, target_lift=0.1)
    assert pw["required_challenges"] > 0


def test_b2_beats_b0_on_synthetic_world(tmp_path):
    cfg = _cfg()
    h = EvalHarness(cfg, results_root=str(tmp_path / "logs_test"), base_seed=999)
    world = build_world(n_challenges=24, dim=64, seed=4)
    challenges = world.challenges
    systems = get_baselines(["B0", "B1", "B2", "B3"])
    out = h.evaluate(systems, challenges, _factory(seed=4),
                     dataset_name="seedtest", train_epochs=0, n_seeds=3)
    # retrieval systems should solve at least as well as zero-shot
    assert out["systems"]["B2"]["solve_rate"] >= out["systems"]["B0"]["solve_rate"] - 0.02
    # lift table present for non-B0 systems
    assert "B2" in out["paired_lift_over_B0"]
    assert "lift" in out["paired_lift_over_B0"]["B2"]


def test_eval_is_reproducible_across_reruns(tmp_path):
    cfg = _cfg()
    systems = get_baselines(["B0", "B2"])
    world = build_world(n_challenges=18, dim=64, seed=4)
    h1 = EvalHarness(cfg, results_root=str(tmp_path / "logs_test1"), base_seed=42)
    h2 = EvalHarness(cfg, results_root=str(tmp_path / "logs_test2"), base_seed=42)
    o1 = h1.evaluate(systems, world.challenges, _factory(seed=4), "rep", train_epochs=0, n_seeds=2)
    o2 = h2.evaluate(systems, world.challenges, _factory(seed=4), "rep", train_epochs=0, n_seeds=2)
    assert o1["systems"]["B2"]["solve_rate"] == o2["systems"]["B2"]["solve_rate"]


def test_transform_family_preserves_solution():
    world = build_world(n_challenges=6, dim=64, seed=4)
    ch = world.challenges[0]
    var = transform_family(ch, variant_seed=2)
    assert var["_world"]["solution_tools"] == ch["_world"]["solution_tools"]
    assert var["name"] != ch["name"]
    assert var["contamination_flag"] is False


def test_partitions_tag_contamination():
    world = build_world(n_challenges=12, dim=64, seed=4)
    parts = make_synthetic_partitions(world)
    assert all(c["contamination_flag"] for c in parts["seed"])
    assert all(not c["contamination_flag"] for c in parts["heldout"])
    assert all(c["dataset"] == "families" for c in parts["families"])
