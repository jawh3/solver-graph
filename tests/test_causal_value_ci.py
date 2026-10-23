"""
Phase 1a tests - per-lesson bootstrap CIs, per-lesson AIPW/DR, label-level &
promotion CIs, and the determinism fix (sha1 propensity seed, not per-process hash()).
All mock / CPU; no GPU.
"""

import subprocess
import sys

import numpy as np

from ope.causal_value import (
    _stable_seed,
    observational_lesson_values,
    observational_lesson_values_aipw,
    summarize_by_label_ci,
    promotion_rate_ci,
    promotion_gap_ci,
)


def _make_runs(n, seed=0, good_effect=1.0, m=2):
    """Fabricate runs where node 'g' raises reward when sampled, 'b' is inert (poison
    the agent recovers from), 'd*' are distractors. Sampling is uniform over the pool."""
    rng = np.random.RandomState(seed)
    nodes = ["g", "b", "d1", "d2"]
    runs = []
    for i in range(n):
        sampled = list(rng.choice(nodes, size=m, replace=False))
        R = good_effect if "g" in sampled else 0.0
        pool = [{"node_id": nid, "score": 1.0, "cosine": 1.0, "prior": 1.0,
                 "Q": None, "n": 0.0} for nid in nodes]
        runs.append({"run_id": f"r{i}", "candidate_pool": pool, "sampled_set": sampled,
                     "R_final": float(R), "temperature": 1.0, "context_node_id": "c0"})
    return runs


TRUTH = {"g": ("helpful", "x"), "b": ("poison", "x"),
         "d1": ("distractor", "x"), "d2": ("distractor", "x")}


def test_stable_seed_is_process_independent():
    # same string -> same seed in THIS process, AND in a fresh process (the bug was
    # CPython's randomized str hash()).
    here = _stable_seed("afee6acba504")
    other = subprocess.check_output(
        [sys.executable, "-c",
         "from ope.causal_value import _stable_seed; print(_stable_seed('afee6acba504'))"],
        text=True).strip()
    assert here == int(other)
    assert _stable_seed("x", salt=0) != _stable_seed("x", salt=1)


def test_observational_values_deterministic():
    runs = _make_runs(200, seed=1)
    a = observational_lesson_values(runs, truth=TRUTH)
    b = observational_lesson_values(runs, truth=TRUTH)
    for k in a:
        assert a[k]["ate"] == b[k]["ate"]


def test_per_lesson_ci_separates_helpful_from_poison():
    runs = _make_runs(400, seed=2)
    vals = observational_lesson_values(runs, truth=TRUTH, n_boot=2000)
    # 'g' raises reward -> positive ATE whose CI excludes 0 (well-powered per lesson).
    assert vals["g"]["ate"] > 0.3 and vals["g"]["ate_excl0"] is True
    assert vals["g"]["ate_ci_lo"] > 0
    # 'b' is inert but in fixed-M sampling its inclusion DISPLACES the helpful node
    # (mutual exclusivity -> the marginal ATE the estimator reports is <= 0, NOT a
    # false positive). This is the SUTVA/displacement caveat, and the desired behaviour:
    # an inert/poison lesson is never assigned a promotable (>0) value.
    assert vals["b"]["ate"] < vals["g"]["ate"]
    assert vals["b"]["ate"] <= 0  # not promoted
    # ci bounds bracket the point estimate
    assert vals["g"]["ate_ci_lo"] <= vals["g"]["ate"] <= vals["g"]["ate_ci_hi"]
    # ESS surfaced for the positivity table
    assert vals["g"]["ess_treat"] > 0 and vals["g"]["ess_control"] > 0


def test_aipw_recovers_sign():
    runs = _make_runs(400, seed=3)
    vals = observational_lesson_values_aipw(runs, truth=TRUTH, n_boot=1000)
    assert vals["g"]["ate"] > vals["b"]["ate"]
    assert vals["g"]["ate"] > 0


def test_label_level_ci_shape():
    runs = _make_runs(300, seed=4)
    vals = observational_lesson_values(runs, truth=TRUTH)
    summ = summarize_by_label_ci(vals, n_boot=2000)
    assert summ["helpful"]["ci_lo"] <= summ["helpful"]["mean"] <= summ["helpful"]["ci_hi"]
    assert summ["helpful"]["mean"] > summ["poison"]["mean"]


def test_promotion_rate_and_gap_ci():
    # outcome-Q-like signal: both helpful and poison "promotable" (>0);
    # causal-ATE-like signal: only helpful >0. Gap should be +1 for poison.
    qval = {"g": 0.4, "b": 0.3, "d1": -0.1, "d2": 0.2}
    ate = {"g": 0.3, "b": -0.1, "d1": -0.2, "d2": -0.1}
    lab = {k: TRUTH[k][0] for k in TRUTH}
    pr = promotion_rate_ci(qval, lab, theta=0.0)
    assert pr["poison"]["rate"] == 1.0
    gap = promotion_gap_ci(qval, ate, lab, theta=0.0)
    assert gap["poison"]["gap"] == 1.0
    assert gap["helpful"]["gap"] == 0.0
    assert gap["poison"]["ci_lo"] <= gap["poison"]["gap"] <= gap["poison"]["ci_hi"]
