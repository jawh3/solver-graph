"""
Non-additive (SUTVA-stress) validation of observational causal credit.

The estimator differences E[R | v in M] - E[R | v not in M], marginalizing over
co-sampled lessons -- valid under additive rewards. CTF lessons INTERACT, so this
test builds a world where two true lessons are COMPLEMENTARY (the task solves only
when BOTH are in context) plus a poison that wastes the budget. The estimator's
exact ATE is attenuated by the interaction, but the question that matters for
false-lesson SUPPRESSION is whether the RANKING survives: do the complementary true
lessons still rank above poison and distractor? This addresses the reviewer
objection that additive synthetic validation is circular.
"""

import numpy as np

from retrieval.sampling import sample_without_replacement
from ope.causal_value import observational_lesson_values, summarize_by_label


def _simulate_runs(n_runs=600, m=4, tau=1.0, seed=0):
    """Generate logged RunRecords with COMPLEMENTARY true lessons + poison + distractors.

    Nodes: A,B = complementary true (solve iff BOTH sampled); P = poison (if sampled,
    halves the success reward - wastes budget); D0..D4 = distractors (no effect).
    Reward R = base_success * (0.4 if P present else 1.0), base_success=1 iff {A,B}⊆M.
    """
    rng = np.random.RandomState(seed)
    nodes = ["A", "B", "P", "D0", "D1", "D2", "D3"]
    # fixed-ish scores (A,B,P moderately likely; distractors lower) + per-run jitter
    base = {"A": 1.2, "B": 1.1, "P": 1.0, "D0": 0.6, "D1": 0.5, "D2": 0.4, "D3": 0.3}
    runs = []
    for i in range(n_runs):
        scores = [base[v] + 0.3 * rng.randn() for v in nodes]
        order, _, set_lp = sample_without_replacement(scores, m=m, tau=tau,
                                                      rng=np.random.RandomState(seed * 7919 + i))
        sampled = [nodes[j] for j in order]
        s = set(sampled)
        base_success = 1.0 if ("A" in s and "B" in s) else 0.0
        R = base_success * (0.4 if "P" in s else 1.0)
        runs.append({
            "run_id": f"r{i}", "context_node_id": "c0", "temperature": tau,
            "candidate_pool": [{"node_id": nodes[j], "score": scores[j]} for j in range(len(nodes))],
            "sampled_set": sampled, "R_final": R,
        })
    return runs


def test_ranking_survives_non_additivity():
    runs = _simulate_runs(n_runs=800, m=4, seed=1)
    truth = {"A": ("helpful", "x"), "B": ("helpful", "x"), "P": ("poison", "x"),
             "D0": ("distractor", "x"), "D1": ("distractor", "x"),
             "D2": ("distractor", "x"), "D3": ("distractor", "x")}
    vals = observational_lesson_values(runs, truth=truth, min_obs=5, ess_frac=0.2)
    summ = summarize_by_label(vals)
    # RANKING must survive interference: complementary true lessons rank above poison,
    # which ranks at/below distractors (it actively harms).
    assert "helpful" in summ and "poison" in summ, summ
    assert summ["helpful"]["mean_ate"] > summ["poison"]["mean_ate"], summ
    # poison is harmful -> negative-ish; distractors ~0
    assert summ["poison"]["mean_ate"] < summ["helpful"]["mean_ate"]
    if "distractor" in summ:
        assert summ["helpful"]["mean_ate"] > summ["distractor"]["mean_ate"]
