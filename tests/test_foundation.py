"""Tests for config.py and the instrumentation layer (foundation)."""

import json

from config import (
    DEFAULT_CONFIG,
    SolverConfig,
    ContextResolution,
    CreditMode,
)
from instrumentation import (
    ExperimentLogger,
    RunRecord,
    CitationRecord,
    UpdateRecord,
)


def test_config_hash_is_stable_and_sensitive():
    c1 = SolverConfig()
    c2 = SolverConfig()
    assert c1.hash() == c2.hash()                      # deterministic
    c3 = c1.with_overrides(M=12)
    assert c3.M == 12
    assert c3.hash() != c1.hash()                      # sensitive to a real knob
    # doc-only metadata must not affect the hash
    assert "sweep_range" not in c1.policy_dict()


def test_config_overrides_coerce_enums():
    c = DEFAULT_CONFIG.with_overrides(
        context_resolution="category", credit_mode="all_m"
    )
    assert c.context_resolution is ContextResolution.CATEGORY
    assert c.credit_mode is CreditMode.ALL_M
    # round-trips through the hash without raising
    assert isinstance(c.hash(), str)


def test_policy_dict_is_json_serializable():
    blob = json.dumps(DEFAULT_CONFIG.policy_dict(), sort_keys=True)
    assert "context_resolution" in blob
    assert "cluster" in blob  # enum serialized to its value


def test_experiment_logger_roundtrip(tmp_path):
    logger = ExperimentLogger("exp_test", root=tmp_path)
    run = RunRecord(
        run_id="r1", challenge_id="ch1",
        policy_config_hash=DEFAULT_CONFIG.hash(),
        sampled_set=["n1", "n2"],
        per_draw_propensity=[{"node_id": "n1", "p": 0.6}, {"node_id": "n2", "p": 0.4}],
        set_logprob=-1.02, outcome=1, R_final=1.3,
    )
    logger.log_run(run)
    logger.log_citation(CitationRecord(run_id="r1", node_id="n1", cited=True,
                                       match_evidence="strings"))
    logger.log_update(UpdateRecord(run_id="r1", context_node_id="c0", node_id="n1",
                                   pre_Q=0.0, post_Q=0.13, credit_eligible=True))

    runs = logger.read("run")
    assert len(runs) == 1
    assert runs[0]["sampled_set"] == ["n1", "n2"]
    assert runs[0]["per_draw_propensity"][0]["p"] == 0.6
    cites = logger.read("citation")
    assert cites[0]["cited"] is True
    updates = logger.read("update")
    assert updates[0]["post_Q"] == 0.13


def test_logger_is_append_only(tmp_path):
    logger = ExperimentLogger("exp_append", root=tmp_path)
    for i in range(3):
        logger.log_run(RunRecord(run_id=f"r{i}", challenge_id="ch"))
    assert len(logger.read("run")) == 3   # nothing overwritten
