"""The execution-verifier (positive contrast) must flag ~100% of poison (sibling decode produces
no well-formed flag) and ~0% of helpful — the deployable answer the outcome estimator cannot give."""
import benchmark.hard_ctf as hard
from experiments.execution_verifier import verify, verify_malignant


def test_verifier_flags_poison_not_helpful():
    fams = hard.grid_family_names(n=10)
    res = verify(fams, n_per_family=5, seed=1)
    assert res["helpful_verified_rate"] >= 0.99, res          # correct recipe always recovers a flag
    assert res["poison_verified_rate"] <= 0.01, res           # wrong (sibling) recipe never does
    assert res["verifier_poison_FLAG_rate"] >= 0.99, res


def test_verifier_flags_every_malignant_arm():
    """T1-1: the deployable verifier must flag ALL floor-violating arms at ~1.0 — destroy/exhaust
    fail well-formedness, wrongflag fails input-sensitivity (constant output)."""
    fams = hard.grid_family_names(n=12)
    res = verify_malignant(fams, n_per_family=6, seed=2)
    for arm in hard.MALIGNANT_ARMS:
        d = res["by_arm"][arm]
        assert d["verifier_FLAG_rate"] >= 0.99, (arm, d)      # every malignant arm caught
        assert d["verified_rate"] <= 0.01, (arm, d)
