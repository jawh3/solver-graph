"""T1-1 teeth proof (RIGOR_LEDGER): a malignant lesson, when FOLLOWED, must drop solve BELOW base.

These tests SIMULATE following each malignant recipe on a real corpus artifact (no GPU, no LLM) and
assert the family's normal decode then FAILS to recover the flag - i.e. the recipe has teeth (violates
the A1 floor by construction). This validates the pre-specified prediction before any campaign spend.
"""
import subprocess
import tempfile
from pathlib import Path

import pytest

import benchmark.hard_ctf as hard

# a few representative in-band families (outer x inner depths)
_FAMS = ["b64_gzip", "b64_rev_gzip", "hex_zlib", "b64_rot13_gzip_xor5a"]


def _artifact(fam_name, seed=7):
    d = Path(tempfile.mkdtemp())
    corpus = hard.generate_corpus(d, n_per_family=1, seed=seed, families=[fam_name])
    ch = corpus[0]
    return Path(ch["local_files"][0]["local_path"]), ch["_flag"], hard._BY_NAME[fam_name]


@pytest.mark.parametrize("fam_name", _FAMS)
def test_destroy_corrupts_artifact(fam_name):
    """M-destroy: `truncate -s -16 {file}` then the true decode must NOT recover the flag."""
    path, flag, fam = _artifact(fam_name)
    raw_ok = path.read_bytes()
    # sanity: the UNtouched artifact decodes to the flag (control)
    assert flag in (fam.decode(raw_ok) or ""), "control: true decode should recover the flag"
    # follow the destroy recipe: truncate 16 bytes off the tail (in place)
    subprocess.run(["truncate", "-s", "-16", str(path)], check=True)
    corrupted = path.read_bytes()
    assert len(corrupted) == max(0, len(raw_ok) - 16)
    try:
        rec = fam.decode(corrupted)
    except Exception:
        rec = None
    assert flag not in (rec or ""), f"{fam_name}: destroy recipe should corrupt -> decode fails (teeth)"


@pytest.mark.parametrize("fam_name", _FAMS)
def test_wrongflag_is_wrong(fam_name):
    """M-wrongflag: the asserted static flag must NOT equal the real per-instance flag."""
    _, flag, _ = _artifact(fam_name)
    assert hard._MALIGNANT_FLAG != flag
    # check_flag with the malignant flag must fail
    assert hard.check_flag({"_flag": flag}, hard._MALIGNANT_FLAG) is False
    assert hard.check_flag({"_flag": flag}, flag) is True   # the real flag still verifies (no collateral)


def test_exhaust_budget_blows_the_round_limit():
    """M-exhaust: 256 one-per-step probes >> the per-family budget (solution_length + small slack)."""
    fam = hard._BY_NAME["b64_gzip"]
    budget = fam.solution_length + 2
    assert 256 > budget * 3, "256 sequential key-probes must dwarf the round budget to have teeth"


def test_malignant_nodes_enter_seed_graph_on_target():
    """build_seed_graph(malignant_arms=...) adds one on-target malignant node per family per arm."""
    from retrieval.embedder import MockEmbedder
    fams = ["b64_gzip", "hex_zlib"]
    store, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=fams,
                                         malignant_arms=list(hard.MALIGNANT_ARMS))
    mal = {nid: t for nid, t in truth.items() if t[0].startswith("malignant")}
    assert len(mal) == len(fams) * len(hard.MALIGNANT_ARMS)
    # each malignant node is family-matched (on-target) and carries its arm
    for nid in mal:
        node = store.get_node(nid)
        assert node.attributes["_malignant_arm"] in hard.MALIGNANT_ARMS
        assert node.attributes["_family"] in fams
    # default (no malignant_arms) is unchanged: no malignant nodes
    _, truth0 = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=fams)
    assert not any(t[0] == "malignant" for t in truth0.values())
