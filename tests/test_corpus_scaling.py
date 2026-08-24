"""
Phase 1c tests — the scalable family generator must (a) reach >=150 lessons/label,
(b) round-trip every family (self-verified codecs), (c) carry a wrong-but-plausible
poison, and (d) NOT change the default (families=None) scope so campaign_v2 still
reproduces from the original 6 core families.
"""

import numpy as np

import benchmark.hard_ctf as hard
from retrieval.embedder import MockEmbedder


def test_grid_reaches_150_lessons_per_label():
    names = hard.grid_family_names(n=160, seed=1)
    _, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=names)
    from collections import Counter
    labels = Counter(t[0] for t in truth.values())
    assert labels["helpful"] >= 150
    assert labels["poison"] >= 150


def test_every_family_round_trips():
    rng = np.random.RandomState(123)
    for f in hard._ALL_FAMILIES:
        flag = hard._rand_flag(rng)
        raw = f.encode(flag, rng)
        assert isinstance(raw, (bytes, bytearray)) and len(raw) > 0
        assert f.decode is not None and f.decode(raw) == flag, f.name


def test_poison_present_and_wrong():
    for f in hard.GRID_FAMILIES:
        assert f.poison_recipe and f.poison_recipe != f.true_recipe, f.name
        assert f.poison_sig


def test_default_scope_is_core_only_backcompat():
    # families=None must still resolve to the original 6 core families (campaign_v2 anchor)
    fams = hard._resolve_families(None)
    assert [f.name for f in fams] == [f.name for f in hard._CORE_FAMILIES]
    assert len(fams) == 6
    _, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1))  # default
    from collections import Counter
    labels = Counter(t[0] for t in truth.values())
    assert labels["helpful"] == 6 and labels["poison"] == 6 and labels["distractor"] == 4


def test_named_grid_family_resolves_and_generates():
    name = hard.grid_family_names(difficulty=["easy", "medium"], n=1, seed=2)[0]
    import tempfile
    corpus = hard.generate_corpus(tempfile.mkdtemp(), n_per_family=2, seed=5, families=[name])
    assert len(corpus) == 2
    ch = corpus[0]
    raw = open(ch["local_files"][0]["local_path"], "rb").read()
    assert hard._BY_NAME[name].decode(raw) == ch["_flag"]
