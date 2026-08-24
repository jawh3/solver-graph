"""Regression tests for the 2026-06-03 benchmark-audit fixes.

Guards the bugs the audit found so they cannot silently return:
- build_seed_graph poison nodes carry `_poison` AND classify into the POISON bucket
  (the mock `_poison` contract divergence that would have demoted them to DISTRACTOR).
- POISON raises mock steps (effort teeth fire) while HELPFUL cuts them — the
  effort-channel detection the thesis rests on (previously untested end-to-end).
- generated grid family names are unique (duplicate `('rev','bz2')` _TWO_LAYER entry removed).
- canonical_name does NOT telegraph correctness ('_wrong'/'_recipe' leak removed).
"""
import numpy as np

import benchmark.hard_ctf as hard
from retrieval.embedder import MockEmbedder
from solver.mock_solver import classify_lessons, mock_solve_challenge


def _family():
    return hard.grid_family_names(n=1)[0]


def _nodes(fam):
    store, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=[fam])
    pick = lambda lab: [store.get_node(nid) for nid, (l, _) in truth.items() if l == lab][0]
    return pick("helpful"), pick("poison"), pick("distractor")


def _world_challenge(fam, helpful):
    return {"name": fam, "_world": {
        "solution_tools": list(helpful.attributes["invocation_signature"]),
        "steps_base": 10, "step_floor": 2}}


def test_seedgraph_poison_classifies_as_poison():
    fam = _family()
    helpful, poison, distract = _nodes(fam)
    h, p, d = classify_lessons(_world_challenge(fam, helpful), [helpful, poison, distract])
    assert poison in p, "poison node must land in the POISON bucket (regression: was demoted to distractor)"
    assert helpful in h
    assert distract in d
    assert poison.attributes.get("_poison") is True, "build_seed_graph poison node must carry _poison"


def test_poison_raises_steps_helpful_cuts_steps():
    """Effort-channel detection: in the unclamped regime poison costs MORE steps; helpful FEWER."""
    fam = _family()
    helpful, poison, _ = _nodes(fam)
    ch = _world_challenge(fam, helpful)
    mean_steps = lambda nodes: float(np.mean(
        [mock_solve_challenge(ch, sampled_nodes=nodes, seed=s, max_rounds=18)["rounds_used"]
         for s in range(500)]))
    base = mean_steps([])
    assert mean_steps([poison]) > base + 1.5, "poison teeth (+3 steps) must fire (unclamped)"
    assert mean_steps([helpful]) < base - 0.5, "helpful must cut steps"


def test_grid_family_names_unique():
    names = hard.grid_family_names()
    assert len(names) == len(set(names)), "duplicate grid family names (regression: _TWO_LAYER dup)"
    gnames = [f.name for f in hard.GRID_FAMILIES]
    assert len(gnames) == len(set(gnames)), "duplicate names in the generated grid"


def test_canonical_name_does_not_leak_label():
    fam = _family()
    helpful, poison, _ = _nodes(fam)
    for n in (helpful, poison):
        cn = n.attributes["canonical_name"].lower()
        assert "wrong" not in cn and "recipe" not in cn, f"canonical_name leaks label: {cn}"
