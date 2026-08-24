"""Tests for C3: citation matcher, reward-hacking guard, context assembly, mock solver."""

from knowledge_graph.models import Node, NodeType
from solver.citation import (
    match_citations, detect_non_exploit_capture, citation_quality,
)
from solver.context_assembly import assemble_context, render_node
from solver.mock_solver import mock_solve_challenge, classify_lessons


def _tool(node_id, sig, name=None):
    return Node(id=node_id, type=NodeType.TOOL,
                attributes={"canonical_name": name or sig, "invocation_signature": sig})


def _tech(node_id, name, keywords=None):
    return Node(id=node_id, type=NodeType.TECHNIQUE,
                attributes={"canonical_name": name, "keywords": keywords or []})


def test_citation_matches_executed_commands_not_self_report():
    nodes = [_tool("t_r2", "r2"), _tool("t_ropper", "ropper"), _tool("t_steghide", "steghide")]
    trace = [
        {"round": 1, "commands": ["r2 -q -c 'aaa; pdf @ main' target"]},
        {"round": 2, "commands": ["echo step_2"]},
    ]
    matches = {m.node_id: m for m in match_citations(nodes, trace)}
    assert matches["t_r2"].cited is True
    assert "r2" in matches["t_r2"].match_evidence
    # ropper and steghide were in context but never executed -> NOT cited (recovery masking)
    assert matches["t_ropper"].cited is False
    assert matches["t_steghide"].cited is False


def test_citation_quality_against_key():
    nodes = [_tool("a", "r2"), _tool("b", "ropper"), _tech("c", "rsa", ["rsa", "gmpy2"])]
    trace = [{"round": 1, "commands": ["r2 target", "python3 -c 'import gmpy2; rsa'"]}]
    matches = match_citations(nodes, trace)
    key = {"a": True, "b": False, "c": True}
    q = citation_quality(matches, key)
    assert q["precision"] == 1.0
    assert q["recall"] == 1.0


def test_non_exploit_capture_guard():
    pwn = {"name": "ch", "category": "pwn"}
    trace_trivial = [{"round": 1, "commands": ["cat flag.txt"]}]
    assert detect_non_exploit_capture(pwn, "picoCTF{x}", trace_trivial) is True
    # forensics is allowed to read files trivially
    forensics = {"name": "ch", "category": "forensics"}
    assert detect_non_exploit_capture(forensics, "picoCTF{x}", trace_trivial) is False
    # no flag -> not a capture
    assert detect_non_exploit_capture(pwn, None, trace_trivial) is False


def test_context_assembly_marks_provisional_and_applicability():
    trusted = Node(id="t1", type=NodeType.TECHNIQUE,
                   attributes={"canonical_name": "ret2win", "text": "overwrite RIP",
                               "tier": "trusted",
                               "applicability": {"categories": ["pwn"],
                                                 "preconditions": ["no PIE"]}})
    prov = Node(id="p1", type=NodeType.TECHNIQUE,
                attributes={"canonical_name": "guess", "text": "try it", "tier": "provisional"})
    ctx = assemble_context([prov, trusted])
    assert "ret2win" in ctx and "guess" in ctx
    assert "PROVISIONAL" in ctx
    assert "applies when" in ctx
    # trusted rendered before provisional (stable ordering)
    assert ctx.index("ret2win") < ctx.index("guess")


def test_mock_solver_helpful_lesson_cited_and_faster():
    challenge = {"name": "rev1", "category": "rev",
                 "_world": {"solution_tools": ["r2"], "steps_base": 12, "p_solve_base": 0.5}}
    helpful = _tool("t_r2", "r2")
    distractor = _tool("t_x", "steghide")
    helpful, poison, dist = classify_lessons(challenge, [helpful, distractor])
    assert len(helpful) == 1 and len(dist) == 1

    res_help = mock_solve_challenge(challenge, sampled_nodes=[_tool("t_r2", "r2")], seed=1)
    res_none = mock_solve_challenge(challenge, sampled_nodes=[], seed=1)
    # helpful lesson should appear in executed commands (citable)
    cmds = [c for r in res_help["trace"] for c in r["commands"]]
    assert any("r2" in c for c in cmds)
    # and generally cut steps vs no lesson (deterministic seed)
    assert res_help["rounds_used"] <= res_none["rounds_used"]


def test_mock_solver_poison_increases_steps():
    challenge = {"name": "pwn1", "category": "pwn",
                 "_world": {"solution_tools": ["ropper"], "steps_base": 8, "p_solve_base": 0.6}}
    good = _tool("g", "ropper")
    poison = Node(id="p", type=NodeType.TECHNIQUE,
                  attributes={"canonical_name": "wrong_premise", "_poison": True,
                              "invocation_signature": "ropper"})
    res_good = mock_solve_challenge(challenge, sampled_nodes=[good], seed=5)
    res_poison = mock_solve_challenge(challenge, sampled_nodes=[poison], seed=5)
    # poison routes the long way -> more steps than the genuinely-good lesson
    assert res_poison["rounds_used"] > res_good["rounds_used"]


def test_mock_solver_result_shape():
    res = mock_solve_challenge({"name": "x", "_world": {}}, sampled_nodes=[], seed=0)
    assert set(res.keys()) == {"flag", "trace", "rounds_used", "gave_up_reason", "total_commands"}
