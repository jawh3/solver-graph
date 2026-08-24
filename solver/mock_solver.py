"""
solver/mock_solver.py — deterministic mock solver + synthetic CTF world (C3, §2 mock duality).

The real solver (`solver/solver.py::solve_challenge`) calls Qwen3.5 over vLLM and
is the reproducibility anchor — untouched. This mock is its GPU-free twin: it
returns the SAME result shape `{flag, trace, rounds_used, gave_up_reason,
total_commands}` so the orchestrator, eval harness, bandit, and write-back can be
exercised and tested end-to-end with no GPU.

It is also a *research instrument*: a controllable world in which the three
false-lesson modes (§11) are ground-truth-known, so the suppression mechanisms
can be validated before any real run.

World model (encoded on each challenge dict under key `_world`):
    {
      "solution_tools": ["r2", "ropper"],   # tools whose use advances the solve
      "steps_base": 10,                       # steps with no helpful lesson
      "p_solve_base": 0.4,                    # solve prob with no helpful lesson
      "step_floor": 2,
    }

A sampled lesson node interacts with the world by its attributes:
  - HELPFUL  : its tool/canonical_name ∈ solution_tools. Used (=> cited via an
               executed command carrying its invocation signature), cuts steps,
               raises solve prob.
  - POISON   : attribute `_poison=True`. It IS used (cited) but routes the agent
               the long way: solve prob rises slightly yet steps INCREASE — the
               latent-poison signature that only a step-count reward can see (§9.1).
  - DISTRACTOR/FALSE: not in solution and not poison. Sits in context; with some
               probability wastes a step but is NOT cited (no command emitted) —
               the recovery-masking case (in-context, non-causal).

Determinism: outcome/steps are drawn from a seeded RNG keyed by
(challenge_id, sampled_ids, seed), so paired/common-random-seed eval (§14.4) is exact.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

import numpy as np

from knowledge_graph.models import Node, NodeType


def _world(challenge: Dict) -> Dict:
    w = dict(challenge.get("_world", {}))
    w.setdefault("solution_tools", [])
    w.setdefault("steps_base", 10)
    w.setdefault("p_solve_base", 0.4)
    w.setdefault("step_floor", 2)
    return w


def _node_tool(node: Node) -> str:
    a = node.attributes
    return str(a.get("canonical_name") or a.get("invocation_signature") or node.id)


def _seed_for(challenge_id: str, sampled_ids: List[str], seed: int) -> int:
    blob = f"{seed}|{challenge_id}|{'/'.join(sampled_ids)}".encode("utf-8")
    return int(hashlib.sha1(blob).hexdigest(), 16) % (2**32 - 1)


def classify_lessons(challenge: Dict, sampled_nodes: List[Node]):
    """Split the sampled nodes into helpful / poison / distractor (ground truth)."""
    w = _world(challenge)
    sol = set(t.lower() for t in w["solution_tools"])
    helpful, poison, distractor = [], [], []
    for n in sampled_nodes:
        truth = n.attributes.get("_truth")              # build_seed_graph sets this
        if n.attributes.get("_poison") or truth == "poison":
            poison.append(n)
        elif truth == "helpful" or _node_tool(n).lower() in sol:
            helpful.append(n)
        else:
            distractor.append(n)
    return helpful, poison, distractor


def mock_solve_challenge(
    challenge: Dict,
    sampled_nodes: Optional[List[Node]] = None,
    seed: int = 42,
    max_rounds: int = 15,
    **_ignored,
) -> Dict:
    """Deterministic mock of `solve_challenge`. Returns the same result shape."""
    sampled_nodes = sampled_nodes or []
    sampled_ids = [n.id for n in sampled_nodes]
    w = _world(challenge)
    rng = np.random.RandomState(_seed_for(challenge.get("name", "?"), sampled_ids, seed))

    helpful, poison, distractor = classify_lessons(challenge, sampled_nodes)

    # --- solve probability ---
    p = w["p_solve_base"]
    p += 0.18 * len(helpful)          # each helpful lesson helps
    p += 0.05 * len(poison)           # poison helps a *little* (false premise still wins sometimes)
    p -= 0.04 * len(distractor)       # distractors mildly hurt (context dilution)
    p = float(min(0.97, max(0.02, p)))
    solved = bool(rng.random() < p)

    # --- step count ---
    steps = w["steps_base"]
    steps -= 2 * len(helpful)         # helpful lessons cut steps
    steps += 3 * len(poison)          # POISON: longer route (the latent-poison signal)
    steps += 1 * len(distractor)      # distractors add a little wasted work
    steps += int(rng.randint(-1, 2))  # small noise
    steps = max(w["step_floor"], min(max_rounds, steps))
    if not solved:
        steps = min(max_rounds, max(steps, w["steps_base"]))  # failures burn the budget

    # --- build a trace whose executed commands ground citations (§9.3) ---
    # Only HELPFUL and POISON lessons are actually "used" => emit their signatures.
    used = helpful + poison
    trace: List[Dict] = []
    total_commands = 0
    for i in range(steps):
        cmds: List[str] = []
        if i < len(used):
            node = used[i]
            sig = node.attributes.get("invocation_signature") or _node_tool(node)
            cmds = [f"{sig} target_artifact   # round {i+1}"]
        else:
            cmds = [f"echo step_{i+1}"]
        total_commands += len(cmds)
        trace.append({"round": i + 1, "thinking": "", "action": "run_commands",
                      "commands": cmds, "results_summary": "", "flag": None})

    flag = None
    gave_up_reason = None
    if solved:
        flag = f"picoCTF{{mock_{challenge.get('name','x')}_{rng.randint(10**6):06d}}}"
        if trace:
            trace[-1]["flag"] = flag
            trace[-1]["action"] = "submit_flag"
    else:
        gave_up_reason = "rounds_exhausted"

    return {
        "flag": flag,
        "trace": trace,
        "rounds_used": len(trace),
        "gave_up_reason": gave_up_reason,
        "total_commands": total_commands,
    }


class MockSolver:
    """Object wrapper so the orchestrator can hold a `solver` with `.solve(...)`."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model_name = "mock-solver"

    def solve(self, challenge: Dict, sampled_nodes: List[Node],
              max_rounds: int = 15, seed: int = None) -> Dict:
        return mock_solve_challenge(challenge, sampled_nodes=sampled_nodes,
                                    seed=(self.seed if seed is None else seed),
                                    max_rounds=max_rounds)
