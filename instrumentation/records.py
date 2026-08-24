"""
instrumentation/records.py — the §12 logging schema as typed dataclasses.

One dataclass per record type in CTF_SOLVER_ARCHITECTURE_v2.md §12. Each is a
plain, JSON-serializable record; `ExperimentLogger` (jsonl_logger.py) writes them
to append-only JSONL.

Why typed records instead of ad-hoc dicts: the off-policy evaluator (C8) and the
eval harness (C4) both consume these logs months after they are written. A typo
in a key name silently breaks IPS/DR (it would read `R=0`). The dataclasses are
the contract.

Design-citation notes for the *fields that matter for the research claim*:
  - `per_draw_propensity` / `set_logprob`: mandatory for unbiased OPE
    (Dudík et al., Doubly-Robust 1103.4601; OfflineEval/replay 1003.5956). The
    sequential without-replacement set probability — NOT a plain per-item softmax
    (BUILD_CHECKLIST C2 pitfall).
  - `CitationRecord.cited` via `match_evidence` from executed actions, not
    self-report: the recovery-masking lever (§9.3; MemoryR2 2605.21768).
  - `time_to_flag_steps` (not wall-clock): the latent-poison signal
    (§9.1; SpuriousRewards 2601.11061).
  - `would_have_retrieved`: free counterfactual refresh instrumentation (§8).
  - `non_exploit_capture`: reward-hacking guard (§11; LLMsGamingVerifiers
    2604.15149, RewardHackingBench 2605.02964).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunRecord:
    """One solve attempt under one policy config (§12 RunRecord).

    This is the central record the OPE estimators and the eval harness read.
    """
    run_id: str
    challenge_id: str
    experiment_id: str = "adhoc"
    policy_config_hash: str = ""
    git_commit: str = ""
    ts: str = field(default_factory=_now_iso)

    # challenge metadata (§3.1 / §14.1)
    challenge_meta: Dict[str, Any] = field(default_factory=dict)
    # ^ {category, difficulty, difficulty_source, contamination_flag, dataset}

    # retrieval (§7)
    context_node_id: str = ""
    context_resolution: str = ""
    candidate_pool: List[Dict[str, Any]] = field(default_factory=list)
    # ^ [{node_id, cosine, rerank_score, mmr_kept}]
    sampled_set: List[str] = field(default_factory=list)
    per_draw_propensity: List[Dict[str, Any]] = field(default_factory=list)
    # ^ [{node_id, p}] in realized draw order — the conditional p at each draw
    set_logprob: float = 0.0       # log P(realized set | c) under logging policy
    temperature: float = 1.0

    # outcome (§9.1)
    outcome: int = 0               # 1 = flag verified, 0 otherwise
    non_exploit_capture: bool = False   # reward-hacking guard (§11)
    steps: int = 0
    tokens: int = 0
    total_commands: int = 0        # executed shell commands (effort signal, D1; real solver fills it)
    wall_clock_s: float = 0.0      # SECONDARY only — never a bandit signal (C-5)
    time_to_flag_steps: Optional[int] = None

    # reward (§9.1)
    R_raw: float = 0.0             # outcome term o
    R_efficiency_e: float = 0.0    # e = -z_steps
    R_final: float = 0.0           # R = o*(1+λe)
    baseline_b: float = 0.0

    # provenance / reproducibility (§3.3, §12)
    seed: int = 0
    solver_model: str = ""
    embedder_model: str = ""
    reranker_model: str = ""
    container_image: str = ""
    tool_versions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepRecord:
    """One agent step within a run (§12 StepRecord)."""
    run_id: str
    step_idx: int
    action_type: str = ""          # run_commands | query_memory | submit_flag | give_up
    tool: str = ""
    command: str = ""
    observation_hash: str = ""
    would_have_retrieved: List[str] = field(default_factory=list)  # counterfactual (§8)
    stagnation_flag: bool = False
    ts: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CitationRecord:
    """Whether a sampled node was action-grounded-cited in a run (§12, §9.3)."""
    run_id: str
    node_id: str
    cited: bool = False
    match_evidence: str = ""       # the command/keyword that grounded the citation
    confidence: str = "deterministic"   # "deterministic" | "llm"
    sampled: bool = True           # was it in the M set (vs only in pool)?

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UpdateRecord:
    """One bandit Q-update for one credited node (§12, §9.3)."""
    run_id: str
    context_node_id: str
    node_id: str
    pre_Q: float = 0.0
    post_Q: float = 0.0
    pre_n: float = 0.0
    post_n: float = 0.0
    distance_scale: float = 1.0
    credit_eligible: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WriteBackRecord:
    """One node written back to the graph from a trajectory (§12, §9.4)."""
    run_id: str
    new_node_id: str
    type: str = ""
    derived_from_run: str = ""
    failure_derived: bool = False
    applicability: Dict[str, Any] = field(default_factory=dict)
    init_weight: float = 0.0
    tier: str = "provisional"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphSnapshot:
    """A pointer to a serialized graph version, per experiment (§12)."""
    experiment_id: str
    version: str
    node_count: int = 0
    edge_count: int = 0
    path: str = ""
    ts: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
