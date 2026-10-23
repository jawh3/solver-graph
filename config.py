"""
config.py - Single source of truth for swept/ablated parameters.

This mirrors CTF_SOLVER_ARCHITECTURE_v2.md §15 ("Configuration & swept
parameters"). Every default here is the spec default; the *range* of each
`[SWEEP]` knob and the choices of each `[ABLATION]` knob are recorded alongside,
so the offline sweep runner (C8, `ope/sweep.py`) and the eval harness (C4) can
enumerate them without re-deriving them from prose.

RULE (BUILD_CHECKLIST global rules): do not silently change a `[SWEEP]`/
`[ABLATION]` default. Construct a `SolverConfig`, override the field, and record
the deviation in OPEN_DECISIONS.md. `policy_config_hash` (see `.hash()`) makes
every logged run attributable to the exact config that produced it (§12).

Paper-citation map for individual knobs lives in docs/DESIGN_CITATIONS.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Ablation enums (string-valued so they serialize cleanly into the config hash)
# ---------------------------------------------------------------------------

class ContextResolution(str, Enum):
    """How the query maps to a bandit context node (spec §6, C-1)."""
    CATEGORY = "category"          # ~6 CTF categories
    CLUSTER = "cluster"            # k-means over Challenge embeddings (default)
    PARAMETRIC = "parametric"      # LinUCB on continuous query embedding (§6.3)


class InitMode(str, Enum):
    """Cold-start init for bandit cells / new nodes (spec §6.2, §9.5)."""
    ZERO = "zero"
    SEED_WARM = "seed_warm"        # inherit a quality prior; log alignment covariate
    OPTIMISTIC = "optimistic"      # Q0 > 0 forces exploration (default)


class DistanceScaling(str, Enum):
    """Embedding-distance scaling of credit (spec §9.3, A-04)."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"      # blame near on failure, reward far on success
    NONE = "none"                  # F41: no distance scaling (scale = 1; needed for an exact running mean)


class WritebackMode(str, Enum):
    """Which trajectories get decomposed into new nodes (spec §9.4, A-03)."""
    OFF = "off"
    SUCCESSFUL = "successful"
    BOTH = "both"                  # default - FORGE supports failure decomposition


class CreditSignal(str, Enum):
    """WHAT the bandit credit is computed from (spec §9.3 + C9 causal credit).

    OUTCOME: credit = the run reward R (cheap, but confounded by recovery masking -
             a cited lesson on a recovered success is credited as if it helped).
    CAUSAL:  credit = the LEAVE-ONE-OUT marginal value - re-roll the solve without
             the lesson (shared seed) and credit the marginal flag difference. This
             is uncfounded: a poison the agent recovers from yields ~0 marginal and
             is driven down. Expensive (extra re-rolls), but a verifiable-reward
             domain can afford it (re-roll + check the flag is oracle-free).
    HYBRID:  outcome for the first `n` exposures (cheap warm-up), then causal for
             nodes that become suspicious (high Q + cited) - bounds cost.
    """
    OUTCOME = "outcome"
    CAUSAL = "causal"
    HYBRID = "hybrid"
    OBSERVATIONAL = "observational"   # batch IPW from logged sampling; no per-run update, no re-rolls


class CreditMode(str, Enum):
    """Credit assignment scope (spec §9.3, C-4 - the recovery-masking lever)."""
    CITED_ONLY = "cited_only"      # default - only action-grounded cited nodes
    ALL_M = "all_m"                # uniform credit (recovery-masking baseline)
    # F40 (mock ladder 2026-09-05): credit ONLY the cited nodes, but give every other sampled node an
    # EXPOSURE update with zero advantage so its cold-start similarity prior decays (spec §6.2: "keep
    # similarity as a DECAYING prior"). Under plain cited_only an ignored lesson kept its prior forever.
    CITED_ONLY_DECAY = "cited_only_decay"


class RefreshMode(str, Enum):
    """Context refresh policy (spec §8 - future; static is the only built one)."""
    STATIC = "static"
    PERIODIC_L = "periodic_L"
    HYBRID_K = "hybrid_K"


# ---------------------------------------------------------------------------
# The config object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolverConfig:
    """All tunables in one place. Frozen so a run's config can't mutate mid-run.

    Field defaults == spec §15 defaults. `sweep_range` / `ablation_choices`
    document the legal space for the sweep runner; they are class-level metadata,
    not per-instance state.
    """

    # --- Retrieval pipeline (§7) ---
    N: int = 200                   # [SWEEP 100-500] recall pool; NOT grown to fight lures (C-3)
    R: int = 50                    # [SWEEP 20-100] cross-encoder rerank keep
    lambda_mmr: float = 0.5        # [SWEEP 0.3-0.8] MMR diversity/relevance trade-off
    M: int = 8                     # [SWEEP 4-16] nodes entering context per run

    # --- Exploration / sampling (§6.2) ---
    tau: float = 1.0               # [SWEEP 0.5-2.0] softmax temperature
    tau_min: float = 0.0           # exploration floor; 0 = none (F3: 0.3 made deployed tau impossible)
    epsilon_explore: float = 0.0   # F3: fraction of runs drawn from the exploration stream (mixture propensity)
    tau_explore: float = 1.0       # F3: temperature of the exploration stream (design="softmax" only)
    # review 2026-09-04: the exploration stream must be a FIXED randomized design (history-free) -
    #   "softmax"    legacy: softmax(score/tau_explore) over bandit-blended scores (adaptive)
    #   "uniform"    uniform M-subset of the pool; pi = M/|pool| exactly
    #   "stratified" each ON-TARGET lesson (same `_family` as the challenge) included w.p. 0.5,
    #                rest uniform; pi(on-target) = 0.5 exactly -> per-lesson identification power
    exploration_design: str = "stratified"
    kappa: float = 5.0             # [SWEEP 1-20] sim-prior strength (pseudo-count, C-2)

    # --- Bandit update (§9.1, §9.3) ---
    alpha: float = 0.1             # [SWEEP 0.02-0.3] bandit learning rate
    lam: float = 0.5               # [SWEEP 0-1.0] efficiency weight λ in R=o(1+λe)
    failure_reward: float = 0.0    # [ABLATION 0 | small negative] reward on failure
    use_baseline_b: bool = True    # center target by running mean reward in context c

    # --- Standardization / partial pooling (§9.2) ---
    tau0: float = 5.0              # [SWEEP] partial-pooling pseudo-count toward global

    # --- Context / init / scaling ablations (§6, §9) ---
    n_clusters: int = 32           # [SWEEP 20-50] k for cluster context nodes
    context_resolution: ContextResolution = ContextResolution.CLUSTER
    init_mode: InitMode = InitMode.OPTIMISTIC
    optimistic_q0: float = 0.5     # [SWEEP] optimistic init value
    distance_scaling: DistanceScaling = DistanceScaling.ASYMMETRIC

    # --- Write-back gate (§9.4) ---
    writeback_mode: WritebackMode = WritebackMode.BOTH
    credit_mode: CreditMode = CreditMode.CITED_ONLY
    citation_mode: str = "distinctive"   # F2: "distinctive" | "any" (legacy matcher)
    prior_source: str = "cosine"         # review 2026-09-05 F-2: "cosine" (legacy) | "rerank" | "none" (F41: prior = 0, Q-only)
    alpha_schedule: str = "ema"          # F41: "ema" (legacy, step = alpha) | "mean" (step = 1/(n+1): exact running mean)
    # C9 causal credit (the fix for recovery masking): credit signal + re-roll budget
    credit_signal: CreditSignal = CreditSignal.OUTCOME
    causal_reroll_seeds: int = 1          # re-rolls per cited node to estimate marginal value
    causal_parallel_workers: int = 6      # 2026-09-07: re-rolls of one node run concurrently (1 = sequential)
    causal_warmup_n: float = 2            # HYBRID: use outcome credit until n exposures, then causal
    causal_q_gate: float = 0.3            # HYBRID: only causal-credit nodes whose Q exceeds this
    k_promote: int = 3             # [SWEEP 2-5] distinct successful cited runs to promote
    c_promote: int = 2             # [SWEEP 1-3] distinct challenges to promote
    provisional_weight_cap: float = 0.3   # provisional nodes capped in score (§9.4)

    # --- Graph hygiene: retraction / decay (§10, built in C9) ---
    theta_retract: float = -0.15  # [SWEEP] Q below which a node is a retraction candidate
    k_retract: int = 5            # [SWEEP] exposures below θ before demotion
    cited_failure_retract_rate: float = 0.6  # cited-failure rate that triggers retraction

    # --- Refresh (future; instrument only - §8) ---
    refresh_mode: RefreshMode = RefreshMode.STATIC

    # --- Solver run limits (§3, C3) ---
    max_steps: int = 15            # agent steps/tool-rounds before timeout

    # --- Determinism (§3.3) ---
    seed: int = 42
    embedding_dim: int = 1536      # retrieval embedder dim (mock default; real = model dim)

    # -- documentation-only metadata (legal sweep space; not part of the hash) --
    sweep_range: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "N": (100, 500), "R": (20, 100), "lambda_mmr": (0.3, 0.8), "M": (4, 16),
        "tau": (0.5, 2.0), "tau_min": (0.1, 0.5), "kappa": (1, 20),
        "alpha": (0.02, 0.3), "lam": (0.0, 1.0), "tau0": (1, 20),
        "n_clusters": (20, 50), "k_promote": (2, 5), "c_promote": (1, 3),
    }, compare=False, repr=False)

    ablation_choices: Dict[str, List[str]] = field(default_factory=lambda: {
        "context_resolution": [e.value for e in ContextResolution],
        "init_mode": [e.value for e in InitMode],
        "distance_scaling": [e.value for e in DistanceScaling],
        "writeback_mode": [e.value for e in WritebackMode],
        "credit_mode": [e.value for e in CreditMode],
        "refresh_mode": [e.value for e in RefreshMode],
        "failure_reward": ["0", "small_negative"],
    }, compare=False, repr=False)

    # ------------------------------------------------------------------
    def policy_dict(self) -> Dict:
        """The subset of fields that define the *policy* (excludes doc metadata).

        Used both for `policy_config_hash` (§12 RunRecord) and for human-readable
        logging of exactly which knobs were active.
        """
        d = asdict(self)
        d.pop("sweep_range", None)
        d.pop("ablation_choices", None)
        # Enums -> their string values for stable hashing/serialization
        for k, v in list(d.items()):
            if isinstance(v, Enum):
                d[k] = v.value
        return d

    def hash(self) -> str:
        """Stable short hash of the policy config (§12 `policy_config_hash`)."""
        blob = json.dumps(self.policy_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:12]

    def with_overrides(self, **kwargs) -> "SolverConfig":
        """Return a copy with fields overridden (frozen dataclass-friendly).

        Any deviation from a default should be paired with an OPEN_DECISIONS.md
        line; this method is the only sanctioned way to change a knob.
        """
        base = self.policy_dict()
        base.update(kwargs)
        # Re-coerce enums from strings if needed
        coercions = {
            "context_resolution": ContextResolution,
            "init_mode": InitMode,
            "distance_scaling": DistanceScaling,
            "writeback_mode": WritebackMode,
            "credit_mode": CreditMode,
            "credit_signal": CreditSignal,
            "refresh_mode": RefreshMode,
        }
        for key, enum_cls in coercions.items():
            if key in base and not isinstance(base[key], enum_cls):
                base[key] = enum_cls(base[key])
        return SolverConfig(**base)


# A module-level default instance for convenience.
DEFAULT_CONFIG = SolverConfig()
