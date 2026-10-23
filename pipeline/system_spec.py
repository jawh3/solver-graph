"""
pipeline/system_spec.py - a SystemSpec enacts one row of the §14.2 baseline matrix.

SolverConfig holds the *numeric* knobs (M, τ, κ, …). SystemSpec holds the
*structural* toggles that turn pipeline stages on/off, so a single Orchestrator
can run any baseline B0-B6/BC by flipping booleans. This separation keeps the
"what mechanism is on" question (research design) distinct from "what value is the
knob" (tuning).

Baseline matrix (spec §14.2):
  B0 zero-shot/no-graph        - use_graph=False
  B1 similarity-only static    - graph on, no rerank/MMR, no bandit, no writeback
  B2 + rerank + diversity      - adds rerank + MMR
  B3 raw/episodic-only         - retrieve raw chunks, no decomposition (the FaultyMemory null)
  B4 + contextual bandit       - adds bandit credit
  B5 + gated write-back        - full system
  B6 ungated write-back        - gating ablation
  BC FORGE-style competitor    - champion-broadcast (C9)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    use_graph: bool = True          # B0 turns this off
    use_rerank: bool = True         # B1 off; B2+ on
    use_mmr: bool = True            # B1 off; B2+ on
    use_bandit: bool = False        # B4+ on
    use_writeback: bool = False     # B5/B6 on
    writeback_gated: bool = True    # B6 sets False (ungated)
    episodic_only: bool = False     # B3 (no decomposition; raw retrieval)
    forge_competitor: bool = False  # BC (C9)
    deterministic: bool = False     # F3: top-M by rerank order, propensity 1/0 (the DEPLOYED static arm)
    force_on_target: str = ""       # review: "helpful" (oracle-static) | "poison" (worst-static): the
                                    # challenge family's lesson of that label is forced into the M-set
    description: str = ""
