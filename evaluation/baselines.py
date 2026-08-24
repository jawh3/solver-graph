"""
evaluation/baselines.py — the §14.2 baseline matrix as concrete SystemSpecs.

Each baseline isolates one mechanism; the eval reports paired lift over B0 (§14.3).
The ordering is a strict ablation ladder B0→B5, plus B6 (gating ablation) and BC
(external competitor). Build order (§16) requires B0–B3 to exist BEFORE the bandit
(B4) and write-back (B5) are evaluated — this module defines all of them but the
eval runner only needs B0–B3 to satisfy C4.
"""

from __future__ import annotations

from typing import Dict, List

from pipeline.system_spec import SystemSpec

BASELINES: Dict[str, SystemSpec] = {
    "B0": SystemSpec("B0", use_graph=False,
                     description="Zero-shot, no graph — the contamination floor / lift denominator."),
    "B1": SystemSpec("B1", use_rerank=False, use_mmr=False, use_bandit=False,
                     description="Seed graph, similarity-only static retrieval."),
    "B2": SystemSpec("B2", use_rerank=True, use_mmr=True, use_bandit=False,
                     description="B1 + cross-encoder rerank + MMR diversity."),
    "B3": SystemSpec("B3", use_rerank=True, use_mmr=True, use_bandit=False, episodic_only=True,
                     description="Raw/episodic-only (no decomposed lessons) — the Faulty-Memory null."),
    "B4": SystemSpec("B4", use_rerank=True, use_mmr=True, use_bandit=True,
                     description="B2 + contextual bandit (no write-back)."),
    "B5": SystemSpec("B5", use_rerank=True, use_mmr=True, use_bandit=True,
                     use_writeback=True, writeback_gated=True,
                     description="Full system: B4 + gated write-back."),
    "B6": SystemSpec("B6", use_rerank=True, use_mmr=True, use_bandit=True,
                     use_writeback=True, writeback_gated=False,
                     description="B5 with write-back UNGATED (gating ablation)."),
    "BC": SystemSpec("BC", use_rerank=True, use_mmr=True, use_bandit=True,
                     use_writeback=True, writeback_gated=True, forge_competitor=True,
                     description="FORGE-style champion-broadcast competitor (C9)."),
}

# The set required to satisfy C4 (before bandit/write-back exist in eval).
C4_REQUIRED = ["B0", "B1", "B2", "B3"]
FULL_LADDER = ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]


def get_baselines(ids: List[str]) -> List[SystemSpec]:
    return [BASELINES[i] for i in ids]
