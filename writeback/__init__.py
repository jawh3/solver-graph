"""
writeback/ — gated lifelong write-back (spec §9.4, C-8/C7).

The LLM abstraction step is where false lessons are BORN, so write-back is gated:
new nodes enter `provisional` (weight-capped, flagged in-context) and are promoted
to `trusted` only after demonstrating generalizable utility (cited in ≥k_promote
successful runs across ≥c_promote distinct challenges). This is the birth-defect
lever (§11). Failures are decomposed too (tagged), per FORGE (2605.16233).
"""

from .writeback import (
    WriteBackManager,
    TrajectoryDecomposer,
    MockTrajectoryDecomposer,
    PromotionTracker,
)

__all__ = [
    "WriteBackManager",
    "TrajectoryDecomposer",
    "MockTrajectoryDecomposer",
    "PromotionTracker",
]
