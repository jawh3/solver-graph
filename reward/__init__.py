"""
reward/ - step-count reward + standardization + difficulty grading (spec §9.1-9.2, C5).

Primary efficiency signal is STEP COUNT, not wall-clock (C-5): wall-clock on a
shared GPU is dominated by queueing and is irreproducible. The reward
R = o*(1+λe) with e = -z_steps is the latent-poison lever (§11) - a slow success
earns less than a fast one.
"""

from .reward import RewardStandardizer, compute_reward, RewardBreakdown
from .grader import DifficultyGrader, MockDifficultyGrader, GraderCalibration

__all__ = [
    "RewardStandardizer",
    "compute_reward",
    "RewardBreakdown",
    "DifficultyGrader",
    "MockDifficultyGrader",
    "GraderCalibration",
]
