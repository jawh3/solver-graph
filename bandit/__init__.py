"""
bandit/ - the contextual bandit over (context_node -> candidate_node) edges.

This is the core change C-1 in CTF_SOLVER_ARCHITECTURE_v2.md: the bandit context
is a coarse Category/Cluster node, NOT a per-challenge analog (which is too
sparse to learn - §6.1). The tabular value bandit is the default; LinUCB is the
optional parametric variant (§6.3).
"""

from .bandit import TabularBandit, BanditCell, sample_efficiency_budget
from .linucb import LinUCBBandit

__all__ = ["TabularBandit", "BanditCell", "sample_efficiency_budget", "LinUCBBandit"]
