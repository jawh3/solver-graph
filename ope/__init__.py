"""
ope/ - off-policy evaluation from logged propensities (spec §13, C8).

Because per-draw propensities + set log-probs are logged (C-7, non-reconstructable),
alternative policies π′ (different M, τ, κ, scaling, init) can be scored on
ALREADY-collected runs without new GPU. This multiplies effective sample size -
essential given the scarce, coordinated vLLM. The sweep runner emits OPE estimates
with bootstrap CIs BEFORE any real-run sweep is committed.
"""

from .ope import ips_estimate, dr_estimate, OPEResult, recompute_policy_logprob
from .sweep import offline_sweep

__all__ = ["ips_estimate", "dr_estimate", "OPEResult", "recompute_policy_logprob",
           "offline_sweep"]
