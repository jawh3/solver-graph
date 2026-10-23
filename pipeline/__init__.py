"""
pipeline/ - the single-run orchestrator wiring C2-C7 together (spec §7-§9).

One `Orchestrator.run_one(...)` = the full loop for one challenge under one
SystemSpec: retrieve → assemble → solve → cite → reward → bandit-update →
write-back, emitting every §12 record. SystemSpec toggles enact baselines B0-B6.
"""

from .system_spec import SystemSpec
from .orchestrator import Orchestrator

__all__ = ["SystemSpec", "Orchestrator"]
