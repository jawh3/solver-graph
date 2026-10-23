"""
instrumentation/ - paper-grade logging (CTF_SOLVER_ARCHITECTURE_v2.md §12).

Append-only JSONL, one file per record type per experiment run. These logs ARE
the paper data; propensities and action-grounded citations are
*non-reconstructable* and MUST be written at run time (BUILD_CHECKLIST C-7).
"""

from .records import (
    RunRecord,
    StepRecord,
    CitationRecord,
    UpdateRecord,
    WriteBackRecord,
    GraphSnapshot,
)
from .jsonl_logger import ExperimentLogger

__all__ = [
    "RunRecord",
    "StepRecord",
    "CitationRecord",
    "UpdateRecord",
    "WriteBackRecord",
    "GraphSnapshot",
    "ExperimentLogger",
]
