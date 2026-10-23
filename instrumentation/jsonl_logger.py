"""
instrumentation/jsonl_logger.py - append-only JSONL writer (§12, §4).

One file per record type per experiment run; never overwrite (logs are paper
data, §4). The logger is the only sanctioned way to persist records, so the
on-disk layout is consistent for the OPE/eval consumers.

Layout:
    {root}/{experiment_id}/runs.jsonl
    {root}/{experiment_id}/steps.jsonl
    {root}/{experiment_id}/citations.jsonl
    {root}/{experiment_id}/updates.jsonl
    {root}/{experiment_id}/writebacks.jsonl
    {root}/{experiment_id}/snapshots.jsonl

Thread-/process-safety: each `log_*` call opens, appends one line, and closes
(O_APPEND is atomic for small writes on POSIX), so multiple coordinated worker
processes on the GPU machine can append to the same files without a lock.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

from .records import (
    CitationRecord,
    GraphSnapshot,
    RunRecord,
    StepRecord,
    UpdateRecord,
    WriteBackRecord,
)

_FILES = {
    "run": "runs.jsonl",
    "step": "steps.jsonl",
    "citation": "citations.jsonl",
    "update": "updates.jsonl",
    "writeback": "writebacks.jsonl",
    "snapshot": "snapshots.jsonl",
}


class ExperimentLogger:
    """Append-only JSONL logger scoped to one experiment_id."""

    def __init__(self, experiment_id: str, root: str | Path = "results/logs"):
        self.experiment_id = experiment_id
        self.dir = Path(root) / experiment_id
        self.dir.mkdir(parents=True, exist_ok=True)

    # --- low-level ---
    def _append(self, kind: str, record: dict) -> None:
        path = self.dir / _FILES[kind]
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # --- typed writers ---
    def log_run(self, rec: RunRecord) -> None:
        self._append("run", rec.to_dict())

    def log_step(self, rec: StepRecord) -> None:
        self._append("step", rec.to_dict())

    def log_citation(self, rec: CitationRecord) -> None:
        self._append("citation", rec.to_dict())

    def log_update(self, rec: UpdateRecord) -> None:
        self._append("update", rec.to_dict())

    def log_writeback(self, rec: WriteBackRecord) -> None:
        self._append("writeback", rec.to_dict())

    def log_snapshot(self, rec: GraphSnapshot) -> None:
        self._append("snapshot", rec.to_dict())

    # --- readers (used by C4 eval + C8 OPE) ---
    def read(self, kind: str) -> List[dict]:
        return list(iter_jsonl(self.dir / _FILES[kind]))

    def path_for(self, kind: str) -> Path:
        return self.dir / _FILES[kind]


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield parsed records from a JSONL file (skips blank/corrupt lines)."""
    path = Path(path)
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_runs(experiment_id: str, root: str | Path = "results/logs") -> List[dict]:
    """Convenience: load all RunRecords for an experiment (for OPE/eval)."""
    return list(iter_jsonl(Path(root) / experiment_id / _FILES["run"]))
