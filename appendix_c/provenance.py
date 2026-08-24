#!/usr/bin/env python3
"""appendix_c/provenance.py — TASK_BRIEF §4.2 companion records.

Every results artifact under `appendix_c/results/` gets a sibling
`<name>.provenance.json` recording what produced it: run id, config hash, seed set,
model, timestamp, and the producing script. Written by one helper so no table can
quietly ship without one.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=str(Path(__file__).resolve().parent.parent), timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                             cwd=str(Path(__file__).resolve().parent.parent), timeout=10)
        return bool(out.stdout.strip())
    except Exception:
        return False


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_result(name: str, payload: Any, *, script: str, seeds: Sequence[int],
                 config_hash: str = "", model: str = "none (CPU-only, deterministic)",
                 extra: Optional[Dict] = None, results_dir: Optional[Path] = None) -> Path:
    """Write `results/<name>.json` plus `results/<name>.provenance.json`; return the
    payload path."""
    d = Path(results_dir) if results_dir else RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2, default=str))

    prov = {
        "artifact": out.name,
        "artifact_sha256": file_sha256(out),
        "run_id": uuid.uuid4().hex[:12],
        "produced_by": script,
        "argv": sys.argv,
        "seeds": list(seeds),
        "config_hash": config_hash,
        "model": model,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if extra:
        prov["extra"] = extra
    (d / f"{name}.provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    return out
