"""Shared test setup: the runner's cross-process service locks go to a temp dir, never results/_work/locks."""
import pytest


@pytest.fixture(autouse=True)
def _realbench_lock_dir(tmp_path, monkeypatch):
    from realbench import run_cybench
    monkeypatch.setattr(run_cybench, "LOCK_DIR", tmp_path / "locks")
