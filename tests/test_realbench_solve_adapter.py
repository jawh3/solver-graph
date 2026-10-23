"""Cybench solve adapter + runner: plumbing without Docker or a model (fake session, fake solver, synthetic task)."""
import json
from pathlib import Path

import pytest

from realbench import cybench as cb
from realbench.solve_adapter import solve_cybench_task
from realbench import run_cybench
from tests.fakes_realbench import FakeSession, fake_solve
from tests.test_realbench_cybench import _mk_task


def test_adapter_end_to_end_with_fakes(tmp_path):
    t = _mk_task(tmp_path, flag="HTB{s0me_fl4g}", handout_text="HTB{s0me_fl4g}")
    rec = solve_cybench_task(t, tmp_path / "work", docker=None, session_factory=FakeSession, solve_impl=fake_solve, llm_seed=7)
    assert rec.error is None, rec.error
    assert rec.solved and rec.flag_exact and rec.flag_returned == "HTB{s0me_fl4g}"
    s = FakeSession.instances[-1]
    assert s.started is False and s.stopped                      # session stopped after the solve
    assert s.seen and s.seen[0][0] == "cat handout.txt"           # the command ran through the backend
    assert Path(s.seen[0][1]).name == "agent" and s.workdir == s.cwd == s.seen[0][1]   # only the agent dir is mounted; cwd = agent dir
    assert rec.display_name.startswith("task-") and rec.handout_changes == [] and rec.llm_seed == 7
    assert not (tmp_path / "work").exists() or not any((tmp_path / "work").iterdir())   # torn down


def test_adapter_reports_wrong_flag_and_brace_content(tmp_path):
    t = _mk_task(tmp_path, flag="HTB{right}", handout_text="b'HTB{right}'")   # brace content matches, exact does not
    rec = solve_cybench_task(t, tmp_path / "work", docker=None, session_factory=FakeSession, solve_impl=fake_solve)
    assert rec.error is None and not rec.solved and rec.flag_returned is None   # the checker rejected it -> no flag returned
    t2 = _mk_task(tmp_path / "b", flag="HTB{right}", handout_text="HTB{wrong}")
    rec2 = solve_cybench_task(t2, tmp_path / "work2", docker=None, session_factory=FakeSession, solve_impl=fake_solve)
    assert not rec2.solved and rec2.flag_returned is None


def test_adapter_survives_a_crashing_solver(tmp_path):
    t = _mk_task(tmp_path)
    def boom(*a, **k): raise RuntimeError("model down")
    rec = solve_cybench_task(t, tmp_path / "work", docker=None, session_factory=FakeSession, solve_impl=boom)
    assert rec.error.startswith("RuntimeError") and not rec.solved and FakeSession.instances[-1].stopped
    assert not (tmp_path / "work" ).exists() or not any((tmp_path / "work").iterdir())


def test_runner_mock_mode_writes_records_and_summary(tmp_path, monkeypatch):
    t = _mk_task(tmp_path, flag="HTB{s0me_fl4g}", handout_text="HTB{s0me_fl4g}")
    (tmp_path / "task_list.txt").write_text(t.id + "\n")          # the runner enumerates from the root's task list
    root = tmp_path; monkeypatch.setenv("MODEL_NAME", "fake-model")
    out = tmp_path / "res" / "bake.jsonl"
    rc = run_cybench.main(["--root", str(root), "--cyber-zero-root", "", "--tasks", "all", "--trials", "2", "--parallel", "2",
                          "--mock-solve", "--workroot", str(tmp_path / "w"), "--out", str(out)])
    assert rc == 0
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == 2 and all(r["solved"] for r in recs) and {r["extra"]["trial"] for r in recs} == {0, 1}
    summ = json.loads(out.with_suffix(".summary.json").read_text())
    assert summ["runs"] == 2 and summ["solves"] == 2 and summ["tasks_solved_at_least_once"] == 1 and summ["model"] == "fake-model"
    assert run_cybench.wilson(1, 4)[0] == 0.25 and run_cybench.wilson(0, 0) == (0.0, 0.0, 0.0)


def test_select_tasks_specs(tmp_path):
    t = _mk_task(tmp_path)
    assert run_cybench.select_tasks([t], "file-only") == [t] and run_cybench.select_tasks([t], "service") == []
    assert run_cybench.select_tasks([t], t.display_name) == [t] and run_cybench.select_tasks([t], "nope") == []


def test_session_kwargs_reach_the_session_and_root_mode_reowns_files(tmp_path):
    from realbench.solve_adapter import solve_cybench_task
    t = _mk_task(tmp_path, flag="HTB{s0me_fl4g}", handout_text="HTB{s0me_fl4g}")
    calls = []
    class FakeDocker(cb.DockerOps):
        def ensure_network(self, name): calls.append(("net", name))
        def chown_tree(self, path, uid, gid): calls.append(("chown", str(path), uid, gid))
    class RecordingSession(FakeSession):
        def __init__(self, workdir, **kw):
            super().__init__(workdir, **{k: v for k, v in kw.items() if k in ("image", "network", "cwd", "command_timeout_s")})
            self.kw = kw
    rec = solve_cybench_task(t, tmp_path / "work", docker=FakeDocker(), session_factory=RecordingSession, solve_impl=fake_solve,
                             command_timeout_s=300, session_kwargs={"memory": "8g", "cpus": "4", "user": "0:0"})
    assert rec.error is None and rec.solved
    s = FakeSession.instances[-1]
    assert s.kw["memory"] == "8g" and s.kw["cpus"] == "4" and s.kw["user"] == "0:0" and s.command_timeout_s == 300
    assert any(c[0] == "chown" and c[1].endswith("/agent") for c in calls)      # root-owned outputs re-owned before teardown


def test_runner_aborts_a_model_whose_first_solves_never_parse(tmp_path, monkeypatch):
    """2026-09-16: a broken quantized build emitted gibberish for 2.5 h; the runner now stops after N action-less solves."""
    import realbench.run_cybench as rc
    t = _mk_task(tmp_path, flag="HTB{s0me_fl4g}", handout_text="HTB{s0me_fl4g}")
    (tmp_path / "task_list.txt").write_text(t.id + "\n"); monkeypatch.setenv("MODEL_NAME", "broken")
    def gibberish_solve(challenge, **kw):
        return {"flag": None, "trace": [{"round": i + 1, "action": "parse_error", "commands": []} for i in range(3)], "rounds_used": 3, "gave_up_reason": "rounds_exhausted", "total_commands": 0}
    monkeypatch.setattr("tests.fakes_realbench.fake_solve", gibberish_solve)
    out = tmp_path / "r" / "b.jsonl"
    rc.main(["--root", str(tmp_path), "--cyber-zero-root", "", "--tasks", "all", "--trials", "6", "--parallel", "1", "--mock-solve",
             "--workroot", str(tmp_path / "w"), "--out", str(out), "--degenerate-after", "2"])
    summ = json.loads(out.with_suffix(".summary.json").read_text())
    assert summ["degenerate"] is True and summ["runs"] == 2


def test_select_models_rule(tmp_path):
    import json as _json
    from realbench.select_models import select
    def summ(model, rate, lo, errors=0, degenerate=False):
        d = tmp_path / model.replace("/", "_"); d.mkdir()
        (d / "bakeoff.summary.json").write_text(_json.dumps({"model": model, "solve_rate": rate, "wilson95": [lo, min(1, rate + 0.15)], "runs": 51, "errors": errors, "mean_wall_s": 100, "degenerate": degenerate}))
    summ("Qwen/Qwen3.8-27B-FP8", 0.35, 0.24); summ("Qwen/Qwen3.6-27B-FP8", 0.29, 0.19); summ("Qwen/Qwen2.5-32B-Instruct-AWQ", 0.08, 0.03)
    summ("openai/gpt-oss-20b", 0.18, 0.10); summ("cyankiwi/GLM-4.7-Flash-AWQ-4bit", 0.14, 0.07); summ("google/gemma-4-31B-it", 0.16, 0.09)
    summ("unsloth/Meta-Llama-3.1-8B-Instruct", 0.06, 0.02); summ("RedHatAI/Qwen3.8-27B-INT4", 0.0, 0.0, degenerate=True)
    res = select(str(tmp_path))
    assert res["chosen"] == ["Qwen/Qwen3.8-27B-FP8", "Qwen/Qwen2.5-32B-Instruct-AWQ", "openai/gpt-oss-20b", "google/gemma-4-31B-it"]
    assert res["flagged_below_bar"] == [] and all(r["model"] != "RedHatAI/Qwen3.8-27B-INT4" for r in res["rows"])
    # two non-Qwen of the SAME origin never both get in; fewer than two qualifiers -> best two anyway, flagged
    (tmp_path / "google_gemma-4-31B-it" / "bakeoff.summary.json").write_text(_json.dumps({"model": "google/gemma-4-31B-it", "solve_rate": 0.05, "wilson95": [0.02, 0.2], "runs": 51, "errors": 0, "mean_wall_s": 100}))
    (tmp_path / "cyankiwi_GLM-4.7-Flash-AWQ-4bit" / "bakeoff.summary.json").write_text(_json.dumps({"model": "cyankiwi/GLM-4.7-Flash-AWQ-4bit", "solve_rate": 0.05, "wilson95": [0.02, 0.2], "runs": 51, "errors": 0, "mean_wall_s": 100}))
    res = select(str(tmp_path))
    assert res["chosen"][2] == "openai/gpt-oss-20b" and len(res["chosen"]) == 4 and res["flagged_below_bar"] == [res["chosen"][3]]


def test_retry_errors_reruns_only_errored_pairs_and_keeps_clean_records(tmp_path, monkeypatch):
    import realbench.run_cybench as rc
    t = _mk_task(tmp_path, flag="HTB{s0me_fl4g}", handout_text="HTB{s0me_fl4g}")
    (tmp_path / "task_list.txt").write_text(t.id + "\n"); monkeypatch.setenv("MODEL_NAME", "m")
    out = tmp_path / "r" / "b.jsonl"; out.parent.mkdir()
    clean = {"task_id": t.id, "display_name": t.display_name, "category": "misc", "needs_service": False, "solved": True, "flag_returned": "HTB{s0me_fl4g}",
             "flag_exact": True, "flag_brace_content": True, "rounds_used": 1, "total_commands": 1, "gave_up_reason": None, "error": None, "wall_s": 1.0,
             "solve_wall_s": 1.0, "commands_run": 1, "command_timeouts": 0, "handout_changes": [], "provision_warnings": [], "blocked_init_commands": [],
             "host_aliases": {}, "service_health": {}, "name_leaks": [], "flag_leaks": [], "container": "", "llm_seed": 1000, "trace": None, "extra": {"trial": 0}}
    bad = dict(clean, solved=False, flag_returned=None, flag_exact=False, error="ProvisionError: boom", llm_seed=1001, extra={"trial": 1})
    out.write_text(json.dumps(clean) + "\n" + json.dumps(bad) + "\n")
    rc.main(["--root", str(tmp_path), "--cyber-zero-root", "", "--tasks", "all", "--trials", "3", "--parallel", "1", "--mock-solve",
             "--workroot", str(tmp_path / "w"), "--out", str(out), "--retry-errors"])
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    # trial 0 kept as is, trial 1 (errored) re-run, trial 2 (never recorded: a killed partial run) run
    assert len(recs) == 3 and {r["extra"]["trial"] for r in recs} == {0, 1, 2} and all(not r["error"] for r in recs)
    assert [r for r in recs if r["extra"]["trial"] == 0][0]["wall_s"] == 1.0


def test_jobs_are_trial_major(tmp_path, monkeypatch):
    import realbench.run_cybench as rc
    t = _mk_task(tmp_path, flag="HTB{s0me_fl4g}", handout_text="HTB{s0me_fl4g}")
    (tmp_path / "task_list.txt").write_text(t.id + "\n"); monkeypatch.setenv("MODEL_NAME", "m")
    out = tmp_path / "r" / "b.jsonl"
    rc.main(["--root", str(tmp_path), "--cyber-zero-root", "", "--tasks", "all", "--trials", "3", "--parallel", "1", "--mock-solve", "--workroot", str(tmp_path / "w"), "--out", str(out)])
    assert [json.loads(l)["extra"]["trial"] for l in out.read_text().splitlines()] == [0, 1, 2]


def test_service_tasks_are_serialized_by_the_gate(tmp_path, monkeypatch):
    """Service tasks publish fixed host ports and container names; the runner must never run two at once."""
    import threading, time as _time
    import realbench.run_cybench as rc
    t = _mk_task(tmp_path, flag="HTB{s0me_fl4g}", handout_text="HTB{s0me_fl4g}", target_host="svc:1337")
    (tmp_path / "task_list.txt").write_text(t.id + "\n"); monkeypatch.setenv("MODEL_NAME", "m")
    assert t.needs_service
    inflight = {"n": 0, "max": 0}; lock = threading.Lock()
    def slow_solve(task, workroot, **kw):
        from realbench.solve_adapter import SolveRecord
        with lock: inflight["n"] += 1; inflight["max"] = max(inflight["max"], inflight["n"])
        _time.sleep(0.2)
        with lock: inflight["n"] -= 1
        return SolveRecord(task_id=task.id, display_name=task.display_name, category=task.category, needs_service=True, extra=dict(kw.get("extra") or {}))
    monkeypatch.setattr(rc, "solve_cybench_task", slow_solve)
    out = tmp_path / "r" / "b.jsonl"
    rc.main(["--root", str(tmp_path), "--cyber-zero-root", "", "--tasks", "all", "--trials", "4", "--parallel", "4", "--mock-solve", "--workroot", str(tmp_path / "w"), "--out", str(out)])
    assert inflight["max"] == 1
