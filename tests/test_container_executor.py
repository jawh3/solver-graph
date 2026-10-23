"""Container executor (Phase 3): argv shapes, backend dispatch, timeout mapping, tool probing - with a fake docker
runner; plus one real round trip that runs only when Docker and the slim image are available."""
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from solver import tool_executor as te
from solver.container_executor import ContainerSession, DEFAULT_IMAGE


class FakeRunner:
    """Records argv; answers `docker run -d` with a container id, `docker exec` from a script, `docker kill` ok."""
    def __init__(self, exec_results=None):
        self.calls, self.exec_results = [], list(exec_results or [])
        self.raise_timeout_next = False
    def __call__(self, argv, **kw):
        self.calls.append((list(argv), kw))
        if argv[1] == "run":
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        if argv[1] == "exec":
            if self.raise_timeout_next:
                self.raise_timeout_next = False
                raise subprocess.TimeoutExpired(argv, kw.get("timeout", 0))
            if self.exec_results:
                return self.exec_results.pop(0)
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_start_argv_mounts_workdir_at_same_path_and_pins_limits(tmp_path):
    s = ContainerSession(tmp_path, network="shared_net", cwd=tmp_path / "agent", runner=FakeRunner(), user="1000:1000")
    argv = s.start_argv()
    assert argv[:4] == ["docker", "run", "-d", "--rm"]
    assert f"{tmp_path.resolve()}:{tmp_path.resolve()}" in argv           # same absolute path inside and outside
    assert argv[argv.index("-w") + 1] == str((tmp_path / "agent").resolve())
    assert argv[argv.index("--network") + 1] == "shared_net"
    assert "--memory" in argv and "--pids-limit" in argv and "--cpus" in argv
    assert argv[-4:] == ["--entrypoint", "sleep", DEFAULT_IMAGE, "infinity"]     # bypasses any image entrypoint


def test_exec_argv_uses_container_side_timeout_and_working_dir(tmp_path):
    s = ContainerSession(tmp_path, runner=FakeRunner(), command_timeout_s=42)
    argv = s.exec_argv("cat data.txt | head", tmp_path / "agent")
    assert argv[:2] == ["docker", "exec"] and argv[argv.index("-w") + 1] == str((tmp_path / "agent").resolve())
    assert argv[argv.index("timeout"):argv.index("bash")] == ["timeout", "-s", "KILL", "42"]
    assert argv[-1].endswith("cat data.txt | head") and "ulimit -f" in argv[-1]


def test_matches_only_inside_the_mounted_tree(tmp_path):
    s = ContainerSession(tmp_path / "w", runner=FakeRunner())
    (tmp_path / "w" / "agent").mkdir(parents=True)
    assert s.matches(tmp_path / "w") and s.matches(tmp_path / "w" / "agent")
    assert not s.matches(tmp_path) and not s.matches(str(tmp_path / "w2"))


def test_run_command_dispatches_to_backend_and_fails_closed_outside_it(tmp_path):
    fake = FakeRunner(exec_results=[SimpleNamespace(returncode=0, stdout="from-container\n", stderr="")])
    s = ContainerSession(tmp_path, runner=fake).start()
    ub = te.use_backend(s)
    with ub:
        r = te.run_command("echo x", tmp_path)                      # inside the tree -> container
        assert r["stdout"] == "from-container\n" and r["returncode"] == 0 and r["error"] is None
        r2 = te.run_command("echo host", "/tmp")                    # outside -> BLOCKED (audit N1: never the host)
        assert r2["stdout"] == "" and r2["error"].startswith("BLOCKED")
    assert ub.stats == {"dispatched": 1, "rejected": 1} and s.commands_run == 1
    assert te.get_backend() is None
    assert te.run_command("echo host", "/tmp")["stdout"].strip() == "host"   # no backend, sandbox not required: host path unchanged
    assert fake.calls[0][0][1] == "run" and fake.calls[1][0][1] == "exec" and len(fake.calls) == 2
    s.stop(); assert fake.calls[-1][0][:2] == ["docker", "kill"]


def test_blocked_commands_never_reach_the_container(tmp_path):
    fake = FakeRunner(); s = ContainerSession(tmp_path, runner=fake).start()
    with te.use_backend(s):
        r = te.run_command("rm -rf /", tmp_path)
    assert r["error"].startswith("BLOCKED") and all(c[0][1] != "exec" for c in fake.calls)


def test_timeouts_are_reported_both_ways(tmp_path):
    fake = FakeRunner(exec_results=[SimpleNamespace(returncode=124, stdout="partial", stderr="")])
    s = ContainerSession(tmp_path, runner=fake, command_timeout_s=3).start()
    r = s.run("sleep 10", tmp_path)
    assert r["error"] == "TIMEOUT after 3s" and r["stdout"] == "partial" and s.timeouts == 1
    fake.raise_timeout_next = True
    r = s.run("sleep 10", tmp_path)
    assert "TIMEOUT" in r["error"] and r["returncode"] == -1 and s.timeouts == 2


def test_output_is_capped_and_hints_apply_through_the_shared_finalizer(tmp_path):
    fake = FakeRunner(exec_results=[SimpleNamespace(returncode=127, stdout="", stderr="bash: binwalk: command not found")])
    s = ContainerSession(tmp_path, runner=fake, max_output_bytes=10).start()
    with te.use_backend(s):
        r = te.run_command("binwalk x", tmp_path)
    assert r["returncode"] == 127 and "[HINT]" in r["stderr"]


def test_available_tools_probes_inside_the_container(tmp_path):
    fake = FakeRunner(exec_results=[SimpleNamespace(returncode=0, stdout="python3\nxxd\n", stderr="")])
    s = ContainerSession(tmp_path, runner=fake).start()
    with te.use_backend(s):
        assert te.get_available_tools() == ["python3", "xxd"]
    assert "command -v" in fake.calls[-1][0][-1]
    assert s.probe_runs == 1 and s.commands_run == 0               # the probe is not a solver command


def _docker_ready():
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "image", "inspect", DEFAULT_IMAGE], capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _docker_ready(), reason="docker or realbench-solver:slim not available")
def test_real_round_trip_same_path_mount_and_timeout(tmp_path):
    work = tmp_path / "w"; (work / "agent").mkdir(parents=True)
    (work / "agent" / "data.txt").write_text("hello from host\n")
    with ContainerSession(work, cwd=work / "agent", command_timeout_s=2) as s, te.use_backend(s):
        r = te.run_command("cat data.txt; pwd", work / "agent")
        assert "hello from host" in r["stdout"] and str((work / "agent").resolve()) in r["stdout"]
        r = te.run_command("echo written-inside > out.txt", work / "agent")
        assert r["returncode"] == 0 and (work / "agent" / "out.txt").read_text().strip() == "written-inside"
        r = te.run_command("sleep 30", work / "agent")
        assert r["error"] and r["error"].startswith("TIMEOUT")
        tools = te.get_available_tools()
        assert "python3" in tools and "xxd" in tools
    # the container is gone after the block
    out = subprocess.run(["docker", "ps", "-a", "--filter", f"name={s.name}", "--format", "{{.Names}}"], capture_output=True, text=True)
    assert s.name not in out.stdout


def test_real_docker_ops_passes_entrypoint_override(tmp_path):
    from realbench.cybench import RealDockerOps
    seen = {}
    def runner(argv, **kw):
        seen["argv"] = list(argv); return SimpleNamespace(returncode=0, stdout="", stderr="")
    RealDockerOps(runner=runner).run_in_container("img", ["solution.sh"], entrypoint="bash", name="n")
    a = seen["argv"]; assert a[a.index("--entrypoint") + 1] == "bash" and a[-2:] == ["img", "solution.sh"]
