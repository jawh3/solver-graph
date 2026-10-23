"""solver/container_executor.py - run the solver's shell commands INSIDE a per-solve Docker container.

Phase 3 (real benchmarks): one long-lived `realbench-solver:slim` container per solve, attached to the task's
docker network, with the solve's work directory bind-mounted at the SAME absolute path it has on the host, so
every path the model sees (the file listing shows basenames; the working directory is the agent dir) is valid
in both places. Each command goes through `docker exec` wrapped in a container-side `timeout -s KILL`, which
signals the whole process group, so nothing outlives the command; the host side waits a little longer and
caps the captured output like the host executor does. A session registered as the tool-executor backend
(`tool_executor.use_backend(session)`) receives every `run_command` whose working directory lies inside the
mounted tree; with no backend registered the host path is byte-identical to before (the reproducibility
anchor). Mock+real duality: `runner` is injectable (tests pass a fake `subprocess.run`).
"""
from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

DEFAULT_IMAGE = "realbench-solver:slim"
COMMON_TOOLS = ["exiftool", "binwalk", "zsteg", "steghide", "strings", "grep", "find", "nasm", "gcc", "g++", "nc",
                "ssh", "python3", "identify", "convert", "foremost", "tesseract", "jq", "curl", "wget", "nmap",
                "smbclient", "sqlmap", "nikto", "gobuster", "dirb", "xxd", "file", "base64", "openssl", "git", "unzip"]


class ContainerSession:
    """A running container that executes commands for one solve. Use as a context manager."""

    def __init__(self, workdir: str | Path, *, image: str = DEFAULT_IMAGE, network: Optional[str] = None,
                 cwd: Optional[str | Path] = None, name: Optional[str] = None, user: Optional[str] = None,
                 memory: str = "2g", pids_limit: int = 512, cpus: str = "2", env: Optional[Dict[str, str]] = None,
                 runner: Optional[Callable] = None, docker: str = "docker", command_timeout_s: float = 60.0,
                 max_output_bytes: int = 50_000, extra_mounts: Optional[Dict[str, str]] = None):
        self.workdir = str(Path(workdir).resolve())
        self.cwd = str(Path(cwd).resolve()) if cwd else self.workdir
        self.image, self.network = image, network
        self.name = name or f"realbench-agent-{uuid.uuid4().hex[:10]}"
        self.user = user or f"{os.getuid()}:{os.getgid()}"
        self.memory, self.pids_limit, self.cpus = memory, int(pids_limit), str(cpus)
        self.env = {"HOME": "/tmp", "LANG": "C.UTF-8", "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1",
                    "PIP_NO_CACHE_DIR": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", **(env or {})}
        self.extra_mounts = dict(extra_mounts or {})
        self.runner = runner or subprocess.run
        self.docker = docker
        self.command_timeout_s = float(command_timeout_s)
        self.max_output_bytes = int(max_output_bytes)
        self.started = False
        self.commands_run = 0       # solver commands executed in this container (the available-tools probe is not one)
        self.probe_runs = 0
        self.timeouts = 0

    # ------------------------------------------------------------------ argv builders (unit-testable)
    def start_argv(self) -> List[str]:
        argv = [self.docker, "run", "-d", "--rm", "--name", self.name, "--user", self.user,
                "--memory", self.memory, "--pids-limit", str(self.pids_limit), "--cpus", self.cpus,
                "-v", f"{self.workdir}:{self.workdir}", "-w", self.cwd]
        for h, c in self.extra_mounts.items():
            argv += ["-v", f"{h}:{c}"]
        if self.network:
            argv += ["--network", self.network]
        for k, v in self.env.items():
            argv += ["-e", f"{k}={v}"]
        # --entrypoint: the Cybench agent image's ENTRYPOINT starts a Docker daemon and never reaches its command without
        # --privileged; the slim image has no entrypoint, so the override is harmless there.
        return argv + ["--entrypoint", "sleep", self.image, "infinity"]

    def exec_argv(self, cmd: str, working_dir: str | Path, timeout_s: Optional[float] = None) -> List[str]:
        t = int(timeout_s if timeout_s is not None else self.command_timeout_s)
        wd = str(Path(working_dir).resolve())
        # `timeout` (coreutils) signals the whole process group of the command on expiry; `ulimit -f` caps any file
        # a command writes (the host executor caps it with RLIMIT_FSIZE) so a decompression bomb cannot fill the disk.
        inner = f"ulimit -f 409600 2>/dev/null; {cmd}"
        return [self.docker, "exec", "-w", wd, self.name, "timeout", "-s", "KILL", str(t), "bash", "-c", inner]

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "ContainerSession":
        res = self.runner(self.start_argv(), capture_output=True, text=True, errors="replace", timeout=180)
        if getattr(res, "returncode", 1) != 0:
            raise RuntimeError(f"container start failed (rc={res.returncode}): {(res.stderr or '')[-800:]}")
        self.started = True
        return self

    def stop(self) -> None:
        if not self.started:
            return
        try:
            self.runner([self.docker, "kill", self.name], capture_output=True, text=True, errors="replace", timeout=60)
        except Exception:
            pass
        self.started = False

    def __enter__(self) -> "ContainerSession":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------ backend protocol
    def matches(self, working_dir: str | Path) -> bool:
        wd = str(Path(working_dir).resolve())
        return wd == self.workdir or wd.startswith(self.workdir + os.sep)

    def run(self, cmd: str, working_dir: str | Path, *, probe: bool = False) -> dict:
        """Execute one (already validated) command; returns {stdout, stderr, returncode, error}.
        `probe=True` (the available-tools probe) is counted in `probe_runs`, not `commands_run`, so a solve can check
        commands_run == the commands the tool executor dispatched."""
        if not self.started:
            return {"stdout": "", "stderr": "", "returncode": -1, "error": "container session not started"}
        argv = self.exec_argv(cmd, working_dir)
        if probe:
            self.probe_runs += 1
        else:
            self.commands_run += 1
        try:
            res = self.runner(argv, capture_output=True, text=True, errors="replace",
                              stdin=subprocess.DEVNULL, timeout=self.command_timeout_s + 15)
        except subprocess.TimeoutExpired:
            self.timeouts += 1
            return {"stdout": "", "stderr": "", "returncode": -1,
                    "error": f"TIMEOUT after {int(self.command_timeout_s)}s (docker exec did not return)"}
        out, err = (res.stdout or ""), (res.stderr or "")
        if len(out) > self.max_output_bytes:
            out = out[:self.max_output_bytes]
        if len(err) > self.max_output_bytes:
            err = err[:self.max_output_bytes]
        error = None
        if res.returncode in (124, 137) and "timeout" not in err.lower():   # coreutils timeout -> 124; KILL -> 137
            error = f"TIMEOUT after {int(self.command_timeout_s)}s"
        elif res.returncode == 124:
            error = f"TIMEOUT after {int(self.command_timeout_s)}s"
        if error:
            self.timeouts += 1
        return {"stdout": out, "stderr": err, "returncode": int(res.returncode), "error": error}

    def available_tools(self, candidates: Iterable[str] = COMMON_TOOLS) -> List[str]:
        names = " ".join(candidates)
        probe = f"for t in {names}; do command -v \"$t\" >/dev/null 2>&1 && echo \"$t\"; done"
        r = self.run(probe, self.cwd, probe=True)
        return [t for t in (r.get("stdout") or "").split() if t]
