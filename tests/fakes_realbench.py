"""Fakes for the Cybench plumbing tests (no Docker, no model)."""
from pathlib import Path


class FakeSession:
    """Looks like solver.container_executor.ContainerSession; records the lifecycle; runs nothing."""
    instances = []

    def __init__(self, workdir, *, image=None, network=None, cwd=None, command_timeout_s=60.0, **kw):
        self.workdir = str(Path(workdir).resolve()); self.cwd = str(Path(cwd or workdir).resolve())
        self.image, self.network, self.command_timeout_s = image, network, command_timeout_s
        self.started = False; self.stopped = False; self.name = "fake-container"; self.commands_run = 0; self.timeouts = 0
        self.seen = []; FakeSession.instances.append(self)

    def start(self):
        self.started = True; return self

    def stop(self):
        self.stopped = True; self.started = False

    def matches(self, working_dir):
        wd = str(Path(working_dir).resolve()); return wd == self.workdir or wd.startswith(self.workdir + "/")

    def run(self, cmd, working_dir):
        self.commands_run += 1; self.seen.append((cmd, str(working_dir)))
        if cmd.startswith("cat "):
            p = Path(working_dir) / cmd.split()[1]
            return {"stdout": p.read_text() if p.exists() else "", "stderr": "", "returncode": 0, "error": None}
        return {"stdout": "fake\n", "stderr": "", "returncode": 0, "error": None}

    def available_tools(self, candidates=None):
        return ["python3"]


def fake_solve(challenge, **kw):
    """A 'solver' that reads the handout through the tool executor (so dispatch is exercised) and returns its content
    if it looks like a flag; asserts the neutrality contract."""
    from solver import tool_executor
    assert not any(str(k).startswith("_") for k in challenge), "private field leaked to the solver"
    assert challenge["difficulty"] == "unknown" and challenge["points"] == 0
    wd = kw["working_dir"]
    files = [f["text"] for f in challenge.get("local_files", [])]
    out = ""
    for name in files:
        r = tool_executor.run_command(f"cat {name}", wd); out += r["stdout"]
    flag = out.strip() if "{" in out else None
    if kw.get("flag_checker") and flag and not kw["flag_checker"](flag):
        flag = None
    return {"flag": flag, "trace": [{"round": 1, "commands": [f"cat {n}" for n in files]}], "rounds_used": 1,
            "gave_up_reason": None if flag else "rounds_exhausted", "total_commands": len(files)}
