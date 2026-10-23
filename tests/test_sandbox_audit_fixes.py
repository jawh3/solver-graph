"""Audit fixes of 2026-10-19 (docs/reviews/AUDIT_NYU_MEMORY_2026-10-19.md), without Docker or a model.

N1   concurrent solves never run a command on the host or in another solve's container (per-thread backend, fail closed,
     require_sandbox), and the adapter checks every dispatched command ran in its container.
F11  an LLM error ends the solve as an error, not a failure.
F04  training LLM/retrieval seeds differ per epoch.
F16/F17/F01  citation: technique tags are not keywords, tokens match literally, generic CLI tokens never ground a citation.
F31  the flag scrubber catches `csawctf{...}`-style prefixes.
"""
import json
import subprocess
import threading
from pathlib import Path

import pytest

from realbench import learn as LN
from realbench import lessons as L
from realbench.solve_adapter import solve_cybench_task
from solver import citation, tool_executor as te
from tests.fakes_realbench import FakeSession, fake_solve
from tests.test_realbench_cybench import _mk_task


@pytest.fixture(autouse=True)
def _sandbox_off_after():
    yield
    te.require_sandbox(False)
    te.set_backend(None)


def _no_host(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a command reached the HOST subprocess path")
    monkeypatch.setattr(te.subprocess, "Popen", boom)


def test_concurrent_solves_only_reach_their_own_container(tmp_path, monkeypatch):
    _no_host(monkeypatch)
    n, rounds = 6, 50
    sessions = [FakeSession(tmp_path / f"agent{i}") for i in range(n)]
    for s in sessions:
        Path(s.workdir).mkdir(parents=True, exist_ok=True); s.start()
    barrier, stats, errors = threading.Barrier(n), {}, []

    def solve(i):
        s = sessions[i]
        ub = te.use_backend(s)
        with ub:
            barrier.wait()                       # every thread has entered use_backend before anyone runs a command
            for k in range(rounds):
                r = te.run_command(f"echo t{i}-{k}", s.workdir)
                if r["error"]:
                    errors.append(r["error"])
        stats[i] = ub.stats

    ths = [threading.Thread(target=solve, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert not errors
    for i, s in enumerate(sessions):
        assert [c for c, _wd in s.seen] == [f"echo t{i}-{k}" for k in range(rounds)]     # all its own, none foreign
        assert s.commands_run == rounds and stats[i] == {"dispatched": rounds, "rejected": 0}
    assert te.get_backend() is None


def test_outside_the_sandbox_is_blocked_and_counted(tmp_path, monkeypatch):
    _no_host(monkeypatch)
    s = FakeSession(tmp_path / "a"); Path(s.workdir).mkdir(parents=True); s.start()
    ub = te.use_backend(s)
    with ub:
        r = te.run_command("id", str(tmp_path / "elsewhere"))
    assert r["error"].startswith("BLOCKED") and s.seen == [] and ub.stats == {"dispatched": 0, "rejected": 1}


def test_require_sandbox_blocks_a_thread_without_backend(tmp_path, monkeypatch):
    _no_host(monkeypatch)
    te.require_sandbox()
    before = te.sandbox_violations()
    r = te.run_command("hostname", tmp_path)
    assert r["error"].startswith("BLOCKED") and te.sandbox_violations() == before + 1
    assert te.get_available_tools() == []          # the host's PATH is never described to a sandboxed solver


def test_adapter_records_sandbox_stats(tmp_path):
    t = _mk_task(tmp_path, flag="HTB{s0me_fl4g}", handout_text="HTB{s0me_fl4g}")
    rec = solve_cybench_task(t, tmp_path / "work", docker=None, session_factory=FakeSession, solve_impl=fake_solve)
    assert rec.error is None and rec.solved
    assert rec.extra["sandbox"] == {"dispatched": 1, "container_runs": 1, "rejected": 0}


def test_adapter_errors_when_a_command_escaped_the_container(tmp_path):
    t = _mk_task(tmp_path)

    def escaping_solve(challenge, **kw):
        te.run_command("ls", kw["working_dir"])
        te.run_command("cat /etc/hostname", "/")       # outside the agent dir -> blocked, and the solve is an error
        return {"flag": None, "trace": [{"round": 1, "commands": ["ls", "cat /etc/hostname"]}], "rounds_used": 1,
                "gave_up_reason": "rounds_exhausted", "total_commands": 2}
    rec = solve_cybench_task(t, tmp_path / "work", docker=None, session_factory=FakeSession, solve_impl=escaping_solve)
    assert rec.error and rec.error.startswith("sandbox:") and rec.extra["sandbox"]["rejected"] == 1


@pytest.mark.parametrize("reason", ["llm_bad_request", "llm_error"])
def test_llm_errors_are_errors_not_failures(tmp_path, reason):
    t = _mk_task(tmp_path)

    def overflow(challenge, **kw):
        return {"flag": None, "trace": [{"round": 1, "action": "error", "commands": []}], "rounds_used": 1,
                "gave_up_reason": reason, "total_commands": 0}
    rec = solve_cybench_task(t, tmp_path / "work", docker=None, session_factory=FakeSession, solve_impl=overflow)
    assert rec.error == f"llm: {reason}" and not rec.solved


def test_rounds_meta_counts_parse_errors_and_truncations(tmp_path):
    t = _mk_task(tmp_path)

    def messy(challenge, **kw):
        return {"flag": None, "rounds_used": 3, "gave_up_reason": "rounds_exhausted", "total_commands": 0,
                "trace": [{"action": "parse_error", "finish_reason": "length"}, {"action": "parse_error"}, {"action": "run_commands", "commands": []}]}
    rec = solve_cybench_task(t, tmp_path / "work", docker=None, session_factory=FakeSession, solve_impl=messy)
    assert rec.extra["rounds_meta"] == {"parse_error_rounds": 2, "length_truncations": 1}


def test_training_seeds_differ_per_epoch():
    assert LN.train_llm_seed(2000, 0) == 2000 and LN.train_llm_seed(2001, 0) == 2001
    seeds = {LN.train_llm_seed(s, e) for s in (2000, 2001) for e in range(3)}
    assert len(seeds) == 6 and not seeds & {1000, 1001, 1002}
    a = LN._seed_rng("task-x", LN.train_llm_seed(2000, 0)).rand(4)
    b = LN._seed_rng("task-x", LN.train_llm_seed(2000, 1)).rand(4)
    assert not (a == b).all()                       # the retrieval draw changes with the epoch


def test_literal_matching_and_generic_tokens():
    m = citation._matches
    assert m("main()", "gdb -batch -ex 'b main' ./a.out", word_boundary=True, literal=True) is None
    assert m("main()", "int main() {", word_boundary=True, literal=True) == "main()"
    assert m("ct[i:i+16]", "for c in ct: print(c)", word_boundary=True, literal=True) is None
    assert m("ct[i:i+16]", "blk = ct[i:i+16]", word_boundary=True, literal=True) == "ct[i:i+16]"
    assert m("rev", "reverse.py", word_boundary=True, literal=True) is None
    # the legacy regex path is unchanged for the Phase-2 microbenchmark
    assert m("ct[i:i+16]", "for c in ct: print(c)", word_boundary=True) is not None

    lesson = {"schema": 1, "task_id": "t", "display_name": "task-1", "title": "pcap creds", "technique": "flag-hex-zip",
              "category": "forensics", "recognise": [], "recipe": ["a", "b"],
              "invocation_signature": ["objdump -d", "checksec --file=", "aaa; afl", "tshark -r", "file"],
              "applicability": {"categories": ["forensics"], "preconditions": [], "artifacts_required": []}, "pitfalls": []}
    node = L.lesson_node(lesson)
    assert node.attributes["keywords"] == lesson["invocation_signature"]          # F16: no technique-tag words
    gen = set(L.generic_keywords([node]))
    assert {"objdump -d", "checksec --file=", "aaa; afl", "file"} <= gen and "tshark -r" not in gen
    trace_generic = [{"commands": ["file x.pcap", "objdump -d a.out", "r2 -c 'aaa; afl' bin"]}]
    trace_real = [{"commands": ["tshark -r cap.pcap -Y http"]}]
    cite = lambda tr: citation.match_citations([node], tr, mode="distinctive", literal=True, generic_tokens=sorted(gen))[0].cited
    assert not cite(trace_generic) and cite(trace_real)


def test_chance_citeability_flags_tokens_every_run_types():
    traces = [[{"commands": ["file a", "strings a | grep x"]}], [{"commands": ["file b"]}], [{"commands": ["tshark -r c"]}]]
    ch = citation.chance_citeability(["file", "tshark -r", "steghide"], traces)
    assert ch == {"file": 2 / 3, "tshark -r": 1 / 3, "steghide": 0.0}
    node = L.lesson_node({"schema": 1, "task_id": "t", "display_name": "task-2", "title": "x", "technique": "t", "category": "misc",
                          "recognise": [], "recipe": ["a", "b"], "invocation_signature": ["tshark -r", "steghide"],
                          "applicability": {"categories": ["misc"], "preconditions": [], "artifacts_required": []}, "pitfalls": []})
    assert "tshark -r" in L.generic_keywords([node], chance=ch, min_chance=0.3)
    assert "steghide" not in L.generic_keywords([node], chance=ch, min_chance=0.3)


def test_scrub_catches_suffix_flag_prefixes():
    text = "flags look like csawctf{abc_123} or csaw_ctf{zz} or myflag{q}; LaTeX c_{i,0} and \\begin{align} survive"
    out, n = L.scrub_flags(text, [], ["HTB"])
    assert n == 3 and "csawctf{" not in out and "csaw_ctf{" not in out and "myflag{" not in out
    assert "c_{i,0}" in out and "\\begin{align}" in out


def test_training_persists_records_with_propensities(tmp_path, monkeypatch):
    from tests.test_realbench_learn import _fake_solve, _mem, _pool_and_tasks
    pool, tasks = _pool_and_tasks(tmp_path)
    monkeypatch.setattr("realbench.solve_adapter.ContainerSession", FakeSession)
    mem = _mem(pool)
    base = tmp_path / "b"
    LN.train(mem, tasks[:2], docker=None, image="x", workroot=str(tmp_path / "w"), seeds=[2000], epochs=2, parallel=2,
             service_parallel=1, bandit_path=f"{base}.json", done_path=f"{base}.train_done.jsonl",
             trainlog_path=f"{base}.train_log.jsonl", solve_kwargs={"solve_impl": _fake_solve}, log=lambda s: None)
    recs = [json.loads(l) for l in Path(f"{base}.train_records.jsonl").read_text().splitlines()]
    assert len(recs) == 4
    assert {r["llm_seed"] for r in recs} == {2000, 12000}                          # F04: epoch 1 has its own seed
    m = recs[0]["extra"]["memory"]
    assert m["per_draw_propensity"] and all(0 < d["p"] <= 1 for d in m["per_draw_propensity"]) and "set_logprob" in m


def _net_runner(state):
    """Fake `docker network` CLI over {name: (internal, n_containers[, isolated gateway])}."""
    from types import SimpleNamespace
    calls = []

    def run(argv, **kw):
        calls.append(list(argv))
        if argv[1:3] == ["network", "inspect"]:
            name = argv[3]
            if name not in state:
                return SimpleNamespace(returncode=1, stdout="", stderr="No such network")
            internal, n, *iso = state[name]
            return SimpleNamespace(returncode=0, stdout=f"{str(internal).lower()} {n} {'isolated' if iso and iso[0] else '<no value>'}\n", stderr="")
        if argv[1:3] == ["network", "rm"]:
            state.pop(argv[3]); return SimpleNamespace(returncode=0, stdout="", stderr="")
        if argv[1:3] == ["network", "create"]:
            state[argv[-1]] = ("--internal" in argv, 0, f"{cb_opt()}=isolated" in argv); return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(argv)
    return run, calls


def cb_opt():
    from realbench import cybench as cb
    return cb.GATEWAY_MODE_OPT


def test_egress_blocked_makes_the_shared_network_internal():
    from realbench import cybench as cb
    state = {"ctfnet": (False, 0)}
    run, calls = _net_runner(state)
    cb.RealDockerOps(runner=run, internal=True).ensure_network("ctfnet")          # plain + empty -> recreated internal + isolated
    assert state["ctfnet"] == (True, 0, True)
    assert ["docker", "network", "create", "--internal", "-o", f"{cb.GATEWAY_MODE_OPT}=isolated", "ctfnet"] in calls
    cb.RealDockerOps(runner=run, internal=True).ensure_network("newnet")          # missing -> created internal + isolated
    assert state["newnet"] == (True, 0, True)
    state["gw"] = (True, 0, False)                                                # internal, host reachable via the gateway
    cb.RealDockerOps(runner=run, internal=True).ensure_network("gw")              # -> recreated isolated (2026-10-20)
    assert state["gw"] == (True, 0, True)
    state["gwbusy"] = (True, 3, False)
    with pytest.raises(cb.ProvisionError, match="gateway"):
        cb.RealDockerOps(runner=run, internal=True).ensure_network("gwbusy")
    cb.RealDockerOps(runner=run, internal=False).ensure_network("ctfnet")         # egress open again -> plain bridge
    assert state["ctfnet"] == (False, 0, False)
    # two solves racing to recreate the same shared network: rm "not found" and create "already exists" are tolerated
    from types import SimpleNamespace
    racing = {"x": (True, 0, False)}
    run2, _ = _net_runner(racing)
    def run_race(argv, **kw):
        if argv[1:3] == ["network", "rm"]:
            racing.pop(argv[3]); racing[argv[3]] = (True, 0, True)               # the other solve already recreated it
            return SimpleNamespace(returncode=1, stdout="", stderr="Error response from daemon: network x not found")
        if argv[1:3] == ["network", "create"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="network with name x already exists")
        return run2(argv, **kw)
    cb.RealDockerOps(runner=run_race, internal=True).ensure_network("x")
    racing["x"] = (True, 0, False)
    def run_race_bad(argv, **kw):
        if argv[1:3] == ["network", "rm"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="network x not found")
        if argv[1:3] == ["network", "create"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="network with name x already exists")
        return run2(argv, **kw)
    with pytest.raises(cb.ProvisionError, match="concurrently"):                  # ... but the survivor must be isolated
        cb.RealDockerOps(runner=run_race_bad, internal=True).ensure_network("x")
    state["busy"] = (False, 2)
    with pytest.raises(cb.ProvisionError):                                        # never pulls a network out from under containers
        cb.RealDockerOps(runner=run, internal=True).ensure_network("busy")
    n = len(calls)
    cb.RealDockerOps(runner=run).ensure_network("busy")                           # legacy: inspect only
    assert len(calls) == n + 1


def test_egress_violations_flags_noninternal_networks_and_socket_mounts():
    """The fail-closed egress/socket check (2026-10-20 review): a stack container on a non-internal network, or that
    bind-mounts the Docker socket, is reported; a container only on internal networks with no socket is clean."""
    from types import SimpleNamespace
    from realbench import cybench as cb
    import json as _json
    # network inspect of the per-solve net -> one container id; inspect of that container; inspect of each of its networks
    internal = {"solvenet": True, "dbnet": True, "ctfnet-x": True, "philnet": False}
    container = {"Id": "cid1", "Name": "/svc", "NetworkSettings": {"Networks": {"solvenet": {"NetworkID": "solvenet"}, "philnet": {"NetworkID": "philnet"}}},
                 "Mounts": [{"Source": "/var/run/docker.sock", "Destination": "/var/run/docker.sock"}]}
    def run(argv, **kw):
        if argv[1:3] == ["network", "inspect"] and "{{range $id" in argv[-1]:
            return SimpleNamespace(returncode=0, stdout="cid1 ", stderr="")
        if argv[1:3] == ["network", "inspect"]:
            return SimpleNamespace(returncode=0, stdout=f"{str(internal.get(argv[3], True)).lower()}\n", stderr="")
        if argv[1] == "inspect":
            return SimpleNamespace(returncode=0, stdout=_json.dumps([container]), stderr="")
        raise AssertionError(argv)
    v = cb.RealDockerOps(runner=run, internal=True).egress_violations("solvenet")
    assert any("philnet" in x and "egress open" in x for x in v)
    assert any("docker.sock" in x and "host-root" in x for x in v)
    # clean container: only internal networks, no socket
    container["NetworkSettings"]["Networks"] = {"solvenet": {"NetworkID": "solvenet"}, "dbnet": {"NetworkID": "dbnet"}}
    container["Mounts"] = [{"Source": "/data", "Destination": "/data"}]
    assert cb.RealDockerOps(runner=run, internal=True).egress_violations("solvenet") == []
    # empty network -> nothing to check
    def run_empty(argv, **kw):
        return SimpleNamespace(returncode=0, stdout=" ", stderr="")
    assert cb.RealDockerOps(runner=run_empty, internal=True).egress_violations("solvenet") == []


def test_egress_violations_fails_closed_and_checks_the_whole_compose_project():
    """Review follow-up: an inspect that fails is a violation, not a clean result; a sidecar of the same compose project
    that is not on the solve network is still checked; the socket check runs even when egress is not checked."""
    from types import SimpleNamespace
    from realbench import cybench as cb
    import json as _json
    internal = {"solvenet": True, "proj_default": False}
    svc = {"Id": "cid1", "Name": "/svc", "Config": {"Labels": {"com.docker.compose.project": "nyu-x"}},
           "NetworkSettings": {"Networks": {"solvenet": {"NetworkID": "solvenet"}}}, "Mounts": []}
    side = {"Id": "cid2", "Name": "/db", "Config": {"Labels": {"com.docker.compose.project": "nyu-x"}},
            "NetworkSettings": {"Networks": {"proj_default": {"NetworkID": "proj_default"}, "none": {}}},
            "Mounts": [{"Source": "/run/docker.sock", "Destination": "/var/run/docker.sock"}]}
    byid = {"cid1": svc, "cid2": side}
    fail = set()
    def run(argv, **kw):
        if argv[1:3] == ["network", "inspect"] and "{{range $id" in argv[-1]:
            return SimpleNamespace(returncode=1 if "list" in fail else 0, stdout="cid1 ", stderr="boom")
        if argv[1:3] == ["network", "inspect"]:
            assert argv[3] != "none"
            return SimpleNamespace(returncode=0, stdout=f"{str(internal[argv[3]]).lower()}\n", stderr="")
        if argv[1] == "ps":
            assert "label=com.docker.compose.project=nyu-x" in argv
            return SimpleNamespace(returncode=0, stdout="cid1\ncid2\n", stderr="")
        if argv[1] == "inspect":
            if "inspect" in fail:
                return SimpleNamespace(returncode=1, stdout="[]", stderr="No such container")
            return SimpleNamespace(returncode=0, stdout=_json.dumps([byid[c] for c in argv[4:]]), stderr="")
        raise AssertionError(argv)
    d = cb.RealDockerOps(runner=run, internal=True)
    v = d.egress_violations("solvenet")
    assert any(x.startswith("db is on non-internal network 'proj_default'") for x in v)
    assert any(x.startswith("db bind-mounts the Docker socket") for x in v) and not any("none" in x for x in v)
    assert [x for x in d.egress_violations("solvenet", check_egress=False) if "egress open" in x] == []
    assert any("docker.sock" in x for x in d.egress_violations("solvenet", check_egress=False))
    fail.add("inspect")
    assert any("could not inspect" in x for x in d.egress_violations("solvenet"))
    fail.add("list")
    assert any("could not list" in x for x in d.egress_violations("solvenet"))


def test_sweep_networks_removes_only_empty_old_matching_networks():
    from types import SimpleNamespace
    from realbench import cybench as cb
    import time as _time
    old, young = int(_time.time()) - 3600, int(_time.time()) - 5
    nets = {"ctfnet-aaaaaa": (0, old), "ctfnet-bbbbbb": (2, old), "ctfnet-cccccc": (0, young), "ctfnet": (0, old), "shared_net": (0, old)}
    removed = []
    def run(argv, **kw):
        if argv[1:3] == ["network", "ls"]:
            return SimpleNamespace(returncode=0, stdout="\n".join(nets) + "\n", stderr="")
        if argv[1:3] == ["network", "inspect"]:
            return SimpleNamespace(returncode=0, stdout="%d %d\n" % nets[argv[3]], stderr="")
        if argv[1:3] == ["network", "rm"]:
            removed.append(argv[3]); nets.pop(argv[3]); return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(argv)
    n = cb.RealDockerOps(runner=run, internal=True).sweep_networks("ctfnet-")
    # not the one with 2 containers, not the empty one another runner created 5 s ago, not "ctfnet", not "shared_net"
    assert n == 1 and removed == ["ctfnet-aaaaaa"]


def test_teardown_force_removes_leftover_containers_and_survives_a_hung_stop_script(tmp_path):
    """2026-10-22: a leftover service (compose down failed or hung) is force-removed from the solve's network, and an
    exception from stop_docker.sh no longer skips the cleanup."""
    from realbench import cybench as cb
    calls = []
    class D(cb.DockerOps):
        def run_script(self, script, cwd, timeout_s):
            raise subprocess.TimeoutExpired("stop_docker.sh", timeout_s)
        def remove_containers_on(self, network): calls.append(("rm-containers", network)); return 2
        def remove_network(self, name): calls.append(("rm-net", name)); return True
    snap = tmp_path / "task"; snap.mkdir(); (snap / "stop_docker.sh").write_text("#!/bin/bash\n")
    wd = tmp_path / "wd"; wd.mkdir()
    prov = cb.Provisioned(task=None, workdir=wd, agent_dir=wd, snapshot_dir=snap, handouts={}, source_manifest={},
                          services_started=True, owned_networks=("ctfnet-abc",), _docker=D())
    prov.teardown(verify_source=False)
    assert calls == [("rm-containers", "ctfnet-abc"), ("rm-net", "ctfnet-abc")]
    assert any("stop_docker.sh failed" in w for w in prov.warnings) and any("force-removed 2" in w for w in prov.warnings)


def test_every_service_gets_a_process_cap(tmp_path):
    from realbench import nyuctf as ny
    from tests.test_realbench_nyuctf import _mk_root
    root = _mk_root(tmp_path, compose=True)
    (t,) = ny.enumerate_tasks(root, split="test")
    assert f"    pids_limit: {ny.SERVICE_PIDS_LIMIT}\n" in ny.compose_override(t) and ny.SERVICE_PIDS_LIMIT == 4096


def test_isolation_problem_refuses_docker_older_than_28():
    """Docker < 28 stores gateway_mode_ipv4=isolated without enforcing it, so the policy read-back would pass."""
    from types import SimpleNamespace
    from realbench import cybench as cb
    def ver(out, rc=0):
        return cb.RealDockerOps(runner=lambda argv, **kw: SimpleNamespace(returncode=rc, stdout=out, stderr="err"), internal=True)
    assert ver("28.1.1\n").isolation_problem() == ""
    assert "needs >= 28" in ver("27.5.1\n").isolation_problem()
    assert "could not read" in ver("", rc=1).isolation_problem()
    assert "could not read" in ver("podman\n").isolation_problem()
