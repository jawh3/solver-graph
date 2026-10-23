"""NYU CTF Bench adapter without Docker or a model: enumeration + neutral fields, flag classification and pre-declared
exclusions, provisioning (handout subpaths, snapshot excludes, compose scripts on ctfnet), the solver-visible dict, the
runner in --bench nyuctf mock mode, duck-typed flag matching, the reachability check, and distillation sources."""
import collections
import json
import subprocess
from pathlib import Path

import pytest

from realbench import cybench as cb
from realbench import nyuctf as ny
from realbench import run_cybench
from tests.test_realbench_cybench import FakeDockerOps


def _mk_root(tmp_path, *, compose=False, box="pwn.chal.csaw.io", port=5000, flag="flag{x_y_z}", files=("handout.txt", "sub/inner.bin"),
             name="Maze Runner", tid="2021f-pwn-maze_runner", desc=None, alias=None, docker_sock=False, handout_text="hello"):
    root = tmp_path / "nyu"
    rel = "test/2021/CSAW-Finals/pwn/maze_runner"
    tdir = root / rel; tdir.mkdir(parents=True)
    for f in files:
        p = tdir / f; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(handout_text)
    (tdir / "flag.txt").write_text(flag + "\n")                      # server-side, never a handout
    (tdir / "solve.py").write_text("print('reference')\n")
    (tdir / "hints").mkdir(); (tdir / "hints" / "solution.md").write_text("the intended solution")
    (tdir / "test_solver").mkdir(); (tdir / "test_solver" / "test.sh").write_text("# Run the solver (your code here)\nexit 1\n")
    (tdir / "README.md").write_text("write-up for Maze Runner\n")
    cj = {"name": name, "category": "pwn", "description": desc if desc is not None else "Escape Maze Runner. `nc {box} {port}`",
          "flag": flag, "files": list(files), "box": box, "internal_port": port, "compose": compose, "points": 100}
    (tdir / "challenge.json").write_text(json.dumps(cj))
    if compose:
        vol = "    volumes:\n      - /var/run/docker.sock:/var/run/docker.sock\n" if docker_sock else ""
        (tdir / "docker-compose.yml").write_text(
            "services:\n  game-server:\n    image: llmctf/2021f-pwn-maze_runner:latest\n" + vol +
            f"    networks:\n      ctfnet:\n        aliases:\n          - {alias or box}\n    ports:\n      - \"5000:5000\"\n"
            "networks:\n  ctfnet:\n    external: true\n")
    (root / "test_dataset.json").write_text(json.dumps({tid: {"year": "2021", "event": "CSAW-Finals", "category": "pwn",
                                                              "challenge": "maze_runner", "path": rel}}))
    return root


def test_enumerate_neutral_fields_and_formats(tmp_path):
    root = _mk_root(tmp_path, compose=True)
    (t,) = ny.enumerate_tasks(root, split="test")
    assert t.id == "2021f-pwn-maze_runner" and t.display_name.startswith("task-") and t.category == "pwn"
    assert t.needs_service and t.target_host == "pwn.chal.csaw.io:5000" and t.target_hosts == (("pwn.chal.csaw.io", 5000),)
    assert t.answer_format == "flag{...}" and t.competition_date == "2021-11" and t.post_cutoff() is False
    assert "maze_runner" in t.aliases and t.solver_extractable and t.exclusion_reason == "" and t.server_type == "nc"
    assert t.handout_files == ("handout.txt", "sub/inner.bin") and not t.docker_sock
    assert cb._flag_format_of(t.answer_format) == "flag"


def test_flag_classification_and_exclusions(tmp_path):
    (t,) = ny.enumerate_tasks(_mk_root(tmp_path, flag="flag{has a space}"), split="test")
    assert not t.solver_extractable and "extractor" in t.exclusion_reason
    (t,) = ny.enumerate_tasks(_mk_root(tmp_path / "b", flag="deadbeef00"), split="test")
    assert not t.solver_extractable and t.answer_format == "" and cb._flag_format_of(t.answer_format) == "any"
    (t,) = ny.enumerate_tasks(_mk_root(tmp_path / "c", compose=False, box="pwn.chal.csaw.io"), split="test")
    assert not t.needs_service and t.target_host == "" and t.server_type == "" and t.exclusion_reason == ""   # box without compose: file-only
    root = _mk_root(tmp_path / "d", tid="2023f-cry-nervcenter")
    (t,) = ny.enumerate_tasks(root, split="test")
    assert t.exclusion_reason.startswith("depends on an external")
    (t,) = ny.enumerate_tasks(_mk_root(tmp_path / "e", compose=True, docker_sock=True), split="test")
    assert t.docker_sock and t.exclusion_reason == ""                                     # included by run policy, flagged
    assert ny.solver_can_extract("csawctf{a_b-c}") and not ny.solver_can_extract("flag{a{b}}") and not ny.solver_can_extract("flag{a b}")
    assert ny.solver_can_extract("flag{a{b}")                       # one inner `{`: the solver's own class `[^\s}]` allows it
    from solver.solver import compile_flag_pattern
    assert compile_flag_pattern("flag").search("flag{a{b}").group(0) == "flag{a{b}"


def test_description_only_task_runs_with_an_empty_agent_dir(tmp_path):
    root = _mk_root(tmp_path, compose=False, files=(), desc="Crack the hash `abc`. Submit it as `flag{{<pw>}}`.")
    (t,) = ny.enumerate_tasks(root, split="test")
    assert t.exclusion_reason == "" and t.handout_files == ()
    prov = ny.provision(t, tmp_path / "work", docker=None)
    assert not prov.handouts and prov.network == "ctfnet"
    desc = ny.to_challenge_dict(t, prov)["description"]
    assert "flag{<pw>}" in desc and "flag{{" not in desc and "No files are provided" in desc   # format_map-style brace collapse
    prov.teardown()
    root2 = _mk_root(tmp_path / "e", compose=False, files=(), desc="")
    (t2,) = ny.enumerate_tasks(root2, split="test")
    assert t2.exclusion_reason.startswith("no handouts, no service and no description")


def test_tree_lists_images_and_file_only_solves_join_ctfnet(tmp_path):
    (t,) = ny.enumerate_tasks(_mk_root(tmp_path, compose=False, files=("pic.png", "notes.txt")), split="test")
    prov = ny.provision(t, tmp_path / "work", docker=None)
    assert prov.network == "ctfnet"                                                       # not Cybench's shared_net
    desc = ny.to_challenge_dict(t, prov)["description"]
    assert "pic.png" in desc and "notes.txt" in desc                                       # upstream lists every handout
    assert "pic.png" not in cb.render_tree(prov.agent_dir) and "pic.png" in cb.render_tree(prov.agent_dir, hide_ext=())
    prov.teardown()


def test_compose_port_map_corrects_internal_port_and_fixed_names_are_precleaned(tmp_path):
    root = _mk_root(tmp_path, compose=True, port=3000)
    (root / "test/2021/CSAW-Finals/pwn/maze_runner/docker-compose.yml").write_text(
        "services:\n  web:\n    image: llmctf/x:latest\n    container_name: phil-front\n    ports:\n      - 3000:80\n"
        "    networks:\n      philnet:\n      ctfnet:\n        aliases:\n          - pwn.chal.csaw.io\n          - maze_runner\n"
        "networks:\n  philnet:\n    name: philnet\n  ctfnet:\n    external: true\n")
    (t,) = ny.enumerate_tasks(root, split="test")
    assert t.internal_port == 80 and t.target_host == "pwn.chal.csaw.io:80"              # 3000 is the HOST side of 3000:80
    facts = ny._compose_facts(t.compose_file)
    assert facts["aliases"] == ["pwn.chal.csaw.io", "maze_runner"] and facts["container_names"] == ["phil-front"]
    assert facts["network_names"] == ["philnet"] and facts["port_map"] == [(3000, 80)] and facts["images"] == ["llmctf/x:latest"]
    scripts = ny.compose_scripts(t, "nyu-p"); start = scripts["start_docker.sh"]
    assert "docker rm -f phil-front" in start and "docker network rm philnet" in start and start.index("rm -f") < start.index("compose -p")
    assert facts["service_names"] == ["web"]
    assert scripts["nyu-override.yml"] == ('services:\n  "web":\n    ports: !reset []\n    pids_limit: 4096\n'
        'networks:\n  "philnet":\n    internal: true\n    driver_opts:\n      com.docker.network.bridge.gateway_mode_ipv4: isolated\n')
    assert ny._container_port(5000, [(5000, 5000)]) == 5000 and ny._container_port(80, [(3000, 80)]) == 80 and ny._container_port(None, [(1, 2)]) is None


def test_missing_handout_excludes_the_task_and_init_script_exits_3(tmp_path):
    import subprocess
    root = _mk_root(tmp_path, compose=False, files=("handout.txt", "gone.bin"))
    tdir = root / "test/2021/CSAW-Finals/pwn/maze_runner"; (tdir / "gone.bin").unlink()
    (t,) = ny.enumerate_tasks(root, split="test")
    assert t.exclusion_reason.startswith("listed handout missing") and "gone.bin" in t.exclusion_reason
    script = tdir / "init_script.sh"; script.write_text(ny.init_script(t))
    r = subprocess.run(["bash", str(script), str(tmp_path / "agent")], cwd=str(tdir), capture_output=True, text=True)
    assert r.returncode == 3 and "handout missing: gone.bin" in r.stderr and (tmp_path / "agent" / "handout.txt").exists()


def test_gather_sources_skips_a_readme_that_is_a_handout(tmp_path):
    (t,) = ny.enumerate_tasks(_mk_root(tmp_path, files=("README.md",)), split="test")
    src = ny.gather_sources(t)
    assert "### README.md" not in src.get("writeups", "") and "solution.md" in src["writeups"]


def test_select_tasks_needs_a_separator_boundary(tmp_path):
    (a,) = ny.enumerate_tasks(_mk_root(tmp_path / "a", tid="2021f-rev-maze"), split="test")
    (b,) = ny.enumerate_tasks(_mk_root(tmp_path / "b", tid="2022q-msc-ezmaze"), split="test")
    assert run_cybench.select_tasks([a, b], "maze") == [a] and run_cybench.select_tasks([a, b], "ezmaze") == [b]
    assert run_cybench.select_tasks([a, b], "2021f-rev-maze," + b.display_name) == [a, b] and run_cybench.select_tasks([a, b], "aze") == []


def test_service_jobs_are_gated_per_hostname_and_interleaved(tmp_path, monkeypatch):
    """--service-parallel 2: two service tasks on DIFFERENT hostnames run concurrently, two on the SAME hostname never do;
    interleave_jobs orders service jobs round-robin over hostname groups with file-only jobs spread between them."""
    import threading, time as _time
    from realbench.solve_adapter import SolveRecord
    root = _mk_root(tmp_path, compose=True, files=("handout.txt",))
    (base,) = ny.enumerate_tasks(root, split="test")
    mk = lambda tid, host, svc=True: base.__class__(**{**base.__dict__, "id": tid, "needs_service": svc, "target_host": f"{host}:5000" if svc else "", "box": host})
    pwn1, pwn2, pwn3 = mk("2021f-pwn-a", "pwn.chal.csaw.io"), mk("2021f-pwn-b", "pwn.chal.csaw.io"), mk("2021f-pwn-c", "pwn.chal.csaw.io")
    web1 = mk("2021f-web-a", "web.chal.csaw.io"); f1, f2 = mk("2021f-rev-f1", "", False), mk("2021f-rev-f2", "", False)
    order = run_cybench.interleave_jobs([(t, 0) for t in (pwn1, pwn2, pwn3, web1, f1, f2)])
    assert [t.id for t, _k in order] == ["2021f-pwn-a", "2021f-web-a", "2021f-rev-f1", "2021f-pwn-b", "2021f-rev-f2", "2021f-pwn-c"]
    assert [k for _t, k in run_cybench.interleave_jobs([(pwn1, 1), (web1, 0), (pwn2, 0)])] == [0, 0, 1]          # trial-major kept
    monkeypatch.setattr(run_cybench, "load_bench", lambda args: (ny, [pwn1, pwn2, web1]))
    monkeypatch.setattr(ny, "PER_SOLVE_NETWORK", False)            # the shared-network protocol (per-solve: next test)
    monkeypatch.setattr(run_cybench, "LOCK_DIR", tmp_path / "locks")
    monkeypatch.setenv("MODEL_NAME", "m")
    live = collections.defaultdict(int); peak = collections.defaultdict(int); lock = threading.Lock()
    def slow(task, workroot, **kw):
        h = task.box
        with lock:
            live[h] += 1; live["all"] += 1; peak[h] = max(peak[h], live[h]); peak["all"] = max(peak["all"], live["all"])
        _time.sleep(0.2)
        with lock: live[h] -= 1; live["all"] -= 1
        return SolveRecord(task_id=task.id, display_name=task.display_name, category=task.category, needs_service=True, extra=dict(kw.get("extra") or {}))
    monkeypatch.setattr(run_cybench, "solve_cybench_task", slow)
    out = tmp_path / "r" / "g.jsonl"
    rc = run_cybench.main(["--bench", "nyuctf", "--nyu-root", str(root), "--tasks", "all", "--trials", "2", "--parallel", "3", "--service-parallel", "2",
                           "--degenerate-after", "0", "--mock-solve", "--workroot", str(tmp_path / "w"), "--out", str(out)])
    assert rc == 0 and peak["pwn.chal.csaw.io"] == 1 and peak["all"] == 2 and len(out.read_text().splitlines()) == 6


def test_per_solve_network_lets_same_hostname_tasks_overlap_and_locks_pinned_names(tmp_path, monkeypatch):
    """2026-10-20: with per-solve networks, two service tasks on the SAME alias may run at once (no shared network to
    collide on); only names a compose file pins machine-wide are locked, and that lock also holds across processes."""
    import threading, time as _time, multiprocessing as mp
    from realbench.solve_adapter import SolveRecord
    root = _mk_root(tmp_path, compose=True, files=("handout.txt",))
    (base,) = ny.enumerate_tasks(root, split="test")
    mk = lambda tid: base.__class__(**{**base.__dict__, "id": tid, "needs_service": True, "target_host": "pwn.chal.csaw.io:5000", "box": "pwn.chal.csaw.io"})
    a, b = mk("2021f-pwn-a"), mk("2021f-pwn-b")
    assert run_cybench.service_lock_keys(a, ny) == []                       # nothing pinned in the fixture compose file
    monkeypatch.setattr(run_cybench, "load_bench", lambda args: (ny, [a, b]))
    monkeypatch.setattr(run_cybench, "LOCK_DIR", tmp_path / "locks")
    monkeypatch.setenv("MODEL_NAME", "m")
    live = {"n": 0, "peak": 0}; lock = threading.Lock()
    def slow(task, workroot, **kw):
        with lock: live["n"] += 1; live["peak"] = max(live["peak"], live["n"])
        _time.sleep(0.2)
        with lock: live["n"] -= 1
        return SolveRecord(task_id=task.id, display_name=task.display_name, category=task.category, needs_service=True, extra=dict(kw.get("extra") or {}))
    monkeypatch.setattr(run_cybench, "solve_cybench_task", slow)
    rc = run_cybench.main(["--bench", "nyuctf", "--nyu-root", str(root), "--tasks", "all", "--trials", "1", "--parallel", "2", "--service-parallel", "2",
                           "--degenerate-after", "0", "--mock-solve", "--workroot", str(tmp_path / "w"), "--out", str(tmp_path / "o.jsonl")])
    assert rc == 0 and live["peak"] == 2
    # pinned names are the lock keys; a NamedLock held here blocks another process
    facts = {"container_names": ["phil-db"], "network_names": ["philnet"]}
    monkeypatch.setattr(ny, "_compose_facts", lambda f: facts)
    assert run_cybench.service_lock_keys(a, ny) == ["phil-db", "philnet"]
    q = mp.get_context("fork").Queue()
    def other():
        t0 = _time.time()
        with run_cybench.NamedLock("phil-db"):
            q.put(_time.time() - t0)
    with run_cybench.NamedLock("phil-db"):
        pr = mp.get_context("fork").Process(target=other); pr.start(); _time.sleep(0.6)
    pr.join(10)
    assert q.get(timeout=5) >= 0.5


def test_override_points_ctfnet_at_the_per_solve_network_and_teardown_removes_it(tmp_path):
    root = _mk_root(tmp_path, compose=True, files=("handout.txt",))
    (t,) = ny.enumerate_tasks(root, split="test")
    ov = ny.compose_override(t, "ctfnet-abc123")
    assert 'networks:\n  "ctfnet":\n    name: "ctfnet-abc123"\n    external: true' in ov
    assert "ctfnet-abc123" not in ny.compose_override(t)
    calls = []
    class D(cb.DockerOps):
        def ensure_network(self, name): calls.append(("net", name))
        def remove_network(self, name): calls.append(("rm", name)); return True
        def run_script(self, script, cwd, timeout_s):
            calls.append(("script", script.name))
            return subprocess.CompletedProcess([], 0, "", "")
        def wait_for_hosts(self, hosts, network, timeout_s=180.0, interval_s=2.0):
            calls.append(("health", network)); return {f"{h}:{p}": True for h, p in hosts}
        def add_network_alias(self, network, hostname, alias): return True
    prov = ny.provision(t, tmp_path / "wd", docker=D())
    assert prov.network.startswith("ctfnet-") and prov.owned_networks == (prov.network,)
    assert ("net", prov.network) in calls and ("health", prov.network) in calls
    assert f'name: "{prov.network}"' in (prov.snapshot_dir / ny.NOPORTS_OVERRIDE).read_text()
    prov.teardown(verify_source=False)
    assert calls[-1] == ("rm", prov.network)
    legacy = ny.provision(t, tmp_path / "wd2", docker=D(), per_solve_network=False)
    assert legacy.network == "ctfnet" and legacy.owned_networks == ()
    legacy.teardown(verify_source=False)


def test_start_timeout_still_stops_the_stack_and_per_solve_solves_skip_shared_net(tmp_path):
    """Review follow-up: a start_docker.sh that times out may have started part of the stack, so teardown runs
    stop_docker.sh before removing the per-solve network (else the containers stay attached and the network leaks).
    A per-solve service solve never ensures Cybench's shared_net (it never joins it)."""
    root = _mk_root(tmp_path, compose=True, files=("handout.txt",))
    (t,) = ny.enumerate_tasks(root, split="test")
    calls = []
    class D(cb.DockerOps):
        def ensure_network(self, name): calls.append(("net", name))
        def remove_network(self, name): calls.append(("rm", name)); return True
        def run_script(self, script, cwd, timeout_s):
            calls.append(("script", script.name))
            if script.name == "start_docker.sh":
                raise subprocess.TimeoutExpired(["bash", str(script)], timeout_s)
            return subprocess.CompletedProcess([], 0, "", "")
    with pytest.raises(subprocess.TimeoutExpired):
        ny.provision(t, tmp_path / "wd", docker=D())
    nets = [c[1] for c in calls if c[0] == "net"]
    assert len(nets) == 1 and nets[0].startswith("ctfnet-") and cb.SHARED_NET not in nets
    assert calls[-2:] == [("script", "stop_docker.sh"), ("rm", nets[0])]
    calls.clear()
    legacy = ny.provision(t, tmp_path / "wd2", docker=FakeDockerOps(), per_solve_network=False)   # shared ctfnet: unchanged
    assert legacy.networks == ("ctfnet", cb.SHARED_NET)
    legacy.teardown(verify_source=False)


def test_override_drops_the_host_docker_socket_for_every_task(tmp_path):
    """2026-10-20 (review): the host Docker socket is never mounted into a challenge; the one task that needs it is
    EXCLUDED, so there is no keep-the-socket exception any more."""
    root = _mk_root(tmp_path, compose=True, docker_sock=True)
    (t,) = ny.enumerate_tasks(root, split="test")
    assert "docker.sock" in t.compose_file.read_text()
    assert '  "game-server":\n    ports: !reset []\n    pids_limit: 4096\n    volumes: !reset []\n' in ny.compose_override(t)
    # other mounts on the same service are kept; only the socket is removed
    t.compose_file.write_text(t.compose_file.read_text().replace("      - /var/run/docker.sock:/var/run/docker.sock\n",
                                                                "      - /var/run/docker.sock:/var/run/docker.sock\n      - ./data:/data\n"))
    assert '    volumes: !override\n      - "./data:/data"\n' in ny.compose_override(t)


def test_es1337_is_excluded_as_a_host_root_risk():
    """The one NYU image that genuinely needs the host Docker socket is excluded rather than given host-root."""
    if not (ny.DEFAULT_NYU_ROOT / "test_dataset.json").exists():
        pytest.skip("needs a NYU CTF Bench checkout ($NYUCTF_ROOT)")
    ts = {t.id: t for t in ny.enumerate_tasks(split="test")}
    es = ts.get("2018f-pwn-es1337")
    assert es is not None and es.exclusion_reason and "socket" in es.exclusion_reason.lower()
    # and it is absent from the usable test list the campaign runs on (column 1 = display name, last column = task id)
    rows = [l.rstrip("\n").split("\t") for l in open(Path(__file__).resolve().parents[1] / "docs/phase3/NYU_USABLE_test.txt") if l.strip()]
    assert len(rows) > 100 and es.display_name not in {r[0] for r in rows} and es.id not in {r[-1] for r in rows}


def test_override_marks_compose_defined_networks_internal(tmp_path):
    """A non-external network a compose file creates itself is an ordinary bridge with internet; the override marks it
    internal + isolated gateway so a service on it (philnet, shreeramquest) cannot reach out."""
    root = _mk_root(tmp_path, compose=True)
    (t,) = ny.enumerate_tasks(root, split="test")
    t.compose_file.write_text(t.compose_file.read_text().replace(
        "networks:\n  ctfnet:\n    external: true\n",
        "    networks:\n      philnet:\n      ctfnet:\n        aliases:\n          - pwn.chal.csaw.io\n"
        "networks:\n  philnet:\n    driver: bridge\n  ctfnet:\n    external: true\n"))
    assert ny._nonexternal_networks(t) == ["philnet"]
    ov = ny.compose_override(t, "ctfnet-abc")
    assert f'  "philnet":\n    internal: true\n    driver_opts:\n      {cb.GATEWAY_MODE_OPT}: isolated\n' in ov
    assert ov.count("networks:") == 1          # one networks: block, both ctfnet and philnet under it


def test_runner_uses_separate_pools_for_service_and_file_jobs(tmp_path, monkeypatch):
    """parallel 3 / service-parallel 1 with both kinds queued: file-only jobs keep 2 workers busy while service jobs
    run one at a time (previously two workers sat blocked on the gate)."""
    import threading, time as _time
    from realbench.solve_adapter import SolveRecord
    root = _mk_root(tmp_path, compose=True, files=("handout.txt",))
    (t_svc,) = ny.enumerate_tasks(root, split="test")
    t_file = t_svc.__class__(**{**t_svc.__dict__, "id": "2021f-pwn-other", "needs_service": False, "target_host": "", "compose_file": None})
    monkeypatch.setattr(run_cybench, "load_bench", lambda args: (ny, [t_svc, t_file]))
    monkeypatch.setenv("MODEL_NAME", "m")
    seen = {"svc": 0, "file": 0, "svc_max": 0, "file_max": 0}; lock = threading.Lock()
    def slow(task, workroot, **kw):
        kind = "svc" if task.needs_service else "file"
        with lock: seen[kind] += 1; seen[kind + "_max"] = max(seen[kind + "_max"], seen[kind])
        _time.sleep(0.15)
        with lock: seen[kind] -= 1
        return SolveRecord(task_id=task.id, display_name=task.display_name, category=task.category, needs_service=task.needs_service, extra=dict(kw.get("extra") or {}))
    monkeypatch.setattr(run_cybench, "solve_cybench_task", slow)
    out = tmp_path / "r" / "n.jsonl"
    rc = run_cybench.main(["--bench", "nyuctf", "--nyu-root", str(root), "--tasks", "all", "--trials", "4", "--parallel", "3", "--service-parallel", "1", "--degenerate-after", "0",
                           "--mock-solve", "--workroot", str(tmp_path / "w"), "--out", str(out)])
    assert rc == 0 and seen["svc_max"] == 1 and seen["file_max"] == 2 and len(out.read_text().splitlines()) == 8


def test_provision_file_only_preserves_subpaths_and_hides_solution_material(tmp_path):
    (t,) = ny.enumerate_tasks(_mk_root(tmp_path, compose=False), split="test")
    prov = ny.provision(t, tmp_path / "work", docker=None)
    agent = {p.relative_to(prov.agent_dir).as_posix() for p in prov.agent_dir.rglob("*") if p.is_file()}
    assert agent == {"handout.txt", "sub/inner.bin"}                                     # only the handout list, subpath kept
    snap = {p.relative_to(prov.snapshot_dir).as_posix() for p in prov.snapshot_dir.rglob("*") if p.is_file()}
    assert "flag.txt" in snap and "solve.py" in snap                                       # snapshot keeps build inputs; agent never sees it
    assert not any(s.startswith(("hints/", "test_solver/")) for s in snap)
    assert prov.init_returncode == 0 and prov.blocked_commands == () and prov.flag_leaks == ()
    prov.teardown()
    assert not (tmp_path / "work").exists()


def test_provision_service_runs_compose_on_ctfnet_with_unique_project(tmp_path):
    (t,) = ny.enumerate_tasks(_mk_root(tmp_path, compose=True), split="test")
    d = FakeDockerOps()
    prov = ny.provision(t, tmp_path / "work", docker=d)
    assert prov.network.startswith("ctfnet-") and ("ensure_network", prov.network) in d.calls and prov.services_started   # per-solve network
    assert ("run_script", "start_docker.sh", str(prov.snapshot_dir)) in d.calls
    assert ("wait_for_hosts", (("pwn.chal.csaw.io", 5000),), prov.network) in d.calls and prov.health == {"pwn.chal.csaw.io:5000": True}
    start = (prov.snapshot_dir / "start_docker.sh").read_text()
    assert "-p nyu-task-" in start and "--pull never" in start and "-f docker-compose.yml -f nyu-override.yml up" in start
    assert (prov.snapshot_dir / "nyu-override.yml").read_text() == ('services:\n  "game-server":\n    ports: !reset []\n    pids_limit: 4096\n'   # host ports dropped
        f'networks:\n  "ctfnet":\n    name: "{prov.network}"\n    external: true\n')                          # per-solve network
    assert "-f docker-compose.yml -f nyu-override.yml down" in (prov.snapshot_dir / "stop_docker.sh").read_text()
    assert not any(c[0] == "add_network_alias" for c in d.calls)                          # generic hostname: nothing to hide
    prov.teardown()
    assert ("run_script", "stop_docker.sh", str(prov.snapshot_dir)) in d.calls


def test_leaking_hostname_gets_a_neutral_alias(tmp_path):
    (t,) = ny.enumerate_tasks(_mk_root(tmp_path, compose=True, box="maze-runner.csaw.io", alias="maze-runner.csaw.io"), split="test")
    d = FakeDockerOps()
    prov = ny.provision(t, tmp_path / "work", docker=d)
    assert prov.host_aliases == {"maze-runner.csaw.io": t.display_name}
    desc = ny.to_challenge_dict(t, prov)["description"]
    assert "maze-runner.csaw.io" not in desc and f"nc {t.display_name} 5000" in desc
    ovr = (prov.snapshot_dir / "nyu-override.yml").read_text()                            # alias installed AT START, no reconnect
    assert ovr == ('services:\n  "game-server":\n    ports: !reset []\n    pids_limit: 4096\n    networks:\n      "ctfnet":\n        aliases:\n'
                   f'          - "maze-runner.csaw.io"\n          - "{t.display_name}"\n'
                   f'networks:\n  "ctfnet":\n    name: "{prov.network}"\n    external: true\n')
    assert ny.neutral_alias(t) == t.display_name
    prov.teardown()
    (t0,) = ny.enumerate_tasks(_mk_root(tmp_path / "g", compose=True), split="test")
    assert ny.neutral_alias(t0) == "" and "aliases" not in ny.compose_override(t0)         # generic hostname: ports only


def test_challenge_dict_is_neutral_and_complete(tmp_path):
    (t,) = ny.enumerate_tasks(_mk_root(tmp_path, compose=True), split="test")
    prov = ny.provision(t, tmp_path / "work", docker=FakeDockerOps())
    ch = ny.to_challenge_dict(t, prov)
    shown = {k: v for k, v in ch.items() if not k.startswith("_")}
    assert shown["name"] == t.display_name and shown["dataset"] == "nyuctf" and shown["flag_format"] == "flag"
    assert shown["points"] == 0 and shown["difficulty"] == "unknown" and shown["hints"] == []
    desc = shown["description"]
    assert not cb.leaks_task_name(desc, t) and "Maze Runner" not in desc
    assert "nc pwn.chal.csaw.io 5000" in desc and "The flag format is flag{...}." in desc and "handout.txt" in desc and "inner.bin" in desc
    assert [f["text"] for f in shown["local_files"]] == ["handout.txt", "sub/inner.bin"]
    assert ch["_flag"] == "flag{x_y_z}" and ch["_network"].startswith("ctfnet-") and ch["_server_type"] == "nc" and ch["contamination_flag"] is True
    prov.teardown()
    assert desc.count("pwn.chal.csaw.io 5000") == 1                                       # description already says it: no second server line
    (t2,) = ny.enumerate_tasks(_mk_root(tmp_path / "w", compose=True, box="web.chal.csaw.io", desc="A notes app."), split="test")
    t2 = t2.__class__(**{**t2.__dict__, "category_raw": "web", "server_type": "web"})
    assert "http://web.chal.csaw.io:5000/" in ny.render_description(t2, None)
    t3 = t2.__class__(**{**t2.__dict__, "internal_port": 443, "target_host": "web.chal.csaw.io:443"})
    assert "https://web.chal.csaw.io:443/" in ny.render_description(t3, None)
    assert ch["_name_redactions"] == 1                                                     # "Maze Runner" once in the description


def test_flag_match_is_duck_typed(tmp_path):
    (t,) = ny.enumerate_tasks(_mk_root(tmp_path), split="test")
    assert ny.check_flag(t, "flag{x_y_z}") and cb.flag_match(t, " flag{x_y_z}\n").exact and not ny.check_flag(t, "flag{x}")
    assert cb.check_flag("flag{a}", "flag{a}") and not cb.check_flag("flag{a}", "flag{b}")


def test_runner_mock_mode_with_bench_nyuctf(tmp_path, monkeypatch):
    root = _mk_root(tmp_path, compose=False, files=("handout.txt",), handout_text="flag{x_y_z}")   # one handout: the mock solver cats every file
    monkeypatch.setenv("MODEL_NAME", "fake-model")
    out = tmp_path / "res" / "nyu.jsonl"
    rc = run_cybench.main(["--bench", "nyuctf", "--nyu-root", str(root), "--split", "test", "--tasks", "all", "--trials", "2",
                           "--parallel", "2", "--mock-solve", "--workroot", str(tmp_path / "w"), "--out", str(out)])
    assert rc == 0
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == 2 and all(r["solved"] for r in recs) and all(r["error"] is None for r in recs)
    assert recs[0]["task_id"] == "2021f-pwn-maze_runner" and recs[0]["display_name"].startswith("task-") and recs[0]["category"] == "pwn"
    summ = json.loads(out.with_suffix(".summary.json").read_text())
    assert summ["runs"] == 2 and summ["solves"] == 2 and summ["model"] == "fake-model"


def test_runner_skips_pre_excluded_tasks_unless_asked(tmp_path, monkeypatch, capsys):
    root = _mk_root(tmp_path, compose=False, tid="2019f-msc-alive", files=("handout.txt",), handout_text="flag{x_y_z}")
    monkeypatch.setenv("MODEL_NAME", "fake-model")
    out = tmp_path / "res" / "nyu.jsonl"
    try:
        run_cybench.main(["--bench", "nyuctf", "--nyu-root", str(root), "--tasks", "all", "--mock-solve", "--workroot", str(tmp_path / "w"), "--out", str(out)])
        assert False, "should exit with no tasks"
    except SystemExit as e:
        assert "no tasks" in str(e)
    rc = run_cybench.main(["--bench", "nyuctf", "--nyu-root", str(root), "--tasks", "all", "--include-excluded", "--mock-solve",
                           "--workroot", str(tmp_path / "w2"), "--out", str(out)])
    assert rc == 0 and len(out.read_text().splitlines()) == 1


def test_health_check_and_usable(tmp_path):
    (t,) = ny.enumerate_tasks(_mk_root(tmp_path, compose=True), split="test")
    r = ny.health_check(t, FakeDockerOps(), tmp_path / "h")
    assert r["provisioned_ok"] and r["services_started"] and r["health"] == {"pwn.chal.csaw.io:5000": True} and ny.usable(r) and r["n_handouts"] == 2
    r2 = ny.health_check(t, FakeDockerOps(healthy=False), tmp_path / "h2")
    assert not r2["provisioned_ok"] and r2["error"] and not ny.usable(r2)                # provision raises on a failed health check
    (tx,) = ny.enumerate_tasks(_mk_root(tmp_path / "x", tid="2023f-cry-nervcenter"), split="test")
    rx = ny.health_check(tx, FakeDockerOps(), tmp_path / "hx")
    assert rx["exclusion_reason"] and not ny.usable(rx) and not (tmp_path / "hx").exists()


def test_gather_sources_reads_reference_material_not_handouts(tmp_path):
    (t,) = ny.enumerate_tasks(_mk_root(tmp_path), split="test")
    src = ny.gather_sources(t)
    assert set(src) == {"challenge_prompt", "writeups", "solver_scripts"}
    assert "hello" not in json.dumps(src)                                                   # handout content never read
    assert "solution.md" in src["writeups"] and "solve.py" in src["solver_scripts"] and "test.sh" not in src["solver_scripts"]
    assert not cb.leaks_task_name(json.dumps(src), t)                                        # task name redacted everywhere


def test_retry_errors_honours_the_one_retry_ledger(tmp_path, monkeypatch):
    """2026-10-22: --retry-errors re-runs errored and missing pairs, except pairs listed in <out>.no_retry (already retried
    once): those keep their record, errored or not, so every errored pair gets exactly one retry."""
    from realbench.solve_adapter import SolveRecord
    root = _mk_root(tmp_path, compose=False)
    (base,) = ny.enumerate_tasks(root, split="test")
    mk = lambda tid: base.__class__(**{**base.__dict__, "id": tid})
    A, B, C, D = mk("2021f-rev-a"), mk("2021f-rev-b"), mk("2021f-rev-c"), mk("2021f-rev-d")
    monkeypatch.setattr(run_cybench, "load_bench", lambda args: (ny, [A, B, C, D]))
    monkeypatch.setenv("MODEL_NAME", "m")
    ran = []
    def fake(task, workroot, **kw):
        ran.append(task.id)
        return SolveRecord(task_id=task.id, display_name=task.display_name, category=task.category, needs_service=False, extra=dict(kw.get("extra") or {}))
    monkeypatch.setattr(run_cybench, "solve_cybench_task", fake)
    out = tmp_path / "s.jsonl"
    def rec(tid, err):
        d = SolveRecord(task_id=tid, display_name=tid, category="reverse", needs_service=False, extra={"trial": 0}).to_dict()
        d["error"] = err
        return json.dumps(d) + "\n"
    out.write_text(rec("2021f-rev-a", "llm: llm_bad_request") + rec("2021f-rev-b", "llm: llm_bad_request") + rec("2021f-rev-c", None))
    (tmp_path / "s.jsonl.no_retry").write_text("2021f-rev-a\t0\n")
    rc = run_cybench.main(["--bench", "nyuctf", "--nyu-root", str(root), "--tasks", "all", "--trials", "1", "--parallel", "1",
                           "--degenerate-after", "0", "--mock-solve", "--retry-errors", "--workroot", str(tmp_path / "w"), "--out", str(out)])
    assert rc == 0 and sorted(ran) == ["2021f-rev-b", "2021f-rev-d"]          # A already retried once; C clean
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert {(r["task_id"], bool(r.get("error"))) for r in lines} == {("2021f-rev-a", True), ("2021f-rev-b", False), ("2021f-rev-c", False), ("2021f-rev-d", False)}
