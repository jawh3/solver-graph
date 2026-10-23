"""ON-learned arm without a model/Docker/GPU: the bandit credit update, that training raises the value of a lesson whose
use correlates with solving, greedy eval selection, save/load, resumable training, and the --bandit runner path."""
import json
from pathlib import Path

from realbench import cybench as cb
from realbench import lessons as L
from realbench import learn as LN
from realbench import run_cybench
from tests.fakes_realbench import FakeSession
from tests.test_realbench_cybench import _mk_task


GOOD = "goodtok"


def _lesson(task_id, dn, sig, cat="misc", title=None):
    # the signature token appears in the RECIPE (as in a real distilled lesson) so it renders into the injected context;
    # render_node only surfaces `invocation_signature` for TOOL nodes, and these are TECHNIQUE nodes.
    return {"schema": 1, "task_id": task_id, "display_name": dn, "task_category": cat, "title": title or f"lesson {dn}", "technique": "t",
            "category": cat, "recognise": ["x"], "recipe": [f"run {sig[0]} to proceed", "then finish"], "invocation_signature": list(sig),
            "applicability": {"categories": [cat], "preconditions": [], "artifacts_required": []}, "pitfalls": [], "model": "fake"}


def _pool_and_tasks(tmp_path):
    """5 tasks A-E under ONE root (so the runner can re-enumerate), each with its own distractor lesson (sig bad0..bad4),
    plus one shared GOOD lesson (sig goodtok)."""
    tasks = [_mk_task(tmp_path, name=f"[Very Easy] Task{i}", flag="HTB{win}") for i in range(5)]
    lessons = [_lesson(t.id, t.display_name, [f"bad{i}"]) for i, t in enumerate(tasks)]
    lessons.append(_lesson("ghost", "task-90000000", [GOOD], title="the good lesson"))
    return L.LessonPool(lessons), tasks


def _fake_solve(challenge, **kw):
    """The GOOD lesson helps 70% of the time it is present (deterministic in (task, seed)); nothing else ever solves.
    The 70% (not 100%) keeps the reward baseline below 1 so the learned advantage of the good lesson is non-zero - on
    real Cybench base solve rates are low, so this never saturates; a deterministic-always fake would.  GOOD is executed
    against (`echo goodtok`, so it is cited) only on the runs it actually solves."""
    import hashlib
    ctx = kw.get("injected_rag_context", "") or ""
    seed = kw.get("llm_seed", 0) or 0
    hit = int(hashlib.sha1(f"{challenge.get('name')}|{seed}".encode()).hexdigest(), 16) % 10 < 7
    solved = (GOOD in ctx) and hit
    cmds = ["echo goodtok"] if solved else ["ls -la"]
    return {"flag": "HTB{win}" if solved else None, "rounds_used": 2, "total_commands": len(cmds),
            "trace": [{"round": 1, "commands": cmds}], "gave_up_reason": None if solved else "rounds_exhausted"}


def _mem(pool):
    return LN.BanditMemory(pool, L.build_embedder("mock"), L.build_reranker("mock"), m=3)


def _rec(solved, lessons, cited):
    class Rec:
        pass
    r = Rec(); r.solved = solved; r.rounds_used = 2
    r.extra = {"memory": {"context_node_id": "ctx::cat::misc", "lessons": lessons, "cited": cited,
                          "sampled_cosines": {n: 0.6 for n in lessons}}}
    return r


def test_training_gates_service_solves_per_hostname(tmp_path, monkeypatch):
    """Two dev tasks on the SAME hostname never solve concurrently; different hostnames may (service_parallel 2)."""
    import threading, time as _time
    import realbench.solve_adapter as sa
    from realbench.solve_adapter import SolveRecord
    a = _mk_task(tmp_path, name="[Very Easy] SvcA", flag="HTB{w}", target_host="pwn.chal:5000")
    b = _mk_task(tmp_path, name="[Very Easy] SvcB", flag="HTB{w}", target_host="pwn.chal:5000")
    c = _mk_task(tmp_path, name="[Very Easy] SvcC", flag="HTB{w}", target_host="web.chal:80")
    assert a.needs_service and c.needs_service
    live = {}; peak = {}; lock = threading.Lock()
    def slow(task, workroot, **kw):
        h = task.target_hosts[0][0]
        with lock:
            live[h] = live.get(h, 0) + 1; live["all"] = live.get("all", 0) + 1
            peak[h] = max(peak.get(h, 0), live[h]); peak["all"] = max(peak.get("all", 0), live["all"])
        _time.sleep(0.15)
        with lock: live[h] -= 1; live["all"] -= 1
        return SolveRecord(task_id=task.id, display_name=task.display_name, category=task.category, needs_service=True, error="fake: no update", extra=dict(kw.get("extra") or {}))
    monkeypatch.setattr(sa, "solve_cybench_task", slow)
    pool = L.LessonPool([_lesson("ghost", "task-90000000", [GOOD])]); mem = _mem(pool)
    summ = LN.train(mem, [a, b, c], docker=None, image="x", workroot=str(tmp_path / "w"), seeds=[2000, 2001], epochs=1, parallel=3, service_parallel=2,
                    bandit_path=str(tmp_path / "b.json"), done_path=str(tmp_path / "done.jsonl"), trainlog_path=str(tmp_path / "t.jsonl"), solve_kwargs={}, log=lambda s: None)
    assert summ["train_solves"] == 6 and peak["pwn.chal"] == 1 and peak["all"] == 2
    assert len((tmp_path / "done.jsonl").read_text().splitlines()) == 6


def test_update_credits_cited_lesson_and_exposes_uncited(tmp_path):
    pool, tasks = _pool_and_tasks(tmp_path)
    mem = _mem(pool)
    G = "lesson::task-90000000"; D = "lesson::" + tasks[1].display_name
    # one update: both incremented; the uncited node's credit is exactly the context baseline (advantage 0)
    log = mem.update(_rec(True, [G, D], [G]), tasks[0])
    assert log["solved"] and log["R"] > 0
    ups = {u["node"]: u for u in log["updates"]}
    assert ups[G]["cited"] and not ups[D]["cited"]
    assert abs(ups[D]["reward"] - mem.bandit.baseline("ctx::cat::misc")) < 1e-9   # uncited -> exposure at the baseline
    assert mem.bandit.cells["ctx::cat::misc"][G].n == 1 and mem.bandit.cells["ctx::cat::misc"][D].n == 1
    # over mixed solved/unsolved runs the baseline settles below 1, so the cited-on-solve lesson climbs above an exposure-only one
    mem.update(_rec(False, [G, D], []), tasks[2])           # an unsolved run pulls the baseline down
    for t in (tasks[3], tasks[4], tasks[0]):
        mem.update(_rec(True, [G, D], [G]), t)              # good is cited on solves; D is never cited (exposure only)
    g = mem.bandit.cells["ctx::cat::misc"][G]; d = mem.bandit.cells["ctx::cat::misc"][D]
    assert g.Q > d.Q + 0.02                                 # learned: the cited-useful lesson is worth more than the exposure-only one


def test_training_raises_the_good_lesson_and_greedy_eval_picks_it(tmp_path):
    pool, tasks = _pool_and_tasks(tmp_path)
    mem = _mem(pool)
    skw = {"session_factory": FakeSession, "solve_impl": _fake_solve}
    base = tmp_path / "bandit"
    summ = LN.train(mem, tasks, docker=None, image="x", workroot=str(tmp_path / "w"), seeds=[2000, 2001], epochs=3,
                    parallel=2, service_parallel=1, bandit_path=str(base) + ".json",
                    done_path=str(base) + ".done.jsonl", trainlog_path=str(base) + ".log.jsonl", solve_kwargs=skw, log=lambda *_: None)
    cell = mem.bandit.cells["ctx::cat::misc"]
    good = cell["lesson::task-90000000"]
    distractors = [c.Q for nid, c in cell.items() if nid != "lesson::task-90000000"]
    assert good.n > 0 and good.Q > max(distractors)        # the good lesson learned the highest value
    # greedy eval retrieves the good lesson for every task (leave-one-out keeps it, it is no task's own)
    prov = mem.eval_provider()
    for t in tasks:
        shown = {"name": t.display_name, "category": "misc", "description": "solve it"}
        _ctx, nodes, meta = prov(shown, t, 1000)
        assert "lesson::task-90000000" in [n.id for n in nodes] and GOOD in _ctx
    assert summ["train_solves"] == 30


def test_learned_eval_beats_untrained_through_the_runner(tmp_path, monkeypatch):
    pool, tasks = _pool_and_tasks(tmp_path)
    (tmp_path / "task_list.txt").write_text("\n".join(t.id for t in tasks) + "\n")
    monkeypatch.setenv("MODEL_NAME", "fake")
    lessons = tmp_path / "lessons.jsonl"; lessons.write_text("\n".join(json.dumps(l) for l in pool.lessons) + "\n")
    # train a bandit and save it
    mem = _mem(pool)
    base = tmp_path / "bandit"
    LN.train(mem, tasks, docker=None, image="x", workroot=str(tmp_path / "w"), seeds=[2000, 2001], epochs=3, parallel=2, service_parallel=1,
             bandit_path=str(base) + ".json", done_path=str(base) + ".done.jsonl", trainlog_path=str(base) + ".log.jsonl",
             solve_kwargs={"session_factory": FakeSession, "solve_impl": _fake_solve}, log=lambda *_: None)
    # eval through run_cybench with --bandit, mock-solve path (the runner imports fake_solve from fakes_realbench; patch it to ours)
    import realbench.run_cybench as rc
    import tests.fakes_realbench as fr
    monkeypatch.setattr(fr, "fake_solve", _fake_solve)
    out = tmp_path / "on_learned.jsonl"
    rc.main(["--root", str(tmp_path), "--cyber-zero-root", "", "--tasks", "all", "--trials", "1", "--parallel", "2", "--mock-solve",
             "--workroot", str(tmp_path / "we"), "--out", str(out), "--lessons", str(lessons), "--bandit", str(base) + ".json",
             "--embedder", "mock", "--reranker", "mock", "--embed-cache", ""])
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert recs and all(r["extra"]["condition"] == "learned" for r in recs)
    # learning made the greedy eval put the good lesson in front of every task (it is the only lesson that ever solves)
    assert all("lesson::task-90000000" in r["extra"]["memory"]["lessons"] for r in recs)
    assert sum(r["solved"] for r in recs) >= 3            # good is retrieved everywhere -> most solve (70% hit rate)


def test_save_load_roundtrips_and_training_resumes(tmp_path):
    pool, tasks = _pool_and_tasks(tmp_path)
    mem = _mem(pool)
    base = tmp_path / "bandit"
    skw = {"session_factory": FakeSession, "solve_impl": _fake_solve}
    # first pass: 1 epoch
    LN.train(mem, tasks, docker=None, image="x", workroot=str(tmp_path / "w"), seeds=[2000], epochs=1, parallel=2, service_parallel=1,
             bandit_path=str(base) + ".json", done_path=str(base) + ".done.jsonl", trainlog_path=str(base) + ".log.jsonl", solve_kwargs=skw, log=lambda *_: None)
    n_done_1 = len((Path(str(base) + ".done.jsonl")).read_text().splitlines())
    assert n_done_1 == 5
    saved = json.loads((Path(str(base) + ".json")).read_text())
    mem2 = LN.BanditMemory.load(str(base) + ".json", pool, L.build_embedder("mock"), L.build_reranker("mock"))
    assert mem2.bandit.to_dict()["cells"] == saved["bandit"]["cells"]
    # resume: a fresh BanditMemory over the same paths should skip the done steps and only run the new seed
    mem3 = _mem(pool)
    LN.train(mem3, tasks, docker=None, image="x", workroot=str(tmp_path / "w2"), seeds=[2000, 2001], epochs=1, parallel=2, service_parallel=1,
             bandit_path=str(base) + ".json", done_path=str(base) + ".done.jsonl", trainlog_path=str(base) + ".log.jsonl", solve_kwargs=skw, log=lambda *_: None)
    n_done_2 = len((Path(str(base) + ".done.jsonl")).read_text().splitlines())
    assert n_done_2 == 10                                  # 5 already done + 5 new (seed 2001), not 15
