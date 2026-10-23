"""Memory campaign plumbing without a model, Docker or a GPU: distiller hygiene, leave-one-task-out retrieval, the
adapter/runner memory path and the paired analysis."""
import json
from pathlib import Path
from types import SimpleNamespace

from realbench import cybench as cb
from realbench import lessons as L
from realbench import memory_analysis as MA
from realbench import run_cybench
from realbench.solve_adapter import solve_cybench_task
from tests.fakes_realbench import FakeSession
from tests.test_realbench_cybench import _mk_task


def _lesson(task_id, dn, cat="misc", title="Read the handout with python", sig=("long_to_bytes", "pow(e, -1"), model="fake"):
    return {"schema": 1, "task_id": task_id, "display_name": dn, "task_category": cat, "title": title, "technique": "rsa-prime-modulus",
            "category": cat, "recognise": ["n is prime"], "recipe": ["read n,e,c", "phi = n-1", "d = pow(e,-1,phi)"],
            "invocation_signature": list(sig), "applicability": {"categories": [cat], "preconditions": ["isPrime(n)"], "artifacts_required": ["output.txt"]},
            "pitfalls": [], "scrubbed": 0, "name_redacted": False, "model": model}


def test_scrub_flags_known_prefixes_only():
    text = "flag HTB{abc_1} and hkcert22{x-y} and exact SEKAI{real} but LaTeX c_{i,0} \\begin{align} stay"
    out, n = L.scrub_flags(text, ["SEKAI{real}"], ["HTB", "hkcert22"])
    assert n == 3 and "HTB{" not in out and "hkcert22{" not in out and "SEKAI{real}" not in out
    assert "c_{i,0}" in out and "\\begin{align}" in out and out.count("[FLAG]") == 3


def test_extract_json_handles_think_fences_and_orphans():
    raw = "<think>reasoning {not json}</think>```json\n{\"title\": \"T\", \"recipe\": [\"a\", \"b\"], \"x\": {\"y\": 1}}\n```"
    assert L._extract_json_object(raw)["title"] == "T"
    assert L._extract_json_object("thinking ... </think> here: {\"a\": \"line\nbreak\"} tail")["a"] == "line\nbreak"
    assert L._extract_json_object("no object here") is None and L._extract_json_object("{broken") is None


def test_validate_lesson_scrubs_flag_and_task_name(tmp_path):
    t = _mk_task(tmp_path, name="[Very Easy] Synthetic Thing", flag="HTB{s0me_fl4g}")
    raw = {"title": "Solve Synthetic Thing by reading HTB{s0me_fl4g}", "technique": "Read File", "category": "crypto",
           "recipe": ["cat the file", "print HTB{s0me_fl4g}"], "invocation_signature": ["cat", "strings", "HTB{s0me_fl4g}"],
           "applicability": {"categories": ["crypto", "bogus"], "preconditions": ["a text file"]}, "pitfalls": []}
    les, problems = L.validate_lesson(raw, t, ["HTB{s0me_fl4g}"], ["HTB"])
    assert problems == [] and les["scrubbed"] >= 2 and les["name_redacted"]
    from solver.context_assembly import render_node
    shown = render_node(L.lesson_node(les))                                        # what the solver would see
    assert "HTB{" not in shown and not cb.leaks_task_name(shown, t) and "[redacted]" in les["title"] and "HTB{" not in json.dumps(les)
    assert les["invocation_signature"] == ["cat", "strings"] and les["applicability"]["categories"] == ["crypto"]
    assert les["category"] == "crypto" and les["technique"] == "read-file"
    bad, problems = L.validate_lesson({"title": "x", "recipe": ["one"], "invocation_signature": ["a"]}, t, [], [])
    assert bad is None and len(problems) == 2


class _FakeClient:
    """OpenAI-shaped client: replies from a queue; records the prompts it saw."""
    def __init__(self, replies):
        self.replies = list(replies); self.seen = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.seen.append(kw)
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_distill_retries_then_stores_a_clean_lesson(tmp_path):
    t = _mk_task(tmp_path, name="[Very Easy] Synthetic Thing", flag="HTB{s0me_fl4g}")
    good = json.dumps({"title": "Prime RSA modulus", "technique": "rsa-prime", "category": "misc", "recognise": ["n prime"],
                       "recipe": ["read", "phi=n-1", "decrypt"], "invocation_signature": ["long_to_bytes", "isPrime"],
                       "applicability": {"categories": ["crypto"], "preconditions": [], "artifacts_required": []}, "pitfalls": []})
    client = _FakeClient(["<think>hm</think>not json at all", good])
    r = L.distill_task(t, client, "fake-model", known_flags=["HTB{s0me_fl4g}"], prefixes=["HTB"], max_attempts=3)
    assert r.lesson is not None and r.attempts == 2 and r.lesson["title"] == "Prime RSA modulus"
    first_prompt = client.seen[0]["messages"][1]["content"]
    assert "HTB{s0me_fl4g}" not in first_prompt and "[FLAG]" in first_prompt          # README carries the flag -> scrubbed
    assert not cb.leaks_task_name(first_prompt, t)                                    # hard_prompt names the task -> redacted
    assert client.seen[1]["messages"][2]["role"] == "assistant" and "not usable" in client.seen[1]["messages"][3]["content"]
    out = tmp_path / "lessons.jsonl"
    summ = L.distill_all([t], _FakeClient([good]), "fake-model", out, parallel=1, log=lambda *_: None)
    assert summ["n_ok"] == 1 and L.LessonPool.load(out).nodes[0].attributes["_task_id"] == t.id
    summ2 = L.distill_all([t], _FakeClient([]), "fake-model", out, parallel=1, log=lambda *_: None)     # resumable: no call made
    assert summ2["n_ok"] == 1
    gave_up = L.distill_task(t, _FakeClient(["x", "y", "z"]), "m", known_flags=[], prefixes=[], max_attempts=3)
    assert gave_up.lesson is None and gave_up.attempts == 3


def test_leave_one_out_retrieval_is_deterministic_and_never_returns_own_lesson():
    pool = L.LessonPool([_lesson(f"t{i}", f"task-{i:08d}", cat=("crypto" if i % 2 else "web"), title=f"lesson {i} about {'rsa' if i % 2 else 'sqli'}") for i in range(6)])
    retr = L.LeaveOneOutRetriever(pool, L.build_embedder("mock"), L.build_reranker("mock"), m=3)
    shown = {"name": "task-00000001", "category": "crypto", "description": "capture the flag from output.txt and source.py rsa"}
    nodes, meta = retr.retrieve(shown, "t1")
    assert len(nodes) == 3 and all(n.attributes["_task_id"] != "t1" for n in nodes) and meta["pool_size"] == 5 and meta["excluded"]
    nodes2, _ = retr.retrieve(shown, "t1")
    assert [n.id for n in nodes] == [n.id for n in nodes2]                          # deterministic
    assert meta["policy"]["deterministic"] and meta["context_node_id"] == "ctx::cat::crypto"
    small = L.LeaveOneOutRetriever(L.LessonPool(pool.lessons[:2]), L.build_embedder("mock"), L.build_reranker("mock"), m=3)
    assert len(small.retrieve(shown, "t0")[0]) == 1                                  # pool of 2 minus own = 1
    provider = L.make_context_provider(retr)
    ctx, ns, m = provider(shown, SimpleNamespace(id="t1"), 1000)
    assert ctx.startswith("Retrieved lessons") and len(ns) == 3 and m["lessons"] == [n.id for n in ns] and m["context_chars"] == len(ctx)
    ctx_b, _, _ = provider(shown, SimpleNamespace(id="t1"), 1000)
    assert ctx_b == ctx                                                              # same (task, seed) -> same rendered block


def test_provider_through_adapter_logs_lessons_and_citations(tmp_path):
    t = _mk_task(tmp_path, flag="HTB{s0me_fl4g}", handout_text="HTB{s0me_fl4g}")
    pool = L.LessonPool([_lesson(t.id, t.display_name, title="own lesson"),
                         _lesson("other-1", "task-aaaaaaaa", title="other lesson one", sig=("long_to_bytes", "isPrime")),
                         _lesson("other-2", "task-bbbbbbbb", title="other lesson two", sig=("binwalk -e", "foremost"))])
    provider = L.make_context_provider(L.LeaveOneOutRetriever(pool, L.build_embedder("mock"), L.build_reranker("mock"), m=3))
    seen = {}

    def solve(challenge, **kw):
        seen["ctx"] = kw.get("injected_rag_context", "")
        return {"flag": "HTB{s0me_fl4g}", "rounds_used": 2, "total_commands": 2,
                "trace": [{"round": 1, "commands": ["python3 -c 'from Crypto.Util.number import long_to_bytes'"]}, {"round": 2, "commands": ["ls"]}]}
    rec = solve_cybench_task(t, tmp_path / "work", docker=None, session_factory=FakeSession, solve_impl=solve, llm_seed=1000, context_provider=provider)
    assert rec.error is None and rec.solved
    mem = rec.extra["memory"]
    assert set(mem["lessons"]) == {"lesson::task-aaaaaaaa", "lesson::task-bbbbbbbb"}       # own lesson excluded
    assert mem["cited"] == ["lesson::task-aaaaaaaa"] and "long_to_bytes" in mem["citation_evidence"]["lesson::task-aaaaaaaa"]
    assert "other lesson one" in seen["ctx"] and "own lesson" not in seen["ctx"]


def test_runner_lessons_mode_end_to_end(tmp_path, monkeypatch):
    t = _mk_task(tmp_path, flag="HTB{s0me_fl4g}", handout_text="HTB{s0me_fl4g}")
    (tmp_path / "task_list.txt").write_text(t.id + "\n"); monkeypatch.setenv("MODEL_NAME", "fake-model")
    lessons = tmp_path / "lessons.jsonl"
    lessons.write_text("\n".join(json.dumps(l) for l in [_lesson(t.id, t.display_name), _lesson("o1", "task-aaaaaaaa", sig=("cat", "handout.txt"))]) + "\n")
    out = tmp_path / "res" / "on.jsonl"
    rc = run_cybench.main(["--root", str(tmp_path), "--cyber-zero-root", "", "--tasks", "all", "--trials", "2", "--parallel", "2", "--mock-solve",
                           "--workroot", str(tmp_path / "w"), "--out", str(out), "--lessons", str(lessons), "--embedder", "mock", "--reranker", "mock",
                           "--embed-cache", ""])
    assert rc == 0
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == 2 and all(r["extra"]["condition"] == "on" for r in recs)
    assert all(r["extra"]["memory"]["lessons"] == ["lesson::task-aaaaaaaa"] for r in recs)
    summ = json.loads(out.with_suffix(".summary.json").read_text())
    assert summ["memory_runs"] == 2 and summ["same_category_lesson_runs"] == 2 and summ["solves"] == 2


def test_memory_analysis_paired_rd_and_markdown(tmp_path):
    def rec(tid, trial, solved, cat="crypto", mem=None):
        return {"task_id": tid, "category": cat, "solved": solved, "rounds_used": 3, "error": None,
                "extra": {"trial": trial, **({"memory": mem} if mem else {})}}
    off = [rec("A", 0, 0), rec("A", 1, 0), rec("B", 0, 1), rec("B", 1, 0), rec("C", 0, 0, "web"), rec("C", 1, 0, "web"), rec("C", 2, 0, "web")]
    on = [rec("A", 0, 1, mem={"lesson_categories": ["crypto"], "cited": ["x"]}), rec("A", 1, 1, mem={"lesson_categories": ["crypto"], "cited": []}),
          rec("B", 0, 1, mem={"lesson_categories": ["web"], "cited": []}), rec("B", 1, 1, mem={"lesson_categories": ["web"], "cited": []}),
          rec("C", 0, 0, "web", mem={"lesson_categories": ["crypto"], "cited": []}), rec("C", 1, 0, "web", mem={"lesson_categories": ["crypto"], "cited": []}),
          rec("C", 9, 1, "web", mem={})]      # trial 9 has no OFF partner -> not paired
    roff, ron = tmp_path / "off" / "m1", tmp_path / "on" / "m1"; roff.mkdir(parents=True); ron.mkdir(parents=True)
    (roff / "bakeoff.jsonl").write_text("\n".join(json.dumps(r) for r in off) + "\n"); (ron / "on_static.jsonl").write_text("\n".join(json.dumps(r) for r in on) + "\n")
    rows = MA.pair(MA.load(roff / "bakeoff.jsonl"), MA.load(ron / "on_static.jsonl"))
    assert len(rows) == 6 and sum(r["off"] for r in rows) == 1 and sum(r["on"] for r in rows) == 4
    c = MA.cell(rows)
    assert abs(c["rd"] - 0.5) < 1e-9 and c["rd_ci"][0] <= 0.5 <= c["rd_ci"][1] and c["on_only"] == 3 and c["off_only"] == 0 and c["tasks_pos"] == 2
    out = tmp_path / "report.md"
    rc = MA.main(["--root-off", str(tmp_path / "off"), "--root-on", str(tmp_path / "on"), "--no-contamination", "--out", str(out), "--json", str(tmp_path / "r.json")])
    assert rc == 0 and "| m1 |" in out.read_text() and "same_category_lesson" in out.read_text()
    j = json.loads((tmp_path / "r.json").read_text()); assert j["per_model"]["m1"]["all"]["pairs"] == 6 and j["pooled"] is None
