#!/usr/bin/env python3
"""realbench/run_cybench.py - run a solver over Cybench tasks (the Phase-3 bake-off and, with lessons, the memory
campaign). One container per solve, N trials per task, records to JSONL, summary with Wilson intervals.

    export MODEL_NAME=... VLLM_URL=http://127.0.0.1:8004/v1
    python -m realbench.run_cybench --tasks file-only --trials 3 --parallel 6 --out results/realbench/<model>/bakeoff.jsonl

`--tasks` = file-only | service | all | comma-separated task ids (or suffixes). `--dry-run` lists what would run.
Memory ON (the campaign): `--lessons results/realbench_mem/<model>/lessons.jsonl` retrieves `--lessons-m` lessons per solve through the Phase-2
pipeline, leave-one-task-out (`realbench.lessons`); `extra.memory` in each record logs the lessons, the candidate pool and the citations.
Mock: `--mock-solve` answers every task with a wrong flag through a fake session (plumbing test, no Docker, no model).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List

from realbench import cybench as cb
from realbench.solve_adapter import solve_cybench_task


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


LOCK_DIR = Path(os.environ.get("REALBENCH_LOCK_DIR", "results/_work/locks"))


class NamedLock:
    """A lock per name that also holds across processes (fcntl.flock on LOCK_DIR/<name>.lock), so two runners that
    share a machine (two arms run side by side, 2026-10-20) never start the same pinned container or alias at once."""

    def __init__(self, name: str):
        import threading
        self.path = LOCK_DIR / (re.sub(r"[^A-Za-z0-9_.-]", "_", name) + ".lock"); self._t = threading.Lock(); self._fh = None

    def __enter__(self):
        import fcntl
        self._t.acquire()
        try:
            LOCK_DIR.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a"); fcntl.flock(self._fh, fcntl.LOCK_EX)
        except BaseException:                 # never leave the thread lock held if open/flock fails (would hang the pool)
            if self._fh is not None:
                self._fh.close(); self._fh = None
            self._t.release()
            raise
        return self

    def __exit__(self, *exc):
        import fcntl
        fcntl.flock(self._fh, fcntl.LOCK_UN); self._fh.close(); self._fh = None; self._t.release()
        return False


def service_lock_keys(t, bench) -> List[str]:
    """What a service solve must hold while it runs.  On a shared network: every target hostname (aliases are shared
    across tasks).  With per-solve networks (nyuctf.PER_SOLVE_NETWORK): only names a compose file pins machine-wide
    (`container_name`, a named network) - the same task in two arms would otherwise remove the other's container."""
    if getattr(bench, "PER_SOLVE_NETWORK", False) and getattr(t, "compose_file", None):
        f = bench._compose_facts(t.compose_file)
        return sorted(set(f.get("container_names") or []) | set(f.get("network_names") or []))
    return sorted({h for h, _p in t.target_hosts}) or [t.id]


def interleave_jobs(jobs):
    """Within each trial: service jobs round-robin over their hostname groups (largest group first, so the group that
    bounds the makespan starts early), with the file-only jobs spread evenly between them.  Consecutive jobs then
    rarely share a hostname, so workers rarely block on a host lock.  Trial-major order is preserved."""
    out = []
    for k in sorted({k for _t, k in jobs}):
        svc = [(t, j) for t, j in jobs if j == k and t.needs_service]
        files = [(t, j) for t, j in jobs if j == k and not t.needs_service]
        groups: Dict[str, list] = defaultdict(list)
        for t, j in svc:
            groups[" ".join(sorted(h for h, _p in t.target_hosts)) or t.id].append((t, j))
        order = sorted(groups.values(), key=len, reverse=True)
        rr = []
        while any(order):
            for g in order:
                if g:
                    rr.append(g.pop(0))
        merged, fi = [], 0
        for i, s in enumerate(rr):
            merged.append(s)
            want = round(len(files) * (i + 1) / max(1, len(rr)))
            while fi < want:
                merged.append(files[fi]); fi += 1
        merged += files[fi:]
        out += merged
    return out


def select_tasks(tasks: List[cb.CybenchTask], spec: str) -> List[cb.CybenchTask]:
    if spec == "all":
        return list(tasks)
    if spec == "file-only":
        return [t for t in tasks if not t.needs_service]
    if spec == "service":
        return [t for t in tasks if t.needs_service]
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    # suffix matches need a separator boundary: NYU ids are dash-joined ("...-maze" must not match "...-ezmaze");
    # Cybench ids are path-shaped ("/[Very Easy] Dynastic")
    return [t for t in tasks if any(t.id == w or t.id.endswith("/" + w) or t.id.endswith("-" + w) or t.display_name == w for w in wanted)]


def add_bench_args(ap: argparse.ArgumentParser) -> None:
    """Benchmark selection shared by run_cybench / distill / learn."""
    ap.add_argument("--bench", default="cybench", choices=["cybench", "nyuctf"])
    ap.add_argument("--split", default="test", choices=["test", "development"], help="nyuctf only")
    ap.add_argument("--nyu-root", default=os.environ.get("NYUCTF_ROOT", "<path>"))
    ap.add_argument("--include-excluded", action="store_true", help="nyuctf: also run tasks with a pre-declared exclusion reason")


def add_harness_args(ap: argparse.ArgumentParser) -> None:
    """Harness options shared by run_cybench / learn (audit follow-up, 2026-10-19)."""
    ap.add_argument("--generic-tokens-from", default="", help="comma-separated memory-OFF records (JSONL); lesson tokens found in at least "
                    "--generic-min-chance of their traces never ground a citation (on top of realbench.lessons.generic_keywords)")
    ap.add_argument("--generic-min-chance", type=float, default=0.2)
    ap.add_argument("--egress", default="blocked", choices=["blocked", "open"],
                    help="blocked (default) = the task networks are Docker --internal: the solver reaches the task's services but not the "
                         "internet (agents searched GitHub/DuckDuckGo for write-ups and grepped results for the flag); open = the "
                         "pre-2026-10 protocol")


def load_generic_tokens(args, pool) -> List[str]:
    if not getattr(args, "generic_tokens_from", ""):
        return []
    from realbench.lessons import generic_keywords
    from solver.citation import chance_citeability
    traces = []
    for p in args.generic_tokens_from.split(","):
        traces += [(r.get("trace") or []) for r in map(json.loads, Path(p.strip()).read_text().splitlines()) if r.get("trace")]
    kws = [k for n in pool.nodes for k in (n.attributes.get("invocation_signature") or [])]
    chance = chance_citeability(kws, traces)
    return generic_keywords(pool.nodes, chance=chance, min_chance=args.generic_min_chance)


def check_workroot(workroot: str) -> None:
    """Audit F26: the agent directory lives under the workroot and its absolute path is visible to the model, so a workroot
    named after the arm (`..._mem/_work`, `..._static`) tells the solver which condition it is in. Warn loudly."""
    low = str(Path(workroot).resolve()).lower()
    hits = [w for w in ("mem", "static", "learned", "lesson", "bandit", "_off", "screen") if w in low]
    if hits:
        print(f"[harness] WARNING: workroot {workroot} contains {hits} - visible to the solver; use one neutral workroot for every arm", flush=True)


def load_bench(args):
    """(benchmark module, selected tasks).  The module exposes enumerate_tasks/provision/to_challenge_dict/check_flag/
    flag_match/SHARED_NET (+ gather_sources for distillation); every task exposes id/display_name/category/needs_service."""
    if getattr(args, "bench", "cybench") == "nyuctf":
        from realbench import nyuctf as bench
        tasks = bench.enumerate_tasks(Path(args.nyu_root), split=args.split)
        sel = select_tasks(tasks, args.tasks)
        if not getattr(args, "include_excluded", False):      # 2026-10-20: honour exclusions for an explicit id list too (es1337 host-root risk)
            dropped = [t for t in sel if t.exclusion_reason]
            sel = [t for t in sel if not t.exclusion_reason]
            if dropped:
                print(f"[load_bench] nyuctf {args.split}: {len(dropped)} pre-excluded task(s) skipped ({', '.join(t.id for t in dropped)}; --include-excluded to run them)", flush=True)
        return bench, sel
    from realbench import cybench as bench
    cz = getattr(args, "cyber_zero_root", "")
    tasks = bench.enumerate_tasks(Path(args.root), cyber_zero_root=(Path(cz) if cz else None))
    return bench, select_tasks(tasks, args.tasks)


def summarize(records: List[dict]) -> dict:
    by_task = {}
    for r in records:
        by_task.setdefault(r["task_id"], []).append(r)
    n = len(records); k = sum(1 for r in records if r["solved"]); p, lo, hi = wilson(k, n)
    tasks_solved = sum(1 for rs in by_task.values() if any(r["solved"] for r in rs))
    per_cat = {}
    for r in records:
        c = per_cat.setdefault(r["category"], [0, 0]); c[0] += 1; c[1] += int(r["solved"])
    mem = [r for r in records if (r.get("extra") or {}).get("memory")]
    sb = [(r.get("extra") or {}).get("sandbox") or {} for r in records]
    return {"runs": n, "solves": k, "solve_rate": p, "wilson95": [lo, hi], "tasks": len(by_task), "tasks_solved_at_least_once": tasks_solved,
            "sandbox": {"records_with_stats": sum(1 for s in sb if s), "dispatched": sum(s.get("dispatched", 0) for s in sb),
                        "container_runs": sum(s.get("container_runs", 0) for s in sb), "rejected": sum(s.get("rejected", 0) for s in sb)},
            "memory_runs": len(mem), "cited_runs": sum(1 for r in mem if r["extra"]["memory"].get("cited")),
            "same_category_lesson_runs": sum(1 for r in mem if r["category"] in (r["extra"]["memory"].get("lesson_categories") or [])),
            "errors": sum(1 for r in records if r.get("error")), "brace_only": sum(1 for r in records if r.get("flag_brace_content") and not r["solved"]),
            "mean_rounds": (sum(r["rounds_used"] for r in records) / n) if n else 0.0,
            "mean_wall_s": (sum(r["wall_s"] for r in records) / n) if n else 0.0,
            "per_category": {c: {"runs": v[0], "solves": v[1]} for c, v in sorted(per_cat.items())}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("CYBENCH_ROOT", "<path>"))
    ap.add_argument("--cyber-zero-root", default=os.environ.get("CYBER_ZERO_CYBENCH_ROOT", "<path>"))
    ap.add_argument("--tasks", default="file-only"); ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=4); ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--max-output-tokens", type=int, default=512); ap.add_argument("--context-window", type=int, default=8192)
    ap.add_argument("--trim-fraction", type=float, default=0.7); ap.add_argument("--command-timeout", type=float, default=60.0)
    ap.add_argument("--image", default=os.environ.get("REALBENCH_IMAGE", "realbench-solver:slim"))
    ap.add_argument("--container-memory", default="2g"); ap.add_argument("--container-cpus", default="2")
    ap.add_argument("--container-user", default="", help="'' = host uid:gid (slim image); 'root' for the Cybench agent image")
    ap.add_argument("--tool-calling", action="store_true", help="offer the actions as OpenAI tools (models that answer via function calls, e.g. gpt-oss)")
    ap.add_argument("--service-parallel", type=int, default=1, help="max service tasks in flight at once (their compose files publish fixed host ports and container names; 1 = serialize them)")
    ap.add_argument("--retry-errors", action="store_true", help="re-run only the (task, trial) pairs whose record carries an error (e.g. a provisioning failure), keeping the clean records")
    ap.add_argument("--degenerate-after", type=int, default=6, help="abort the model if its first N solves produced NO parseable action in any round (0 = never); 2026-09-16: a broken INT4 build emitted gibberish for 2.5 h")
    ap.add_argument("--workroot", default="results/_work/agents"); ap.add_argument("--out", required=True)
    ap.add_argument("--seed0", type=int, default=1000); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock-solve", action="store_true", help="plumbing test: fake session + fake solver, no Docker/model")
    ap.add_argument("--rag-context", default="", help="path to a text file injected as the lesson context (memory ON, one static block)")
    ap.add_argument("--lessons", default="", help="memory ON (campaign): lessons JSONL from realbench.distill; leave-one-task-out retrieval of --lessons-m per solve")
    ap.add_argument("--lessons-m", type=int, default=3); ap.add_argument("--embedder", default="sentence_transformers", choices=["sentence_transformers", "mock"])
    ap.add_argument("--reranker", default="cross_encoder", choices=["cross_encoder", "mock"]); ap.add_argument("--embed-cache", default="results/embed_cache")
    ap.add_argument("--fixed-lesson-order", action="store_true", help="render lessons in the label-neutral content order instead of the per-(task, seed) shuffle")
    ap.add_argument("--bandit", default="", help="memory ON-learned eval: a saved bandit (realbench.learn) - retrieve the greedy top-M by the LEARNED score (needs --lessons)")
    add_bench_args(ap)
    add_harness_args(ap)
    args = ap.parse_args(argv)
    bench, tasks = load_bench(args)
    if not tasks:
        sys.exit("no tasks selected")
    if args.dry_run:
        for t in tasks:
            print(f"{t.display_name}  {t.category:10s} service={t.needs_service}  {t.id}")
        print(f"{len(tasks)} tasks x {args.trials} trials = {len(tasks) * args.trials} solves"); return 0
    rag = Path(args.rag_context).read_text() if args.rag_context else ""
    provider = None
    generic_tokens: List[str] = []
    if args.lessons:
        from realbench.lessons import LessonPool, LeaveOneOutRetriever, build_embedder, build_reranker, make_context_provider
        pool = LessonPool.load(args.lessons)
        missing = [t.display_name for t in tasks if t.id not in pool.by_task]
        emb = build_embedder(args.embedder, cache_dir=args.embed_cache or None); rer = build_reranker(args.reranker)
        mode = "ON-learned (greedy top-M by learned bandit score)" if args.bandit else "ON-static (deterministic top-M)"
        print(f"[run_cybench] memory {mode}: {len(pool.nodes)} lessons from {args.lessons}, M={args.lessons_m}, embedder {args.embedder}, reranker {args.reranker}"
              + (f"; {len(missing)} selected tasks have no own lesson (fine: leave-one-out needs none): {missing}" if missing else ""), flush=True)
        if args.bandit:
            from realbench.learn import BanditMemory
            provider = BanditMemory.load(args.bandit, pool, emb, rer, m=args.lessons_m).eval_provider(randomize_order=not args.fixed_lesson_order)
        else:
            provider = make_context_provider(LeaveOneOutRetriever(pool, emb, rer, m=args.lessons_m), randomize_order=not args.fixed_lesson_order)
        generic_tokens = load_generic_tokens(args, pool)
        if generic_tokens:
            print(f"[run_cybench] citation: {len(generic_tokens)} generic tokens (rule + chance >= {args.generic_min_chance} on {args.generic_tokens_from})", flush=True)
    docker = None if args.mock_solve else cb.RealDockerOps(internal=(args.egress == "blocked"))
    session_factory = solve_impl = None
    if args.mock_solve:
        from tests.fakes_realbench import FakeSession, fake_solve   # type: ignore
        session_factory, solve_impl = FakeSession, fake_solve
    else:
        from solver import tool_executor
        tool_executor.require_sandbox()            # audit N1: no solver command may ever run on the host
        check_workroot(args.workroot)
        problem = docker.isolation_problem() if docker.internal else ""   # review 2026-10-20: never record an unenforced egress: blocked
        if problem:
            sys.exit(f"--egress blocked refused: {problem}")
        swept = docker.sweep_networks(bench.SHARED_NET + "-")   # 2026-10-20: remove empty per-solve networks a killed run leaked (else the address pool fills up)
        if swept:
            print(f"[run_cybench] swept {swept} empty {bench.SHARED_NET}-* networks (>= 10 min old) left by an earlier run", flush=True)
        docker.ensure_network(bench.SHARED_NET)
        if not docker.image_exists(args.image):
            sys.exit(f"image {args.image} missing - build it: docker build -t {args.image} realbench/docker")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    # trial-major order (2026-09-17): a task's trials must not provision concurrently - the same compose project and ports
    # collide when two trials of one service task start together (3 of 36 service runs errored at parallel 3)
    jobs = [(t, k) for k in range(args.trials) for t in tasks]
    if args.retry_errors:
        # resume: (task, trial) pairs that errored OR were never recorded are (re-)run; clean records are kept
        # (2026-09-23: a killed partial run used to lose its unrun tasks, since only errored pairs were re-run)
        # 2026-10-22: one retry per errored pair. Keys listed in <out>.no_retry ("task_id<TAB>trial", already retried once)
        # keep their record, errored or not, instead of being retried again.
        prior = [json.loads(l) for l in open(out) if l.strip()] if Path(out).exists() else []
        nr_path = Path(str(out) + ".no_retry")
        noretry = {(l.split("\t")[0], int(l.split("\t")[1])) for l in nr_path.read_text().splitlines() if l.strip()} if nr_path.exists() else set()
        key = lambda r: (r["task_id"], int(r["extra"].get("trial", 0)))
        bad = {key(r) for r in prior if r.get("error")} - noretry
        keep = [r for r in prior if not r.get("error") or key(r) in noretry]
        have = {key(r) for r in keep}
        jobs = [(t, k) for (t, k) in jobs if (t.id, k) in bad or (t.id, k) not in have]
        print(f"[run_cybench] retry-errors: {len(bad)} errored + {len(jobs) - len(bad)} missing (task, trial) pairs re-run; {len(keep)} records kept "
              f"({sum(1 for r in keep if r.get('error'))} errored, already retried once per {nr_path.name})", flush=True)
        Path(out).write_text("".join(json.dumps(r) + "\n" for r in keep)); records.extend(keep)
    jobs = interleave_jobs(jobs)
    print(f"[run_cybench] {args.bench}{'/' + args.split if args.bench == 'nyuctf' else ''}: {len(tasks)} tasks x {args.trials} trials, parallel {args.parallel}, model {os.environ.get('MODEL_NAME', '?')}, image {args.image}, "
          f"cmd timeout {args.command_timeout:g}s, container {args.container_memory}/{args.container_cpus}cpu/{args.container_user or 'host-uid'}, egress {args.egress}", flush=True)

    skw = {"memory": args.container_memory, "cpus": args.container_cpus}
    if args.container_user:
        skw["user"] = "0:0" if args.container_user == "root" else args.container_user

    import threading
    service_gate = threading.Semaphore(max(1, args.service_parallel))
    host_locks = defaultdict(lambda: None)       # 2026-09-23: at most one live service per hostname (shared aliases)
    _locks_guard = threading.Lock()

    def named_lock(h):
        with _locks_guard:
            if host_locks[h] is None:
                host_locks[h] = NamedLock(h)
            return host_locks[h]

    def one(t, k):
        if t.needs_service:
            # 2026-09-17: service tasks never overlap beyond --service-parallel (fixed ports/names then; ports are dropped
            # for NYU now).  2026-09-23: two tasks that share a hostname alias (pwn.chal.csaw.io x36) would resolve to the
            # wrong container -> a lock per hostname; tasks with different hostnames may run concurrently.
            # 2026-10-20: NYU service solves get their own network, so only pinned names are locked (service_lock_keys);
            # the locks hold across processes too.
            with ExitStack() as st:
                for h in service_lock_keys(t, bench):
                    st.enter_context(named_lock(h))
                with service_gate:
                    return _one(t, k)
        return _one(t, k)

    def _one(t, k):
        return solve_cybench_task(t, args.workroot, docker=docker, image=args.image, solve_impl=solve_impl, session_factory=session_factory, session_kwargs=skw, tool_calling=args.tool_calling,
                                  injected_rag_context=rag, max_rounds=args.max_rounds, max_output_tokens=args.max_output_tokens,
                                  context_window=args.context_window, trim_fraction=args.trim_fraction, llm_seed=args.seed0 + k,
                                  command_timeout_s=args.command_timeout, context_provider=provider, bench=bench,
                                  citation_generic_tokens=generic_tokens,
                                  extra={"trial": k, "model": os.environ.get("MODEL_NAME", ""), "egress": args.egress, "condition": ("learned" if (provider is not None and args.bandit) else ("on" if provider is not None else ("static" if rag else "off")))})
    # 2026-09-22: with a mix of service and file-only jobs, a single pool of `parallel` workers gated by the semaphore leaves
    # (parallel - service_parallel) workers blocked whenever service jobs are queued (NYU: 122 service vs 57 file-only tasks).
    # Two pools instead: service jobs get their own `service_parallel` workers, file-only jobs the rest; total concurrency
    # is unchanged. A single pool (+ gate) remains when there is only one kind of job or parallel <= service_parallel.
    n_svc = max(1, args.service_parallel)
    svc_jobs = [(t, k) for t, k in jobs if t.needs_service]; file_jobs = [(t, k) for t, k in jobs if not t.needs_service]
    if svc_jobs and file_jobs and args.parallel > n_svc:
        pools = [(n_svc, svc_jobs), (args.parallel - n_svc, file_jobs)]
    else:
        pools = [(max(1, args.parallel), jobs)]
    t0 = time.time(); degenerate = False
    with ExitStack() as stack, open(out, "a") as fh:
        futs = []
        for n_workers, pool_jobs in pools:
            ex = stack.enter_context(ThreadPoolExecutor(max_workers=n_workers))
            futs += [ex.submit(one, t, k) for t, k in pool_jobs]
        for fut in as_completed(futs):
            rec = fut.result().to_dict(); records.append(rec)
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if args.degenerate_after and len(records) == args.degenerate_after and not any(
                    t.get("action") not in (None, "parse_error", "error") for r in records for t in (r.get("trace") or [])):
                degenerate = True
                print(f"[run_cybench] DEGENERATE: no parseable action in any round of the first {len(records)} solves - aborting this model", flush=True)
                for f in futs:
                    f.cancel()
                break
            mem = rec["extra"].get("memory") or {}
            print(f"  {rec['display_name']:14s} trial {rec['extra'].get('trial')}: {'SOLVED' if rec['solved'] else 'no'}"
                  f"{' brace-only' if rec['flag_brace_content'] and not rec['solved'] else ''} rounds {rec['rounds_used']} "
                  f"{rec['wall_s']:.0f}s{(' ERROR ' + rec['error']) if rec.get('error') else ''}"
                  f"{(' mem ' + ','.join(l.split('::')[-1] for l in mem.get('lessons', [])) + ' cited ' + str(len(mem.get('cited', [])))) if mem else ''}", flush=True)
    summ = summarize(records); summ["elapsed_s"] = time.time() - t0; summ["model"] = os.environ.get("MODEL_NAME", ""); summ["degenerate"] = degenerate
    summ["egress"] = args.egress
    if not args.mock_solve:
        from solver import tool_executor
        summ["sandbox"]["violations_no_backend"] = tool_executor.sandbox_violations()
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "per_category"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
