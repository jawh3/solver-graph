"""realbench/solve_adapter.py - one Cybench solve attempt end to end (Phase 3 / Tier B).

    provision(task) -> ContainerSession on the task's network (only the agent dir is mounted, at its host path)
      -> solve_challenge(neutral name, no private fields, flag_format from answer_format, flag_checker = ground truth,
                         working_dir = agent dir; every shell command runs inside the container)
      -> SolveRecord (exact flag match is the only solve; brace-content matches are reported, never counted)
      -> teardown (services stopped, work dir removed, source verified).

Mock + real duality: `docker` (realbench.cybench.DockerOps), `session_factory` and `solve_impl` are injectable, so the
plumbing is tested without Docker or a model. Lessons are injected through `injected_rag_context` exactly as in Phase 2.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from realbench import cybench as cb
from realbench.cybench import CybenchTask, DockerOps, Provisioned, check_flag, flag_match, provision, to_challenge_dict
from solver import tool_executor
from solver.container_executor import DEFAULT_IMAGE, ContainerSession


@dataclass
class SolveRecord:
    task_id: str
    display_name: str
    category: str
    needs_service: bool
    solved: bool = False
    flag_returned: Optional[str] = None
    flag_exact: bool = False
    flag_brace_content: bool = False
    rounds_used: int = 0
    total_commands: int = 0
    gave_up_reason: Optional[str] = None
    error: Optional[str] = None
    wall_s: float = 0.0
    solve_wall_s: float = 0.0
    commands_run: int = 0
    command_timeouts: int = 0
    handout_changes: List[str] = field(default_factory=list)
    provision_warnings: List[str] = field(default_factory=list)
    blocked_init_commands: List[str] = field(default_factory=list)
    host_aliases: Dict[str, str] = field(default_factory=dict)
    service_health: Dict[str, bool] = field(default_factory=dict)
    name_leaks: List[str] = field(default_factory=list)
    flag_leaks: List[str] = field(default_factory=list)
    container: str = ""
    llm_seed: Optional[int] = None
    trace: Optional[List[dict]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def solve_cybench_task(task: CybenchTask, workroot: str | Path, *, docker: Optional[DockerOps] = None,
                       image: str = DEFAULT_IMAGE, solve_impl: Optional[Callable] = None,
                       session_factory: Optional[Callable] = None, injected_rag_context: str = "",
                       max_rounds: int = 8, max_output_tokens: int = 512, context_window: int = 8192,
                       trim_fraction: float = 0.7, llm_seed: Optional[int] = None, service_timeout_s: float = 180.0,
                       command_timeout_s: float = 60.0, keep_trace: bool = True, log=None,
                       extra: Optional[Dict[str, Any]] = None, session_kwargs: Optional[Dict[str, Any]] = None,
                       tool_calling: bool = False, context_provider: Optional[Callable] = None, bench=None,
                       citation_generic_tokens: Optional[Sequence[str]] = None) -> SolveRecord:
    """`session_kwargs` reach the container session (memory, cpus, user, ...). When the container runs as root
    (the full Cybench agent image expects it), files it creates in the agent dir are re-owned to the host user
    before verification and teardown.
    `context_provider(shown_challenge, task, llm_seed) -> (context_str, nodes, meta)` (memory ON, `realbench.lessons`):
    the rendered lessons replace `injected_rag_context`, `meta` is logged under `extra["memory"]`, and after the solve the
    action-grounded citation matcher (mode "distinctive", literal tokens, minus `realbench.lessons.generic_keywords` and
    any `citation_generic_tokens`, e.g. the high-chance tokens of a memory-OFF run) marks which of the injected nodes
    were executed against.

    2026-10-19 (audit N1, F11): `extra["sandbox"]` records the commands dispatched to the container, the commands the
    container ran and any command blocked for being outside it; a solve where they disagree, or one that ended on an LLM
    error (`llm_bad_request` = e.g. a context overflow, `llm_error` = the server gave up), carries `error` and is re-run by
    `--retry-errors` / excluded from paired analysis."""
    if bench is None:
        from realbench import cybench as bench                   # default benchmark module; realbench.nyuctf plugs in the same way
    t0 = time.monotonic()
    rec = SolveRecord(task_id=task.id, display_name=task.display_name, category=task.category,
                      needs_service=task.needs_service, llm_seed=llm_seed, extra=dict(extra or {}))
    workdir = Path(workroot) / f"{task.display_name}-{uuid.uuid4().hex[:6]}"
    prov: Optional[Provisioned] = None
    session = None
    try:
        try:
            prov = bench.provision(task, workdir, docker=docker, service_timeout_s=service_timeout_s)
        except Exception as e1:  # noqa - one retry after a pause: service starts collide when trials of a task overlap
            import shutil as _sh
            _sh.rmtree(workdir, ignore_errors=True); time.sleep(20)
            workdir = Path(workroot) / f"{task.display_name}-{uuid.uuid4().hex[:6]}"
            rec.provision_warnings.append(f"provision retried after: {type(e1).__name__}: {str(e1)[:120]}")
            prov = bench.provision(task, workdir, docker=docker, service_timeout_s=service_timeout_s)
        rec.provision_warnings = list(prov.warnings)
        rec.blocked_init_commands = list(prov.blocked_commands)
        rec.host_aliases = dict(prov.host_aliases)
        rec.service_health = dict(prov.health)
        rec.flag_leaks = list(prov.flag_leaks)
        ch = bench.to_challenge_dict(task, prov)
        rec.name_leaks = list(ch.get("_name_leaks") or [])
        shown = {k: v for k, v in ch.items() if not str(k).startswith("_")}     # never hand a private field to the solver
        memory_nodes = []
        if context_provider is not None:
            injected_rag_context, memory_nodes, mem_meta = context_provider(shown, task, llm_seed)
            rec.extra["memory"] = dict(mem_meta)
        factory = session_factory or ContainerSession
        skw = dict(session_kwargs or {})
        session = factory(prov.agent_dir, image=image, network=prov.network, cwd=prov.agent_dir,
                          command_timeout_s=command_timeout_s, **skw)
        session.start()
        rec.container = getattr(session, "name", "")
        impl = solve_impl
        if impl is None:
            from solver.solver import solve_challenge      # real model path only
            impl = solve_challenge
        t1 = time.monotonic()
        ub = tool_executor.use_backend(session)
        with ub:
            res = impl(shown, use_tools=True, use_memory=False, injected_rag_context=injected_rag_context,
                       max_rounds=max_rounds, max_output_tokens=max_output_tokens, context_window=context_window,
                       llm_seed=llm_seed, advertise_memory_tools=False, trim_fraction=trim_fraction,
                       flag_checker=(lambda f, _t=task, _b=bench: _b.check_flag(_t, f)), working_dir=str(prov.agent_dir),
                       flag_format=ch.get("flag_format"), tool_calling=tool_calling)
        rec.solve_wall_s = time.monotonic() - t1
        rec.commands_run, rec.command_timeouts = getattr(session, "commands_run", 0), getattr(session, "timeouts", 0)
        rec.extra["sandbox"] = {"dispatched": ub.stats["dispatched"], "container_runs": rec.commands_run,
                                "rejected": ub.stats["rejected"]}
        session.stop()
        if docker is not None and str(skw.get("user", "")) in ("root", "0", "0:0"):
            import os as _os
            docker.chown_tree(prov.agent_dir, _os.getuid(), _os.getgid())   # root-owned outputs -> host user
        flag = (res or {}).get("flag")
        fm = bench.flag_match(task, flag)
        rec.flag_returned, rec.flag_exact, rec.flag_brace_content = flag, bool(fm.exact), bool(fm.brace_content)
        rec.solved = bool(fm.exact)
        rec.rounds_used = int((res or {}).get("rounds_used") or 0)
        rec.total_commands = int((res or {}).get("total_commands") or 0)
        rec.gave_up_reason = (res or {}).get("gave_up_reason")
        trace = (res or {}).get("trace") or []
        rec.extra["rounds_meta"] = {"parse_error_rounds": sum(1 for t in trace if t.get("action") == "parse_error"),
                                    "length_truncations": sum(1 for t in trace if t.get("finish_reason") == "length")}
        if keep_trace:
            rec.trace = (res or {}).get("trace")
        sb = rec.extra["sandbox"]
        if sb["rejected"] or sb["dispatched"] != sb["container_runs"]:
            rec.error = (f"sandbox: {sb['rejected']} command(s) blocked outside the container, "
                         f"{sb['dispatched']} dispatched vs {sb['container_runs']} run in it")
        elif rec.gave_up_reason in ("llm_bad_request", "llm_error"):
            rec.error = f"llm: {rec.gave_up_reason}"       # F11: not a solver failure; re-run or exclude, never score
        if memory_nodes:
            from solver.citation import match_citations
            from realbench.lessons import generic_keywords
            generic = sorted(set(generic_keywords(memory_nodes)) | {str(t).lower() for t in (citation_generic_tokens or [])})
            ms = match_citations(memory_nodes, trace, mode="distinctive", literal=True, generic_tokens=generic)
            rec.extra.setdefault("memory", {})["cited"] = [m.node_id for m in ms if m.cited]
            rec.extra["memory"]["citation_evidence"] = {m.node_id: m.match_evidence for m in ms if m.cited}
            rec.extra["memory"]["n_distinctive"] = {m.node_id: m.n_distinctive for m in ms}
            rec.extra["memory"]["citation_matcher"] = {"mode": "distinctive", "literal": True, "n_generic_tokens": len(generic)}
        try:
            rec.handout_changes = list(prov.verify_handouts())
        except Exception as e:  # noqa
            rec.handout_changes = [f"verify_handouts failed: {e!r}"]
    except Exception as e:  # noqa - one failed solve must not stop a campaign; the record says why
        rec.error = f"{type(e).__name__}: {e}"[:500]
    finally:
        if session is not None:
            try:
                session.stop()
            except Exception:
                pass
        if prov is not None:
            try:
                prov.teardown()
            except Exception as e:  # noqa
                rec.error = (rec.error or "") + f" | teardown: {type(e).__name__}: {e}"[:300]
        rec.wall_s = time.monotonic() - t0
    return rec
