"""
realbench/oracle_check.py - validate Cybench provisioning END TO END without any LLM.

For each selected task: provision it (fresh work dir, init_script.sh -> handouts, services if any),
run the task's own reference `metadata/solution/solution.sh` inside a slim disposable container
(`realbench-solver:slim`, built from realbench/docker/Dockerfile) with the work dir mounted and the
container on the task's docker network, and compare the produced flag (last non-empty stdout line,
whitespace stripped - upstream run_solution.sh semantics) with the metadata flag.

A task is reported VALIDATED only when the exact match holds.  `brace` reports the looser
"same {...} content" match separately (e.g. a `b'HTB{...}'` bytes-repr).

  python -m realbench.oracle_check --list
  python -m realbench.oracle_check --quick 3 --max-minutes 30 --workroot /some/scratch/dir
  python -m realbench.oracle_check --tasks "[Very Easy] Dynastic" "It Has Begun" --json out.json

CPU only.  Never starts vLLM.  Services are only started for tasks selected with --allow-service.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from realbench.cybench import (
    DEFAULT_CYBER_ZERO_ROOT, DEFAULT_UPSTREAM_ROOT, SHARED_NET, SLIM_IMAGE, CybenchTask, DockerOps,
    ProvisionError, RealDockerOps, SourceCorrupted, enumerate_tasks, find_task, flag_match, provision,
)

DOCKERFILE_DIR = Path(__file__).resolve().parent / "docker"


@dataclass
class OracleResult:
    task_id: str
    display_name: str
    category: str
    difficulty: int
    needs_service: bool
    first_solve_minutes: Optional[float]
    provisioned_ok: bool = False
    init_seconds: Optional[float] = None
    n_handouts: int = 0
    services_started: bool = False
    solution_ran_ok: bool = False
    solution_returncode: Optional[int] = None
    produced: str = ""
    flag_exact: bool = False
    flag_brace_content: bool = False
    matched_reference: Optional[str] = None
    seconds: float = 0.0
    error: str = ""
    warnings: List[str] = field(default_factory=list)
    blocked_commands: List[str] = field(default_factory=list)
    stderr_tail: str = ""

    @property
    def validated(self) -> bool:
        return self.provisioned_ok and self.solution_ran_ok and self.flag_exact


def _last_nonempty_line(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def ensure_image(docker: DockerOps, image: str, build: bool, log=print) -> Optional[int]:
    if docker.image_exists(image):
        return docker.image_size_bytes(image)
    if not build:
        raise SystemExit(f"image {image} is missing; run with --build-image (context: {DOCKERFILE_DIR})")
    log(f"[oracle] building {image} from {DOCKERFILE_DIR} ...")
    t0 = time.monotonic()
    res = docker.build_image(image, DOCKERFILE_DIR)
    if res.returncode != 0:
        raise SystemExit(f"docker build failed (rc={res.returncode}):\n{(res.stderr or '')[-3000:]}")
    log(f"[oracle] built {image} in {time.monotonic() - t0:.0f}s")
    return docker.image_size_bytes(image)


def run_oracle(task: CybenchTask, docker: DockerOps, workroot: Path, image: str = SLIM_IMAGE,
               task_timeout_s: float = 900.0, service_timeout_s: float = 180.0, keep: bool = False,
               log=print) -> OracleResult:
    r = OracleResult(task_id=task.id, display_name=task.display_name, category=task.category,
                     difficulty=task.difficulty, needs_service=task.needs_service,
                     first_solve_minutes=task.first_solve_minutes)
    t_start = time.monotonic()
    workdir = workroot / f"{task.display_name}-{uuid.uuid4().hex[:6]}"
    prov = None
    try:
        prov = provision(task, workdir, docker=docker, with_solution=True, service_timeout_s=service_timeout_s,
                         keep_on_failure=keep)
        r.provisioned_ok = True
        r.init_seconds = prov.timings_s.get("init")
        r.n_handouts = len(prov.handouts)
        r.services_started = prov.services_started
        r.warnings = list(prov.warnings)
        r.blocked_commands = list(prov.blocked_commands)
        if prov.solution_dir is None:
            raise ProvisionError("no metadata/solution in the snapshot")
        docker.ensure_network(prov.network)   # file-only tasks never created it during provisioning
        res = docker.run_in_container(
            image, ["solution.sh"], entrypoint="bash",      # bypass the image's entrypoint (dockerd loop in cybench/cybench)
            network=prov.network,
            mounts={str(prov.workdir): "/work"},
            workdir="/work/task/metadata/solution",
            env={"HOME": "/tmp", "TMP_DIR": "/work/agent", "PYTHONDONTWRITEBYTECODE": "1",
                 "PIP_NO_CACHE_DIR": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "LANG": "C.UTF-8"},
            user=f"{os.getuid()}:{os.getgid()}",
            name=f"realbench-oracle-{uuid.uuid4().hex[:8]}",
            timeout_s=task_timeout_s,
        )
        r.solution_returncode = res.returncode
        r.solution_ran_ok = res.returncode == 0
        r.produced = _last_nonempty_line(res.stdout)
        r.stderr_tail = (res.stderr or "")[-600:]
        fm = flag_match(task, r.produced)
        r.flag_exact, r.flag_brace_content, r.matched_reference = fm.exact, fm.brace_content, fm.matched_reference
    except (ProvisionError, SourceCorrupted) as e:
        r.error = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001 - report, never abort the batch
        r.error = f"{type(e).__name__}: {e}"
    finally:
        if prov is not None and not keep:
            try:
                prov.teardown(verify_source=True)
            except SourceCorrupted as e:
                r.error = (r.error + " | " if r.error else "") + f"SourceCorrupted: {e}"
            except Exception as e:  # noqa: BLE001
                r.warnings.append(f"teardown: {type(e).__name__}: {e}")
        r.seconds = round(time.monotonic() - t_start, 2)
    status = "VALIDATED" if r.validated else ("brace-only" if r.flag_brace_content else "FAILED")
    log(f"[oracle] {task.name[:32]:32s} {status:10s} prov={r.provisioned_ok} ran={r.solution_ran_ok} "
        f"exact={r.flag_exact} brace={r.flag_brace_content} {r.seconds:6.1f}s" + (f"  err={r.error[:160]}" if r.error else ""))
    return r


def select_tasks(tasks: List[CybenchTask], names: Optional[List[str]], quick: Optional[int],
                 allow_service: bool) -> List[CybenchTask]:
    if names:
        chosen = [find_task(tasks, n) for n in names]
    else:
        pool = [t for t in tasks if t.has_reference_solution and (allow_service or not t.needs_service)]
        pool.sort(key=lambda t: (t.first_solve_minutes is None, t.first_solve_minutes or 0.0, t.difficulty))
        chosen = pool[: (quick or 3)]
    if not allow_service:
        skipped = [t.name for t in chosen if t.needs_service]
        if skipped:
            print(f"[oracle] skipping service tasks (use --allow-service): {skipped}")
        chosen = [t for t in chosen if not t.needs_service]
    return chosen


def print_list(tasks: List[CybenchTask]) -> None:
    print(f"{'display':13s} {'cat':9s} {'diff':4s} {'svc':3s} {'sol':3s} {'src':10s} {'1st-solve(min)':14s} name")
    for t in tasks:
        fs = f"{t.first_solve_minutes:.0f}" if t.first_solve_minutes is not None else "?"
        print(f"{t.display_name:13s} {t.category:9s} {t.difficulty:<4d} {'Y' if t.needs_service else '-':3s} "
              f"{'Y' if t.has_reference_solution else '-':3s} {t.source:10s} {fs:14s} {t.name}")
    n_svc = sum(t.needs_service for t in tasks)
    print(f"\n{len(tasks)} tasks; {n_svc} need a service, {len(tasks) - n_svc} are file-only")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_UPSTREAM_ROOT))
    ap.add_argument("--cyber-zero-root", default=str(DEFAULT_CYBER_ZERO_ROOT))
    ap.add_argument("--list", action="store_true", help="enumerate and exit")
    ap.add_argument("--tasks", nargs="*", help="task names / ids / display names (default: --quick selection)")
    ap.add_argument("--quick", type=int, default=3, help="N fastest-first-solve file-only tasks with a solution (default 3)")
    ap.add_argument("--allow-service", action="store_true", help="also run tasks that need a docker service")
    ap.add_argument("--max-minutes", type=float, default=30.0, help="wall-clock budget; no new task starts after it")
    ap.add_argument("--task-timeout", type=float, default=900.0, help="seconds per solution.sh container")
    ap.add_argument("--service-timeout", type=float, default=180.0, help="seconds to wait for target hosts")
    ap.add_argument("--workroot", default=None, help="parent dir for the per-task work dirs (default: $TMPDIR/realbench-oracle)")
    ap.add_argument("--image", default=SLIM_IMAGE)
    ap.add_argument("--no-build", action="store_true", help="fail instead of building a missing image")
    ap.add_argument("--json", default=None, help="write per-task results here (default: <workroot>/oracle_<ts>.json)")
    ap.add_argument("--keep", action="store_true", help="keep work dirs (debugging)")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)     # progress lines reach a redirected log as they happen
    except Exception:  # noqa: BLE001
        pass

    tasks = enumerate_tasks(args.root, args.cyber_zero_root)
    if args.list:
        print_list(tasks)
        return 0
    chosen = select_tasks(tasks, args.tasks, args.quick, args.allow_service)
    if not chosen:
        print("[oracle] nothing selected")
        return 2
    workroot = Path(args.workroot or (Path(tempfile.gettempdir()) / "realbench-oracle")).resolve()
    workroot.mkdir(parents=True, exist_ok=True)
    docker = RealDockerOps()
    docker.ensure_network(SHARED_NET)
    t_all = time.monotonic()
    image_size = ensure_image(docker, args.image, build=not args.no_build)
    build_s = round(time.monotonic() - t_all, 1)
    print(f"[oracle] image {args.image}: {(image_size or 0) / 1e6:.0f} MB (ensure took {build_s}s); "
          f"{len(chosen)} task(s); budget {args.max_minutes:.0f} min; workroot {workroot}")

    results: List[OracleResult] = []
    for t in chosen:
        elapsed_min = (time.monotonic() - t_all) / 60.0
        if elapsed_min >= args.max_minutes:
            print(f"[oracle] budget exhausted ({elapsed_min:.1f} min) - not starting {t.name}")
            break
        results.append(run_oracle(t, docker, workroot, image=args.image, task_timeout_s=args.task_timeout,
                                  service_timeout_s=args.service_timeout, keep=args.keep))

    out = Path(args.json) if args.json else workroot / f"oracle_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "image": args.image, "image_size_bytes": image_size, "budget_minutes": args.max_minutes,
        "total_seconds": round(time.monotonic() - t_all, 1),
        "results": [dict(asdict(r), validated=r.validated) for r in results],
    }, indent=2))
    n_ok = sum(r.validated for r in results)
    print(f"\n[oracle] {n_ok}/{len(results)} validated (exact flag match); "
          f"{sum(r.flag_brace_content and not r.flag_exact for r in results)} brace-only; results -> {out}")
    print(f"{'task':34s} {'prov':5s} {'ran':5s} {'exact':6s} {'brace':6s} {'sec':7s} note")
    for r, t in zip(results, chosen):
        note = r.error[:70] if r.error else (f"produced={r.produced[:48]!r}" if not r.flag_exact else "")
        print(f"{t.name[:34]:34s} {str(r.provisioned_ok):5s} {str(r.solution_ran_ok):5s} {str(r.flag_exact):6s} "
              f"{str(r.flag_brace_content):6s} {r.seconds:7.1f} {note}")
    return 0 if results and n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
