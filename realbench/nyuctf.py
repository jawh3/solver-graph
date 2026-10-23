"""realbench/nyuctf.py - NYU CTF Bench (CSAW challenges) as a second benchmark for the Phase-3 harness.

Mirrors `realbench.cybench`: enumerate tasks from the dataset JSON, provision one solve attempt into a fresh work dir
(handouts copied into the agent dir, services started from a snapshot with `docker compose`), show the solver a neutral
description, check the flag by exact match, tear down.  Everything Cybench-agnostic in `cybench.py` (neutral names, leak
guards, flag matching, DockerOps, Provisioned, snapshot + manifest, render_tree) is reused; the NYU deltas are:

  * handouts are an explicit `files` list (relative subpaths preserved) - an init script generated per provision copies them;
  * services come from the task's docker-compose.yml on the EXTERNAL network `ctfnet`, with the challenge hostname as a network
    alias.  Aliases are category-generic (pwn.chal.csaw.io is shared by 45 tasks) and host ports are fixed, so service tasks
    must run one at a time (`--service-parallel 1`) and every compose invocation gets a unique project name;
  * no uniform reference solution exists, so the pre-run check is an LLM-free provisioning + reachability check
    (`python -m realbench.nyuctf --check`), not an oracle solve;
  * flags whose text the solver's extractor cannot represent (inner whitespace, no braces, nested braces) are marked
    `solver_extractable=False` and excluded up front - a pre-declared ceiling, not a silent failure;
  * a task with no handouts and no service but a non-empty description is a description-only task and RUNS (the official
    benchmark runs it); a listed handout missing from the checkout excludes the task up front.

Run policy recorded here (user decisions, 2026-09-22): tasks whose compose mounts the host Docker socket are INCLUDED, as the
official benchmark runs them; the one task that puts a container on the host network to rewrite host iptables is excluded.
Review fixes 2026-09-22 (adversarial review of the adapter): file-only solves run on `ctfnet` too; the handout tree lists
every file (Cybench's image-hiding is a Cybench prompt convention); compose `H:C` port maps correct a challenge.json
`internal_port` that names the host side; fixed `container_name`/network names are removed before `up` so a crashed earlier
solve cannot block the next one.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from realbench import cybench as cb

DEFAULT_NYU_ROOT = Path(os.environ.get("NYUCTF_ROOT", "<path>"))
SHARED_NET = "ctfnet"                                   # the external network every NYU compose file joins
DEFAULT_CONTAMINATION_CUTOFF = cb.DEFAULT_CONTAMINATION_CUTOFF

# The snapshot is only what compose/build and the handout copy need; the solver never sees it.  Reference solutions and
# write-ups are left out; flag.txt at the task root is KEPT because the 7 `build:` tasks COPY it into their images.
NYU_SNAPSHOT_EXCLUDES: Tuple[str, ...] = ("test_solver", "hints", "writeup", "Writeup", "writeups", "solution", "Solution",
                                          "solutions", "solver", "solve", "soln", "solutionScript", "testing", "exp")

# hostname the compose alias actually installs when challenge.json's `box` disagrees with it
HOST_FIXES: Dict[str, str] = {"2023q-web-cookie_injection": "web.chal.csaw.io"}
# server type when the category rule (crypto/pwn/rev -> nc) is wrong for the service the compose file actually runs
SERVER_TYPE_FIXES: Dict[str, str] = {"2023f-cry-textbook_rsa": "web"}      # Flask app on container port 80 (compose 3000:80)

# pre-declared exclusions (reason shown in the usable-task table); everything else is decided by the reachability check
EXCLUDED: Dict[str, str] = {
    "2023f-web-triathlon_or_sprint": "compose side-car runs on the HOST network and rewrites host iptables",
    "2023f-for-forensings": "listed handout missing from the benchmark checkout (external download)",
    "2019f-msc-alive": "welcome task: the flag is printed in the description",
    "2018f-msc-leaked_flag": "welcome task: the flag is printed in the description",
    "2021f-pwn-chatterbox": "depends on an external remote host that no longer exists",
    "2021q-web-scp_terminal": "depends on an external remote host; no handouts",
    "2020f-web-snailrace1": "depends on an external remote host that no longer exists",
    "2020f-web-sharkfacts": "depends on an external remote host that no longer exists",
    "2023f-cry-nervcenter": "depends on an external remote host (nervcenter.moyix.net)",
    "2023q-web-rainbow_notes": "TLS-only service (port 443) with a second hostname that embeds the task name; the one-alias "
                              "neutralisation cannot render the admin-bot host the description points at",
    "2013q-web-guess_harder": "no port in its metadata and the description's URL path embeds the task name",
    "2018f-pwn-es1337": "the challenge image needs the HOST Docker socket (starts a container per connection); "
                        "mounting it would give a solver with a shell host-root, so the task is excluded rather than run",
}

_SOLVER_FLAG_RE = re.compile(r"^[A-Za-z0-9_]{2,24}\{[^\s}]{3,100}\}$")   # solver.solver._ANY_FLAG_PATTERN, anchored


def solver_can_extract(flag: str) -> bool:
    """True iff the solver's flag regex (PREFIX{...}, no whitespace or `}` inside, 3-100 chars) matches this flag VERBATIM -
    the same character class as solver.solver.compile_flag_pattern, anchored at both ends (so `flag{a{b}}` is rejected:
    the extractor would return `flag{a{b}`)."""
    return bool(_SOLVER_FLAG_RE.match(flag or ""))


@dataclass(frozen=True)
class NYUTask:
    id: str                          # dataset key, e.g. "2021f-rev-maze" (the official canonical name; LOGGING + keys only)
    name: str                        # challenge.json name - never shown to the solver
    split: str                       # "test" | "development"
    year: int
    event: str                       # "CSAW-Quals" | "CSAW-Finals"
    category: str                    # normalized (rev -> reverse), the bandit context and summary key
    category_raw: str
    points: int
    needs_service: bool              # compose file present (== challenge.json `compose`)
    target_host: str                 # "" or "host:port" in Cybench's form (so parse_target_hosts / provision work unchanged)
    box: str
    internal_port: Optional[int]
    proto: str
    server_type: str                 # "nc" | "web" | ""  (official CTFChallenge.server_type)
    flag: str
    handout_files: Tuple[str, ...]   # `files` with a leading "./" stripped; relative subpaths preserved
    task_dir: Path
    compose_file: Optional[Path]
    description_raw: str             # with {box}/{port} placeholders unresolved
    reference: str
    aliases: Tuple[str, ...]         # other public names for the leak guard
    competition_date: str            # YYYY-MM (Quals: September, Finals: November)
    docker_sock: bool = False        # compose mounts /var/run/docker.sock
    privileged: bool = False
    host_network: bool = False
    solver_extractable: bool = True
    exclusion_reason: str = ""       # non-empty => pre-declared exclusion
    source: str = "nyuctf"
    difficulty: int = -1
    first_solve_minutes: Optional[float] = None
    has_start_docker: bool = False
    has_reference_solution: bool = False

    @property
    def display_name(self) -> str:
        return cb.neutral_display_name(self.id)

    @property
    def categories(self) -> Tuple[str, ...]:
        return (self.category,)

    @property
    def accepted_flags(self) -> Tuple[str, ...]:
        return (self.flag,)

    @property
    def answer_format(self) -> str:
        return re.sub(r"\{.*\}", "{...}", self.flag, count=1) if "{" in self.flag and "}" in self.flag else ""

    @property
    def target_hosts(self) -> Tuple[Tuple[str, int], ...]:
        return cb.parse_target_hosts(self.target_host)

    @property
    def difficulty_label(self) -> str:
        return "unknown"

    def post_cutoff(self, cutoff: str = DEFAULT_CONTAMINATION_CUTOFF) -> Optional[bool]:
        return self.competition_date >= cutoff if self.competition_date else None


def _compose_facts(compose_file: Path) -> Dict[str, Any]:
    """The compose facts the adapter needs: every network alias (in file order), images, fixed `container_name`s and
    explicitly named non-external networks (removed before `up`), `H:C` port maps, and the policy flags.  Parsed with
    PyYAML; a file YAML cannot parse falls back to a regex read of the same fields."""
    t = compose_file.read_text(errors="replace")
    facts: Dict[str, Any] = {"docker_sock": "docker.sock" in t, "privileged": bool(re.search(r"^\s*privileged:\s*true", t, re.M)),
                             "host_network": bool(re.search(r"^\s*network_mode:\s*[\"']?host", t, re.M)),
                             "has_build": bool(re.search(r"^\s*build:", t, re.M)), "images": [], "aliases": [],
                             "container_names": [], "network_names": [], "port_map": [], "service_names": [],
                             "service_aliases": {}}          # service -> network -> [aliases]
    y = None
    try:
        import yaml
        y = yaml.safe_load(t)
    except Exception:  # noqa
        y = None
    if isinstance(y, dict) and isinstance(y.get("services"), dict):
        facts["service_names"] = [str(k) for k in y["services"].keys()]
        for sname, svc in y["services"].items():
            if not isinstance(svc, dict):
                continue
            if svc.get("image"):
                facts["images"].append(str(svc["image"]))
            if svc.get("container_name"):
                facts["container_names"].append(str(svc["container_name"]))
            nets = svc.get("networks")
            if isinstance(nets, dict):
                for net, cfg in nets.items():
                    if isinstance(cfg, dict) and cfg.get("aliases"):
                        al = [str(a) for a in cfg["aliases"]]
                        facts["aliases"] += al
                        facts["service_aliases"].setdefault(str(sname), {})[str(net)] = al
            for p in svc.get("ports") or []:
                m = re.match(r"^(?:[\d.]+:)?(\d+):(\d+)(?:/\w+)?$", str(p).strip()) if not isinstance(p, dict) else None
                if m:
                    facts["port_map"].append((int(m.group(1)), int(m.group(2))))
                elif isinstance(p, dict) and p.get("published") and p.get("target"):
                    facts["port_map"].append((int(p["published"]), int(p["target"])))
        for key, cfg in (y.get("networks") or {}).items() if isinstance(y.get("networks"), dict) else []:
            if isinstance(cfg, dict) and cfg.get("name") and not cfg.get("external"):
                facts["network_names"].append(str(cfg["name"]))
    else:
        facts["images"] = re.findall(r"image:\s*['\"]?([^\s'\"]+)", t)
        facts["aliases"] = re.findall(r"aliases:\s*\n\s*-\s*([\w.\-]+)", t) + re.findall(r"aliases:\s*\[\s*([\w.\-]+)", t)
        facts["container_names"] = re.findall(r"^\s*container_name:\s*['\"]?([\w.\-]+)", t, re.M)
        facts["port_map"] = [(int(h), int(c)) for h, c in re.findall(r"^\s*-\s*['\"]?(?:[\d.]+:)?(\d+):(\d+)", t, re.M)]
        m = re.search(r"^services:\s*$(.*?)(?=^\S|\Z)", t, flags=re.M | re.S)
        facts["service_names"] = re.findall(r"^  ([\w.\-]+):\s*$", m.group(1), re.M) if m else []
    facts["images"] = list(dict.fromkeys(facts["images"]))
    facts["aliases"] = list(dict.fromkeys(facts["aliases"]))
    return facts


def _container_port(port: Optional[int], port_map: Sequence[Tuple[int, int]]) -> Optional[int]:
    """challenge.json's `internal_port` sometimes names the HOST side of a compose `H:C` map (2023f-cry-textbook_rsa:
    3000 for `3000:80`).  In-network the target is the container side, so map it when the port is only a host side."""
    if port is None or not port_map or any(c == port for _h, c in port_map):
        return port
    for h, c in port_map:
        if h == port:
            return c
    return port


def enumerate_tasks(root: Path | str = DEFAULT_NYU_ROOT, split: str = "test", task_ids: Optional[Sequence[str]] = None,
                    strict: bool = True) -> List[NYUTask]:
    """Tasks of one split, in dataset order.  Reads `<root>/<split>_dataset.json` + each task's challenge.json directly
    (the official package would git-clone into ~/.nyuctf and raises on a {box} placeholder without a box)."""
    root = Path(root)
    ds = json.loads((root / f"{split}_dataset.json").read_text())
    ids = list(task_ids) if task_ids is not None else list(ds.keys())
    out: List[NYUTask] = []
    for tid in ids:
        if tid not in ds:
            if strict:
                raise KeyError(f"{tid!r} is not in {split}_dataset.json")
            continue
        entry = ds[tid]
        tdir = root / entry["path"]
        cj_path = tdir / "challenge.json"
        if not cj_path.exists():
            if strict:
                raise FileNotFoundError(cj_path)
            continue
        cj = json.loads(cj_path.read_text())
        compose_file = next((p for p in (tdir / "docker-compose.yml", tdir / "docker-compose.yaml") if p.exists()), None)
        needs_service = bool(cj.get("compose", False)) and compose_file is not None
        facts = _compose_facts(compose_file) if compose_file else {}
        port_raw = cj.get("internal_port", cj.get("port"))
        try:
            port = int(str(port_raw).strip()) if port_raw not in (None, "") else None
        except ValueError:
            port = None
        if needs_service:
            port = _container_port(port, facts.get("port_map") or [])
        box = str(cj.get("box") or "").strip()
        if tid in HOST_FIXES:
            box = HOST_FIXES[tid]
        elif needs_service and not box and facts.get("aliases"):
            box = facts["aliases"][0]                     # compose without a box: the alias is what resolves
        target_host = f"{box}:{port}" if (needs_service and box and port) else ""
        cat_raw = str(cj.get("category") or entry.get("category") or "misc").strip().lower()
        proto = str(cj.get("proto") or "").strip().lower()
        if not (needs_service and box and port):
            server_type = ""
        elif tid in SERVER_TYPE_FIXES:
            server_type = SERVER_TYPE_FIXES[tid]
        elif cat_raw in ("web", "misc") and proto != "nc":
            server_type = "web"
        else:
            server_type = "nc"
        files = tuple(str(f)[2:] if str(f).startswith("./") else str(f) for f in (cj.get("files") or []))
        flag = str(cj.get("flag") or "")
        year = int(entry["year"]); event = str(entry["event"])
        description_raw = str(cj.get("description") or "")
        missing = [f for f in files if not (tdir / f).exists()]
        reason = EXCLUDED.get(tid, "")
        if not reason and not solver_can_extract(flag):
            reason = "flag text is outside what the solver's extractor can represent (whitespace, no braces, or nested braces)"
        if not reason and needs_service and not target_host:
            reason = "service task without a resolvable hostname:port in its metadata"
        if not reason and missing:
            reason = f"listed handout missing from the benchmark checkout: {', '.join(missing)[:80]}"
        if not reason and not files and not needs_service and not description_raw.strip():
            reason = "no handouts, no service and no description"
        name = str(cj.get("name") or entry["challenge"])
        aliases = tuple(dict.fromkeys(a for a in (entry["challenge"], tdir.name, re.sub(r"[^\w]+", "_", name)) if a and a != name))
        out.append(NYUTask(
            id=tid, name=name, split=split, year=year, event=event, category=cb.normalize_category(cat_raw), category_raw=cat_raw,
            points=int(cj.get("points") or cj.get("initial") or 0), needs_service=needs_service, target_host=target_host,
            box=box, internal_port=port, proto=proto, server_type=server_type, flag=flag, handout_files=files, task_dir=tdir,
            compose_file=compose_file, description_raw=description_raw, reference=str(cj.get("reference") or ""),
            aliases=aliases, competition_date=f"{year}-{'11' if 'Final' in event else '09'}",
            docker_sock=bool(facts.get("docker_sock")), privileged=bool(facts.get("privileged")),
            host_network=bool(facts.get("host_network")), solver_extractable=solver_can_extract(flag), exclusion_reason=reason,
            has_start_docker=needs_service,
            has_reference_solution=any((tdir / n).exists() for n in ("solve.py", "solver.py", "solution.py", "solve.txt"))))
    return out


def find_task(tasks: Iterable[NYUTask], needle: str) -> NYUTask:
    return cb.find_task(tasks, needle)   # duck-typed: id / name / display_name


# --------------------------------------------------------------------------- #
# provisioning                                                                 #
# --------------------------------------------------------------------------- #

def init_script(task: NYUTask) -> str:
    """Copies the handout list into $1 (the agent dir), preserving relative subpaths; runs from the snapshot dir.
    A missing handout is reported on stderr and the script exits 3, which cb.provision records as a warning (the run
    continues with the handouts that exist; enumerate_tasks already excludes a task whose listed handout is absent)."""
    lines = ["#!/bin/bash", "set -e", 'TMP_DIR="$1"', 'mkdir -p "$TMP_DIR"', "missing=0"]
    for rel in task.handout_files:
        q = shlex.quote(rel)
        lines.append(f"if [ -e {q} ]; then cp -r --parents {q} \"$TMP_DIR\"/; else echo \"handout missing: {rel}\" >&2; missing=1; fi")
    lines.append('[ "$missing" = 0 ] || exit 3')
    return "\n".join(lines) + "\n"


NOPORTS_OVERRIDE = "nyu-override.yml"
# 2026-10-20: every service solve gets its own network (`ctfnet-<tag>`, internal under --egress blocked). Before, all
# service stacks joined `ctfnet`, so solves whose services share an alias had to run one at a time (a per-hostname lock):
# with two arms in parallel the 36 crypto.chal.csaw.io tasks alone would serialize ~20 h. It also stops a solve from
# reaching another solve's services.
PER_SOLVE_NETWORK = True
# 2026-10-22: every challenge service gets a process cap. A leftover 2019f-rev-mystery_boi service (its solve had finished)
# grew to ~315,000 processes and 40 GB, twice, and stalled the shared host; normal services use a few dozen (gunicorn: 41).
SERVICE_PIDS_LIMIT = 4096
# 2026-10-20: 58 of the 118 usable test service stacks bind-mount the HOST's /var/run/docker.sock into the challenge
# container (upstream compose template). A solver that gets a shell in the service (the intended path for pwn) would then
# control the host's Docker daemon. The override drops the mount for every task. The one image that actually uses it
# (es1337, which starts containers per connection; checked image by image) is EXCLUDED instead.
def _socket_free_volumes(task: NYUTask) -> Dict[str, list]:
    """service -> its volume list without the docker.sock mount, for every service that mounts it. The host Docker socket
    is never mounted into a challenge (the one task that needs it, es1337, is EXCLUDED). The post-up egress/socket check
    in cybench.provision fails the solve closed if a socket mount somehow survives, so this is best-effort, not the guard."""
    if not task.compose_file:
        return {}
    try:
        import yaml
        y = yaml.safe_load(task.compose_file.read_text()) or {}
    except Exception:
        return {}
    out = {}
    for name, svc in (y.get("services") or {}).items():
        vols = (svc or {}).get("volumes") or []
        if any("docker.sock" in str(v) for v in vols):
            out[str(name)] = [v for v in vols if "docker.sock" not in str(v)]
    return out


def _nonexternal_networks(task: NYUTask) -> list:
    """Names of networks the compose file defines itself (not `external: true`): docker compose creates these as ordinary
    bridges WITH internet, so the override marks them internal (the renamed ctfnet is already internal)."""
    if not task.compose_file:
        return []
    try:
        import yaml
        y = yaml.safe_load(task.compose_file.read_text()) or {}
    except Exception:
        return []
    return [str(n) for n, c in (y.get("networks") or {}).items() if not (c or {}).get("external")]


def neutral_alias(task: NYUTask) -> str:
    """The neutral network alias cb.provision installs for a hostname that leaks the task name (single target host)."""
    return task.display_name if (task.box and cb.leaks_task_name(task.box, task)) else ""


def compose_override(task: NYUTask, network: str = "") -> str:
    """Compose override written into the snapshot beside the task's (never edited) compose file:
      * `ports: !reset []` for every service (compose >= 2.24): the agent reaches a service in-network by its alias; a
        published host port only collides - with whatever else runs on a shared host (port 80 is bound here by another
        user's process, which broke one task's `up`) and between tasks;
      * for a hostname that leaks the task name, the neutral alias is added to that service's alias list (original
        aliases repeated, so the result is the same under merge-or-replace semantics) - the container STARTS with it.
        cb.provision's fallback (disconnect + reconnect the live container) leaves a window with no DNS, and a server
        that resolves its own hostname at import (2023q-msc-android_dropper's gunicorn workers) dies in it.
      * the host's docker.sock is not mounted into the challenge (the one task needing it, es1337, is EXCLUDED);
      * every network the compose file defines itself (not external) is marked `internal` + isolated gateway, so a
        service on it (philnet, shreeramquest) has no internet - the renamed ctfnet is already internal;
      * `network` (2026-10-20): the external `ctfnet` is renamed to this solve's own network, so two solves whose
        services share an alias (crypto.chal.csaw.io x36) can run at once without resolving to each other's container."""
    facts = _compose_facts(task.compose_file) if task.compose_file else {}
    names = facts.get("service_names") or []
    net_lines = ["networks:", f"  {json.dumps(SHARED_NET)}:", f"    name: {json.dumps(network)}", "    external: true"] if network else []
    for nn in _nonexternal_networks(task):     # close the egress gap: compose-created bridges are internal + isolated
        net_lines += [f"  {json.dumps(nn)}:", "    internal: true",
                      "    driver_opts:", f"      {cb.GATEWAY_MODE_OPT}: isolated"]
    if net_lines and net_lines[0] != "networks:":
        net_lines = ["networks:"] + net_lines
    if not names:
        return "\n".join(net_lines) + "\n" if net_lines else ""
    alias = neutral_alias(task)
    keep_vols = _socket_free_volumes(task)
    lines = ["services:"]
    for n in names:
        lines += [f"  {json.dumps(n)}:", "    ports: !reset []", f"    pids_limit: {SERVICE_PIDS_LIMIT}"]
        if n in keep_vols:            # host docker.sock dropped; any other mounts kept as they were
            lines += ["    volumes: !reset []"] if not keep_vols[n] else ["    volumes: !override"] + [f"      - {json.dumps(v)}" for v in keep_vols[n]]
        for net, al in (facts.get("service_aliases") or {}).get(n, {}).items():
            if alias and task.box in al:
                lines += ["    networks:", f"      {json.dumps(net)}:", "        aliases:"]
                lines += [f"          - {json.dumps(a)}" for a in [*al, alias]]
    return "\n".join(lines + net_lines) + "\n"


def compose_scripts(task: NYUTask, project: str, network: str = "") -> Dict[str, str]:
    """start/stop scripts run FROM the snapshot dir.  Unique project name -> no container-name collisions between solves;
    `--pull never` -> a missing image fails loudly instead of downloading mid-run (use `--pull` first).  Stacks that pin a
    `container_name` or a named network would still collide with a leftover from a crashed earlier solve (service tasks
    run one at a time, so those names can only be ours): remove them before `up`.  Host ports: see compose_override."""
    f = task.compose_file.name if task.compose_file else "docker-compose.yml"
    p = shlex.quote(project)
    facts = _compose_facts(task.compose_file) if task.compose_file else {}
    pre = [f"docker rm -f {shlex.quote(n)} >/dev/null 2>&1 || true" for n in facts.get("container_names") or []]
    pre += [f"docker network rm {shlex.quote(n)} >/dev/null 2>&1 || true" for n in facts.get("network_names") or []]
    override = compose_override(task, network)
    files = f"-f {f} -f {NOPORTS_OVERRIDE}" if override else f"-f {f}"
    out = {"start_docker.sh": "#!/bin/bash\nset -e\n" + "".join(l + "\n" for l in pre) +
                              f"docker compose -p {p} {files} up -d --force-recreate --pull never\n",
           "stop_docker.sh": f"#!/bin/bash\ndocker compose -p {p} {files} down --volumes --remove-orphans --timeout 20\n"}
    if override:
        out[NOPORTS_OVERRIDE] = override
    return out


def provision(task: NYUTask, workdir: Path | str, *, docker: Optional[cb.DockerOps] = None, with_solution: bool = False,
              service_timeout_s: float = 180.0, keep_on_failure: bool = False, neutralize_flag_files: bool = True,
              project: Optional[str] = None, per_solve_network: Optional[bool] = None) -> cb.Provisioned:
    """`per_solve_network` (default PER_SOLVE_NETWORK): a service task's stack and the agent container share a network
    created for this solve alone instead of the common `ctfnet` (file-only tasks still join `ctfnet`)."""
    scripts = {"init_script.sh": init_script(task)}
    net_map = None
    if task.needs_service:
        tag = uuid.uuid4().hex[:6]
        psn = PER_SOLVE_NETWORK if per_solve_network is None else per_solve_network
        net = f"{SHARED_NET}-{tag}" if psn else ""
        net_map = {SHARED_NET: net} if net else None
        scripts.update(compose_scripts(task, project or f"nyu-{task.display_name}-{tag}", net))
    prov = cb.provision(task, workdir, docker=docker, with_solution=with_solution, service_timeout_s=service_timeout_s,
                        snapshot_excludes=() if with_solution else NYU_SNAPSHOT_EXCLUDES, keep_on_failure=keep_on_failure,
                        neutralize_flag_files=neutralize_flag_files, service_scripts=scripts,
                        allow_empty=not task.handout_files,          # description-only task: an empty agent dir is by design
                        network_map=net_map)
    if not prov.services_started:
        prov.network = SHARED_NET                         # file-only: the agent container joins ctfnet, not Cybench's shared_net
    return prov


# --------------------------------------------------------------------------- #
# what the solver sees                                                         #
# --------------------------------------------------------------------------- #

def render_description(task: NYUTask, agent_dir: Optional[Path], *, target_host: Optional[str] = None, redact: bool = True) -> str:
    """The upstream description with {box}/{port} resolved to the host the solver may use, a server line by server type,
    the handout tree, and the flag format in NYU's own `flag{...}` form.  The task name is redacted (F1)."""
    th = cb.neutral_target_host(task) if target_host is None else target_host
    host, port = (th.split()[0].rsplit(":", 1) + [""])[:2] if th else ("", "")
    desc = task.description_raw.strip()
    desc = desc.replace("{box}", host or "the challenge server").replace("{port}", port or "the challenge port")
    desc = desc.replace("{{", "{").replace("}}", "}")     # upstream renders with str.format_map, which collapses the escapes
    if redact:
        for h, _p in task.target_hosts:
            if cb.leaks_task_name(h, task) and host:
                desc = desc.replace(h, host)
        desc = cb.redact_task_name(desc, task)
    parts = [desc] if desc else []
    if host and port and f"{host}:{port}" not in desc and f"{host} {port}" not in desc:   # the resolved nc/URL line already says it
        if task.server_type == "web":
            scheme = "https" if port == "443" else "http"
            parts.append(f"The challenge web service is reachable at {scheme}://{host}:{port}/ (use this host exactly).")
        else:
            parts.append(f"The challenge service is reachable with `nc {host} {port}` (use this host exactly).")
    has_files = bool(agent_dir) and any(Path(agent_dir).rglob("*"))
    if has_files:
        parts.append(f"Files provided in your working directory:\n{cb.render_tree(agent_dir, hide_ext=())}")   # every file, as upstream lists them
    else:
        parts.append("No files are provided in your working directory.")
    if task.answer_format:
        parts.append(f"The flag format is {task.answer_format}.")
    return "\n\n".join(parts)


def name_redaction_count(task: NYUTask) -> int:
    """How many task-name tokens the F1 guard removes from this description (audit field; the name is never shown)."""
    return len(cb.name_leak_pattern(task).findall(task.description_raw))


def to_challenge_dict(task: NYUTask, provisioned: Optional[cb.Provisioned] = None, *, cutoff: str = DEFAULT_CONTAMINATION_CUTOFF,
                      **_ignored) -> dict:
    agent_dir = Path(provisioned.agent_dir) if provisioned else None
    shown_host = provisioned.shown_target_host if provisioned else cb.neutral_target_host(task)
    d = cb.to_challenge_dict(task, provisioned, cutoff=cutoff,
                             description=render_description(task, agent_dir, target_host=shown_host), dataset="nyuctf")
    d.update({"_split": task.split, "_year": task.year, "_event": task.event, "_points": task.points,
              "_server_type": task.server_type, "_docker_sock": task.docker_sock, "_name_redactions": name_redaction_count(task)})
    return d


flag_match = cb.flag_match
check_flag = cb.check_flag


# --------------------------------------------------------------------------- #
# lesson-distillation sources (for the memory campaign)                        #
# --------------------------------------------------------------------------- #

def gather_sources(task: NYUTask, *, max_chars_per_source: int = 6000, max_script_bytes: int = 120_000) -> Dict[str, str]:
    """Reference material for distilling a lesson: the public description, README / hints / write-ups and reference solver
    scripts shipped in the task dir (never the handouts), task name redacted.  Handouts are never read."""
    d = Path(task.task_dir)
    src: Dict[str, str] = {"challenge_prompt": task.description_raw}
    docs = []
    handouts = set(task.handout_files)
    for p in sorted(d.glob("*")):
        if p.is_file() and p.name.lower().startswith(("readme", "writeup")) and p.suffix.lower() in (".md", ".txt", "") \
                and p.name not in handouts:                 # a README that IS a handout is prompt material, not a write-up
            docs.append(f"### {p.name}\n" + p.read_text(errors="replace"))
    for p in sorted((d / "hints").glob("*.md")) if (d / "hints").is_dir() else []:
        docs.append(f"### hints/{p.name}\n" + p.read_text(errors="replace"))
    for sub in ("writeup", "Writeup", "writeups"):
        for p in sorted((d / sub).rglob("*")) if (d / sub).is_dir() else []:
            if p.is_file() and p.suffix.lower() in (".md", ".txt"):
                docs.append(f"### {sub}/{p.name}\n" + p.read_text(errors="replace"))
    if docs:
        src["writeups"] = cb_clip("\n\n".join(docs), max_chars_per_source)
    scripts = []
    for p in sorted(d.glob("*")):
        if p.is_file() and p.suffix in (".py", ".sage", ".sh", ".txt") and re.match(r"(solve|solver|solution|soln|exploit)", p.name, re.I) \
                and p.stat().st_size <= max_script_bytes:
            scripts.append(f"### {p.name}\n" + p.read_text(errors="replace"))
    ts = d / "test_solver"
    if ts.is_dir():
        for p in sorted(ts.rglob("*")):
            if p.is_file() and p.suffix in (".py", ".sage", ".sh") and p.stat().st_size <= max_script_bytes:
                body = p.read_text(errors="replace")
                if "your code here" in body and "exit 1" in body:
                    continue                                  # the upstream stub
                scripts.append(f"### test_solver/{p.relative_to(ts)}\n" + body)
    if scripts:
        src["solver_scripts"] = cb_clip("\n\n".join(scripts), max_chars_per_source * 2)
    return {k: cb.redact_task_name(v, task) for k, v in src.items() if v and v.strip()}


def cb_clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [clipped {len(text) - limit} chars]"


# --------------------------------------------------------------------------- #
# pre-run checks: images, reachability                                         #
# --------------------------------------------------------------------------- #

def compose_images(tasks: Iterable[NYUTask]) -> List[str]:
    imgs: List[str] = []
    for t in tasks:
        if t.compose_file:
            for im in _compose_facts(t.compose_file)["images"]:
                if im not in imgs:
                    imgs.append(im)
    return imgs


def health_check(task: NYUTask, docker: Optional[cb.DockerOps], workroot: Path | str, *, service_timeout_s: float = 180.0) -> dict:
    """Provision + (for services) start + in-network reachability + teardown.  No model, no solving."""
    t0 = time.monotonic()
    rec = {"task_id": task.id, "display_name": task.display_name, "category": task.category, "split": task.split,
           "needs_service": task.needs_service, "docker_sock": task.docker_sock, "exclusion_reason": task.exclusion_reason,
           "provisioned_ok": False, "n_handouts": 0, "services_started": False, "health": {}, "warnings": [], "flag_leaks": [],
           "name_leaks": [], "error": None, "seconds": 0.0}
    if task.exclusion_reason:
        rec["seconds"] = round(time.monotonic() - t0, 2); return rec
    workdir = Path(workroot) / f"{task.display_name}-{uuid.uuid4().hex[:6]}"
    prov = None
    try:
        prov = provision(task, workdir, docker=docker if task.needs_service else None, service_timeout_s=service_timeout_s)
        rec.update({"provisioned_ok": True, "n_handouts": len(prov.handouts), "services_started": prov.services_started,
                    "health": dict(prov.health), "warnings": list(prov.warnings), "flag_leaks": list(prov.flag_leaks),
                    "name_leaks": cb.handout_name_leaks(task, prov.agent_dir)})
        if task.needs_service and task.target_hosts and not all(prov.health.values()):
            rec["error"] = f"unreachable: {prov.health}"
    except Exception as e:  # noqa
        rec["error"] = f"{type(e).__name__}: {e}"[:600]
    finally:
        if prov is not None:
            try:
                prov.teardown(verify_source=False)
            except Exception as e:  # noqa
                rec["warnings"].append(f"teardown: {e}"[:200])
        shutil.rmtree(workdir, ignore_errors=True)
    rec["seconds"] = round(time.monotonic() - t0, 2)
    return rec


def usable(rec: dict) -> bool:
    return not rec["exclusion_reason"] and rec["provisioned_ok"] and not rec["error"] and \
        (not rec["needs_service"] or (rec["services_started"] and all(rec["health"].values())))


def main(argv=None):
    ap = argparse.ArgumentParser(description="NYU CTF Bench adapter: list, pull images, or run the LLM-free reachability check")
    ap.add_argument("--root", "--nyu-root", dest="root", default=str(DEFAULT_NYU_ROOT), help="NYU_CTF_Bench checkout (same flag name as run_cybench --nyu-root)")
    ap.add_argument("--split", default="test", choices=["test", "development"])
    ap.add_argument("--tasks", default="all"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--pull", action="store_true", help="docker pull every compose image that is missing locally")
    ap.add_argument("--check", action="store_true", help="provision + reachability check every selected task (services one at a time)")
    ap.add_argument("--out", default="", help="JSON (and .md/.txt beside it) for --check")
    ap.add_argument("--workroot", default="results/realbench_nyu/_health"); ap.add_argument("--service-timeout", type=float, default=180.0)
    ap.add_argument("--egress", default="blocked", choices=["blocked", "open"], help="check under the network policy the solves will use "
                    "(blocked = ctfnet is a Docker --internal network)")
    args = ap.parse_args(argv)
    from realbench.run_cybench import select_tasks
    tasks = select_tasks(enumerate_tasks(Path(args.root), split=args.split), args.tasks)
    if args.list or not (args.pull or args.check):
        for t in tasks:
            print(f"{t.display_name}\t{t.category:9s}\t{'service' if t.needs_service else 'file-only'}\t{t.id}\t"
                  f"{'EXCLUDED: ' + t.exclusion_reason if t.exclusion_reason else ''}")
        n_ex = sum(1 for t in tasks if t.exclusion_reason)
        print(f"{len(tasks)} tasks ({sum(t.needs_service for t in tasks)} service, {n_ex} pre-excluded, "
              f"{sum(t.docker_sock for t in tasks)} with docker.sock)", file=sys.stderr)
        return 0
    if args.pull:
        imgs = compose_images(t for t in tasks if not t.exclusion_reason)
        missing = [im for im in imgs if subprocess.run(["docker", "image", "inspect", im], capture_output=True).returncode != 0]
        print(f"[nyuctf] {len(imgs)} images, {len(missing)} missing", flush=True)
        bad = []
        for im in missing:
            r = subprocess.run(["docker", "pull", im], capture_output=True, text=True)
            print(("ok   " if r.returncode == 0 else "FAIL ") + im, flush=True)
            if r.returncode != 0:
                bad.append(im)
        print(f"[nyuctf] pull done, {len(bad)} failed", flush=True)
        return 0 if not bad else 2
    if args.check:
        docker = cb.RealDockerOps(internal=(args.egress == "blocked")); docker.ensure_network(SHARED_NET)
        out = Path(args.out) if args.out else Path(f"docs/phase3/NYU_HEALTH_{args.split}_{time.strftime('%Y-%m-%d')}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        done = {}
        if out.exists():
            done = {r["task_id"]: r for r in json.loads(out.read_text()).get("results", [])}
        results = []
        t_start = time.time()
        for i, t in enumerate(tasks, 1):
            prev = done.get(t.id)
            if prev and not prev.get("error") and prev.get("exclusion_reason", "") == t.exclusion_reason:   # resume: unchanged + clean
                results.append(prev); continue
            r = health_check(t, docker, args.workroot, service_timeout_s=args.service_timeout)
            results.append(r)
            print(f"[{i}/{len(tasks)}] {t.display_name} {t.category:9s} {'svc' if t.needs_service else 'file'} "
                  f"{'USABLE' if usable(r) else ('excluded' if r['exclusion_reason'] else 'FAIL')} {r['seconds']:.0f}s "
                  f"{(r['error'] or r['exclusion_reason'] or '')[:90]}", flush=True)
            out.write_text(json.dumps({"split": args.split, "root": str(args.root), "elapsed_s": time.time() - t_start, "results": results}, indent=1))
        ok = [r for r in results if usable(r)]
        lines = [f"{r['display_name']}\t{r['category']}\t{'service' if r['needs_service'] else 'file-only'}\t{r['task_id']}" for r in ok]
        out.with_suffix(".txt").write_text("\n".join(lines) + "\n")
        md = [f"# NYU CTF Bench {args.split}: usable tasks after the reachability check ({time.strftime('%Y-%m-%d')})", "",
              f"{len(ok)} usable of {len(results)} ({sum(r['needs_service'] for r in ok)} service, {sum(1 for r in ok if r['docker_sock'])} of those with docker.sock).", "",
              "| task | category | kind | status | reason |", "|---|---|---|---|---|"]
        for r in results:
            st = "usable" if usable(r) else ("excluded" if r["exclusion_reason"] else "unusable")
            md.append(f"| {r['display_name']} ({r['task_id']}) | {r['category']} | {'service' if r['needs_service'] else 'file-only'} | {st} | {(r['exclusion_reason'] or r['error'] or '')[:120]} |")
        out.with_suffix(".md").write_text("\n".join(md) + "\n")
        print(f"[nyuctf] usable {len(ok)}/{len(results)} -> {out}, {out.with_suffix('.txt')}, {out.with_suffix('.md')}", flush=True)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
