"""realbench/lessons.py - the Cybench memory campaign's lesson layer.

Three pieces, each usable without a GPU (mock embedder/reranker/client in tests):

  1. DISTILLER  `distill_task(task, client, model)` - one typed lesson per Cybench task, written by the served model from the
     task's reference solution and public write-up (never from a handout or a solver trace). The lesson carries no flag
     (every known flag and every `<prefix>{...}` of a known answer format is scrubbed from the inputs AND the output) and
     never names the task (the solver only ever sees the neutral `task-<sha1[:8]>` id; `cybench.redact_task_name`).
  2. POOL       `LessonPool` - the distilled lessons as Phase-2 `Node`s (TECHNIQUE, tier trusted, `invocation_signature`
     + `keywords` for the action-grounded citation matcher, `applicability` rendered by `solver.context_assembly`).
  3. RETRIEVAL  `make_context_provider(...)` - leave-one-task-out retrieval through the Phase-2 pipeline
     (`ContextRetriever`: recall over the whole pool -> cross-encoder rerank -> MMR -> deterministic top-M, bandit
     context = the task's category, no learning). A task's OWN lesson is structurally excluded: the store handed to the
     retriever for task T never contains lesson(T). The provider returns the rendered context, the sampled nodes (for the
     citation matcher) and a logging dict (candidate pool with score components, policy).

The retrieval query is exactly what the solver sees at round 0 (neutral name + category + the unguided description with
the directory tree) - never the task name, the metadata prompts, or the reference solution.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from knowledge_graph.models import Node, NodeType
from knowledge_graph.store import KnowledgeGraphStore
from realbench import cybench as cb

LESSON_SCHEMA = 1
CATEGORIES = ("crypto", "web", "pwn", "reverse", "forensics", "misc")
_GENERIC_FLAG_PREFIXES = ("flag", "ctf")

# Audit F01/F15/F16 (2026-10-19): words that nearly every solve types, lessons or not - on NYU the no-memory traces
# "cited" 61-74% of lesson sets through words like these (file 91, objdump -d 23, nc 19, main() 13, checksec 12). A
# signature token made only of these words, option flags and punctuation never grounds a citation (`generic_keywords`).
GENERIC_CLI_TOKENS = frozenset("""
file strings grep egrep find xxd hexdump od nc ncat netcat curl wget python python3 pip pip3 bash sh cat ls echo head
tail less more sed awk cut sort uniq tr wc base64 unzip tar gzip gunzip 7z chmod cd mkdir rm cp mv touch printf main
main() checksec objdump readelf nm gdb ltrace strace ssh openssl gcc make import from pwn pwntools requests socket open
read write print flag flag.txt ctf key hex rsa web zip image type exploit payload solve solve.py http https localhost
r2 radare2 aaa afl pdf px db while for if chr ord int len range re.search pass user
""".split())


def generic_keywords(nodes: Sequence[Node], base: Iterable[str] = GENERIC_CLI_TOKENS,
                     chance: Optional[Dict[str, float]] = None, min_chance: float = 0.2) -> List[str]:
    """The lower-cased keywords of `nodes` that must not ground a citation: every keyword whose words are all in `base`,
    option flags (`-d`, `--file=`), numbers or punctuation; plus, when `chance` (keyword -> share of memory-OFF traces
    containing it, `solver.citation.chance_citeability`) is given, every keyword at or above `min_chance`."""
    base = {b.lower() for b in base}
    out = set(base)
    for n in nodes:
        a = n.attributes
        sig = a.get("invocation_signature") or []
        for kw in list(sig if isinstance(sig, list) else [sig]) + list(a.get("keywords") or []):
            k = str(kw).strip().lower()
            if not k:
                continue
            parts = [p for p in re.split(r"\s+", k) if p]
            if all(p in base or p.startswith("-") or re.fullmatch(r"[\W\d_]+", p) or p.rstrip("=:,;") in base for p in parts):
                out.add(k)
            if chance is not None and chance.get(str(kw).strip(), 0.0) >= min_chance:
                out.add(k)
    return sorted(out)

# ----------------------------------------------------------------------------------------------------------------------
# source material + scrubbing
# ----------------------------------------------------------------------------------------------------------------------


def flag_prefixes(tasks: Iterable[cb.CybenchTask]) -> List[str]:
    """Distinct `<prefix>` of every task's answer_format (`HTB{****}` -> `HTB`) plus the generic ones."""
    out: List[str] = []
    for t in tasks:
        m = re.match(r"\s*([A-Za-z0-9_]{2,16})\{", t.answer_format or "")
        if m and m.group(1).lower() not in {p.lower() for p in out}:
            out.append(m.group(1))
    for g in _GENERIC_FLAG_PREFIXES:
        if g not in {p.lower() for p in out}:
            out.append(g)
    return out


def scrub_flags(text: str, known_flags: Sequence[str], prefixes: Sequence[str], repl: str = "[FLAG]") -> Tuple[str, int]:
    """Remove every known flag (exact) and every `<known prefix>{...}` string. Returns (text, n_replaced).
    Only known prefixes are matched so LaTeX (`c_{i,0}`, `\\begin{align}`) survives."""
    n = 0
    for f in sorted({f for f in known_flags if f}, key=len, reverse=True):
        c = text.count(f)
        if c:
            text = text.replace(f, repl); n += c
    if prefixes:
        pat = re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(p) for p in prefixes) + r")\{[^}\n]{0,400}\}?", re.IGNORECASE)
        text, c = pat.subn(repl, text); n += c
    # audit F31: the generic prefixes also end longer identifiers (`csawctf{`, `csaw_ctf{`, `myflag{`)
    gen = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z0-9_]{0,24}(?:" + "|".join(_GENERIC_FLAG_PREFIXES) + r")\{[^}\n]{0,400}\}?", re.IGNORECASE)
    text, c = gen.subn(repl, text); n += c
    return text, n


def _clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"\n... [clipped {len(text) - limit} chars]"


def _writeup_solution_part(md: str) -> str:
    """HTB READMEs open with a story + skills; the solution part starts at a 'Solution'/'Exploit' heading when present."""
    m = re.search(r"^#+\s*(Solution|Exploitation|Exploit|Solving|Walkthrough|Approach)\b.*$", md, flags=re.M | re.I)
    return md[m.start():] if m else md


def gather_sources(task: cb.CybenchTask, *, max_chars_per_source: int = 6000, max_script_bytes: int = 120_000) -> Dict[str, str]:
    """The distiller's inputs for one task, task name redacted. Keys are stable (they head the prompt sections)."""
    d = Path(task.task_dir); sol = d / "metadata" / "solution"
    src: Dict[str, str] = {}
    meta = json.loads((d / "metadata" / "metadata.json").read_text())
    src["challenge_prompt"] = (meta.get("hard_prompt") or "").replace("$TMP_DIR", "the working directory")
    steps = []
    for i, s in enumerate(meta.get("subtasks") or [], 1):
        bits = [f"step {i}: {s.get('subtask', '')}"]
        if s.get("context"):
            bits.append(f"  why: {s['context']}")
        if s.get("hints"):
            bits.append("  hints: " + " | ".join(str(h) for h in s["hints"]))
        if s.get("commands"):
            bits.append("  commands: " + " ; ".join(str(c) for c in s["commands"]))
        steps.append("\n".join(bits))
    src["annotated_steps"] = "\n".join(steps)
    for name in ("solution.txt", "solution.sh"):
        p = sol / name
        if p.exists():
            src[name] = _clip(p.read_text(errors="replace"), max_chars_per_source)
    scripts = []
    for p in sorted(sol.glob("*")):
        if p.suffix in (".py", ".sage", ".rb", ".js", ".c", ".go") and p.is_file() and p.stat().st_size <= max_script_bytes:
            scripts.append(f"### {p.name}\n" + p.read_text(errors="replace"))
    if scripts:
        src["solver_scripts"] = _clip("\n\n".join(scripts), max_chars_per_source * 2)
    readme = d / "README.md"
    if readme.exists() and readme.stat().st_size:
        src["official_writeup"] = _clip(_writeup_solution_part(readme.read_text(errors="replace")), max_chars_per_source)
    cz = sorted(d.glob("solution/*.md"))
    if cz:
        src["public_writeup"] = _clip("\n\n".join(p.read_text(errors="replace") for p in cz), max_chars_per_source)
    return {k: cb.redact_task_name(v, task) for k, v in src.items() if v and v.strip()}


# ----------------------------------------------------------------------------------------------------------------------
# the distiller
# ----------------------------------------------------------------------------------------------------------------------

DISTILL_SYSTEM = """You write reusable lessons for the long-term memory of an autonomous CTF-solving agent.
The agent works in a Linux shell (python3, pwntools, gdb, radare2, binwalk, sqlmap, curl, ... available) and later reads
the lessons of OTHER challenges before attempting a new one. A good lesson lets it recognise the same situation and
reproduce the technique quickly. Rules:
- Describe the vulnerability class / technique and a concrete recipe with the exact tools, commands, library calls and
  parameters that worked. Concrete beats vague: name the functions, flags and file types.
- Never include a flag, a flag fragment, the challenge's name, the competition, or story text.
- Do not write the recipe for one challenge's file names only; state the recognisable preconditions (what to look for)
  so the lesson can be applied - or rejected - on a different challenge.
- Output ONLY one JSON object, no prose around it."""

DISTILL_USER = """Category: {category}
What the agent was told (the public prompt):
{challenge_prompt}

Reference material (annotated solution steps, the reference solver scripts and the public write-up):
{material}

Write the lesson as JSON with exactly these keys:
{{
  "title": "<= 10 words naming the technique (e.g. 'RSA with a prime modulus: phi = n-1')",
  "technique": "<= 4 words tag (e.g. 'rsa-prime-modulus')",
  "category": one of {categories},
  "recognise": ["2-5 short observations that identify this situation from the files / service alone"],
  "recipe": ["3-8 imperative steps; each names the exact tool/command/library call and parameters"],
  "invocation_signature": ["3-10 distinctive tokens that will literally appear in the shell commands or code when the
                            recipe is used: tool names, python functions, flags (e.g. 'long_to_bytes', 'pow(e, -1', 'binwalk -e')"],
  "applicability": {{"categories": ["..."], "preconditions": ["..."], "artifacts_required": ["file kinds / services needed"]}},
  "pitfalls": ["0-3 things that waste time or fail"]
}}"""


def _extract_json_object(text: str) -> Optional[dict]:
    """The outermost {...} object in an LLM reply (think tags and code fences stripped); tolerant of literal newlines."""
    if not text:
        return None
    text = re.sub(r"<(?:seed:|reasoning_)?think(?:ing)?>.*?</(?:seed:|reasoning_)?think(?:ing)?>", "", text, flags=re.S)
    text = re.sub(r"^.*?</(?:seed:|reasoning_)?think(?:ing)?>", "", text, flags=re.S)   # orphan closing tag
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0; in_str = False; esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
                continue
            if ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1], strict=False)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _str_list(v: Any, *, max_items: int, max_len: int) -> List[str]:
    if isinstance(v, str):
        v = [v]
    out: List[str] = []
    for x in (v or []):
        s = re.sub(r"\s+", " ", str(x)).strip()
        if s and s not in out:
            out.append(s[:max_len])
        if len(out) >= max_items:
            break
    return out


def validate_lesson(raw: dict, task: cb.CybenchTask, known_flags: Sequence[str], prefixes: Sequence[str]) -> Tuple[Optional[dict], List[str]]:
    """Normalise a raw lesson object into the stored schema. Returns (lesson, problems); lesson is None when unusable."""
    problems: List[str] = []
    if not isinstance(raw, dict):
        return None, ["not an object"]
    title = re.sub(r"\s+", " ", str(raw.get("title") or "")).strip()[:120]
    technique = re.sub(r"[^a-z0-9\-]+", "-", str(raw.get("technique") or "").lower()).strip("-")[:48]
    cat = str(raw.get("category") or task.category).strip().lower()
    cat = cb.normalize_category(cat) if cat else task.category
    if cat not in CATEGORIES:
        cat = task.category
    recipe = _str_list(raw.get("recipe"), max_items=8, max_len=300)
    recognise = _str_list(raw.get("recognise"), max_items=5, max_len=200)
    sig = [s for s in _str_list(raw.get("invocation_signature"), max_items=10, max_len=48) if len(s) >= 2]
    pitfalls = _str_list(raw.get("pitfalls"), max_items=3, max_len=200)
    app_raw = raw.get("applicability") if isinstance(raw.get("applicability"), dict) else {}
    app = {"categories": [c for c in _str_list(app_raw.get("categories"), max_items=4, max_len=20) if c.lower() in CATEGORIES] or [cat],
           "preconditions": _str_list(app_raw.get("preconditions"), max_items=5, max_len=200),
           "artifacts_required": _str_list(app_raw.get("artifacts_required"), max_items=5, max_len=80)}
    if not title: problems.append("missing title")
    if len(recipe) < 2: problems.append("recipe has fewer than 2 steps")
    if len(sig) < 2: problems.append("fewer than 2 invocation tokens")
    if problems:
        return None, problems
    lesson = {"schema": LESSON_SCHEMA, "task_id": task.id, "display_name": task.display_name, "task_category": task.category,
              "title": title, "technique": technique or "unnamed", "category": cat, "recognise": recognise, "recipe": recipe,
              "invocation_signature": sig, "applicability": app, "pitfalls": pitfalls, "scrubbed": 0, "name_redacted": False}
    # hygiene: no flag, no task name, anywhere in the RENDERED fields (task_id/display_name are private keys, never rendered)
    rendered_keys = ("title", "technique", "recognise", "recipe", "invocation_signature", "applicability", "pitfalls")
    blob = json.dumps({k: lesson[k] for k in rendered_keys})
    _, n_flag = scrub_flags(blob, known_flags, prefixes)
    if n_flag or cb.leaks_task_name(blob, task):
        def fix(s: str) -> str:
            s2, _ = scrub_flags(s, known_flags, prefixes)
            return cb.redact_task_name(s2, task)
        for k in ("title", "technique"):
            lesson[k] = fix(lesson[k])
        for k in ("recognise", "recipe", "invocation_signature", "pitfalls"):
            lesson[k] = [fix(s) for s in lesson[k]]
        for k in ("preconditions", "artifacts_required"):
            lesson["applicability"][k] = [fix(s) for s in lesson["applicability"][k]]
        lesson["scrubbed"] = int(n_flag); lesson["name_redacted"] = cb.leaks_task_name(blob, task)
        lesson["invocation_signature"] = [s for s in lesson["invocation_signature"] if "[FLAG]" not in s and "[redacted]" not in s]
        if len(lesson["invocation_signature"]) < 2:
            return None, ["invocation signature was a flag/name"]
    return lesson, []


@dataclass
class DistillResult:
    lesson: Optional[dict]
    attempts: int
    problems: List[str] = field(default_factory=list)
    raw: str = ""
    seconds: float = 0.0


def _chat_kwargs(model: str, messages: list, *, max_tokens: int, temperature: float, seed: int) -> dict:
    kw = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": seed}
    if os.environ.get("CTF_NO_CHAT_TEMPLATE_KWARGS") != "1":
        kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
    return kw


def _reply_text(resp) -> str:
    msg = resp.choices[0].message
    content = msg.content or ""
    if not content.strip():
        content = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
    return content


def distill_task(task: cb.CybenchTask, client, model: str, *, known_flags: Sequence[str], prefixes: Sequence[str],
                 max_attempts: int = 3, max_tokens: int = 6000, temperature: float = 0.2, seed: int = 7,
                 sources: Optional[Dict[str, str]] = None) -> DistillResult:
    """One lesson for `task` from the served model. Inputs are flag-scrubbed and name-redacted BEFORE the prompt; the
    output is validated (schema + hygiene) and re-prompted with the problems up to `max_attempts` times."""
    t0 = time.time()
    src = sources if sources is not None else gather_sources(task)
    material = []
    for k, v in src.items():
        if k == "challenge_prompt":
            continue
        v2, _ = scrub_flags(v, known_flags, prefixes)
        material.append(f"## {k}\n{v2}")
    prompt_text, _ = scrub_flags(src.get("challenge_prompt", ""), known_flags, prefixes)
    user = DISTILL_USER.format(category=task.category, challenge_prompt=prompt_text or "(none)", material="\n\n".join(material) or "(none)",
                               categories=list(CATEGORIES))
    messages = [{"role": "system", "content": DISTILL_SYSTEM}, {"role": "user", "content": user}]
    problems: List[str] = []; raw = ""
    for attempt in range(1, max_attempts + 1):
        resp = client.chat.completions.create(**_chat_kwargs(model, messages, max_tokens=max_tokens, temperature=temperature, seed=seed + attempt))
        raw = _reply_text(resp)
        obj = _extract_json_object(raw)
        if obj is None:
            problems = ["no JSON object in the reply"]
        else:
            lesson, problems = validate_lesson(obj, task, known_flags, prefixes)
            if lesson is not None:
                return DistillResult(lesson=lesson, attempts=attempt, raw=raw, seconds=time.time() - t0)
        messages = messages[:2] + [{"role": "assistant", "content": raw[-4000:]},
                                   {"role": "user", "content": "That reply was not usable: " + "; ".join(problems)
                                    + ". Reply again with ONLY the JSON object, all keys present."}]
    return DistillResult(lesson=None, attempts=max_attempts, problems=problems, raw=raw, seconds=time.time() - t0)


def distill_all(tasks: Sequence[Any], client, model: str, out_path: str | Path, *, parallel: int = 4,
                log: Callable[[str], None] = print, sources_fn: Optional[Callable] = None, **kw) -> Dict[str, Any]:
    """Distill every task into `out_path` (JSONL, one lesson per line, resumable: tasks already present are skipped).
    Returns a summary {n_tasks, n_ok, n_failed, failed: [...]}."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    have = {json.loads(l)["task_id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    known = [f for t in tasks for f in t.accepted_flags]
    prefixes = flag_prefixes(tasks)
    todo = [t for t in tasks if t.id not in have]
    log(f"[distill] {len(todo)} tasks to distill ({len(have)} already present) with {model}")
    failed = []; lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex, open(out, "a") as fh:
        futs = {ex.submit(distill_task, t, client, model, known_flags=known, prefixes=prefixes,
                          sources=(sources_fn(t) if sources_fn else None), **kw): t for t in todo}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa - one failure must not stop the pass
                r = DistillResult(lesson=None, attempts=0, problems=[f"{type(e).__name__}: {e}"[:300]])
            if r.lesson is None:
                failed.append({"task_id": t.id, "display_name": t.display_name, "problems": r.problems})
                log(f"  {t.display_name} FAILED after {r.attempts} attempts: {'; '.join(r.problems)[:200]}")
                continue
            rec = dict(r.lesson); rec.update({"model": model, "attempts": r.attempts, "seconds": round(r.seconds, 1),
                                              "distilled_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            with lock:
                fh.write(json.dumps(rec) + "\n"); fh.flush()
            log(f"  {t.display_name} ok ({r.attempts} attempt{'s' if r.attempts > 1 else ''}, {r.seconds:.0f}s): {rec['title']}")
    return {"n_tasks": len(tasks), "n_ok": len(tasks) - len(failed), "n_failed": len(failed), "failed": failed, "path": str(out)}


# ----------------------------------------------------------------------------------------------------------------------
# the pool (lessons as Phase-2 nodes)
# ----------------------------------------------------------------------------------------------------------------------


def lesson_node(lesson: dict) -> Node:
    """A distilled lesson as the TECHNIQUE node the Phase-2 renderer, retriever and citation matcher expect."""
    recipe = " ".join(f"{i}) {s}" for i, s in enumerate(lesson["recipe"], 1))
    recog = ("Recognise: " + "; ".join(lesson.get("recognise") or []) + ". ") if lesson.get("recognise") else ""
    pit = (" Pitfalls: " + "; ".join(lesson["pitfalls"])) if lesson.get("pitfalls") else ""
    text = f"{recog}Recipe: {recipe}{pit}"
    # audit F16 (2026-10-19): the technique tag's words ('flag', 'hex', 'rsa', 'zip', ...) are a label, not something a
    # solve executes - only the distilled invocation signature grounds a citation
    return Node(id=f"lesson::{lesson['display_name']}", type=NodeType.TECHNIQUE, attributes={
        "canonical_name": lesson["title"], "text": text, "category": lesson["category"],
        "invocation_signature": list(lesson["invocation_signature"]),
        "keywords": list(dict.fromkeys(lesson["invocation_signature"])),
        "tier": "trusted", "source": f"distilled:{lesson.get('model', '?')}",
        "applicability": {"categories": lesson["applicability"]["categories"],
                          "preconditions": lesson["applicability"]["preconditions"],
                          "artifacts_required": lesson["applicability"]["artifacts_required"]},
        "_task_id": lesson["task_id"], "_display_name": lesson["display_name"], "_technique": lesson.get("technique", "")})


class LessonPool:
    def __init__(self, lessons: Sequence[dict]):
        self.lessons = list(lessons)
        self.nodes = [lesson_node(l) for l in self.lessons]
        self.by_task = {n.attributes["_task_id"]: n for n in self.nodes}
        if len(self.by_task) != len(self.nodes):
            raise ValueError("duplicate lessons for one task in the pool")

    @classmethod
    def load(cls, path: str | Path) -> "LessonPool":
        return cls([json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()])

    def nodes_excluding(self, task_id: str) -> List[Node]:
        return [n for n in self.nodes if n.attributes["_task_id"] != task_id]


# ----------------------------------------------------------------------------------------------------------------------
# retrieval (leave-one-task-out, Phase-2 pipeline, deterministic top-M)
# ----------------------------------------------------------------------------------------------------------------------


def build_embedder(backend: str = "sentence_transformers", cache_dir: Optional[str] = "results/embed_cache",
                   model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
    """The retrieval embedder (spec §7 keeps it separate from the solver model). CPU by default: vLLM owns the GPUs."""
    from retrieval.embedder import DiskCacheEmbedder, MockEmbedder, SentenceTransformerEmbedder
    if backend == "mock":
        return MockEmbedder(dim=64, seed=1)
    base = SentenceTransformerEmbedder(model_name=model_name, device=device)
    return DiskCacheEmbedder(base, cache_dir=cache_dir, namespace=f"st:{model_name}") if cache_dir else base


def build_reranker(backend: str = "cross_encoder", device: str = "cpu"):
    from retrieval.reranker import CrossEncoderReranker, MockReranker
    return MockReranker() if backend == "mock" else CrossEncoderReranker(device=device)


def query_text_for(shown_challenge: dict) -> str:
    """What the retriever is allowed to see: the solver-visible round-0 fields (F1: never the logging id)."""
    from pipeline.orchestrator import challenge_query_text
    return challenge_query_text(shown_challenge)


class LeaveOneOutRetriever:
    """Phase-2 `ContextRetriever` over the pool minus the task's own lesson; deterministic top-M; category context."""

    def __init__(self, pool: LessonPool, embedder, reranker, *, m: int = 3, embedding_dim: Optional[int] = None):
        from config import ContextResolution, SolverConfig
        self.pool, self.embedder, self.reranker, self.m = pool, embedder, reranker, m
        dim = embedding_dim or int(np.asarray(embedder.embed("dimension probe")).shape[0])
        n = max(1, len(pool.nodes))
        self.config = SolverConfig(N=max(200, n), R=max(50, n), M=m, embedding_dim=dim,
                                   context_resolution=ContextResolution.CATEGORY, epsilon_explore=0.0)
        self._emb: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        for node in pool.nodes:                     # embed once (cached on disk by DiskCacheEmbedder)
            self._emb[node.id] = np.asarray(embedder.embed(_node_text(node)), dtype=np.float32)

    def retrieve(self, shown_challenge: dict, task_id: str, *, bandit=None, explore: bool = False,
                 deterministic: bool = True, rng=None, rank_by_score: bool = False) -> Tuple[List[Node], Dict[str, Any]]:
        """Static (default): bandit=None, deterministic top-M of the rerank/MMR pool - the ON-static arm.
        Learning-train: bandit set, deterministic=False, explore=True, a per-(task,seed) rng - softmax-without-replacement
        sampling over the bandit-blended score (the exploration).  Learning-eval: bandit set, deterministic=True,
        rank_by_score=True - greedy top-M by the LEARNED score (the retriever's deterministic mode keeps MMR order, so the
        selection is re-done here from the candidate pool's score)."""
        from retrieval.retriever import ContextRetriever
        nodes = self.pool.nodes_excluding(task_id)
        store = KnowledgeGraphStore()
        for node in nodes:
            store.add_node(node, self._emb[node.id])
        query = query_text_for(shown_challenge)
        with self._lock:                            # the CPU cross-encoder / bi-encoder are shared across solve threads
            r = ContextRetriever(store, self.embedder, config=self.config, reranker=self.reranker, bandit=bandit).retrieve(
                query, category=str(shown_challenge.get("category") or ""), rng=rng,
                deterministic=deterministic, use_bandit=(bandit is not None), explore=explore)
        sampled = list(r.sampled_nodes)
        if rank_by_score and bandit is not None:    # greedy exploitation: the M highest LEARNED scores
            by_id = {n.id: n for n in nodes}
            ranked = sorted(r.candidate_pool, key=lambda c: -float(c.get("score") or 0.0))
            sampled = [by_id[c["node_id"]] for c in ranked[:self.m] if c["node_id"] in by_id]
        cos = {c["node_id"]: c.get("cosine") for c in r.candidate_pool}
        qn = {c["node_id"]: (c.get("Q"), c.get("n")) for c in r.candidate_pool}
        meta = {"query_chars": len(query), "context_node_id": r.context_node_id, "pool_size": len(nodes),
                "excluded": task_id in self.pool.by_task, "policy": r.policy, "exploration": bool(r.exploration),
                "sampled_cosines": {n.id: cos.get(n.id) for n in sampled},
                "sampled_Qn": {n.id: qn.get(n.id) for n in sampled},
                # audit F20 / spec hard rule 1: the draw's propensities are non-reconstructable, so they are logged
                "per_draw_propensity": [{"node_id": d.get("node_id"), "p": (round(float(d["p"]), 6) if d.get("p") is not None else None)}
                                        for d in (r.per_draw_propensity or [])],
                "set_logprob": float(r.set_logprob or 0.0), "temperature": float(r.temperature or 0.0),
                "candidates": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k in ("node_id", "cosine", "rerank_score", "mmr_kept", "score", "Q", "n")}
                               for c in sorted(r.candidate_pool, key=lambda c: -float(c.get("score") or c.get("rerank_score") or 0.0))[:10]]}
        assert all(n.attributes["_task_id"] != task_id for n in sampled), "leave-one-out violated"
        return sampled, meta


def _node_text(node: Node) -> str:
    a = node.attributes
    return str(a.get("canonical_name") or "") + ". " + str(a.get("text") or "")


def make_context_provider(retriever: LeaveOneOutRetriever, *, randomize_order: bool = True) -> Callable:
    """`provider(shown_challenge, task, llm_seed) -> (context_str, nodes, meta)` for `solve_cybench_task`.
    Lesson order in the rendered block is a hash of (task, llm_seed) - the Phase-2 fix for the primacy confound - so the
    paired OFF/ON runs are reproducible and the order is not tied to retrieval rank."""
    from solver.context_assembly import assemble_context

    def provider(shown: dict, task: cb.CybenchTask, llm_seed: Optional[int]):
        nodes, meta = retriever.retrieve(shown, task.id)
        order_seed = (int(hashlib.sha1(f"order|{task.id}|{llm_seed}".encode()).hexdigest()[:8], 16) if randomize_order else None)
        ctx = assemble_context(nodes, order_seed=order_seed)
        meta.update({"lessons": [n.id for n in nodes], "lesson_titles": [n.attributes.get("canonical_name") for n in nodes],
                     "lesson_categories": [n.attributes.get("category") for n in nodes], "order_seed": order_seed,
                     "context_chars": len(ctx), "m": retriever.m})
        return ctx, nodes, meta
    return provider
