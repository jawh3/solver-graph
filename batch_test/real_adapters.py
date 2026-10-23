"""
batch_test/real_adapters.py - real-GPU backends behind the mock interfaces.

The orchestrator/eval take three pluggable backends: a solve_fn, an embedder, and
a reranker. In dev these are mocks; here they are the real ones. Nothing else in
the pipeline changes - same logging, same OPE, same metrics.

  real_solve_fn(...)      wraps solver.solver.solve_challenge into the orchestrator's
                          solve_fn(challenge, sampled_nodes, max_rounds) signature,
                          rendering the M sampled nodes via assemble_context into
                          `injected_rag_context` (STATIC injection, §7.7 - NOT the
                          MCP dynamic path, which is the future refresh ablation).
  real_embedder(...)      VLLMEmbedder against a DEDICATED embedding endpoint
                          (not the solver model), disk-cached.
  real_reranker(...)      CrossEncoderReranker (sentence-transformers).
  seed_graph_factory(...) loads the serialized seed graph fresh per system so a
                          system's write-back cannot leak into another (eval rule).

These are import-light at module load; heavy deps (sentence-transformers) are
imported lazily inside the real reranker so the module imports on a laptop too.
"""

from __future__ import annotations

import copy
import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from config import SolverConfig
from knowledge_graph.models import Node
from knowledge_graph.store import KnowledgeGraphStore
from retrieval.embedder import VLLMEmbedder, DiskCacheEmbedder
from retrieval.reranker import CrossEncoderReranker
from solver.context_assembly import assemble_context


def real_embedder(endpoint_url: str, model_name: str,
                  cache_dir: str = "results/embed_cache") -> DiskCacheEmbedder:
    """Disk-cached real embedder against a DEDICATED embedding endpoint."""
    base = VLLMEmbedder(endpoint_url=endpoint_url, model_name=model_name)
    return DiskCacheEmbedder(base, cache_dir=cache_dir, namespace=f"vllm:{model_name}")


def real_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name=model_name)


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class ArtifactCorrupted(RuntimeError):
    """The SOURCE artifact changed during a solve. With isolation on this cannot happen unless
    the solver escaped its working directory; it is raised, never logged as a solve outcome."""


def isolate_artifacts(challenge: dict, workroot: str) -> Tuple[dict, Path, Dict[str, str]]:
    """Review finding (2026-09-04, methodology): the solver used to run INSIDE the corpus
    directory, so leftover files (decoded.txt, decode.py, 16 true flags) and in-place damage
    (the malignant `truncate` arm corrupted 41-47% of artifacts) carried over across runs,
    epochs and CONDITIONS. Every solve now gets a fresh directory holding a COPY of each
    artifact; the originals are hash-verified after the solve and the copy is deleted."""
    d = Path(workroot) / ("iso-" + uuid.uuid4().hex[:12])
    d.mkdir(parents=True, exist_ok=True)
    ch = copy.deepcopy(challenge)
    hashes: Dict[str, str] = {}
    for i, lf in enumerate(ch.get("local_files", []) or []):
        src = lf.get("local_path")
        if src and os.path.exists(src):
            dst = d / Path(src).name
            if dst.exists():                         # two artifacts with the same basename
                dst = d / f"{i}_{Path(src).name}"
            shutil.copy2(src, dst)
            lf["local_path"] = str(dst)
            hashes[src] = _sha1_file(src)
    return ch, d, hashes


def real_solve_fn(use_tools: bool = True, max_output_tokens: int = 512,
                  context_window: int = 8192, trim_fraction: float = 0.6,
                  isolate: Optional[bool] = None, workroot: Optional[str] = None) -> Callable:
    """Adapter: orchestrator solve_fn -> solver.solver.solve_challenge (real Qwen).

    The M sampled lesson nodes are rendered to a static RAG block and injected via
    `injected_rag_context`; `use_memory=False` so the MCP dynamic path stays off
    (the future refresh ablation, §8).

    `context_window` is capped (8192 default) for the 8 GB laptop GPU: the solver
    trims history to ~60% of it, bounding the KV cache so generation fits VRAM.
    """
    from solver.solver import solve_challenge   # imports openai; real path only
    if isolate is None:
        isolate = os.environ.get("CTF_SOLVE_ISOLATE", "1") != "0"
    workroot = workroot or os.environ.get("CTF_SOLVE_WORKROOT", "results/_solve_work")

    def _fn(challenge: dict, sampled_nodes: List[Node], max_rounds: int, seed: int = None,
            order_seed: int = None, flag_checker=None) -> dict:
        injected = assemble_context(sampled_nodes, order_seed=order_seed) if sampled_nodes else ""
        # F1 (FIX_CATALOG 2026-09): the solver sees the neutral `_display_name`, never the
        # family-encoding logging id. Copy so the caller's record keeps its id.
        if isolate:
            shown, iso_dir, hashes = isolate_artifacts(challenge, workroot)
        else:
            shown, iso_dir, hashes = dict(challenge), None, {}
        if challenge.get("_display_name"):
            shown["name"] = challenge["_display_name"]
        # review 2026-09-05 F-6: `difficulty` (derived from the inner tags) and `points` were rendered
        # into the prompt; private underscore keys were handed to solve_challenge. The solver sees a
        # neutral difficulty and no private field; the caller's record keeps everything.
        shown["difficulty"] = "unknown"; shown["points"] = 0
        for k in [k for k in shown if str(k).startswith("_")]:
            shown.pop(k, None)
        try:
            res = solve_challenge(
                shown,
                use_tools=use_tools,
                use_memory=False,
                injected_rag_context=injected,
                max_rounds=max_rounds,
                max_output_tokens=max_output_tokens,
                context_window=context_window,
                llm_seed=seed,
                advertise_memory_tools=False,      # F4a: memory tools are off on this path
                flag_checker=flag_checker,         # F4c: wrong candidates do not end the run
                trim_fraction=trim_fraction,       # F4d
            )
        finally:
            if iso_dir is not None:
                shutil.rmtree(iso_dir, ignore_errors=True)
        for src, h in hashes.items():
            if not os.path.exists(src) or _sha1_file(src) != h:
                raise ArtifactCorrupted(f"source artifact changed during solve: {src}")
        res = dict(res); res["isolated"] = bool(isolate)
        return res

    return _fn


def seed_graph_factory(graph_base_path: str) -> Callable[[], Tuple[KnowledgeGraphStore, object]]:
    """Return a factory that loads a FRESH copy of the seed graph per system.

    `graph_base_path` is the base path passed to KnowledgeGraphStore.save/load
    (the .json/.npy pair). The embedder is attached by the caller (eval harness),
    so this returns (store, None) and the caller supplies the embedder - but to
    keep the eval's GraphFactory contract (returns (store, embedder)), we bind the
    embedder via closure in `make_real_factory` below.
    """
    def _factory():
        store = KnowledgeGraphStore.load(graph_base_path)
        return store
    return _factory


def make_real_factory(graph_base_path: str, embedder) -> Callable[[], Tuple[KnowledgeGraphStore, object]]:
    """GraphFactory for the eval harness: (fresh seed store, shared real embedder)."""
    def _factory():
        store = KnowledgeGraphStore.load(graph_base_path)
        return store, embedder
    return _factory
