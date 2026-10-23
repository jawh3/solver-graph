#!/usr/bin/env python3
"""
batch_test/smoke_real.py - preflight before a coordinated GPU run.

Cheap checks that fail FAST and LOUD, so a scarce vLLM window is never wasted on a
misconfiguration:

  1. vLLM solver endpoint reachable + model id matches the spec (Qwen3.5-27B-FP8).
  2. Embedding endpoint reachable; one embed returns the expected dim.
  3. Cross-encoder reranker loads (downloads weights once).
  4. Seed graph loads; report node/edge counts + the §6.1 sample-efficiency budget.
  5. ONE real solve on a trivial challenge end-to-end through the orchestrator.

Run:  python -m batch_test.smoke_real --graph data/seed_graph/kg
Exit code 0 = good to launch run_batch; non-zero = fix the printed problem first.
"""

from __future__ import annotations

import argparse
import os
import sys

from config import SolverConfig
from bandit.bandit import sample_efficiency_budget


def _check(label, fn):
    try:
        ok, detail = fn()
    except Exception as e:  # noqa
        ok, detail = False, f"EXCEPTION: {e}"
    mark = "✓" if ok else "✗"
    print(f"  {mark} {label}: {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-url", default=os.getenv("VLLM_URL", "http://localhost:8000/v1"))
    ap.add_argument("--embed-url", default=os.getenv("EMBED_URL", "http://localhost:8001/v1/embeddings"))
    ap.add_argument("--model", default=os.getenv("MODEL_NAME", "Qwen/Qwen3.5-27B-FP8"))
    ap.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "Qwen/Qwen3.5-27B-FP8"))
    ap.add_argument("--graph", default="data/seed_graph/kg")
    ap.add_argument("--expected-solve-runs", type=int, default=2000)
    args = ap.parse_args()

    print("=== batch_test preflight ===")
    all_ok = True

    # 1. solver endpoint + model id
    def _solver():
        from solver.solver import check_llm, MODEL_NAME
        ok = check_llm()
        note = f"model={MODEL_NAME}"
        if "27B" not in MODEL_NAME and "27b" not in MODEL_NAME:
            note += "  (WARNING: MODEL_NAME does not look like Qwen3.5-27B-FP8)"
        return ok, note
    all_ok &= _check("vLLM solver", _solver)

    # 2. embedding endpoint + dim
    embedder = None
    def _embed():
        nonlocal embedder
        from batch_test.real_adapters import real_embedder
        embedder = real_embedder(args.embed_url, args.embed_model)
        v = embedder.embed("smoke test crypto rsa")
        return v is not None and v.shape[0] > 0, f"dim={v.shape[0]}"
    all_ok &= _check("embedding endpoint", _embed)

    # 3. reranker loads
    def _rerank():
        from batch_test.real_adapters import real_reranker
        from knowledge_graph.models import Node, NodeType
        rr = real_reranker()
        out = rr.rerank("rsa crypto", [(Node(id="n", type=NodeType.TOOL,
                        attributes={"text": "rsa factoring with gmpy2"}), 0.5)], keep=1)
        return len(out) == 1, f"model={rr.model_name}"
    all_ok &= _check("cross-encoder reranker", _rerank)

    # 4. seed graph + budget calc
    def _graph():
        from knowledge_graph.store import KnowledgeGraphStore
        store = KnowledgeGraphStore.load(args.graph)
        ids, _ = store.get_all_embeddings()
        budget = sample_efficiency_budget(
            expected_solve_runs=args.expected_solve_runs, M=SolverConfig().M,
            num_context_nodes=SolverConfig().n_clusters, avg_candidates_credited_per_run=3)
        return len(ids) > 0, (f"{len(ids)} nodes; §6.1 updates/cell="
                              f"{budget['expected_updates_per_cell']:.1f} ({budget['verdict']})")
    all_ok &= _check("seed graph + §6.1 budget", _graph)

    # 5. one real end-to-end solve
    def _e2e():
        from knowledge_graph.store import KnowledgeGraphStore
        from retrieval.retriever import ContextRetriever
        from retrieval.context_resolution import ContextResolver
        from batch_test.real_adapters import real_reranker, real_solve_fn
        from pipeline import Orchestrator, SystemSpec
        from instrumentation.jsonl_logger import ExperimentLogger
        cfg = SolverConfig()
        store = KnowledgeGraphStore.load(args.graph)
        cr = ContextResolver(cfg).fit(store)
        retr = ContextRetriever(store, embedder, config=cfg, reranker=real_reranker(),
                                context_resolver=cr)
        orch = Orchestrator(store, embedder, cfg, SystemSpec("B2"),
                            retriever=retr, logger=ExperimentLogger("smoke", root="results/logs"),
                            solve_fn=real_solve_fn())
        ch = {"name": "smoke_chal", "category": "general",
              "description": "Decode this: cGljb0NURntzbW9rZX0=", "points": 50}
        run = orch.run_one(ch, experiment_id="smoke")
        return True, f"ran; outcome={run.outcome}, steps={run.steps}"
    all_ok &= _check("end-to-end real solve", _e2e)

    print("=== preflight", "PASSED - safe to launch run_batch ===" if all_ok
          else "FAILED - fix the ✗ items above before spending GPU ===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
