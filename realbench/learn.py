#!/usr/bin/env python3
"""realbench/learn.py - the Cybench memory campaign's ON-learned arm.

Instead of serving a fixed top-3 (ON-static), the Phase-2 contextual bandit LEARNS which lessons to retrieve, from the
model's own solve outcomes, with the exact Phase-2 credit: cited-only outcome credit, sim-prior blend (n·Q + κ·prior)/(n+κ),
asymmetric distance scaling, running-mean baseline. Context = the task's category. Leave-one-task-out throughout.

Two phases, per model:
  TRAIN (this module's `train`): over (task, train-seed, epoch), retrieve WITH the bandit by softmax-without-replacement
    sampling (the exploration), solve in the real container, mark which injected lessons were executed against (the citation
    matcher), turn the outcome into reward, and update the bandit on the sampled lessons (cited → outcome reward; uncited →
    exposure, advantage 0 so the prior's weight decays). Batched at the serving parallelism: a batch retrieves under one frozen
    bandit state, then its updates are applied in a deterministic order - a batched bandit, the necessary deviation from a
    strictly-sequential learner because a real solve is minutes, not milliseconds. Resumable: the bandit + a done-log are
    checkpointed after every batch.
  EVAL (`run_cybench --bandit`): freeze the bandit, retrieve the greedy top-M by the LEARNED score (no exploration), solve on
    the held-out eval seeds (the OFF/ON-static seeds), pair against OFF.

The train/eval split is over SEEDS, not items - with 29 tasks there is no held-out ITEM set; the bandit is keyed on category,
so it learns category-level lesson values on the train seeds and is scored on held-out seeds. This weakness is pre-specified.
(NYU CTF Bench, 2026-09-25: `--bench nyuctf --split development` trains on the development split, an item-level split.)

Audit fixes 2026-10-19: per-epoch LLM/retrieval seeds (`train_llm_seed`, F04); every training record with its propensities
persisted to `<bandit>.train_records.jsonl` (F20); the checkpoint is saved before the done-log (F34); the sandbox is
required (N1). Open, NOT fixed (learner design, docs/reviews/AUDIT_NYU_MEMORY_2026-10-19.md): F02 (learned and static rank
on different signals), F05 (an exposed-but-uncited lesson falls below a never-exposed one under zero init), F18, F19, F28.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from realbench import cybench as cb
from realbench.lessons import LessonPool, LeaveOneOutRetriever, build_embedder, build_reranker
from solver.context_assembly import assemble_context


def _seed_rng(task_id: str, llm_seed: Optional[int]) -> np.random.RandomState:
    h = int(hashlib.sha1(f"retr|{task_id}|{llm_seed}".encode()).hexdigest()[:8], 16)
    return np.random.RandomState(h)


EPOCH_SEED_STRIDE = 10_000


def train_llm_seed(seed: int, epoch: int) -> int:
    """Audit F04 (2026-10-19): the LLM seed - and through it the retrieval draw (`_seed_rng`) and the lesson order - must
    differ per epoch; keyed on (task, seed) alone, epoch 1 re-drew epoch 0's lesson sets for 57 of 90 NYU training pairs.
    Epoch 0 keeps the bare seed (2000, 2001, ...); epoch e adds e × 10,000, clear of the eval seeds (1000-1002)."""
    return int(seed) + EPOCH_SEED_STRIDE * int(epoch)


class BanditMemory:
    """A persistent Phase-2 bandit + reward standardizer over the lesson pool, with train/eval retrieval providers."""

    def __init__(self, pool: LessonPool, embedder, reranker, *, m: int = 3):
        from bandit.bandit import TabularBandit
        from config import DistanceScaling, InitMode
        from reward.reward import RewardStandardizer
        import dataclasses
        self.retriever = LeaveOneOutRetriever(pool, embedder, reranker, m=m)
        # The F41 exact-running-mean configuration (deviates from the Phase-2 microbenchmark defaults, which suit a different
        # structure): CATEGORY context, κ=5, and
        #   alpha_schedule="mean"  Q = exact running mean of the credit targets (step 1/(n+1)); the slow EMA (α=0.1) barely
        #                          moves off init in the ~116 training solves this arm can afford.
        #   distance_scaling=NONE  full credit (scale=1); asymmetric down-weights exactly the high-similarity lessons we want
        #                          to learn about, and an exact mean needs scale=1.
        #   init_mode=ZERO         a learned Q is the advantage over the similarity prior; a cold lesson scores at its prior
        #                          (where ON-static starts), so learning only moves a lesson off that. Optimistic init would
        #                          rank the least-tried lesson highest - the opposite of exploitation at eval.
        cfg = dataclasses.replace(self.retriever.config, alpha_schedule="mean",
                                  distance_scaling=DistanceScaling.NONE, init_mode=InitMode.ZERO)   # SolverConfig is frozen
        self.retriever.config = cfg
        self.config = cfg
        self.bandit = TabularBandit(config=cfg)
        self.standardizer = RewardStandardizer(config=cfg)
        self.m = m
        self._lock = threading.Lock()

    # --- providers for solve_cybench_task -----------------------------------------------------------------
    def _provider(self, *, train: bool, randomize_order: bool = True) -> Callable:
        def provider(shown: dict, task, llm_seed: Optional[int]):
            if train:
                nodes, meta = self.retriever.retrieve(shown, task.id, bandit=self.bandit, explore=True,
                                                      deterministic=False, rng=_seed_rng(task.id, llm_seed))
            else:
                nodes, meta = self.retriever.retrieve(shown, task.id, bandit=self.bandit, deterministic=True, rank_by_score=True)
            order_seed = (int(hashlib.sha1(f"order|{task.id}|{llm_seed}".encode()).hexdigest()[:8], 16) if randomize_order else None)
            ctx = assemble_context(nodes, order_seed=order_seed)
            meta.update({"lessons": [n.id for n in nodes], "lesson_titles": [n.attributes.get("canonical_name") for n in nodes],
                         "lesson_categories": [n.attributes.get("category") for n in nodes], "order_seed": order_seed,
                         "context_chars": len(ctx), "m": self.m, "condition": ("learned-train" if train else "learned")})
            return ctx, nodes, meta
        return provider

    def train_provider(self) -> Callable:
        return self._provider(train=True)

    def eval_provider(self, randomize_order: bool = True) -> Callable:
        return self._provider(train=False, randomize_order=randomize_order)

    # --- the credit update (mirrors pipeline.orchestrator, OUTCOME + CITED_ONLY_DECAY) ---------------------
    def update(self, rec, task) -> Optional[dict]:
        """One bandit update from a finished training solve. Returns a compact log dict (or None if no memory)."""
        mem = (rec.extra or {}).get("memory")
        if not mem or not mem.get("lessons"):
            return None
        ctx = mem["context_node_id"]; category = task.category
        outcome = int(bool(rec.solved)); steps = int(rec.rounds_used or 0)
        cited = set(mem.get("cited") or [])
        cosines = mem.get("sampled_cosines") or {}
        with self._lock:                              # updates are applied one batch at a time; the lock guards concurrent batches (there are none) + is cheap
            self.standardizer.observe(category, "unknown", steps)
            rb = self.standardizer.reward(outcome, steps, category, "unknown")
            self.bandit.observe_reward_for_baseline(ctx, rb.R_final)
            ups = []
            for nid in mem["lessons"]:
                is_cited = nid in cited
                reward = rb.R_final if is_cited else self.bandit.baseline(ctx)   # CITED_ONLY_DECAY: uncited -> exposure (advantage 0)
                sim = cosines.get(nid)
                sim = float(sim) if sim is not None else 0.5
                pre_Q, post_Q, pre_n, post_n, scale = self.bandit.update(ctx, nid, reward, similarity=sim, success=bool(outcome), subtract_baseline=True)
                ups.append({"node": nid, "cited": is_cited, "reward": round(reward, 3), "Q": round(post_Q, 3), "n": post_n})
        return {"task": task.display_name, "context": ctx, "solved": bool(outcome), "R": round(rb.R_final, 3), "updates": ups}

    # --- persistence --------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"schema": 1, "m": self.m, "bandit": self.bandit.to_dict(),
                                          "standardizer": self.standardizer.to_dict()}, indent=1))

    @classmethod
    def load(cls, path: str | Path, pool: LessonPool, embedder, reranker, *, m: Optional[int] = None) -> "BanditMemory":
        from bandit.bandit import TabularBandit
        from reward.reward import RewardStandardizer
        data = json.loads(Path(path).read_text())
        self = cls(pool, embedder, reranker, m=(m or data.get("m", 3)))
        self.bandit = TabularBandit.from_dict(data["bandit"], config=self.config)
        self.standardizer = RewardStandardizer.from_dict(data["standardizer"], config=self.config)
        return self

    def learned_table(self) -> Dict[str, List[dict]]:
        """A readable view of what was learned: per category, the lessons by descending Q."""
        pool_by_id = {n.id: n for n in self.retriever.pool.nodes}
        out: Dict[str, List[dict]] = {}
        for ctx, cells in self.bandit.cells.items():
            rows = [{"node": nid, "Q": round(c.Q, 3), "n": c.n,
                     "title": pool_by_id[nid].attributes.get("canonical_name") if nid in pool_by_id else "?"}
                    for nid, c in cells.items()]
            out[ctx] = sorted(rows, key=lambda r: -r["Q"])
        return out


# ----------------------------------------------------------------------------------------------------------
# training loop
# ----------------------------------------------------------------------------------------------------------


def train(mem: BanditMemory, tasks: Sequence[cb.CybenchTask], *, docker, image: str, workroot: str,
          seeds: Sequence[int], epochs: int, parallel: int, service_parallel: int, bandit_path: str | Path,
          done_path: str | Path, trainlog_path: str | Path, solve_kwargs: Dict[str, Any],
          log: Callable[[str], None] = lambda s: print(s, flush=True), records_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """`records_path` (audit F20): every training solve's full record - candidate pool, per-draw propensities, set
    log-prob, citations, trace - is appended there (default: next to the train log)."""
    from realbench.solve_adapter import solve_cybench_task
    done_path, bandit_path, trainlog_path = Path(done_path), Path(bandit_path), Path(trainlog_path)
    records_path = Path(records_path) if records_path else Path(str(trainlog_path).replace(".train_log.jsonl", ".train_records.jsonl"))
    done = set()
    if done_path.exists():
        for l in done_path.read_text().splitlines():
            if l.strip():
                d = json.loads(l); done.add((d["task_id"], int(d["seed"]), int(d["epoch"])))
    if bandit_path.exists() and done:
        mem2 = BanditMemory.load(bandit_path, mem.retriever.pool, mem.retriever.embedder, mem.retriever.reranker, m=mem.m)
        mem.bandit, mem.standardizer = mem2.bandit, mem2.standardizer      # resume from the checkpoint
        log(f"[learn] resumed: {len(done)} train steps already done")
    # 2026-09-25: within each (epoch, seed) the tasks are interleaved over hostname groups (run_cybench.interleave_jobs), and
    # service solves hold a lock per hostname under the --service-parallel gate - same rule as the runner: two live tasks on
    # one shared alias (NYU: pwn.chal.csaw.io) would resolve to the wrong container; different hostnames may overlap.
    from collections import defaultdict
    from contextlib import ExitStack
    from realbench.run_cybench import interleave_jobs
    steps = []
    for e in range(epochs):
        for s in seeds:
            group = [t for t in tasks if (t.id, s, e) not in done]
            steps += [(t, s, e) for t, _k in interleave_jobs([(t, 0) for t in group])]
    provider = mem.train_provider()
    service_gate = threading.Semaphore(max(1, service_parallel))
    host_locks = defaultdict(threading.Lock)
    df = open(done_path, "a"); tf = open(trainlog_path, "a"); rf = open(records_path, "a")

    def one(t, s, e):
        def go():
            return solve_cybench_task(t, workroot, docker=docker, image=image, context_provider=provider, llm_seed=train_llm_seed(s, e),
                                      extra={"seed": s, "epoch": e, "phase": "train", "model": os.environ.get("MODEL_NAME", "")}, **solve_kwargs)
        if t.needs_service:
            hosts = sorted({h for h, _p in t.target_hosts}) or [t.id]
            with ExitStack() as st:
                for h in hosts:
                    st.enter_context(host_locks[h])
                with service_gate:
                    return go()
        return go()

    t0 = time.time(); n_done = len(done); n_total = n_done + len(steps)
    for i in range(0, len(steps), parallel):
        batch = steps[i:i + parallel]
        results = []
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
            futs = {ex.submit(one, t, s, e): (t, s, e) for (t, s, e) in batch}
            for fut in as_completed(futs):
                results.append((futs[fut], fut.result()))
        # apply updates AFTER the batch joins, in a deterministic order (epoch, seed, task) - reproducible
        finished = []
        for (t, s, e), rec in sorted(results, key=lambda x: (x[0][2], x[0][1], x[0][0].id)):
            n_done += 1
            rf.write(json.dumps(rec.to_dict()) + "\n")
            if rec.error:
                log(f"  train {t.display_name} seed {s} ep {e}: ERROR {rec.error} (no update)")
            else:
                upd = mem.update(rec, t)
                tf.write(json.dumps({"task_id": t.id, "seed": s, "epoch": e, "llm_seed": rec.llm_seed, "solved": bool(rec.solved),
                                     "rounds": rec.rounds_used, "update": upd}) + "\n")
            finished.append({"task_id": t.id, "seed": s, "epoch": e})
        rf.flush(); tf.flush()
        mem.save(bandit_path)                          # checkpoint after every batch ...
        for d in finished:                             # ... BEFORE the done-log (audit F34: a crash in between used to
            df.write(json.dumps(d) + "\n")             # resume past updates the checkpoint never received)
        df.flush()
        el = time.time() - t0
        log(f"[learn] {n_done}/{n_total} train solves ({el/60:.0f} min, {el/max(1,(n_done-len(done))):.0f}s/solve); bandit cells: "
            f"{sum(len(v) for v in mem.bandit.cells.values())} over {len(mem.bandit.cells)} categories")
    df.close(); tf.close(); rf.close()
    mem.save(bandit_path)
    summ = {"train_solves": n_total, "epochs": epochs, "seeds": list(seeds), "bandit": str(bandit_path),
            "learned": mem.learned_table()}
    Path(str(bandit_path).replace(".json", ".learned.json")).write_text(json.dumps(summ["learned"], indent=1))
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("CYBENCH_ROOT", "<path>"))
    ap.add_argument("--cyber-zero-root", default=os.environ.get("CYBER_ZERO_CYBENCH_ROOT", "<path>"))
    ap.add_argument("--tasks", default="all"); ap.add_argument("--lessons", required=True)
    ap.add_argument("--out-bandit", required=True, help="where to save the learned bandit (e.g. results/realbench_mem/<slug>/bandit.json)")
    ap.add_argument("--lessons-m", type=int, default=3); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--train-seed0", type=int, default=2000); ap.add_argument("--train-trials", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=3); ap.add_argument("--service-parallel", type=int, default=1)
    ap.add_argument("--max-rounds", type=int, default=10); ap.add_argument("--max-output-tokens", type=int, default=8192)
    ap.add_argument("--context-window", type=int, default=32768); ap.add_argument("--trim-fraction", type=float, default=0.7)
    ap.add_argument("--command-timeout", type=float, default=300.0); ap.add_argument("--image", default=os.environ.get("REALBENCH_IMAGE", "cybench/cybench:latest"))
    ap.add_argument("--container-memory", default="8g"); ap.add_argument("--container-cpus", default="4"); ap.add_argument("--container-user", default="root")
    ap.add_argument("--tool-calling", action="store_true")
    ap.add_argument("--embedder", default="sentence_transformers", choices=["sentence_transformers", "mock"])
    ap.add_argument("--reranker", default="cross_encoder", choices=["cross_encoder", "mock"]); ap.add_argument("--embed-cache", default="results/embed_cache")
    ap.add_argument("--workroot", default="results/_work/agents")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--review", action="store_true", help="print the learned bandit table and exit")
    from realbench.run_cybench import add_bench_args, add_harness_args, check_workroot, load_bench, load_generic_tokens
    add_bench_args(ap)
    add_harness_args(ap)
    args = ap.parse_args(argv)
    bench, tasks = load_bench(args)
    if not tasks:
        sys.exit("no tasks selected")
    pool = LessonPool.load(args.lessons)
    emb = build_embedder(args.embedder, cache_dir=args.embed_cache or None)
    rer = build_reranker(args.reranker)
    if args.review:
        mem = BanditMemory.load(args.out_bandit, pool, emb, rer, m=args.lessons_m)
        print(json.dumps(mem.learned_table(), indent=1)); return 0
    seeds = [args.train_seed0 + k for k in range(args.train_trials)]
    n = len(tasks) * len(seeds) * args.epochs
    if args.dry_run:
        print(f"{len(tasks)} tasks x {len(seeds)} train seeds {seeds} x {args.epochs} epochs = {n} training solves; "
              f"bandit -> {args.out_bandit}; lessons {len(pool.nodes)} from {args.lessons}"); return 0
    from solver import tool_executor
    tool_executor.require_sandbox()                # audit N1: no solver command may ever run on the host
    check_workroot(args.workroot)
    docker = cb.RealDockerOps(internal=(args.egress == "blocked")); docker.ensure_network(bench.SHARED_NET)
    if not docker.image_exists(args.image):
        sys.exit(f"image {args.image} missing")
    mem = BanditMemory(pool, emb, rer, m=args.lessons_m)
    skw = {"memory": args.container_memory, "cpus": args.container_cpus}
    if args.container_user:
        skw["user"] = "0:0" if args.container_user == "root" else args.container_user
    # solve_kwargs are passed to solve_cybench_task ALONGSIDE image= (set inside train.one), so image is NOT in here
    solve_kwargs = dict(bench=bench, session_kwargs=skw, tool_calling=args.tool_calling, max_rounds=args.max_rounds,
                        max_output_tokens=args.max_output_tokens, context_window=args.context_window,
                        trim_fraction=args.trim_fraction, command_timeout_s=args.command_timeout,
                        citation_generic_tokens=load_generic_tokens(args, pool))
    base = Path(args.out_bandit).with_suffix("")
    print(f"[learn] {os.environ.get('MODEL_NAME','?')}: {n} training solves (parallel {args.parallel}), lessons {len(pool.nodes)}, image {args.image}", flush=True)
    summ = train(mem, tasks, docker=docker, image=args.image, workroot=args.workroot, seeds=seeds, epochs=args.epochs,
                 parallel=args.parallel, service_parallel=args.service_parallel, bandit_path=args.out_bandit,
                 done_path=f"{base}.train_done.jsonl", trainlog_path=f"{base}.train_log.jsonl", solve_kwargs=solve_kwargs)
    print(json.dumps({"train_solves": summ["train_solves"], "categories": list(summ["learned"].keys()),
                      "cells": sum(len(v) for v in summ["learned"].values())}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
