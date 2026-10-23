#!/usr/bin/env python3
"""
experiments/run_campaign.py - the big, RESUMABLE causal-vs-outcome credit campaign.

This is the experiment that tests the fix: does folding CAUSAL credit (leave-one-out
re-rollout) into the bandit's online update let it suppress executed poison and beat
static retrieval, where OUTCOME credit could not?

Conditions (headline metric = solve rate):
  B0       no memory
  B2       static retrieval (no bandit)
  outcome  bandit + cited-only OUTCOME credit            (what failed in the pilot)
  causal   bandit + cited-only CAUSAL credit (LOO online) (the hypothesis)
  [optional ablations via --conditions]

Designed to run for a long time, INTERMITTENTLY:
  * Checkpoints the full state (graph, bandit, standardizer, progress) after EVERY
    solve, so you can Ctrl-C / lose the GPU at any moment and resume losslessly.
  * `--resume` continues from the last checkpoint.
  * Tolerates the LLM server being down: it retries, then checkpoints and exits
    cleanly so you can restart the server + rerun with --resume.

Usage (on the GPU box, server up):
  python -m local_llm.server --model <Qwen3-4B-Instruct-2507 path> &          # serve
  VLLM_URL=http://127.0.0.1:8000/v1 MODEL_NAME=Qwen3-4B-Instruct-2507 \
    python -m experiments.run_campaign --out results/campaign --epochs 8 --eval-seeds 3
  # later, after using your machine for other things:
  ... python -m experiments.run_campaign --out results/campaign --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from config import SolverConfig
from knowledge_graph.store import KnowledgeGraphStore
from retrieval.embedder import SentenceTransformerEmbedder
from retrieval.retriever import ContextRetriever
from retrieval.context_resolution import ContextResolver
from retrieval.reranker import MockReranker
from bandit.bandit import TabularBandit
from reward.reward import RewardStandardizer
from writeback.writeback import WriteBackManager
from pipeline import Orchestrator, SystemSpec
from pipeline.orchestrator import challenge_query_text
from instrumentation.jsonl_logger import ExperimentLogger
from batch_test.real_adapters import real_solve_fn
import os as _os, importlib as _il
hard = _il.import_module(_os.environ.get("CTF_DOMAIN", "benchmark.hard_ctf"))  # T2-7 domain switch


# --- condition table: name -> (SystemSpec, config overrides) ---
def _conditions(names):
    table = {
        "B0":      (SystemSpec("B0", use_graph=False), {}),
        # LEGACY stochastic arms: pinned to the pre-2026-09 sampler (tau=1, no exploration stream)
        # so campaign-wide --tau/--epsilon-explore (meant for the deployed arms) cannot alter them.
        "B2":      (SystemSpec("B2", use_rerank=True, use_mmr=True, use_bandit=False), {"tau": 1.0, "epsilon_explore": 0.0}),
        "outcome": (SystemSpec("B4o", use_bandit=True), {"credit_signal": "outcome", "tau": 1.0, "epsilon_explore": 0.0}),
        "causal":  (SystemSpec("B4c", use_bandit=True), {"credit_signal": "causal", "tau": 1.0, "epsilon_explore": 0.0}),
        "observational": (SystemSpec("B4i", use_bandit=True), {"credit_signal": "observational", "tau": 1.0, "epsilon_explore": 0.0}),
        "causal_wb": (SystemSpec("B5c", use_bandit=True, use_writeback=True, writeback_gated=True),
                      {"credit_signal": "causal"}),
        "outcome_allM": (SystemSpec("B4a", use_bandit=True), {"credit_signal": "outcome", "credit_mode": "all_m"}),
        # F3 (FIX_CATALOG 2026-09): the DEPLOYED arms. B2det = deterministic top-M static memory;
        # deployed = bandit at the campaign's --tau with an --epsilon-explore stream for identification.
        # review 2026-09-05 F-3: the deterministic arms take the CROSS-ENCODER top-M (MMR off) so the
        # label "cross-encoder top-M" is true; `B2det_mmr` keeps the smoke-era MMR-ordered variant.
        "B2det":    (SystemSpec("B2d", use_rerank=True, use_mmr=False, use_bandit=False, deterministic=True), {}),
        "B2det_mmr": (SystemSpec("B2dm", use_rerank=True, use_mmr=True, use_bandit=False, deterministic=True), {}),
        "deployed": (SystemSpec("B4d", use_bandit=True), {"credit_signal": "outcome"}),
        "deployed_obs": (SystemSpec("B4do", use_bandit=True), {"credit_signal": "observational"}),
        # user decision 2026-09-07: the paper's remedy for recovery masking, as a DEPLOYED arm (inherits
        # tau/epsilon/design/learner from the campaign; 3 parallel re-rolls per cited lesson)
        "deployed_causal": (SystemSpec("B4dc", use_bandit=True), {"credit_signal": "causal", "causal_reroll_seeds": 3}),
        # review 2026-09-04 §8: bracket the deployed arm - the family's helpful lesson forced in
        # (ceiling) / its poison forced in (floor), on top of deterministic top-M retrieval.
        "oracle_static": (SystemSpec("B2o", use_rerank=True, use_mmr=False, use_bandit=False, deterministic=True,
                                     force_on_target="helpful"), {}),
        "worst_static":  (SystemSpec("B2w", use_rerank=True, use_mmr=False, use_bandit=False, deterministic=True,
                                     force_on_target="poison"), {}),
    }
    return [(n, *table[n]) for n in names]


def recall_all_overrides(store, over):
    """F24: N = R = |store| so every lesson is in the pool (positivity for the exploration design)."""
    n_nodes = len(store.get_all_embeddings()[0])
    over = dict(over); over["N"] = max(n_nodes, 1); over["R"] = max(n_nodes, 1)
    return over


def base_config(seed=1, **over):
    # B1: causal_reroll_seeds>=3 (a single stochastic re-roll is a coin flip).
    # B4/spec: use_baseline_b=True (variance reduction; the outcome condition uses the
    #          advantage R-b, while causal credit is baseline-free by construction).
    # B5: max_steps tight so following a wrong recipe costs the budget (poison teeth);
    #     calibrated empirically (see calib_hard).
    base = dict(embedding_dim=384, context_resolution="cluster", n_clusters=12,
                N=40, R=30, M=6, kappa=4.0, tau=1.0, tau_min=0.0, alpha=0.2, lam=0.5,
                max_steps=4, seed=seed, causal_reroll_seeds=3, use_baseline_b=True)
    base.update(over)
    return SolverConfig().with_overrides(**base)


def make_checked_solver(max_output_tokens=384, context_window=6144, trim_fraction=0.6):
    base = real_solve_fn(max_output_tokens=max_output_tokens, context_window=context_window,
                         trim_fraction=trim_fraction)

    def _solve(challenge, sampled_nodes, max_rounds, seed=None, order_seed=None, flag_checker=None):
        # F4c: the checker is passed INTO the solver so a wrong candidate does not end the run;
        # the post-check below stays as a safety net.
        checker = flag_checker or (lambda f, _c=challenge: hard.check_flag(_c, f))
        res = base(challenge, sampled_nodes, max_rounds, seed=seed, order_seed=order_seed,
                   flag_checker=checker)
        if res.get("flag") and not hard.check_flag(challenge, res["flag"]):
            res = dict(res); res["flag"] = None; res["gave_up_reason"] = "wrong_flag"
        return res
    return _solve


class Campaign:
    def __init__(self, out_dir, conditions, epochs, eval_seeds, n_per_family, world_seed,
                 families=None, max_steps=4, M=None, kappa=None, n_clusters=None, slack=None,
                 malignant_arms=None, leak_names=False, distractor_arm=False,
                 tau=None, tau_min=None, epsilon_explore=None, tau_explore=None,
                 citation_mode=None, poison_kind=None, context_window=6144, max_output_tokens=384,
                 trim_fraction=0.6, randomize_lesson_order=True, reranker="cross",
                 exploration_design=None, query_diagnostics="head", context_key="cluster",
                 recall_all=False, prior_source="cosine", credit_mode=None, init_mode=None,
                 alpha_schedule=None, distance_scaling=None, draw_seed_offset=0):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        # review 2026-09-08 F3: every arm derives its TRAIN-run seed (retrieval rng: the epsilon draw and the stratified
        # design draw; and the solver's llm_seed) from (epoch, item) alone, so arms re-execute the same exploration
        # draws. A non-zero offset gives an arm INDEPENDENT train draws (replication arms); 0 keeps the byte-identical
        # seeds. Eval seeds are never offset: the held-out ladder pairs arms by (item, eval seed).
        self.draw_seed_offset = int(draw_seed_offset or 0)
        self.conditions = conditions
        self.epochs = epochs
        self.eval_seeds = eval_seeds
        self.n_per_family = n_per_family
        self.world_seed = world_seed
        self.families = families
        self.malignant_arms = malignant_arms   # T1-1 positive control: add on-target malignant lessons
        self.max_steps = max_steps
        self.slack = slack       # if set, per-family budget = solution_length + slack (uniform teeth)
        # campaign-wide retrieval overrides (None = keep base_config default)
        self.cfg_over = {k: v for k, v in (("M", M), ("kappa", kappa),
                                           ("n_clusters", n_clusters), ("tau", tau), ("tau_min", tau_min),
                                           ("epsilon_explore", epsilon_explore), ("tau_explore", tau_explore),
                                           ("citation_mode", citation_mode)) if v is not None}
        self.poison_kind = poison_kind or "sibling"       # F5: "sibling" (legacy) | "cross_outer"
        self.reranker_kind = reranker
        if exploration_design is not None:
            self.cfg_over["exploration_design"] = exploration_design
        # F22 context ladder: cluster (legacy k-means on the query embedding; noise under neutral
        # queries) | alphabet (CATEGORY context = outer alphabet class, needs charset diagnostics)
        # | family (oracle CATEGORY context = family id; learning ceiling, never rendered).
        self.context_key = context_key or "cluster"
        if self.context_key in ("global", "alphabet", "family"):
            self.cfg_over["context_resolution"] = "category"
        elif self.context_key != "cluster":
            raise ValueError(f"unknown context_key {context_key!r} (global|cluster|alphabet|family)")
        # review 2026-09-05 F-2c: prior for the stochastic sampler = cosine (legacy) | rerank
        self.prior_source = prior_source or "cosine"
        self.cfg_over["prior_source"] = self.prior_source
        if credit_mode:                                     # F40: cited_only | cited_only_decay | all_m
            self.cfg_over["credit_mode"] = credit_mode
        for k, v in (("init_mode", init_mode), ("alpha_schedule", alpha_schedule), ("distance_scaling", distance_scaling)):
            if v:                                           # F41 learner options (legacy defaults when unset)
                self.cfg_over[k] = v
        # F24: recall the whole store (N = R = |store|) so the exploration design can reach every
        # on-target lesson; under neutral queries similarity recall at N=40 missed 70-80% of them.
        self.recall_all = bool(recall_all)
        self.randomize_lesson_order = randomize_lesson_order   # F7: per-run lesson order
        self.solver_kw = dict(context_window=context_window, max_output_tokens=max_output_tokens,
                              trim_fraction=trim_fraction)
        self.prog_path = self.out / "progress.json"
        self.embedder = SentenceTransformerEmbedder(device="cpu")
        self.leak_names = leak_names           # T2-2 name-leak A/B: reveal family name in filename/desc
        self.distractor_arm = distractor_arm   # T2-3: on-target irrelevant distractor lesson
        # F22 (smoke 2026-09-05): artifact diagnostics mode for the retrieval query. "head" = legacy
        # (`file` + raw head bytes, which the embedder/cross-encoder cannot read as an alphabet);
        # "charset" adds the outer alphabet class in words. Domains without the kwarg keep legacy.
        self.query_diagnostics = query_diagnostics or "head"
        import inspect as _insp
        _gc_kw = dict(n_per_family=n_per_family, seed=world_seed, families=families, leak_names=leak_names)
        if "diagnostics" in _insp.signature(hard.generate_corpus).parameters:
            _gc_kw["diagnostics"] = self.query_diagnostics
        elif self.query_diagnostics != "head":
            raise ValueError(f"{hard.__name__}.generate_corpus has no diagnostics mode; use --query-diagnostics head")
        self.corpus = hard.generate_corpus(self.out / "corpus", **_gc_kw)
        if self.context_key == "family":
            for c in self.corpus:
                c["_retrieval_category"] = c["_family"]
        elif self.context_key == "global":                 # the NULL rung: one context by construction
            for c in self.corpus:
                c["_retrieval_category"] = "global"
        elif self.context_key == "alphabet" and not all(c.get("_retrieval_category") for c in self.corpus):
            raise ValueError("context_key=alphabet needs --query-diagnostics charset (no _retrieval_category)")
        # F1 runtime guard: nothing the solver can see may carry the family string - neither the
        # challenge fields nor the RENDERED lessons (review 2026-09-04: the lesson header leaked).
        if not leak_names:
            from solver.context_assembly import assemble_context as _ac
            _store, _ = hard.build_seed_graph(self.embedder, families=families,
                                              malignant_arms=malignant_arms, distractor_arm=distractor_arm,
                                              poison_kind=self.poison_kind)
            _ids, _ = _store.get_all_embeddings()
            _rendered = _ac([_store.get_node(i) for i in _ids]).lower()
            for _f in sorted({c["_family"] for c in self.corpus}):
                assert _f.lower() not in _rendered, f"F1 leak: family string {_f} in a rendered lesson"
            for c in self.corpus:
                fam = c["_family"].lower()
                visible = " ".join([str(c.get("_display_name", "")), c["description"],
                                    Path(c["local_files"][0]["local_path"]).parent.name,
                                    c["local_files"][0]["text"]]).lower()
                assert fam not in visible and c["name"].lower() not in visible, \
                    f"F1 leak: family string reaches the solver for {c['name']}: {visible[:120]}"
        # PER-FAMILY ROUND BUDGET (panel: keep poison "teeth" uniform across difficulty -
        # budget = solution_length + fixed slack, not a flat round count). slack=None keeps
        # the fixed pilot budget (campaign_v2 reproducibility).
        if slack is not None:
            for c in self.corpus:
                c["_max_steps"] = int(c.get("_solution_length", 2)) + int(slack)
        self.solve_fn = make_checked_solver(**self.solver_kw)
        # B3: split per-family into TRAIN and held-out EVAL (the bandit learns lesson
        # values on train challenges; the headline solve rate is on NEW instances).
        from collections import defaultdict as _dd
        by_fam = _dd(list)
        for c in self.corpus:
            by_fam[c["_family"]].append(c)
        self.train, self.eval = [], []
        n_eval = max(1, n_per_family // 4)              # ~25% held out per family
        for fam, items in by_fam.items():
            self.eval.extend(items[:n_eval])
            self.train.extend(items[n_eval:])
        # cluster resolver fit ONCE on the TRAIN query embeddings only (no eval leak)
        self._cluster_centroids = None

    # ---- progress ----
    def _load_progress(self):
        if self.prog_path.exists():
            return json.loads(self.prog_path.read_text())
        return {"conditions": {}}

    def _save_progress(self, prog):
        tmp = self.prog_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(prog, indent=2)); tmp.replace(self.prog_path)

    # ---- per-condition state ----
    def _cdir(self, name):
        d = self.out / "state" / name; d.mkdir(parents=True, exist_ok=True); return d

    def _save_state(self, name, store, bandit, standardizer, rerolls):
        d = self._cdir(name)
        store.save(d / "graph")
        (d / "bandit.json").write_text(json.dumps(bandit.to_dict()))
        (d / "standardizer.json").write_text(json.dumps(standardizer.to_dict()))
        (d / "rerolls.json").write_text(json.dumps({"rerolls": rerolls}))

    def _load_state(self, name, cfg):
        d = self._cdir(name)
        if not (d / "graph.json").exists():
            return None
        store = KnowledgeGraphStore.load(d / "graph")
        bandit = TabularBandit.from_dict(json.loads((d / "bandit.json").read_text()), cfg) \
            if (d / "bandit.json").exists() else None
        standardizer = RewardStandardizer.from_dict(
            json.loads((d / "standardizer.json").read_text()), cfg)
        rerolls = json.loads((d / "rerolls.json").read_text()).get("rerolls", 0) \
            if (d / "rerolls.json").exists() else 0
        return store, bandit, standardizer, rerolls

    def _fit_resolver(self, cfg):
        cr = ContextResolver(cfg)
        if cfg.context_resolution.value != "cluster":      # F-15: no k-means fit for CATEGORY contexts
            cr._fitted = True
            return cr
        if self._cluster_centroids is None:
            # fit on TRAIN queries only (held-out eval challenges never seen by clustering)
            embs = np.array([self.embedder.embed(challenge_query_text(c)) for c in self.train])
            cr.fit_on_embeddings(embs)
            self._cluster_centroids = cr.centroids
        else:
            cr.centroids = self._cluster_centroids; cr._fitted = True
        return cr

    def _build_orch(self, name, spec, over, store=None, bandit=None, standardizer=None):
        if store is None:
            store, _ = hard.build_seed_graph(self.embedder, families=self.families, malignant_arms=self.malignant_arms,
                                             distractor_arm=self.distractor_arm, poison_kind=self.poison_kind)
        if getattr(self, "recall_all", False):
            over = recall_all_overrides(store, over)
        cfg = base_config(**over)
        if spec.use_bandit and bandit is None:
            bandit = TabularBandit(cfg)
        if standardizer is None:
            standardizer = RewardStandardizer(cfg)
        cr = self._fit_resolver(cfg)
        # review 2026-09-04: the mock (Jaccard) reranker made deterministic top-M a lexical lottery
        # (per family: helpful in top-M 0.00 or 1.00). Real cross-encoder by default.
        if getattr(self, "reranker_kind", "cross") == "cross":
            from retrieval.reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker(device=os.environ.get("CTF_RERANK_DEVICE", "cpu"))
        else:
            reranker = MockReranker()
        retr = ContextRetriever(store, self.embedder, config=cfg, reranker=reranker,
                                context_resolver=cr, bandit=bandit if spec.use_bandit else None)
        logger = ExperimentLogger(f"campaign_{name}", root=str(self.out / "logs"))
        orch = Orchestrator(store, self.embedder, cfg, spec, retriever=retr,
                            bandit=bandit if spec.use_bandit else None,
                            standardizer=standardizer, writeback=WriteBackManager(cfg),
                            logger=logger, solve_fn=self.solve_fn)
        orch.solver_model = os.getenv("MODEL_NAME", "local")
        orch.randomize_lesson_order = self.randomize_lesson_order        # F7
        orch.citation_generic_tokens = getattr(hard, "GENERIC_CITATION_TOKENS", None)   # F2/review
        orch.flag_checker_factory = (lambda ch: (lambda f, _c=ch: hard.check_flag(_c, f)))   # F4c
        return orch, store, bandit, standardizer, cfg

    def train_seed(self, ep, name):
        """The train-run seed of item `name` in epoch `ep` (F3: `draw_seed_offset` 0 reproduces the campaign's seeds)."""
        tag = f"train|{ep}|{name}" if not self.draw_seed_offset else f"train|{ep}|{name}|rep{self.draw_seed_offset}"
        return int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16) % (2**31 - 1)

    def _solve_one(self, orch, ch, seed, max_retries=5):
        """Run one challenge with server-down tolerance. F9: the same `seed` drives the
        retrieval rng AND the solver's llm_seed, so paired conditions share both."""
        for attempt in range(max_retries):
            try:
                return orch.run_one(ch, experiment_id="campaign",
                                    rng=np.random.RandomState(seed), llm_seed=int(seed))
            except Exception as e:  # noqa
                msg = str(e).lower()
                if "connection" in msg or "refused" in msg or "timed out" in msg or "urlopen" in msg:
                    wait = 30 * (attempt + 1)
                    print(f"  [server unreachable: {e}] retry in {wait}s "
                          f"({attempt+1}/{max_retries})", flush=True)
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("server unreachable after retries; checkpoint saved, rerun with --resume")

    def _stop_requested(self) -> bool:
        """F10: the pipeline drops a STOP file in the campaign root at its deadline; we exit at the
        next checkpoint (state is saved after every solve) instead of being killed mid-phase."""
        return (self.out / "STOP").exists() or (self.out.parent / "STOP").exists()

    def run(self):
        prog = self._load_progress()
        for name, spec, over in self.conditions:
            over = {"max_steps": self.max_steps, **self.cfg_over, **over}   # campaign-wide; condition may override
            cprog = prog["conditions"].setdefault(name, {"phase": "train", "epoch": 0,
                                                         "seed": 0, "idx": 0, "done": False,
                                                         "metrics": [], "train_solves": 0})
            if cprog.get("done"):
                print(f"[{name}] already complete; skipping"); continue
            learns = spec.use_bandit or spec.use_writeback
            print(f"\n=== {name} (phase={cprog['phase']} ep={cprog['epoch']} seed={cprog['seed']} idx={cprog['idx']}) ===", flush=True)
            loaded = self._load_state(name, base_config(**over))
            if loaded:
                store, bandit, standardizer, rerolls = loaded
                orch, store, bandit, standardizer, cfg = self._build_orch(
                    name, spec, over, store=store, bandit=bandit, standardizer=standardizer)
                orch._causal_rerolls = rerolls
            else:
                orch, store, bandit, standardizer, cfg = self._build_orch(name, spec, over)

            # --- TRAINING phase: stream the TRAIN split, learning on (B4: bandit conds only) ---
            if cprog["phase"] == "train":
                orch.frozen = False
                n_ep = self.epochs if learns else 0
                for ep in range(cprog["epoch"], n_ep):
                    order = list(range(len(self.train)))
                    np.random.RandomState(7000 + ep).shuffle(order)
                    start = cprog["idx"] if ep == cprog["epoch"] else 0
                    for j in range(start, len(order)):
                        if self._stop_requested():
                            print("[campaign] STOP requested; checkpoint saved, exiting cleanly", flush=True)
                            return
                        ch = self.train[order[j]]
                        # review: `5000 + 137*ep + j` collides across epochs for j >= 137; hash instead
                        run = self._solve_one(orch, ch, seed=self.train_seed(ep, ch["name"]))
                        cprog["metrics"].append({"phase": "train", "epoch": ep, "name": ch["name"],
                                                 "family": ch["_family"], "solved": run.outcome,
                                                 "steps": run.steps})
                        cprog["train_solves"] += 1
                        cprog["idx"] = j + 1; cprog["epoch"] = ep
                        self._save_state(name, store, bandit, standardizer, orch._causal_rerolls)
                        self._save_progress(prog)
                    cprog["idx"] = 0
                    # OBSERVATIONAL: after each epoch, recompute Q from the logged
                    # sampling propensities (IPW) - no per-run updates, no re-rolls.
                    if over.get("credit_signal") == "observational" and bandit is not None:
                        from ope.causal_value import observational_q_table
                        runs = ExperimentLogger(f"campaign_{name}", root=str(self.out / "logs")).read("run")
                        # review 2026-09-08 F6: write n = observations per cell (was n = 1 -> sampled on Q/5)
                        table, counts = observational_q_table(runs, tau0=cfg.tau0, return_counts=True)
                        bandit.set_q_table(table, n_table=counts)
                        self._save_state(name, store, bandit, standardizer, orch._causal_rerolls)
                cprog["phase"] = "eval"; cprog["seed"] = 0; cprog["idx"] = 0
                self._save_progress(prog)

            # --- EVAL phase: FROZEN (no learning), on the HELD-OUT split, same for all conds ---
            if cprog["phase"] == "eval":
                orch.frozen = True                       # B2: clean freeze (Q live, no updates)
                for s in range(cprog["seed"], self.eval_seeds):
                    start = cprog["idx"] if s == cprog["seed"] else 0
                    for j in range(start, len(self.eval)):
                        if self._stop_requested():
                            print("[campaign] STOP requested; checkpoint saved, exiting cleanly", flush=True)
                            return
                        ch = self.eval[j]
                        run = self._solve_one(orch, ch, seed=90000 + 1000 * s + j)
                        cprog["metrics"].append({"phase": "eval", "seed": s, "name": ch["name"],
                                                 "family": ch["_family"], "solved": run.outcome,
                                                 "steps": run.steps})
                        cprog["idx"] = j + 1; cprog["seed"] = s
                        self._save_progress(prog)
                    cprog["idx"] = 0
                cprog["done"] = True
                cprog["causal_rerolls"] = orch._causal_rerolls
                cprog["eval_solves"] = sum(1 for m in cprog["metrics"] if m["phase"] == "eval")
                self._save_progress(prog)
            ev = [m for m in cprog["metrics"] if m["phase"] == "eval"]
            sr = np.mean([m["solved"] for m in ev]) if ev else 0.0
            # B4: report compute so a 'win' is not confused with 'more compute'
            print(f"[{name}] DONE. HELD-OUT solve rate = {sr:.3f}  "
                  f"(train_solves={cprog['train_solves']}, causal_rerolls={orch._causal_rerolls})", flush=True)

        self._write_summary(prog)

    def _write_summary(self, prog):
        from collections import defaultdict as _dd
        summary = {"n_train": len(self.train), "n_eval_heldout": len(self.eval),
                   "families": sorted({c["_family"] for c in self.corpus}),
                   "eval_seeds": self.eval_seeds, "epochs": self.epochs, "draw_seed_offset": self.draw_seed_offset, "conditions": {}}
        for name, cp in prog["conditions"].items():
            ev = [m for m in cp.get("metrics", []) if m["phase"] == "eval"]
            per_fam = _dd(lambda: [0, 0])
            for m in ev:
                per_fam[m["family"]][0] += m["solved"]; per_fam[m["family"]][1] += 1
            summary["conditions"][name] = {
                "heldout_solve_rate": float(np.mean([m["solved"] for m in ev])) if ev else None,
                "eval_n": len(ev),
                "per_family": {f: round(v[0] / v[1], 3) for f, v in sorted(per_fam.items())},
                "train_solves": cp.get("train_solves", 0),
                "causal_rerolls": cp.get("causal_rerolls", 0),
                "total_solves": cp.get("train_solves", 0) + cp.get("causal_rerolls", 0) + len(ev),
                "done": cp.get("done", False),
            }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print("\n=== CAMPAIGN SUMMARY (held-out solve rate; compute disclosed) ===")
        for n, s in summary["conditions"].items():
            print(f"  {n:10s} heldout={s['heldout_solve_rate']} (n={s['eval_n']}) "
                  f"train_solves={s['train_solves']} rerolls={s['causal_rerolls']} "
                  f"total={s['total_solves']} done={s['done']}")
        print(f"wrote {self.out/'summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/campaign")
    ap.add_argument("--conditions", default="B0,B2,outcome,causal")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--n-per-family", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=4,
                    help="solver round budget; 4 keeps the model from exploring its way "
                         "around the lesson (more rounds erode headroom).")
    ap.add_argument("--M", type=int, default=None,
                    help="lessons sampled into context; small M forces the bandit to "
                         "choose helpful vs poison (selection matters).")
    ap.add_argument("--kappa", type=float, default=None,
                    help="similarity-prior pseudo-count; LOW kappa lets the learned Q "
                         "override similarity, so the causal bandit's poison-Q<0 can "
                         "actually exclude poison from sampling (the solve-rate mechanism).")
    ap.add_argument("--n-clusters", type=int, default=None,
                    help="bandit context granularity; keep small for a small corpus so "
                         "each (context,lesson) cell gets enough updates to learn.")
    ap.add_argument("--world-seed", type=int, default=424242)
    ap.add_argument("--draw-seed-offset", type=int, default=0,
                    help="F3 replication arms: non-zero -> independent TRAIN draws (epsilon/design/llm seeds); eval seeds unchanged")
    ap.add_argument("--families", default=None,
                    help="comma-separated family subset (default: all in hard_ctf)")
    ap.add_argument("--grid-families", type=int, default=None,
                    help="lab-scale: use N GENERATED grid families (>=150 -> >=150 "
                         "lessons/label) instead of --families")
    ap.add_argument("--grid-difficulty", default=None,
                    help="restrict grid families to this difficulty band, e.g. "
                         "'easy,medium' (default: all). Calibrate to the model first.")
    ap.add_argument("--malignant-arms", default=None,
                    help="comma list subset of destroy,exhaust,wrongflag - adds on-target MALIGNANT "
                         "lessons (T1-1 positive control). Omit for normal campaigns.")
    ap.add_argument("--slack", type=int, default=None,
                    help="per-family round budget = solution_length + slack (uniform poison "
                         "teeth across difficulty). Unset = fixed --max-steps (pilot mode). "
                         "Pre-specified headline slack = 2; sweep {1,2,3} as robustness.")
    ap.add_argument("--resume", action="store_true", help="(default behavior; checkpoints always resume)")
    ap.add_argument("--leak-names", action="store_true",
                    help="T2-2 name-leak A/B: reveal the family name in the artifact filename + "
                         "description (the encoding chain leaks into the prompt). Default off = neutral.")
    ap.add_argument("--distractor-arm", action="store_true",
                    help="T2-3: add one on-target but irrelevant distractor lesson per family.")
    # F2/F3/F5/F7 (FIX_CATALOG 2026-09)
    ap.add_argument("--tau", type=float, default=None, help="deployed softmax temperature (scores live in ~[0,1])")
    ap.add_argument("--tau-min", type=float, default=None)
    ap.add_argument("--epsilon-explore", type=float, default=None, help="fraction of runs drawn at --tau-explore")
    ap.add_argument("--tau-explore", type=float, default=None)
    ap.add_argument("--citation-mode", default=None, choices=["distinctive", "any"])
    ap.add_argument("--poison-kind", default=None, choices=["sibling", "cross_outer"])
    ap.add_argument("--context-window", type=int, default=6144)
    ap.add_argument("--max-output-tokens", type=int, default=384)
    ap.add_argument("--trim-fraction", type=float, default=0.6)
    ap.add_argument("--fixed-lesson-order", action="store_true", help="disable per-run lesson-order randomization")
    ap.add_argument("--reranker", default="cross", choices=["cross", "mock"])
    ap.add_argument("--exploration-design", default=None, choices=["stratified", "uniform", "softmax"])
    ap.add_argument("--query-diagnostics", default="head", choices=["head", "charset"],
                    help="F22: artifact diagnostics in the retrieval query (head=legacy raw bytes; "
                         "charset=+outer alphabet class in words)")
    ap.add_argument("--context-key", default="cluster", choices=["global", "cluster", "alphabet", "family"],
                    help="F22: bandit context = single global context (null) | k-means cluster (legacy) | "
                         "outer alphabet class | family id (oracle ceiling arm)")
    ap.add_argument("--init-mode", default=None, choices=["zero", "optimistic", "seed_warm"], help="F41 bandit cell init")
    ap.add_argument("--alpha-schedule", default=None, choices=["ema", "mean"], help="F41: ema (legacy) | mean (running mean)")
    ap.add_argument("--distance-scaling", default=None, choices=["symmetric", "asymmetric", "none"], help="F41")
    ap.add_argument("--prior-source", default="cosine", choices=["cosine", "rerank", "none"],
                    help="review F-2: sampling prior for the stochastic arms (cosine=legacy; rerank=sigmoid of "
                         "the cross-encoder logit)")
    ap.add_argument("--recall-all", action="store_true", help="F24: N = R = |store| (recall every lesson)")
    ap.add_argument("--credit-mode", default=None, choices=["cited_only", "cited_only_decay", "all_m"],
                    help="F40: cited_only (legacy) | cited_only_decay (cited credit + zero-advantage exposure "
                         "update for uncited sampled lessons) | all_m (ablation)")
    args = ap.parse_args()
    conds = _conditions([c.strip() for c in args.conditions.split(",")])
    if args.grid_families:
        diff = [d.strip() for d in args.grid_difficulty.split(",")] if args.grid_difficulty else None
        fams = hard.grid_family_names(difficulty=diff, n=args.grid_families, seed=args.world_seed)
        print(f"[grid] {len(fams)} families (difficulty={diff or 'all'}) "
              f"-> ~{len(fams)} lessons/label")
    else:
        fams = [f.strip() for f in args.families.split(",")] if args.families else None
    mal = [a.strip() for a in args.malignant_arms.split(",")] if args.malignant_arms else None
    camp = Campaign(args.out, conds, args.epochs, args.eval_seeds, args.n_per_family,
                    args.world_seed, families=fams, max_steps=args.max_steps,
                    M=args.M, kappa=args.kappa, n_clusters=args.n_clusters, slack=args.slack,
                    malignant_arms=mal, leak_names=args.leak_names,
                    distractor_arm=args.distractor_arm,
                    tau=args.tau, tau_min=args.tau_min, epsilon_explore=args.epsilon_explore,
                    tau_explore=args.tau_explore, citation_mode=args.citation_mode,
                    poison_kind=args.poison_kind, context_window=args.context_window,
                    max_output_tokens=args.max_output_tokens, trim_fraction=args.trim_fraction,
                    randomize_lesson_order=not args.fixed_lesson_order, reranker=args.reranker,
                    exploration_design=args.exploration_design,
                    query_diagnostics=args.query_diagnostics, context_key=args.context_key,
                    recall_all=args.recall_all, prior_source=args.prior_source, credit_mode=args.credit_mode,
                    init_mode=args.init_mode, alpha_schedule=args.alpha_schedule, distance_scaling=args.distance_scaling,
                    draw_seed_offset=args.draw_seed_offset)
    camp.run()


if __name__ == "__main__":
    main()
