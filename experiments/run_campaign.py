#!/usr/bin/env python3
"""
experiments/run_campaign.py — the big, RESUMABLE causal-vs-outcome credit campaign.

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
try:
    from batch_test.real_adapters import real_solve_fn
except ModuleNotFoundError:  # not shipped in the public release
    def real_solve_fn(*a, **k):
        raise RuntimeError(
            "real-GPU backend adapters (batch_test/) are not included in "
            "this release; supply your own solve_fn")
import os as _os, importlib as _il
hard = _il.import_module(_os.environ.get("CTF_DOMAIN", "benchmark.hard_ctf"))  # T2-7 domain switch


# --- condition table: name -> (SystemSpec, config overrides) ---
def _conditions(names):
    table = {
        "B0":      (SystemSpec("B0", use_graph=False), {}),
        "B2":      (SystemSpec("B2", use_rerank=True, use_mmr=True, use_bandit=False), {}),
        "outcome": (SystemSpec("B4o", use_bandit=True), {"credit_signal": "outcome"}),
        "causal":  (SystemSpec("B4c", use_bandit=True), {"credit_signal": "causal"}),
        "observational": (SystemSpec("B4i", use_bandit=True), {"credit_signal": "observational"}),
        "causal_wb": (SystemSpec("B5c", use_bandit=True, use_writeback=True, writeback_gated=True),
                      {"credit_signal": "causal"}),
        "outcome_allM": (SystemSpec("B4a", use_bandit=True), {"credit_signal": "outcome", "credit_mode": "all_m"}),
    }
    return [(n, *table[n]) for n in names]


def base_config(seed=1, **over):
    # B1: causal_reroll_seeds>=3 (a single stochastic re-roll is a coin flip).
    # B4/spec: use_baseline_b=True (variance reduction; the outcome condition uses the
    #          advantage R-b, while causal credit is baseline-free by construction).
    # B5: max_steps tight so following a wrong recipe costs the budget (poison teeth);
    #     calibrated empirically (see calib_hard).
    base = dict(embedding_dim=384, context_resolution="cluster", n_clusters=12,
                N=40, R=30, M=6, kappa=4.0, tau=1.0, tau_min=0.3, alpha=0.2, lam=0.5,
                max_steps=4, seed=seed, causal_reroll_seeds=3, use_baseline_b=True)
    base.update(over)
    return SolverConfig().with_overrides(**base)


def make_checked_solver():
    base = real_solve_fn(max_output_tokens=384, context_window=6144)

    def _solve(challenge, sampled_nodes, max_rounds, seed=None):
        res = base(challenge, sampled_nodes, max_rounds, seed=seed)
        if res.get("flag") and not hard.check_flag(challenge, res["flag"]):
            res = dict(res); res["flag"] = None; res["gave_up_reason"] = "wrong_flag"
        return res
    return _solve


class Campaign:
    def __init__(self, out_dir, conditions, epochs, eval_seeds, n_per_family, world_seed,
                 families=None, max_steps=4, M=None, kappa=None, n_clusters=None, slack=None,
                 malignant_arms=None, leak_names=False, distractor_arm=False):
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
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
                                           ("n_clusters", n_clusters)) if v is not None}
        self.prog_path = self.out / "progress.json"
        self.embedder = SentenceTransformerEmbedder(device="cpu")
        self.leak_names = leak_names           # T2-2 name-leak A/B: reveal family name in filename/desc
        self.distractor_arm = distractor_arm   # T2-3: on-target irrelevant distractor lesson
        self.corpus = hard.generate_corpus(self.out / "corpus", n_per_family=n_per_family,
                                            seed=world_seed, families=families, leak_names=leak_names)
        # PER-FAMILY ROUND BUDGET (panel: keep poison "teeth" uniform across difficulty —
        # budget = solution_length + fixed slack, not a flat round count). slack=None keeps
        # the fixed pilot budget (campaign_v2 reproducibility).
        if slack is not None:
            for c in self.corpus:
                c["_max_steps"] = int(c.get("_solution_length", 2)) + int(slack)
        self.solve_fn = make_checked_solver()
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
        if self._cluster_centroids is None:
            # fit on TRAIN queries only (held-out eval challenges never seen by clustering)
            embs = np.array([self.embedder.embed(challenge_query_text(c)) for c in self.train])
            cr.fit_on_embeddings(embs)
            self._cluster_centroids = cr.centroids
        else:
            cr.centroids = self._cluster_centroids; cr._fitted = True
        return cr

    def _build_orch(self, name, spec, over, store=None, bandit=None, standardizer=None):
        cfg = base_config(**over)
        if store is None:
            store, _ = hard.build_seed_graph(self.embedder, families=self.families, malignant_arms=self.malignant_arms, distractor_arm=self.distractor_arm)
        if spec.use_bandit and bandit is None:
            bandit = TabularBandit(cfg)
        if standardizer is None:
            standardizer = RewardStandardizer(cfg)
        cr = self._fit_resolver(cfg)
        retr = ContextRetriever(store, self.embedder, config=cfg, reranker=MockReranker(),
                                context_resolver=cr, bandit=bandit if spec.use_bandit else None)
        logger = ExperimentLogger(f"campaign_{name}", root=str(self.out / "logs"))
        orch = Orchestrator(store, self.embedder, cfg, spec, retriever=retr,
                            bandit=bandit if spec.use_bandit else None,
                            standardizer=standardizer, writeback=WriteBackManager(cfg),
                            logger=logger, solve_fn=self.solve_fn)
        orch.solver_model = os.getenv("MODEL_NAME", "local")
        return orch, store, bandit, standardizer, cfg

    def _solve_one(self, orch, ch, seed, max_retries=5):
        """Run one challenge with server-down tolerance."""
        for attempt in range(max_retries):
            try:
                return orch.run_one(ch, experiment_id="campaign",
                                    rng=np.random.RandomState(seed))
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
                        ch = self.train[order[j]]
                        run = self._solve_one(orch, ch, seed=5000 + 137 * ep + j)
                        cprog["metrics"].append({"phase": "train", "epoch": ep, "name": ch["name"],
                                                 "family": ch["_family"], "solved": run.outcome,
                                                 "steps": run.steps})
                        cprog["train_solves"] += 1
                        cprog["idx"] = j + 1; cprog["epoch"] = ep
                        self._save_state(name, store, bandit, standardizer, orch._causal_rerolls)
                        self._save_progress(prog)
                    cprog["idx"] = 0
                    # OBSERVATIONAL: after each epoch, recompute Q from the logged
                    # sampling propensities (IPW) — no per-run updates, no re-rolls.
                    if over.get("credit_signal") == "observational" and bandit is not None:
                        from ope.causal_value import observational_q_table
                        runs = ExperimentLogger(f"campaign_{name}", root=str(self.out / "logs")).read("run")
                        table = observational_q_table(runs, tau0=cfg.tau0)
                        bandit.set_q_table(table)
                        self._save_state(name, store, bandit, standardizer, orch._causal_rerolls)
                cprog["phase"] = "eval"; cprog["seed"] = 0; cprog["idx"] = 0
                self._save_progress(prog)

            # --- EVAL phase: FROZEN (no learning), on the HELD-OUT split, same for all conds ---
            if cprog["phase"] == "eval":
                orch.frozen = True                       # B2: clean freeze (Q live, no updates)
                for s in range(cprog["seed"], self.eval_seeds):
                    start = cprog["idx"] if s == cprog["seed"] else 0
                    for j in range(start, len(self.eval)):
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
                   "eval_seeds": self.eval_seeds, "epochs": self.epochs, "conditions": {}}
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
    ap.add_argument("--families", default=None,
                    help="comma-separated family subset (default: all in hard_ctf)")
    ap.add_argument("--grid-families", type=int, default=None,
                    help="lab-scale: use N GENERATED grid families (>=150 -> >=150 "
                         "lessons/label) instead of --families")
    ap.add_argument("--grid-difficulty", default=None,
                    help="restrict grid families to this difficulty band, e.g. "
                         "'easy,medium' (default: all). Calibrate to the model first.")
    ap.add_argument("--malignant-arms", default=None,
                    help="comma list subset of destroy,exhaust,wrongflag — adds on-target MALIGNANT "
                         "lessons (T1-1 positive control). Omit for normal campaigns.")
    ap.add_argument("--slack", type=int, default=None,
                    help="per-family round budget = solution_length + slack (uniform poison "
                         "teeth across difficulty). Unset = fixed --max-steps (pilot mode). "
                         "Pre-registered headline slack = 2; sweep {1,2,3} as robustness.")
    ap.add_argument("--resume", action="store_true", help="(default behavior; checkpoints always resume)")
    ap.add_argument("--leak-names", action="store_true",
                    help="T2-2 name-leak A/B: reveal the family name in the artifact filename + "
                         "description (the encoding chain leaks into the prompt). Default off = neutral.")
    ap.add_argument("--distractor-arm", action="store_true",
                    help="T2-3: add one on-target but irrelevant distractor lesson per family.")
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
                    distractor_arm=args.distractor_arm)
    camp.run()


if __name__ == "__main__":
    main()
