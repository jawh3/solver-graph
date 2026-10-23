"""
evaluation/harness.py - the locked measurement layer (spec §14, C4).

Runs each SystemSpec over a dataset with a PAIRED, common-random-seed design and
emits paper-ready tables: per-system solve rate, paired lift over B0 with bootstrap
CIs, median step-to-flag, recovery-masking rate, and the non-exploit-capture rate.

Built BEFORE the bandit/write-back are *trusted* (build order §16) - but it can
evaluate them too, because it just consumes the orchestrator + the §12 logs.

Design rules honored:
  - Lift is reported over B0, NOT raw solve rate (public challenges are in Qwen's
    pretraining - contamination). Headline = HELD-OUT-NOVEL + TRANSFORMED-FAMILIES.
  - Paired / within-challenge: every system sees the SAME challenges with the SAME
    per-challenge eval seed, so paired bootstrap (stats.py) applies.
  - Learning systems (B4-B6) get `train_epochs` passes first, then a fixed eval
    pass; non-learning systems (B0-B3) ignore training.
  - Each system runs on a FRESH graph/bandit (via a graph_factory) so write-back in
    one system can't leak into another.
  - B5 vs B3 is flagged loudly: if gated write-back does not beat episodic-only,
    write-back is unjustified (Faulty-Memory null, §9.4).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config import SolverConfig
from pipeline import Orchestrator, SystemSpec
from pipeline.orchestrator import Orchestrator as _Orch
from retrieval.retriever import ContextRetriever
from retrieval.reranker import MockReranker
from retrieval.context_resolution import ContextResolver
from bandit.bandit import TabularBandit
from reward.reward import RewardStandardizer
from writeback.writeback import WriteBackManager
from instrumentation.jsonl_logger import ExperimentLogger
from .stats import paired_bootstrap_ci, mcnemar_exact, power_analysis


@dataclass
class RunMetric:
    challenge_id: str
    solved: int
    steps: int
    non_exploit: bool
    sampled: int
    cited: int


@dataclass
class SystemReport:
    system_id: str
    n: int
    solve_rate: float
    solve_rate_excl_non_exploit: float
    median_steps_to_flag: Optional[float]
    recovery_masking_rate: float        # mean (in-context but not cited) on successes
    per_challenge: Dict[str, RunMetric] = field(default_factory=dict)


# A graph_factory returns a fresh (store, embedder) for a system run.
GraphFactory = Callable[[], Tuple[object, object]]


class EvalHarness:
    """Locked measurement layer. Backends are injectable for the real-GPU path.

    `reranker_factory` and `solve_fn` default to the GPU-free mocks; `batch_test/`
    injects the real cross-encoder + real Qwen solve_fn without changing any other
    line of the eval (so mock and real produce identical record schemas).
    """

    def __init__(self, config: SolverConfig, results_root: str = "results/logs",
                 base_seed: int = 12345,
                 reranker_factory: Optional[Callable] = None,
                 solve_fn: Optional[Callable] = None):
        self.config = config
        self.results_root = results_root
        self.base_seed = base_seed
        self.reranker_factory = reranker_factory or (lambda: MockReranker())
        self.solve_fn = solve_fn   # None => Orchestrator's default mock solver

    # ------------------------------------------------------------------
    def _build_orchestrator(self, system: SystemSpec, graph_factory: GraphFactory,
                            experiment_id: str) -> Orchestrator:
        store, embedder = graph_factory()
        cfg = self.config
        bandit = TabularBandit(cfg) if system.use_bandit else None
        cr = ContextResolver(cfg).fit(store)
        retr = ContextRetriever(store, embedder, config=cfg, reranker=self.reranker_factory(),
                                context_resolver=cr, bandit=bandit)
        logger = ExperimentLogger(experiment_id, root=self.results_root)
        return Orchestrator(store, embedder, cfg, system, retriever=retr, bandit=bandit,
                            standardizer=RewardStandardizer(cfg),
                            writeback=WriteBackManager(cfg), logger=logger,
                            solve_fn=self.solve_fn)

    def run_system(
        self,
        system: SystemSpec,
        challenges: List[Dict],
        graph_factory: GraphFactory,
        experiment_id: str,
        train_epochs: int = 0,
        n_seeds: int = 1,
    ) -> SystemReport:
        """Run one system over the dataset (paired eval pass) and report metrics."""
        orch = self._build_orchestrator(system, graph_factory, experiment_id)

        # training passes for learning systems (B4-B6); ignored if no bandit/writeback
        if train_epochs and (system.use_bandit or system.use_writeback):
            for ep in range(train_epochs):
                for i, ch in enumerate(challenges):
                    orch.run_one(ch, experiment_id=experiment_id,
                                 rng=np.random.RandomState(self.base_seed + 7919 * ep + i))

        # fixed, paired eval pass (common random seeds across systems)
        per_challenge: Dict[str, RunMetric] = {}
        from solver.citation import match_citations
        for seed_idx in range(n_seeds):
            for i, ch in enumerate(challenges):
                seed = self.base_seed + 104729 * seed_idx + i   # identical across systems
                run = orch.run_one(ch, experiment_id=experiment_id,
                                   rng=np.random.RandomState(seed))
                sampled = len(run.sampled_set)
                # recompute cited count for this run from the last citation batch
                cited = self._cited_count(orch, run)
                key = f"{ch['name']}#s{seed_idx}"
                per_challenge[key] = RunMetric(
                    challenge_id=ch["name"], solved=run.outcome, steps=run.steps,
                    non_exploit=run.non_exploit_capture, sampled=sampled, cited=cited)

        return self._summarize(system.system_id, per_challenge)

    # ------------------------------------------------------------------
    def _cited_count(self, orch: Orchestrator, run) -> int:
        """Read the citation records this run emitted (cited∩sampled count)."""
        if orch.logger is None:
            return 0
        cites = [c for c in orch.logger.read("citation") if c["run_id"] == run.run_id]
        return sum(1 for c in cites if c["cited"])

    def _summarize(self, system_id: str, per_challenge: Dict[str, RunMetric]) -> SystemReport:
        metrics = list(per_challenge.values())
        n = len(metrics)
        solved = [m.solved for m in metrics]
        solve_rate = float(np.mean(solved)) if n else 0.0
        # solve rate excluding reward-hacked (non-exploit) captures
        solved_clean = [1 if (m.solved and not m.non_exploit) else 0 for m in metrics]
        solve_rate_clean = float(np.mean(solved_clean)) if n else 0.0
        steps_success = [m.steps for m in metrics if m.solved]
        median_steps = float(statistics.median(steps_success)) if steps_success else None
        # recovery masking: on successes, fraction of sampled nodes NOT cited
        rm = []
        for m in metrics:
            if m.solved and m.sampled > 0:
                rm.append((m.sampled - m.cited) / m.sampled)
        recovery_masking = float(np.mean(rm)) if rm else 0.0
        return SystemReport(
            system_id=system_id, n=n, solve_rate=solve_rate,
            solve_rate_excl_non_exploit=solve_rate_clean,
            median_steps_to_flag=median_steps,
            recovery_masking_rate=recovery_masking,
            per_challenge=per_challenge,
        )

    # ------------------------------------------------------------------
    def paired_lift(self, treatment: SystemReport, control: SystemReport,
                    seed: int = 0) -> Dict:
        """Paired solve-rate lift of `treatment` over `control` with bootstrap CI."""
        keys = sorted(set(treatment.per_challenge) & set(control.per_challenge))
        a = [treatment.per_challenge[k].solved for k in keys]
        b = [control.per_challenge[k].solved for k in keys]
        pr = paired_bootstrap_ci(a, b, seed=seed)
        mc = mcnemar_exact(a, b)
        return {
            "treatment": treatment.system_id, "control": control.system_id,
            "n_pairs": pr.n_pairs, "lift": pr.mean_diff,
            "ci_low": pr.ci_low, "ci_high": pr.ci_high,
            "significant": pr.significant, "bootstrap_p": pr.p_value_sign,
            "mcnemar_p": mc["p_value"],
        }

    def evaluate(
        self,
        systems: List[SystemSpec],
        challenges: List[Dict],
        graph_factory: GraphFactory,
        dataset_name: str = "seed",
        train_epochs: int = 5,
        n_seeds: int = 1,
    ) -> Dict:
        """Run all systems, compute paired lift over B0, assemble the results table."""
        reports: Dict[str, SystemReport] = {}
        for sys_ in systems:
            exp_id = f"{dataset_name}_{sys_.system_id}"
            reports[sys_.system_id] = self.run_system(
                sys_, challenges, graph_factory, exp_id,
                train_epochs=train_epochs, n_seeds=n_seeds)

        b0 = reports.get("B0")
        lifts = {}
        if b0 is not None:
            for sid, rep in reports.items():
                if sid == "B0":
                    continue
                lifts[sid] = self.paired_lift(rep, b0, seed=self.base_seed)

        # the §9.4 justification check: B5 must beat B3 (episodic-only)
        writeback_justified = None
        if "B5" in reports and "B3" in reports:
            wb = self.paired_lift(reports["B5"], reports["B3"], seed=self.base_seed)
            writeback_justified = {
                "B5_vs_B3_lift": wb["lift"], "ci": [wb["ci_low"], wb["ci_high"]],
                "justified": wb["lift"] > 0 and wb["significant"],
                "note": ("Gated write-back beats episodic-only." if wb["significant"]
                         else "WARNING: write-back NOT justified vs episodic-only (Faulty-Memory null)."),
            }

        # power analysis for the headline B5-vs-B0 (or best available)
        power = None
        if b0 is not None:
            best = "B5" if "B5" in reports else (sorted(reports)[-1])
            power = power_analysis(baseline_rate=b0.solve_rate, target_lift=0.10)

        return {
            "dataset": dataset_name,
            "systems": {sid: _report_to_row(r) for sid, r in reports.items()},
            "paired_lift_over_B0": lifts,
            "writeback_justification": writeback_justified,
            "power_analysis": power,
            "_reports": reports,   # in-memory handles (not serialized to the table)
        }


def _report_to_row(r: SystemReport) -> Dict:
    return {
        "system_id": r.system_id, "n": r.n, "solve_rate": round(r.solve_rate, 4),
        "solve_rate_excl_non_exploit": round(r.solve_rate_excl_non_exploit, 4),
        "median_steps_to_flag": r.median_steps_to_flag,
        "recovery_masking_rate": round(r.recovery_masking_rate, 4),
    }
