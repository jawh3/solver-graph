"""
pipeline/orchestrator.py - one solve run, end to end (spec §7-§9, §12).

`run_one(challenge)` executes, for one challenge under one SystemSpec:

    retrieve (§7) → assemble context (§7.7) → solve (§7.8) →
    action-grounded citation (§9.3) → reward (§9.1) → bandit update on
    cited∩sampled (§9.3) → promotion registration + gated write-back (§9.4)

and emits the full §12 record set (RunRecord, StepRecord×, CitationRecord×,
UpdateRecord×, WriteBackRecord×). The SystemSpec toggles select B0-B6.

Key correctness points (the things that decide the research result):
  - Credit goes ONLY to S = sampled ∩ cited when credit_mode=cited_only (C-4).
  - Efficiency uses STEPS, never wall-clock (C-5).
  - Provisional write-back nodes are weight-capped (handled in bandit.score) AND
    promotion-gated here (C-8).
  - per-draw propensities + set log-prob are persisted for OPE (C-7).
  - The reward standardizer is updated ONLINE (each run's steps feed the cells);
    the very first runs in a fresh cell standardize against the global pool, which
    partial pooling keeps sane (§9.2).
"""

from __future__ import annotations

import hashlib
import inspect
import time
import uuid
from typing import Callable, Dict, List, Optional

import numpy as np

from config import CreditMode, CreditSignal, SolverConfig
from knowledge_graph.store import KnowledgeGraphStore
from knowledge_graph.models import Node
from retrieval.embedder import Embedder
from retrieval.retriever import ContextRetriever, RetrievalResult
from solver.context_assembly import assemble_context
from solver.citation import match_citations, detect_non_exploit_capture, CitationMatch
from solver.mock_solver import MockSolver
from reward.reward import RewardStandardizer
from reward.grader import DifficultyGrader, MockDifficultyGrader, GraderCalibration
from bandit.bandit import TabularBandit
from writeback.writeback import WriteBackManager, PromotionTracker
from instrumentation.records import (
    RunRecord, StepRecord, CitationRecord, UpdateRecord,
)
from instrumentation.jsonl_logger import ExperimentLogger
from .system_spec import SystemSpec


class SolverUnavailable(RuntimeError):
    """F4b: the LLM backend failed (connection error / server down). Raised instead of logging
    the run as a solve failure. The message contains 'connection error' so the campaign's
    existing retry loop (run_campaign._solve_one) treats it as retryable."""


def _call_solve(fn: Callable, challenge: Dict, nodes: List[Node], max_steps: int, **kw) -> Dict:
    """Call a solve_fn passing only the keyword args its signature accepts (mock solvers and
    test doubles have narrower signatures than the real adapter)."""
    try:
        params = inspect.signature(fn).parameters
        accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        params, accepts_var_kw = {}, False
    kw = {k: v for k, v in kw.items() if v is not None and (accepts_var_kw or k in params)}
    return fn(challenge, nodes, max_steps, **kw)


def _obs_hash(commands: List[str]) -> str:
    return hashlib.sha1("|".join(commands).encode("utf-8")).hexdigest()[:12]


def challenge_query_text(challenge: Dict) -> str:
    """§7 step 1: query = description + artifact static metrics + category hint.

    F1 (FIX_CATALOG 2026-09): the retrieval query uses the solver-visible `_display_name`
    when present, never the logging id, because in the controlled corpus the logging id
    encodes the method (family name) and would turn similarity retrieval into an oracle."""
    parts = [
        str(challenge.get("_display_name") or challenge.get("name", "")),
        str(challenge.get("category", "")),
        str(challenge.get("description", "")),
    ]
    for art in challenge.get("artifacts", []) or []:
        if isinstance(art, dict):
            parts.append(str(art.get("static_metrics", "")))
    return " ".join(p for p in parts if p).strip() or challenge.get("name", "challenge")


class Orchestrator:
    """Runs one challenge through the full pipeline under a SystemSpec."""

    def __init__(
        self,
        store: KnowledgeGraphStore,
        embedder: Embedder,
        config: SolverConfig,
        system: SystemSpec,
        retriever: Optional[ContextRetriever] = None,
        bandit: Optional[TabularBandit] = None,
        standardizer: Optional[RewardStandardizer] = None,
        grader: Optional[DifficultyGrader] = None,
        writeback: Optional[WriteBackManager] = None,
        logger: Optional[ExperimentLogger] = None,
        solve_fn: Optional[Callable] = None,
        git_commit: str = "",
    ):
        self.store = store
        self.embedder = embedder
        self.config = config
        self.system = system
        self.bandit = bandit if bandit is not None else (TabularBandit(config) if system.use_bandit else None)
        self.retriever = retriever or ContextRetriever(store, embedder, config=config, bandit=self.bandit)
        # keep the retriever's bandit in sync with ours
        self.retriever.bandit = self.bandit
        self.standardizer = standardizer or RewardStandardizer(config)
        self.grader = grader or MockDifficultyGrader()
        self.calibration = GraderCalibration()
        # BC competitor (§14.2): champion-broadcast write-back (ungated, immediate trust)
        if system.forge_competitor:
            from writeback.forge_competitor import ForgeChampionWriteBack
            self.writeback = ForgeChampionWriteBack(config)
        else:
            self.writeback = writeback or WriteBackManager(config)
        self.promotion = PromotionTracker(config)
        self.logger = logger
        self.git_commit = git_commit
        self._causal_rerolls = 0          # cost counter: extra solves spent on causal credit
        self._reroll_acct = [0, 0, 0.0]   # per-run re-roll accounting (solves, tokens, wall s); reset in run_one
        self.frozen = False               # eval mode: keep learned Q for sampling, do NO updates
        self.raise_on_llm_error = True    # F4b: an LLM failure raises SolverUnavailable, never logs a failure
        self.flag_checker_factory: Optional[Callable] = None   # F4c: challenge -> callable(flag)->bool
        self.randomize_lesson_order = False   # F7: per-run content-order shuffle of the rendered lessons
        self.citation_generic_tokens = None   # F2/review: domain tokens that never ground a citation
        # default solver = mock; real path injects solver.solver.solve_challenge wrapper
        if solve_fn is not None:
            self._solve = solve_fn
        else:
            mock = MockSolver(seed=config.seed)
            self._solve = lambda ch, nodes, max_rounds, seed=None: mock.solve(
                ch, nodes, max_rounds=max_rounds, seed=seed)

    # ==================================================================
    def run_one(self, challenge: Dict, experiment_id: str = "adhoc",
                run_id: Optional[str] = None,
                rng: Optional[np.random.RandomState] = None,
                llm_seed: Optional[int] = None) -> RunRecord:
        """llm_seed (F9): the solver's sampling seed, chosen by the CAMPAIGN from (challenge,
        eval seed) so that paired conditions share it; independent of the retrieval rng."""
        cfg = self.config
        sys_ = self.system
        run_id = run_id or uuid.uuid4().hex[:12]
        rng = rng or np.random.RandomState(cfg.seed)
        self._reroll_acct = [0, 0, 0.0]          # causal re-roll compute for THIS run (solves, tokens, wall s)

        # difficulty (ground-truth-first, §9.2) -> standardization cell
        grade = self.grader.grade(challenge)
        category = str(challenge.get("category", "unknown"))
        difficulty = grade.difficulty

        # ---------- retrieval (§7) ----------
        retrieval: Optional[RetrievalResult] = None
        sampled_nodes: List[Node] = []
        injected_context = ""
        if sys_.use_graph:
            query = challenge_query_text(challenge)
            # F22: the bandit CATEGORY context may come from a private, non-rendered key
            # (`_retrieval_category`: outer alphabet class, or the family id in the oracle arm)
            retrieval = self.retriever.retrieve(
                query, category=(challenge.get("_retrieval_category") or category), rng=rng,
                use_rerank=sys_.use_rerank, use_mmr=sys_.use_mmr,
                use_bandit=sys_.use_bandit, deterministic=sys_.deterministic,
                explore=not self.frozen,                       # review: epsilon=0 in eval
                on_target_family=challenge.get("_family"),      # stratified exploration (label-blind)
                force_on_target=sys_.force_on_target,           # oracle-/worst-static arms
            )
            sampled_nodes = retrieval.sampled_nodes
            injected_context = assemble_context(sampled_nodes)

        # ---------- solve (§7.8) ----------
        # per-family round budget: budget = solution_length + slack (set by the campaign on
        # the challenge as `_max_steps`); falls back to the fixed cfg.max_steps (pilot mode).
        max_steps = int(challenge.get("_max_steps") or cfg.max_steps)
        t0 = time.time()
        # F7/review: the order seed is a hash of (challenge, llm_seed), NOT a draw from the retrieval
        # rng - conditions consume different numbers of rng draws, which would render the same
        # M-set in different orders across conditions.
        order_seed = (int(hashlib.sha1(f"order|{challenge.get('name')}|{llm_seed}".encode()).hexdigest()[:8], 16)
                      if self.randomize_lesson_order else None)
        checker = self.flag_checker_factory(challenge) if self.flag_checker_factory else None
        solve_result = _call_solve(self._solve, challenge, sampled_nodes, max_steps,
                                   seed=llm_seed, order_seed=order_seed, flag_checker=checker)
        wall = time.time() - t0
        gave_up_reason = str(solve_result.get("gave_up_reason") or "")
        solver_error = gave_up_reason in ("llm_error", "llm_bad_request")
        if gave_up_reason == "llm_error" and self.raise_on_llm_error:
            raise SolverUnavailable(f"LLM backend connection error during run {run_id} "
                                    f"(challenge {challenge.get('name')}); not logged as a failure")
        flag = solve_result.get("flag")
        outcome = 1 if flag else 0
        steps = solve_result.get("rounds_used", 0) or 0
        non_exploit = detect_non_exploit_capture(challenge, flag, solve_result.get("trace", []))

        # ---------- step records + counterfactual would-have-retrieved (§8) ----------
        self._emit_step_records(run_id, challenge, solve_result, sampled_nodes)

        # ---------- action-grounded citation (§9.3) ----------
        matches: List[CitationMatch] = match_citations(
            sampled_nodes, solve_result.get("trace", []),
            mode=getattr(cfg, "citation_mode", "any"), store=self.store,
            generic_tokens=self.citation_generic_tokens)
        cited_ids = {m.node_id for m in matches if m.cited}
        if self.logger:
            for m in matches:
                self.logger.log_citation(CitationRecord(
                    run_id=run_id, node_id=m.node_id, cited=m.cited,
                    match_evidence=m.match_evidence, confidence=m.confidence, sampled=True,
                    mode=m.mode, n_distinctive=m.n_distinctive))

        # ---------- reward (§9.1) ----------
        self.standardizer.observe(category, difficulty, steps)   # online standardization
        rb = self.standardizer.reward(outcome, steps, category, difficulty)
        self.calibration.observe(difficulty, bool(outcome))

        # ---------- bandit update on credit-eligible set (§9.3, + C9 causal credit) ----------
        # `self.frozen` (eval mode) keeps the learned Q live for sampling but performs
        # NO updates and NO causal re-rolls - a clean, idempotent freeze (resume-safe).
        context_id = retrieval.context_node_id if retrieval else ""
        # OBSERVATIONAL credit does NO per-run update - its Q is recomputed in batch
        # from the logged sampling propensities each epoch (ope.causal_value).
        _observational = self.config.credit_signal is CreditSignal.OBSERVATIONAL
        if (self.bandit is not None and sys_.use_bandit and retrieval is not None
                and not self.frozen and not _observational):
            self.bandit.observe_reward_for_baseline(context_id, rb.R_final)
            eligible = self._credit_eligible(sampled_nodes, cited_ids)
            cos_by_id = {c["node_id"]: c["cosine"] for c in retrieval.candidate_pool}
            for node in eligible:
                sim = cos_by_id.get(node.id, 0.5)
                reward, subtract_b, reason = self._node_credit(
                    node, challenge, sampled_nodes, outcome, rb, run_id, rng,
                    context_id=context_id, category=category, difficulty=difficulty,
                    cited_ids=cited_ids)
                pre_Q, post_Q, pre_n, post_n, scale = self.bandit.update(
                    context_id, node.id, reward, similarity=sim, success=bool(outcome),
                    subtract_baseline=subtract_b)
                if self.logger:
                    self.logger.log_update(UpdateRecord(
                        run_id=run_id, context_node_id=context_id, node_id=node.id,
                        pre_Q=pre_Q, post_Q=post_Q, pre_n=pre_n, post_n=post_n,
                        distance_scale=scale, credit_eligible=(reason != "exposure"), reason=reason))

        # ---------- promotion registration for provisional nodes (§9.4) ----------
        if sys_.use_writeback and not self.frozen:
            for node in sampled_nodes:
                if node.attributes.get("tier") == "provisional" and node.id in cited_ids:
                    self.promotion.register_citation(
                        self.store, node.id, challenge.get("name", run_id),
                        success=bool(outcome), ungated=not sys_.writeback_gated)

        # ---------- write-back (§9.4) ----------
        if sys_.use_writeback and not sys_.episodic_only and not self.frozen:
            recs = self.writeback.write_back(challenge, solve_result, run_id,
                                             self.store, self.embedder)
            if self.logger:
                for wr in recs:
                    self.logger.log_writeback(wr)

        # ---------- assemble + persist RunRecord (§12) ----------
        run = RunRecord(
            run_id=run_id, challenge_id=challenge.get("name", run_id),
            experiment_id=experiment_id, policy_config_hash=cfg.hash(),
            git_commit=self.git_commit,
            challenge_meta={"category": category, "difficulty": difficulty,
                            "difficulty_source": grade.source,
                            "contamination_flag": bool(challenge.get("contamination_flag", False)),
                            "dataset": challenge.get("dataset", "unknown"),
                            "solution_length": challenge.get("_solution_length"),  # decode-LAYER count (NOT agent rounds; do NOT use as a step-oracle - layers collapse into one piped round)
                            "max_steps": max_steps,                                # budget (headroom for loops)
                            "system_id": sys_.system_id},
            context_node_id=context_id,
            context_resolution=(retrieval.context_resolution if retrieval else ""),
            candidate_pool=(retrieval.candidate_pool if retrieval else []),
            sampled_set=(retrieval.sampled_ids if retrieval else []),
            per_draw_propensity=(retrieval.per_draw_propensity if retrieval else []),
            set_logprob=(retrieval.set_logprob if retrieval else 0.0),
            temperature=(retrieval.temperature if retrieval else cfg.tau),
            policy=(retrieval.policy if retrieval else {}),
            exploration=(retrieval.exploration if retrieval else False),
            outcome=outcome, gave_up_reason=gave_up_reason, error=solver_error,
            non_exploit_capture=non_exploit,
            steps=steps, tokens=solve_result.get("tokens", 0),  # 0 in mock; real solver fills it
            total_commands=int(solve_result.get("total_commands", 0) or 0),  # effort signal (D1)
            wall_clock_s=wall, time_to_flag_steps=(steps if outcome else None),
            llm_seed=(int(llm_seed) if llm_seed is not None else None), phase=("eval" if self.frozen else "train"),
            reroll_solves=int(self._reroll_acct[0]), reroll_tokens=int(self._reroll_acct[1]),
            reroll_wall_s=float(self._reroll_acct[2]),
            R_raw=float(outcome), R_efficiency_e=rb.efficiency_e, R_final=rb.R_final,
            baseline_b=(self.bandit.baseline(context_id) if self.bandit else 0.0),
            seed=cfg.seed, solver_model=getattr(self, "solver_model", "mock-solver"),
            embedder_model=self.embedder.__class__.__name__,
            reranker_model=self.retriever.reranker.__class__.__name__ if sys_.use_graph else "",
        )
        if self.logger:
            self.logger.log_run(run)
        return run

    # ------------------------------------------------------------------
    def _credit_eligible(self, sampled_nodes: List[Node], cited_ids) -> List[Node]:
        if self.config.credit_mode is CreditMode.ALL_M:
            return list(sampled_nodes)               # recovery-masking baseline
        if self.config.credit_mode is CreditMode.CITED_ONLY_DECAY:
            return list(sampled_nodes)               # uncited nodes get an EXPOSURE update (see _node_credit)
        return [n for n in sampled_nodes if n.id in cited_ids]   # cited_only (default)

    def _node_credit(self, node, challenge, sampled_nodes, outcome, rb, run_id, rng,
                     context_id="", category=None, difficulty=None, cited_ids=None):
        """Per-node credit signal (spec §9.3 + C9). Returns (reward, subtract_baseline, reason).

        OUTCOME: the run reward R (confounded by recovery masking).
        CAUSAL:  the LEAVE-ONE-OUT marginal value - re-roll the solve WITHOUT this
                 lesson and credit the marginal flag difference. A poison the agent
                 recovers from yields marginal ≈ 0 and is driven down; a lesson that
                 was necessary yields marginal ≈ 1. This is the recovery-masking fix.
                 The marginal is already an advantage, so the bandit does not subtract
                 its outcome baseline (subtract_baseline=False).
        HYBRID:  outcome credit until the cell is warm (n ≥ causal_warmup_n) and
                 suspicious (Q ≥ causal_q_gate), then causal - bounds re-roll cost.
        """
        cfg = self.config
        if (cfg.credit_mode is CreditMode.CITED_ONLY_DECAY and cited_ids is not None
                and node.id not in cited_ids):
            # F40: exposure, not credit - reward = the context baseline b, so the advantage is exactly 0:
            # n grows and Q relaxes toward 0 while the prior's weight kappa/(n+kappa) decays. No blame,
            # no credit, for a lesson the solver did not act on.
            return (self.bandit.baseline(context_id) if self.bandit is not None else 0.0), True, "exposure"
        signal = cfg.credit_signal
        use_causal = signal is CreditSignal.CAUSAL
        if signal is CreditSignal.HYBRID:
            cell = self.bandit.cells.get(context_id, {}).get(node.id)
            n = cell.n if cell else 0.0
            q = cell.Q if cell else 0.0
            use_causal = (n >= cfg.causal_warmup_n and q >= cfg.causal_q_gate)
        if not use_causal:
            return rb.R_final, True, f"outcome:{cfg.credit_mode.value}"
        # CAUSAL: strict leave-one-out - re-roll WITHOUT this node, holding all other
        # sampled nodes fixed, averaged over `causal_reroll_seeds` independent re-rolls
        # (the real solver is stochastic; a single re-roll is a coin flip). Each re-roll
        # gets a distinct seed so it samples a different trajectory reproducibly.
        without = [n for n in sampled_nodes if n.id != node.id]
        max_steps = int(challenge.get("_max_steps") or cfg.max_steps)
        checker = self.flag_checker_factory(challenge) if self.flag_checker_factory else None
        K = max(1, cfg.causal_reroll_seeds)
        # seeds are drawn up front, in the same order the old sequential loop drew them, so the
        # parallel execution below reproduces the sequential results exactly
        seeds = [int(rng.randint(0, 2**31 - 1)) if rng is not None else (cfg.seed + k) for k in range(K)]

        def _one(reroll_seed):
            # review #8: re-rolls go through the same path as the main solve (checker, lesson order)
            order_seed = (int(hashlib.sha1(f"order|{challenge.get('name')}|{reroll_seed}".encode()).hexdigest()[:8], 16)
                          if self.randomize_lesson_order else None)
            t0 = time.time()
            res = _call_solve(self._solve, challenge, without, max_steps,
                              seed=reroll_seed, order_seed=order_seed, flag_checker=checker)
            return res, time.time() - t0

        # 2026-09-07 (deployed_causal arm): the K re-rolls run concurrently (the real adapter isolates
        # every solve in its own directory and the client is thread-safe), cutting a run's wall clock
        # from 1 + K·(cited) solves to ≈ 1 + (cited) solves.
        workers = max(1, int(getattr(cfg, "causal_parallel_workers", 6) or 1))
        if workers > 1 and K > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(K, workers)) as ex:
                results = list(ex.map(_one, seeds))
        else:
            results = [_one(sd) for sd in seeds]
        margs = []
        for res, dt in results:
            if str(res.get("gave_up_reason") or "") == "llm_error" and self.raise_on_llm_error:
                raise SolverUnavailable(f"LLM backend connection error during causal re-roll for {node.id}")
            self._causal_rerolls += 1
            self._reroll_acct[0] += 1; self._reroll_acct[1] += int(res.get("tokens") or 0); self._reroll_acct[2] += float(dt)
            solved_without = 1 if res.get("flag") else 0
            steps_without = res.get("rounds_used", max_steps)
            # MARGINAL REWARD (not just outcome): captures both success-change AND
            # speed-change, so a poison the agent recovers from but is *slowed* by gets
            # negative credit (latent poison) even when the outcome is unchanged.
            r_without = self.standardizer.reward(solved_without, steps_without,
                                                 category, difficulty).R_final
            margs.append(rb.R_final - r_without)      # signed: + helpful, 0 irrelevant, - harmful
        marg = sum(margs) / len(margs)                # on the reward scale, already an advantage
        return marg, False, f"causal:margR={marg:+.2f}(n={len(margs)})"

    def _emit_step_records(self, run_id: str, challenge: Dict, solve_result: Dict,
                           sampled_nodes: List[Node]) -> None:
        if not self.logger:
            return
        # counterfactual "would-have-retrieved" (§8): top-M by similarity to the
        # query AUGMENTED with commands run so far - computed, NOT applied (static).
        running_cmds: List[str] = []
        base_query = challenge_query_text(challenge)
        for rec in solve_result.get("trace", []) or []:
            cmds = [str(c) for c in rec.get("commands", []) or []]
            running_cmds.extend(cmds)
            would = self._counterfactual_retrieve(base_query, running_cmds)
            action = rec.get("action", "run_commands")
            self.logger.log_step(StepRecord(
                run_id=run_id, step_idx=rec.get("round", 0),
                action_type=action,
                command=" ; ".join(cmds), observation_hash=_obs_hash(cmds),
                would_have_retrieved=would,
                stagnation_flag=False,
                finish_reason=str(rec.get("finish_reason") or ""),
                llm_response_head=(str(rec.get("llm_response") or "")[:240] if action == "parse_error" else ""),
            ))

    def _counterfactual_retrieve(self, base_query: str, running_cmds: List[str]) -> List[str]:
        """Cheap counterfactual: what static-M would a refresh at this step pick?

        Free instrumentation only (§8): computed from the evolving trajectory
        text, never applied. Returns up to M node ids. Skipped if no graph.
        """
        if not self.system.use_graph:
            return []
        q = base_query + " " + " ".join(running_cmds[-3:])
        try:
            pool = self.retriever.get_top_n_candidates(q, n=self.config.M)
            return [n.id for n, _ in pool]
        except Exception:
            return []
