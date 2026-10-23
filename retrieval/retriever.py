"""
retrieval/retriever.py - the §7 solve-time retrieval pipeline (Component 2).

Pipeline (spec §7), all instrumented for OPE:
    1. query construction (embed)            -> query_emb
    2. high-recall top-N by cosine           -> pool_N   (N fixed; NOT grown, C-3)
    3. cross-encoder rerank to R             -> pool_R   (precision downstream)
    4. MMR diversity to the candidate pool   -> pool_D   (competing alternatives)
    5. context resolution (cluster/category) -> context_node_id   (C-1)
    6. sim-prior-blended stochastic sampling -> M nodes + propensities   (C-2)
    7/8 assembly + solve happen in the solver harness (C3) / orchestrator.

Backward-compatible: the original `get_top_n_candidates` and `sample_top_m`
methods are preserved (used by `experiment_harness.py` and existing tests). The
new `retrieve()` method runs the full pipeline and returns a `RetrievalResult`
carrying everything `RunRecord` (§12) and OPE (§13) need - crucially the
per-draw propensities and per-candidate score COMPONENTS so a counterfactual
policy (different κ/τ/M) can be re-scored offline without a GPU.

Design citations:
  - sim-prior pseudo-count blend (C-2): PURPLE 2601.12078; cold start by
    similarity not uniform - JumpStartFalseStart 2604.02527.
  - rerank+MMR not bigger-N (C-3): LongContextMeetsRAG 2410.05983,
    CrossEncoderVsLLM 2403.10407.
  - propensity logging for OPE (C-7): DoublyRobust 1103.4601, OfflineEval 1003.5956.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import ContextResolution, SolverConfig
from knowledge_graph.store import KnowledgeGraphStore
from knowledge_graph.models import Node

from .embedder import Embedder
from .reranker import Reranker, MockReranker
from .context_resolution import ContextResolver
from .diversity import mmr_select
from .sampling import sample_without_replacement, stratified_design_marginals

log = logging.getLogger(__name__)


def fixed_design_draw(n: int, m: int, rng: np.random.RandomState, on_target_idx: List[int],
                      stratified: bool):
    """History-free exploration draw over a pool of n candidates (review 2026-09-04).

    stratified: each on-target candidate is included independently w.p. 0.5; the remaining
    slots are a uniform subset of the other candidates. uniform: a uniform M-subset.
    Returns (order, per_item_pi, set_logprob, pi_by_index) with EXACT marginal inclusion
    probabilities (no Monte-Carlo needed for these runs)."""
    from math import comb, log as _log
    ot = [i for i in on_target_idx if 0 <= i < n] if stratified else []
    k = len(ot)
    m = min(m, n)
    chosen: List[int] = []
    logp = 0.0
    if k:
        flips = rng.rand(k) < 0.5
        flipped = [i for i, f in zip(ot, flips) if f]
        logp += k * _log(0.5)
        if len(flipped) > m:                     # keep a UNIFORM m-subset (not the first m)
            keep = rng.choice(len(flipped), size=m, replace=False)
            chosen = [flipped[int(i)] for i in keep]
            logp += -_log(comb(len(flipped), m))
        else:
            chosen = list(flipped)
    rest = [i for i in range(n) if i not in ot]
    need = m - len(chosen)
    if need > 0 and rest:
        need = min(need, len(rest))
        pick = rng.choice(len(rest), size=need, replace=False)
        chosen += [rest[int(i)] for i in pick]
        logp += -_log(comb(len(rest), need))
    pi_ot, pi_rest = stratified_design_marginals(n, k, m)   # exact for every (n, k, m)
    pi = np.full(n, pi_rest, dtype=np.float64)
    if k:
        pi[ot] = pi_ot
    order = list(chosen)
    per_item = [float(pi[i]) for i in order]
    return order, per_item, float(logp), pi


def _node_text(node: Node) -> str:
    a = node.attributes
    for key in ("text", "canonical_name", "description", "title"):
        if a.get(key):
            return str(a[key])
    return node.id


def cosine_to_reward_scale(cosine: float) -> float:
    """normalize_to_reward_scale(cosine) - map similarity into the reward range.

    Reward R lives roughly in [0, ~1.5] (R = o*(1+λe)). We map cosine via
    max(0, cosine) so a similarity prior is commensurate with an observed Q and
    with the optimistic init Q0 (~0.5). Negative cosines (irrelevant) -> 0 prior.
    (Logged design choice; see OPEN_DECISIONS 2026-05-31.)
    """
    return max(0.0, float(cosine))


@dataclass
class RetrievalResult:
    """Everything the pipeline produced for one query (feeds RunRecord + OPE)."""
    context_node_id: str
    context_resolution: str
    query_emb: np.ndarray
    candidate_pool: List[Dict] = field(default_factory=list)  # logged pool w/ score components
    pool_d_nodes: List[Node] = field(default_factory=list)    # the diverse pool (sampling input)
    sampled_nodes: List[Node] = field(default_factory=list)   # M nodes in draw order
    sampled_ids: List[str] = field(default_factory=list)
    per_draw_propensity: List[Dict] = field(default_factory=list)  # [{node_id, p}]
    set_logprob: float = 0.0
    temperature: float = 1.0
    policy: Dict = field(default_factory=dict)   # F3: {tau_deploy, tau_explore, epsilon, deterministic}
    exploration: bool = False                    # F3: this run was drawn from the exploration stream


class ContextRetriever:
    """Solve-time retrieval over the knowledge graph."""

    def __init__(
        self,
        store: KnowledgeGraphStore,
        embedder: Embedder,
        config: Optional[SolverConfig] = None,
        reranker: Optional[Reranker] = None,
        context_resolver: Optional[ContextResolver] = None,
        bandit=None,            # bandit.TabularBandit | None (None => similarity-only)
        linucb=None,            # bandit.LinUCBBandit | None (parametric mode)
    ):
        self.store = store
        self.embedder = embedder
        self.config = config or SolverConfig()
        self.reranker = reranker if reranker is not None else MockReranker()
        self.context_resolver = context_resolver or ContextResolver(self.config)
        self.bandit = bandit
        self.linucb = linucb

    # ==================================================================
    # Backward-compatible primitives (kept for existing harness/tests)
    # ==================================================================
    def get_top_n_candidates(self, query: str, n: int) -> List[Tuple[Node, float]]:
        """Embed the query and return Top-N (Node, cosine) by cosine similarity."""
        query_vec = self.embedder.embed(query)
        return self._top_n_from_vec(query_vec, n)

    def _top_n_from_vec(self, query_vec: np.ndarray, n: int) -> List[Tuple[Node, float]]:
        node_ids, embeddings_matrix = self.store.get_all_embeddings()
        if len(node_ids) == 0:
            return []
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            similarities = np.zeros(len(node_ids))
        else:
            qn = query_vec / query_norm
            matrix_norms = np.linalg.norm(embeddings_matrix, axis=1)
            matrix_norms[matrix_norms == 0] = 1.0
            normalized_matrix = embeddings_matrix / matrix_norms[:, np.newaxis]
            similarities = np.dot(normalized_matrix, qn)
        n_actual = min(n, len(node_ids))
        top_indices = np.argsort(similarities)[-n_actual:][::-1]
        results = []
        for idx in top_indices:
            node = self.store.get_node(node_ids[idx])
            if node is not None:
                results.append((node, float(similarities[idx])))
        return results

    def sample_top_m(self, candidates: List[Tuple[Node, float]], m: int,
                     exploration_rate: float = 1.0) -> List[Node]:
        """Legacy similarity-only sampler (kept for the temperature sweep harness)."""
        if not candidates:
            return []
        m_actual = min(m, len(candidates))
        if exploration_rate <= 0.0:
            return [node for node, _ in candidates[:m_actual]]
        weights = np.array([w for _, w in candidates], dtype=np.float64)
        scaled = weights / exploration_rate
        shifted = scaled - np.max(scaled)
        exp_w = np.exp(shifted)
        probs = exp_w / np.sum(exp_w)
        chosen = np.random.choice(len(candidates), size=m_actual, replace=False, p=probs)
        return [candidates[i][0] for i in chosen]

    # ==================================================================
    # The full §7 pipeline
    # ==================================================================
    def retrieve(
        self,
        query_text: str,
        category: Optional[str] = None,
        rng: Optional[np.random.RandomState] = None,
        use_rerank: bool = True,
        use_mmr: bool = True,
        use_bandit: bool = True,
        deterministic: bool = False,
        explore: bool = True,
        on_target_family: Optional[str] = None,
        force_on_target: str = "",
    ) -> RetrievalResult:
        """explore=False disables the exploration stream (eval / frozen runs - review 2026-09-04).
        on_target_family: the challenge's family id, used ONLY by the stratified exploration
        design (label-blind: it does not distinguish helpful from poison)."""
        """Run the retrieval pipeline for one challenge query.

        The stage toggles enact the §14.2 baselines:
          B1 similarity-only  -> use_rerank=False, use_mmr=False, use_bandit=False
          B2 + rerank + MMR   -> use_rerank=True,  use_mmr=True,  use_bandit=False
          B4+ + bandit        -> use_bandit=True
        """
        cfg = self.config
        rng = rng or np.random.RandomState(cfg.seed)

        # 1. query construction
        query_emb = self.embedder.embed(query_text)

        # 5. context resolution (done early; needed for bandit scoring)
        context_id = self.context_resolver.resolve(query_emb, category=category)

        # 2. high-recall top-N
        pool_n = self._top_n_from_vec(query_emb, cfg.N)
        if not pool_n:
            return RetrievalResult(context_node_id=context_id,
                                   context_resolution=cfg.context_resolution.value,
                                   query_emb=query_emb, temperature=max(cfg.tau, cfg.tau_min))
        cosine_by_id = {node.id: cos for node, cos in pool_n}

        # 3. cross-encoder rerank -> R  (B1 skips: relevance = cosine order)
        if use_rerank:
            pool_r = self.reranker.rerank(query_text, pool_n, keep=cfg.R)
        else:
            pool_r = [(node, cos) for node, cos in pool_n[:cfg.R]]
        rerank_by_id = {node.id: score for node, score in pool_r}

        # 4. MMR diversity -> candidate pool D  (B1 skips: keep rerank/cosine order)
        keep_d = min(cfg.R, len(pool_r))
        if use_mmr:
            pool_d = mmr_select(pool_r, self._embedding_of, k=keep_d, lambda_mmr=cfg.lambda_mmr)
        else:
            pool_d = pool_r[:keep_d]
        mmr_kept_ids = {node.id for node, _ in pool_d}

        # 6. weight-blended stochastic sampling of M (with τ_min floor)
        temperature = max(cfg.tau, cfg.tau_min)
        scored_pool: List[Tuple[Node, float, float, float]] = []
        # each entry: (node, sampling_score, prior, cosine)
        # review 2026-09-05 F-2: under recall-all the cross-encoder only re-orders the pool, and the
        # stochastic arms sample on (n·Q + κ·prior)/(n+κ) with prior = cosine - so the reranker was
        # inert for them. `prior_source="rerank"` uses sigmoid(rerank logit) as the prior instead
        # (legacy default "cosine" keeps every old log reproducible). Logged in `policy`.
        prior_source = str(getattr(cfg, "prior_source", "cosine") or "cosine")
        use_rerank_prior = (prior_source == "rerank" and use_rerank)
        # F41 (mock ladder 2026-09-05): ms-marco logits for these pairs sit near -9, so the "rerank" prior is
        # ~1e-4 for every lesson, i.e. effectively "none". `none` makes that explicit: prior = 0, the
        # sampler runs on Q alone and exploration comes from the epsilon stream.
        for node, _rel in pool_d:
            cos = cosine_by_id.get(node.id, 0.0)
            if prior_source == "none":
                prior = 0.0
            elif use_rerank_prior:
                rel = float(rerank_by_id.get(node.id, 0.0))
                prior = float(1.0 / (1.0 + np.exp(-rel)))
            else:
                prior = cosine_to_reward_scale(cos)
            score, q, n = self._sampling_score(context_id, node, prior, query_emb,
                                               use_bandit=use_bandit)
            scored_pool.append((node, score, prior, cos))

        scores = [s for _, s, _, _ in scored_pool]
        m = min(cfg.M, len(scored_pool))
        # F3 (FIX_CATALOG 2026-09): three sampling regimes, all logged so OPE can reconstruct
        # the inclusion propensity under the ACTUAL policy:
        #   deterministic  -> top-M of the (rerank/MMR-ordered) pool, p=1 (the deployed static arm)
        #   deployed       -> softmax at tau (with epsilon_explore=0)
        #   mixture        -> with prob epsilon the run is drawn at tau_explore (exploration stream)
        exploration = False
        pi_exact = None
        tau_deploy = max(cfg.tau, cfg.tau_min)
        design = getattr(cfg, "exploration_design", "softmax")
        eps = float(cfg.epsilon_explore) if explore else 0.0
        if deterministic:
            order = list(range(m)); per_draw_p = [1.0] * m; set_logprob = 0.0; temperature = 0.0
        else:
            temperature = tau_deploy
            if eps > 0 and float(rng.rand()) < eps:
                exploration = True
                if design == "softmax":
                    temperature = float(cfg.tau_explore)
                    order, per_draw_p, set_logprob = sample_without_replacement(
                        scores, m=m, tau=temperature, rng=rng)
                else:
                    ot_idx = [i for i, (node, _, _, _) in enumerate(scored_pool)
                              if on_target_family and node.attributes.get("_family") == on_target_family]
                    order, per_draw_p, set_logprob, pi_exact = fixed_design_draw(
                        len(scored_pool), m, rng, ot_idx, stratified=(design == "stratified"))
            else:
                order, per_draw_p, set_logprob = sample_without_replacement(
                    scores, m=m, tau=temperature, rng=rng)
        policy = {"tau_deploy": float(tau_deploy), "tau_explore": float(cfg.tau_explore),
                  "epsilon": float(eps), "deterministic": bool(deterministic),
                  "design": ("softmax" if deterministic else design), "explore_enabled": bool(explore),
                  "prior_source": ("none" if prior_source == "none" else ("rerank" if use_rerank_prior else "cosine")),
                  "pool_order": ("mmr" if use_mmr else ("rerank" if use_rerank else "cosine"))}
        if force_on_target and on_target_family:
            # oracle-static / worst-static arms (review 2026-09-04 §8): the family's lesson with the
            # given ground-truth label is forced into the M-set (replacing the last draw if absent).
            # No randomization -> flagged in the policy and skipped by every observational estimator.
            forced = [i for i, (node, _, _, _) in enumerate(scored_pool)
                      if node.attributes.get("_family") == on_target_family
                      and node.attributes.get("_truth") == force_on_target]
            # review 2026-09-05 F-13: a clean bracket also keeps the OPPOSITE on-target label out
            # (oracle = helpful in, poison out; worst = poison in, helpful out).
            opposite = {"helpful": "poison", "poison": "helpful"}.get(force_on_target)
            excluded = {i for i, (node, _, _, _) in enumerate(scored_pool)
                        if node.attributes.get("_family") == on_target_family
                        and node.attributes.get("_truth") == opposite}
            order = [i for i in order if i not in excluded]
            if forced and forced[0] not in order:
                if len(order) >= m and order:
                    order = order[:-1]
                order = order + [forced[0]]
            fill = [i for i in range(len(scored_pool)) if i not in order and i not in excluded]
            while len(order) < m and fill:
                order.append(fill.pop(0))
            per_draw_p = [1.0] * len(order); set_logprob = 0.0
            policy["forced"] = force_on_target; policy["deterministic"] = True
        sampled_nodes = [scored_pool[i][0] for i in order]
        sampled_ids = [n.id for n in sampled_nodes]
        per_draw_propensity = [
            {"node_id": scored_pool[order[t]][0].id, "p": per_draw_p[t]}
            for t in range(len(order))
        ]

        # build the logged candidate pool (score COMPONENTS for OPE re-scoring)
        candidate_pool = []
        for node, score, prior, cos in scored_pool:
            cell = None
            if self.bandit is not None:
                cell = self.bandit.cells.get(context_id, {}).get(node.id)
            entry = {
                "node_id": node.id,
                "cosine": cos,
                "rerank_score": rerank_by_id.get(node.id, 0.0),
                "mmr_kept": node.id in mmr_kept_ids,
                "prior": prior,
                "Q": (cell.Q if cell else None),
                "n": (cell.n if cell else 0.0),
                "score": score,
                "tier": node.attributes.get("tier", "trusted"),
            }
            if on_target_family:
                entry["on_target"] = bool(node.attributes.get("_family") == on_target_family)
            if pi_exact is not None:
                entry["pi_explore"] = float(pi_exact[len(candidate_pool)])   # exact design propensity
            candidate_pool.append(entry)

        return RetrievalResult(
            context_node_id=context_id,
            context_resolution=cfg.context_resolution.value,
            query_emb=query_emb,
            candidate_pool=candidate_pool,
            pool_d_nodes=[n for n, _ in pool_d],
            sampled_nodes=sampled_nodes,
            sampled_ids=sampled_ids,
            per_draw_propensity=per_draw_propensity,
            set_logprob=set_logprob,
            temperature=temperature,
            policy=policy,
            exploration=exploration,
        )

    # ------------------------------------------------------------------
    def _embedding_of(self, node_id: str) -> np.ndarray:
        vec = self.store.get_embedding(node_id)
        if vec is None:
            return np.zeros(self.config.embedding_dim, dtype=np.float32)
        return vec

    def _sampling_score(self, context_id: str, node: Node, prior: float,
                        query_emb: np.ndarray,
                        use_bandit: bool = True) -> Tuple[float, Optional[float], float]:
        """The §6.2 sampling score for a candidate. Returns (score, Q, n).

        - parametric mode: LinUCB UCB score (if a linucb model is wired).
        - tabular mode: bandit blend score (n*Q + κ*prior)/(n+κ), provisional-capped.
        - no bandit (use_bandit=False or self.bandit is None): similarity prior
          only (the B1/B2 similarity-only baselines).
        """
        provisional = node.attributes.get("tier") == "provisional"
        if (use_bandit and self.config.context_resolution is ContextResolution.PARAMETRIC
                and self.linucb is not None):
            cand_emb = self._embedding_of(node.id)
            return self.linucb.score(query_emb, cand_emb), None, 0.0
        if use_bandit and self.bandit is not None:
            score = self.bandit.score(context_id, node.id, prior, provisional=provisional)
            cell = self.bandit.cells.get(context_id, {}).get(node.id)
            return score, (cell.Q if cell else None), (cell.n if cell else 0.0)
        # similarity-only (no bandit): prior is the score
        if provisional:
            cap = self.config.provisional_weight_cap
            return prior * cap, None, 0.0
        return prior, None, 0.0
