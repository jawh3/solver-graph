"""
retrieval/retriever.py — the §7 solve-time retrieval pipeline (Component 2).

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
carrying everything `RunRecord` (§12) and OPE (§13) need — crucially the
per-draw propensities and per-candidate score COMPONENTS so a counterfactual
policy (different κ/τ/M) can be re-scored offline without a GPU.

Design citations:
  - sim-prior pseudo-count blend (C-2): PURPLE 2601.12078; cold start by
    similarity not uniform — JumpStartFalseStart 2604.02527.
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
from .sampling import sample_without_replacement

log = logging.getLogger(__name__)


def _node_text(node: Node) -> str:
    a = node.attributes
    for key in ("text", "canonical_name", "description", "title"):
        if a.get(key):
            return str(a[key])
    return node.id


def cosine_to_reward_scale(cosine: float) -> float:
    """normalize_to_reward_scale(cosine) — map similarity into the reward range.

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
    ) -> RetrievalResult:
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
        for node, _rel in pool_d:
            cos = cosine_by_id.get(node.id, 0.0)
            prior = cosine_to_reward_scale(cos)
            score, q, n = self._sampling_score(context_id, node, prior, query_emb,
                                               use_bandit=use_bandit)
            scored_pool.append((node, score, prior, cos))

        scores = [s for _, s, _, _ in scored_pool]
        m = min(cfg.M, len(scored_pool))
        order, per_draw_p, set_logprob = sample_without_replacement(
            scores, m=m, tau=temperature, rng=rng
        )
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
            candidate_pool.append({
                "node_id": node.id,
                "cosine": cos,
                "rerank_score": rerank_by_id.get(node.id, 0.0),
                "mmr_kept": node.id in mmr_kept_ids,
                "prior": prior,
                "Q": (cell.Q if cell else None),
                "n": (cell.n if cell else 0.0),
                "score": score,
                "tier": node.attributes.get("tier", "trusted"),
            })

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
