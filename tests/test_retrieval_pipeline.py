"""Tests for the full C2 retrieval pipeline (spec §7) and its acceptance criteria."""

import numpy as np
import pytest

from config import SolverConfig, ContextResolution
from knowledge_graph.models import Node, NodeType
from knowledge_graph.store import KnowledgeGraphStore
from retrieval.embedder import MockEmbedder
from retrieval.reranker import MockReranker
from retrieval.context_resolution import ContextResolver
from retrieval.retriever import ContextRetriever
from retrieval.diversity import mmr_select, mean_pairwise_similarity
from retrieval.sampling import sample_without_replacement, ordered_set_logprob
from bandit import TabularBandit


def _build_store(n=60, dim=64, seed=42):
    store = KnowledgeGraphStore()
    emb = MockEmbedder(dim=dim, seed=seed)
    cats = ["crypto", "pwn", "web", "rev", "forensics", "misc"]
    for i in range(n):
        cat = cats[i % len(cats)]
        text = f"{cat} technique number {i} using tool_{i % 7}"
        node = Node(id=f"tech_{i}", type=NodeType.TECHNIQUE,
                    attributes={"text": text, "category": cat,
                                "canonical_name": f"tool_{i % 7}"})
        store.add_node(node, emb.embed(text))
    return store, emb


def _retriever(store, emb, cfg, bandit=None):
    cr = ContextResolver(cfg).fit(store)
    return ContextRetriever(store, emb, config=cfg, reranker=MockReranker(),
                            context_resolver=cr, bandit=bandit)


def test_pipeline_runs_and_returns_M_nodes():
    store, emb = _build_store()
    cfg = SolverConfig().with_overrides(N=40, R=20, M=8, n_clusters=8, embedding_dim=64)
    r = _retriever(store, emb, cfg)
    res = r.retrieve("crypto technique number 0 using tool_0", category="crypto")
    assert len(res.sampled_nodes) == 8
    assert len(res.per_draw_propensity) == 8
    assert res.context_node_id.startswith("ctx::")
    # candidate pool carries score components for OPE
    assert all("cosine" in c and "score" in c and "prior" in c for c in res.candidate_pool)


def test_retrieval_determinism_under_fixed_seed():
    store, emb = _build_store()
    cfg = SolverConfig().with_overrides(N=40, R=20, M=6, n_clusters=8, embedding_dim=64)
    r = _retriever(store, emb, cfg)
    res1 = r.retrieve("web technique 3", category="web",
                      rng=np.random.RandomState(123))
    res2 = r.retrieve("web technique 3", category="web",
                      rng=np.random.RandomState(123))
    assert res1.sampled_ids == res2.sampled_ids
    assert abs(res1.set_logprob - res2.set_logprob) < 1e-12


def test_logged_propensities_reproduce_realized_sample():
    # Acceptance (b): re-simulate from logged scores+seed => same set + same logprob.
    store, emb = _build_store()
    cfg = SolverConfig().with_overrides(N=40, R=24, M=8, n_clusters=8, embedding_dim=64)
    r = _retriever(store, emb, cfg)
    res = r.retrieve("rev technique 5", category="rev", rng=np.random.RandomState(7))

    # reconstruct the per-candidate scores from the logged pool, in pool order
    scores = [c["score"] for c in res.candidate_pool]
    id_order = [c["node_id"] for c in res.candidate_pool]
    # realized draw order as indices into the logged pool
    realized_idx = [id_order.index(p["node_id"]) for p in res.per_draw_propensity]
    logp = ordered_set_logprob(scores, realized_idx, tau=res.temperature)
    assert abs(logp - res.set_logprob) < 1e-9


def test_mmr_reduces_pairwise_similarity():
    # Acceptance (c): MMR pool is more diverse than the pure top-k.
    store, emb = _build_store(n=80, dim=64)
    cfg = SolverConfig().with_overrides(embedding_dim=64)
    r = _retriever(store, emb, cfg)
    pool_n = r.get_top_n_candidates("crypto technique number 0 using tool_0", n=40)
    reranked = MockReranker().rerank("crypto technique number 0 using tool_0", pool_n, keep=30)
    top_k = reranked[:10]
    mmr_pool = mmr_select(reranked, r._embedding_of, k=10, lambda_mmr=0.3)
    sim_topk = mean_pairwise_similarity(top_k, r._embedding_of)
    sim_mmr = mean_pairwise_similarity(mmr_pool, r._embedding_of)
    assert sim_mmr <= sim_topk + 1e-9


def test_cold_node_sampled_by_similarity_not_uniform():
    # Acceptance (d): with no bandit Q, sampling order tracks similarity.
    store, emb = _build_store()
    cfg = SolverConfig().with_overrides(N=40, R=20, M=10, tau=0.2, n_clusters=8, embedding_dim=64)
    r = _retriever(store, emb, cfg)  # no bandit -> similarity-only
    # low temperature -> near-greedy; the highest-cosine candidate should be drawn early
    res = r.retrieve("crypto technique number 0 using tool_0", category="crypto",
                     rng=np.random.RandomState(1))
    pool_sorted = sorted(res.candidate_pool, key=lambda c: c["cosine"], reverse=True)
    top_cos_id = pool_sorted[0]["node_id"]
    assert top_cos_id in res.sampled_ids[:3]


def test_bandit_changes_sampling_vs_similarity_only():
    store, emb = _build_store()
    cfg = SolverConfig().with_overrides(N=40, R=20, M=8, kappa=2.0, n_clusters=8, embedding_dim=64)
    bandit = TabularBandit(cfg)
    r = _retriever(store, emb, cfg, bandit=bandit)
    # warm up a specific node strongly in its context
    res0 = r.retrieve("pwn technique 1 using tool_1", category="pwn", rng=np.random.RandomState(0))
    ctx = res0.context_node_id
    target = res0.candidate_pool[-1]["node_id"]  # a low-cosine node
    for _ in range(40):
        bandit.update(ctx, target, reward=1.5, similarity=0.3, success=True)
    res1 = r.retrieve("pwn technique 1 using tool_1", category="pwn", rng=np.random.RandomState(0))
    # the warmed node should now score higher than its prior-only score
    c0 = next(c for c in res0.candidate_pool if c["node_id"] == target)
    c1 = next(c for c in res1.candidate_pool if c["node_id"] == target)
    assert c1["score"] > c0["score"]


def test_sampling_without_replacement_is_unique():
    rng = np.random.RandomState(3)
    order, probs, logp = sample_without_replacement([1.0, 0.5, 0.2, 0.1, 2.0], m=3, tau=1.0, rng=rng)
    assert len(order) == len(set(order)) == 3
    assert all(0 < p <= 1 for p in probs)
    assert logp <= 0
