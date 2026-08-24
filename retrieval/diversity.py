r"""
retrieval/diversity.py — Maximal Marginal Relevance (spec §7 step 4, C-3).

MMR (Carbonell & Goldstein, 1998) guarantees the candidate pool contains
*competing, non-duplicate* lessons rather than k near-copies of the top hit. This
matters for the bandit: if the pool is all paraphrases of one lesson, there is no
"better alternative" for the bandit to discover, and a single (possibly false)
lesson gets all the credit. Diversity is the partner to cited-only credit in
suppressing false-lesson lock-in (§6.2/§9.3 rationale; LongContextMeetsRAG
2410.05983 on distractor/passage trade-off).

MMR selection:
    next = argmax_{v in R\S} [ λ·rel(v) − (1−λ)·max_{u in S} sim(v, u) ]

rel(v) is the reranker score (relevance); sim(v,u) is cosine between node
embeddings. λ = lambda_mmr (1.0 = pure relevance, 0.0 = pure diversity).
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from knowledge_graph.models import Node


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mmr_select(
    ranked: List[Tuple[Node, float]],
    embedding_of: Callable[[str], np.ndarray],
    k: int,
    lambda_mmr: float = 0.5,
) -> List[Tuple[Node, float]]:
    """Select k diverse-yet-relevant items via MMR.

    Args:
        ranked: [(Node, relevance_score)] sorted by relevance (reranker output).
        embedding_of: node_id -> embedding vector (from the store).
        k: number to keep.
        lambda_mmr: relevance/diversity trade-off in [0, 1].

    Returns the selected [(Node, relevance_score)] in MMR-selection order.
    """
    if not ranked:
        return []
    k = min(k, len(ranked))

    # Precompute (relevance-normalized) and embeddings
    rels = np.array([r for _, r in ranked], dtype=np.float64)
    # min-max normalize relevance to [0,1] so it is commensurate with cosine
    if rels.max() > rels.min():
        rel_norm = (rels - rels.min()) / (rels.max() - rels.min())
    else:
        rel_norm = np.ones_like(rels)

    embs = [np.asarray(embedding_of(node.id), dtype=np.float64) for node, _ in ranked]

    selected_idx: List[int] = []
    remaining = set(range(len(ranked)))

    while len(selected_idx) < k and remaining:
        best_i = None
        best_val = -np.inf
        for i in sorted(remaining):
            if not selected_idx:
                diversity_penalty = 0.0
            else:
                diversity_penalty = max(_cosine(embs[i], embs[j]) for j in selected_idx)
            val = lambda_mmr * rel_norm[i] - (1.0 - lambda_mmr) * diversity_penalty
            if val > best_val:
                best_val = val
                best_i = i
        selected_idx.append(best_i)
        remaining.discard(best_i)

    return [ranked[i] for i in selected_idx]


def mean_pairwise_similarity(
    items: List[Tuple[Node, float]],
    embedding_of: Callable[[str], np.ndarray],
) -> float:
    """Diagnostic: mean pairwise cosine of a pool (lower = more diverse).

    Used by the C2 acceptance test (c): MMR must reduce this vs no-MMR.
    """
    embs = [np.asarray(embedding_of(node.id), dtype=np.float64) for node, _ in items]
    n = len(embs)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _cosine(embs[i], embs[j])
            count += 1
    return total / max(count, 1)
