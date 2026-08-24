"""
retrieval/context_resolution.py — map a query to a bandit context node (spec §6, §7 step 5).

C-1, the most important change: the bandit context is a COARSE node (Category or
k-means Cluster over Challenge embeddings), NOT a per-challenge analog. Per the
§6.1 budget calc, per-challenge contexts get <1 update/edge and cannot learn —
the most likely cause of the prior non-significant result. Category (~6) and
Cluster (k≈20–50) pass the budget.

Three resolvers (the `context_resolution` ablation):
  - CategoryResolver: context node = the challenge's CTF category.
  - ClusterResolver: context node = nearest k-means centroid over Challenge
    embeddings (default, k=n_clusters). Deterministic k-means++ seeding.
  - (parametric LinUCB has no discrete context node — handled in retriever.)

`bootstrap` node is returned when the graph is empty / clusters unfit, giving a
uniform prior so the very first runs still have a valid context_id (§7 step 5).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import ContextResolution, SolverConfig
from knowledge_graph.models import NodeType
from knowledge_graph.store import KnowledgeGraphStore

BOOTSTRAP_CONTEXT = "ctx::bootstrap"


# ---------------------------------------------------------------------------
# Deterministic k-means (numpy; no sklearn dependency, keeps repo zero-dep)
# ---------------------------------------------------------------------------

def _kmeans(X: np.ndarray, k: int, seed: int, iters: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm with k-means++ init. Returns (centroids, labels).

    Deterministic given seed. Operates on L2-normalized rows so Euclidean
    nearest-centroid ≈ cosine nearest-centroid.
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    k = min(k, n)

    # k-means++ seeding
    centers = [int(rng.randint(n))]
    d2 = np.sum((X - X[centers[0]]) ** 2, axis=1)
    for _ in range(1, k):
        probs = d2 / max(d2.sum(), 1e-12)
        nxt = int(rng.choice(n, p=probs))
        centers.append(nxt)
        d2 = np.minimum(d2, np.sum((X - X[nxt]) ** 2, axis=1))
    centroids = X[centers].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        # assign
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        # update
        for c in range(k):
            members = X[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)
    return centroids, labels


class ContextResolver:
    """Resolves a query (embedding + optional category) to a context node id."""

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self.centroids: Optional[np.ndarray] = None
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, store: KnowledgeGraphStore) -> "ContextResolver":
        """Fit clusters over Challenge-node embeddings (cluster mode only)."""
        if self.config.context_resolution is not ContextResolution.CLUSTER:
            self._fitted = True
            return self

        node_ids, matrix = store.get_all_embeddings()
        if len(node_ids) == 0:
            self._fitted = False
            return self

        # restrict to Challenge nodes when present; else use all (bootstrap-friendly)
        chal_rows = []
        for i, nid in enumerate(node_ids):
            node = store.get_node(nid)
            if node is not None and node.type is NodeType.CHALLENGE:
                chal_rows.append(i)
        rows = chal_rows if chal_rows else list(range(len(node_ids)))
        X = matrix[rows]
        # normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms

        self.centroids, _ = _kmeans(Xn, self.config.n_clusters, seed=self.config.seed)
        self._fitted = True
        return self

    def fit_on_embeddings(self, embeddings: np.ndarray) -> "ContextResolver":
        """Fit cluster centroids directly on a matrix of QUERY (challenge) embeddings.

        Used by the campaign so the bandit context clusters the *query space* (each
        decoding method clusters separately via its artifact diagnostics), without
        adding challenge nodes to the retrievable store. No-op unless CLUSTER mode.
        """
        if self.config.context_resolution is not ContextResolution.CLUSTER:
            self._fitted = True
            return self
        X = np.asarray(embeddings, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            self._fitted = False
            return self
        norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms == 0] = 1.0
        Xn = X / norms
        k = min(self.config.n_clusters, X.shape[0])
        self.centroids, _ = _kmeans(Xn, k, seed=self.config.seed)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def resolve(self, query_emb: np.ndarray, category: Optional[str] = None) -> str:
        """Return the context node id for this query."""
        mode = self.config.context_resolution

        if mode is ContextResolution.CATEGORY:
            if category and category.strip() and category.lower() != "unknown":
                return f"ctx::cat::{category.strip().lower()}"
            return BOOTSTRAP_CONTEXT

        if mode is ContextResolution.PARAMETRIC:
            # LinUCB has no discrete context; a single shared context id is used
            # so the tabular plumbing still has a key (the parametric model uses
            # the continuous query embedding directly).
            return "ctx::parametric"

        # CLUSTER (default)
        if not self._fitted or self.centroids is None or len(self.centroids) == 0:
            return BOOTSTRAP_CONTEXT
        q = np.asarray(query_emb, dtype=np.float64).ravel()
        nq = np.linalg.norm(q)
        if nq == 0:
            return BOOTSTRAP_CONTEXT
        qn = q / nq
        dists = np.linalg.norm(self.centroids - qn[None, :], axis=1)
        cluster_id = int(np.argmin(dists))
        return f"ctx::cluster::{cluster_id}"

    # ------------------------------------------------------------------
    def num_context_nodes(self) -> int:
        """For the §6.1 sample-efficiency budget calc."""
        mode = self.config.context_resolution
        if mode is ContextResolution.CATEGORY:
            return 6  # crypto/pwn/web/rev/forensics/misc
        if mode is ContextResolution.PARAMETRIC:
            return 1
        return self.config.n_clusters
