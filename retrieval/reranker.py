"""
retrieval/reranker.py - cross-encoder reranker (spec §7 step 3, C-3).

Precision is fixed *downstream* of the high-recall pool, NOT by growing N - a
larger N admits more lures (long-context RAG degradation; LongContextMeetsRAG
2410.05983, DynamicContextSelection 2512.14313). The reranker is a
CROSS-ENCODER, not an LLM reranker: cross-encoders are cheaper and at least as
good for this (CrossEncoderVsLLM 2403.10407; BUILD_CHECKLIST C2/D-05).

Mock + real duality (every vLLM/GPU-touching component):
  - MockReranker: deterministic, GPU-free. Scores by lexical token-overlap
    (Jaccard) between the query text and the node text, blended with the incoming
    cosine. This is an *honest* mock: it actually re-orders on a different signal
    than the bi-encoder cosine, so MMR/rerank effects are observable in tests
    without a model. (Acceptance test (c) needs a real re-ordering signal.)
  - CrossEncoderReranker: real sentence-transformers CrossEncoder
    (e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"), lazy-imported so the package
    is only needed in coordinated GPU runs.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Tuple

from knowledge_graph.models import Node


def _node_text(node: Node) -> str:
    attrs = node.attributes
    for key in ("text", "canonical_name", "description", "title"):
        if attrs.get(key):
            return str(attrs[key])
    return node.id


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


class Reranker(ABC):
    """Abstract reranker: (query_text, [(Node, cosine)]) -> [(Node, rerank_score)]."""

    @abstractmethod
    def rerank(self, query_text: str, candidates: List[Tuple[Node, float]],
               keep: int) -> List[Tuple[Node, float]]:
        ...


class MockReranker(Reranker):
    """Deterministic, GPU-free reranker (lexical overlap + cosine blend)."""

    def __init__(self, overlap_weight: float = 0.7):
        self.overlap_weight = overlap_weight

    def rerank(self, query_text: str, candidates: List[Tuple[Node, float]],
               keep: int) -> List[Tuple[Node, float]]:
        q = _tokens(query_text)
        scored: List[Tuple[Node, float]] = []
        for node, cosine in candidates:
            nt = _tokens(_node_text(node))
            if q or nt:
                jacc = len(q & nt) / max(len(q | nt), 1)
            else:
                jacc = 0.0
            score = self.overlap_weight * jacc + (1 - self.overlap_weight) * cosine
            scored.append((node, float(score)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:keep]


class CrossEncoderReranker(Reranker):
    """Real cross-encoder reranker (sentence-transformers). GPU path."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str | None = None):
        self.model_name = model_name
        self._device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy import (GPU dep)
            self._model = CrossEncoder(self.model_name, device=self._device)
        return self._model

    def rerank(self, query_text: str, candidates: List[Tuple[Node, float]],
               keep: int) -> List[Tuple[Node, float]]:
        if not candidates:
            return []
        model = self._ensure_model()
        pairs = [(query_text, _node_text(node)) for node, _ in candidates]
        scores = model.predict(pairs)
        scored = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:keep]


def build_reranker(mock: bool = True, **kwargs) -> Reranker:
    """Factory honoring the mock+real duality (mock default for dev/tests)."""
    return MockReranker(**kwargs) if mock else CrossEncoderReranker(**kwargs)
