"""
evaluation/synthetic.py - a controllable synthetic CTF world for GPU-free validation.

Why this exists: the existing `MockEmbedder` is hash-based, so semantically related
texts get *independent* random vectors - retrieval similarity is meaningless. To
exercise the WHOLE pipeline (retrieval actually surfacing the right lesson, the
bandit learning to prefer it, write-back growing the graph) without a GPU, we need
(a) an embedder where content-related items are genuinely close, and (b) a world
whose ground truth (which lesson helps which challenge, which lessons are
false/poison) is known so suppression can be *measured*.

This module provides both. It is test/eval infrastructure, NOT part of the real
solve path - the real path uses VLLMEmbedder + Qwen. Everything here is seeded and
deterministic so paired/common-random-seed evaluation (§14.4) is exact.

Ground-truth labels embedded in the world (so §14.3 false-lesson metrics have a
key): each Technique/Tool node is tagged `_truth ∈ {helpful, poison, distractor}`
per category, and each challenge's `_world.solution_tools` names the helpful set.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from knowledge_graph.models import Node, NodeType
from knowledge_graph.store import KnowledgeGraphStore
from retrieval.embedder import Embedder

CATEGORIES = ["crypto", "pwn", "web", "rev", "forensics", "misc"]

# A small tool vocabulary per category; the FIRST few are the genuine solvers.
CATEGORY_TOOLS: Dict[str, List[str]] = {
    "crypto":    ["sage", "gmpy2", "z3", "factordb", "hashcat"],
    "pwn":       ["ropper", "pwntools", "gdb", "one_gadget", "checksec"],
    "web":       ["curl", "sqlmap", "burp", "ffuf", "nikto"],
    "rev":       ["r2", "ghidra", "objdump", "angr", "ltrace"],
    "forensics": ["binwalk", "volatility", "exiftool", "wireshark", "foremost"],
    "misc":      ["base64", "strings", "cyberchef", "stegsolve", "zsteg"],
}


def _basis(token: str, dim: int, seed: int) -> np.ndarray:
    h = int(hashlib.sha1(f"{seed}:{token}".encode()).hexdigest(), 16) % (2**32 - 1)
    rng = np.random.RandomState(h)
    v = rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


class FeatureMockEmbedder(Embedder):
    """Feature-additive embedder: recognized tokens map to fixed basis vectors.

    embed(text) = normalize( Σ basis(token) for known tokens + small hashed noise ).
    Content-related items (same category / same tool) end up close in cosine,
    so the retrieval pipeline behaves like a real semantic retriever for testing.
    """

    def __init__(self, dim: int = 64, seed: int = 7, noise: float = 0.15):
        self.dim = dim
        self.seed = seed
        self.noise = noise
        vocab = list(CATEGORIES) + [t for ts in CATEGORY_TOOLS.values() for t in ts]
        self._basis = {tok: _basis(tok, dim, seed) for tok in vocab}

    def embed(self, text: str) -> np.ndarray:
        toks = text.lower().replace("/", " ").replace("_", " ").split()
        vec = np.zeros(self.dim, dtype=np.float32)
        hit = False
        for t in toks:
            if t in self._basis:
                vec += self._basis[t]
                hit = True
        # small text-hash noise so non-feature words still perturb (and unknown
        # texts are not all identical-zero)
        h = int(hashlib.sha1(text.encode()).hexdigest(), 16) % (2**32 - 1)
        vec += self.noise * np.random.RandomState(h).randn(self.dim).astype(np.float32)
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        return vec


@dataclass
class SyntheticWorld:
    store: KnowledgeGraphStore
    embedder: FeatureMockEmbedder
    challenges: List[Dict]
    helpful_node_ids: Dict[str, List[str]]   # category -> genuine solver node ids
    poison_node_ids: List[str]
    distractor_node_ids: List[str]


def build_world(
    n_challenges: int = 60,
    dim: int = 64,
    seed: int = 7,
    poison_per_category: int = 1,
    dataset_tag: str = "seed",
    contamination_flag: bool = True,
) -> SyntheticWorld:
    """Build a deterministic synthetic graph + challenge set with known ground truth."""
    rng = np.random.RandomState(seed)
    embedder = FeatureMockEmbedder(dim=dim, seed=seed)
    store = KnowledgeGraphStore()

    helpful_ids: Dict[str, List[str]] = {c: [] for c in CATEGORIES}
    poison_ids: List[str] = []
    distractor_ids: List[str] = []

    # --- seed lesson nodes ---
    for cat, tools in CATEGORY_TOOLS.items():
        for i, tool in enumerate(tools):
            truth = "helpful" if i < 2 else "distractor"   # first 2 tools genuinely solve
            text = f"use {tool} for {cat} challenges"
            node = Node(
                id=f"tool::{cat}::{tool}", type=NodeType.TOOL,
                attributes={"canonical_name": tool, "invocation_signature": tool,
                            "text": text, "category": cat, "tier": "trusted",
                            "source": "seed_pico", "_truth": truth,
                            "applicability": {"categories": [cat], "preconditions": [],
                                              "artifacts_required": [], "confidence": 0.8}},
            )
            store.add_node(node, embedder.embed(text))
            if truth == "helpful":
                helpful_ids[cat].append(node.id)
            else:
                distractor_ids.append(node.id)

        # --- poison nodes: claim a solver tool but encode a FALSE premise (long route) ---
        for p in range(poison_per_category):
            tool = tools[0]   # masquerades as the genuine solver
            text = f"WRONG premise about {tool} for {cat} (longer route)"
            node = Node(
                id=f"poison::{cat}::{p}", type=NodeType.TECHNIQUE,
                attributes={"canonical_name": f"{tool}_myth", "invocation_signature": tool,
                            "text": text, "category": cat, "tier": "trusted",
                            "source": "seed_github", "_truth": "poison", "_poison": True,
                            "applicability": {"categories": [cat], "preconditions": [],
                                              "artifacts_required": [], "confidence": 0.6}},
            )
            store.add_node(node, embedder.embed(text))
            poison_ids.append(node.id)

    # --- challenges ---
    challenges: List[Dict] = []
    point_tiers = [50, 100, 150, 200, 300, 400, 500]
    for i in range(n_challenges):
        cat = CATEGORIES[i % len(CATEGORIES)]
        solver_tools = [CATEGORY_TOOLS[cat][0], CATEGORY_TOOLS[cat][1]]  # the genuine solvers
        points = int(point_tiers[i % len(point_tiers)])
        desc = f"A {cat} challenge requiring {solver_tools[0]} and {solver_tools[1]}."
        challenges.append({
            "name": f"{cat}_chal_{i}",
            "category": cat,
            "points": points,
            "description": desc,
            "dataset": dataset_tag,
            "contamination_flag": contamination_flag,
            "_world": {
                "solution_tools": solver_tools,
                "steps_base": 12,
                "p_solve_base": 0.35,
                "step_floor": 2,
            },
        })

    return SyntheticWorld(store=store, embedder=embedder, challenges=challenges,
                          helpful_node_ids=helpful_ids, poison_node_ids=poison_ids,
                          distractor_node_ids=distractor_ids)
