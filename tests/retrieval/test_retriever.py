import numpy as np
import pytest

from knowledge_graph.models import Node, NodeType
from knowledge_graph.store import KnowledgeGraphStore
from retrieval.embedder import MockEmbedder
from retrieval.retriever import ContextRetriever


@pytest.fixture
def store():
    s = KnowledgeGraphStore()
    yield s


@pytest.fixture
def embedder():
    return MockEmbedder(dim=4, seed=42)


@pytest.fixture
def populated_store(store, embedder):
    # Add some nodes
    nodes = [
        ("node1", "crypto rsa"),
        ("node2", "pwn buffer overflow"),
        ("node3", "crypto aes cbc"),
        ("node4", "web sql injection"),
        ("node5", "crypto rsa wiener attack")
    ]
    
    for node_id, text in nodes:
        n = Node(id=node_id, type=NodeType.CHALLENGE, attributes={"text": text})
        # For tests, we'll embed the exact text to make similarities predictable
        vec = embedder.embed(text)
        store.add_node(n, vec)
        
    return store


def test_top_n_candidates(populated_store, embedder):
    retriever = ContextRetriever(populated_store, embedder)
    
    # Query matching node1 strongly
    query = "crypto rsa"
    
    candidates = retriever.get_top_n_candidates(query, n=3)
    
    assert len(candidates) == 3
    # The first result should be node1 since the text matches exactly
    # (MockEmbedder uses a hash of the text, so exact string match gives exact vector)
    assert candidates[0][0].id == "node1"
    assert candidates[0][1] > 0.99  # Cosine sim ~ 1.0


def test_sample_top_m_greedy(populated_store, embedder):
    retriever = ContextRetriever(populated_store, embedder)
    query = "crypto rsa"
    candidates = retriever.get_top_n_candidates(query, n=5)
    
    # Greedy (exploration_rate = 0.0) should return exactly the top 2 candidates
    sampled = retriever.sample_top_m(candidates, m=2, exploration_rate=0.0)
    
    assert len(sampled) == 2
    assert sampled[0].id == candidates[0][0].id
    assert sampled[1].id == candidates[1][0].id


def test_sample_top_m_stochastic(populated_store, embedder):
    retriever = ContextRetriever(populated_store, embedder)
    query = "crypto rsa"
    candidates = retriever.get_top_n_candidates(query, n=5)
    
    # Stochastic with high temperature
    # We run it multiple times to ensure it's not strictly deterministic
    sampled_first_nodes = set()
    for _ in range(100):
        sampled = retriever.sample_top_m(candidates, m=1, exploration_rate=100.0)
        sampled_first_nodes.add(sampled[0].id)
        
    # With T=100 and N=5, uniform distribution means we should see multiple different nodes sampled
    assert len(sampled_first_nodes) > 1
