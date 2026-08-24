import os
import pytest
import numpy as np
from pathlib import Path

from knowledge_graph.models import Node, NodeType, Edge
from knowledge_graph.store import KnowledgeGraphStore
from knowledge_graph.embedding_mock import MockEmbeddingGenerator


@pytest.fixture
def mock_embedder():
    return MockEmbeddingGenerator(dim=8, seed=42)


@pytest.fixture
def store():
    return KnowledgeGraphStore()


def test_add_and_get_node(store, mock_embedder):
    node = Node(id="chall_1", type=NodeType.CHALLENGE, attributes={"title": "Test"})
    vec = mock_embedder.generate("test challenge")
    
    store.add_node(node, vec)
    
    retrieved = store.get_node("chall_1")
    assert retrieved is not None
    assert retrieved.id == "chall_1"
    assert retrieved.type == NodeType.CHALLENGE
    assert retrieved.attributes["title"] == "Test"
    
    retrieved_vec = store.get_embedding("chall_1")
    assert retrieved_vec is not None
    assert np.array_equal(retrieved_vec, vec)


def test_add_and_update_edge(store, mock_embedder):
    node1 = Node(id="chall_1", type=NodeType.CHALLENGE)
    node2 = Node(id="tech_1", type=NodeType.TECHNIQUE)
    
    store.add_node(node1, mock_embedder.generate("c1"))
    store.add_node(node2, mock_embedder.generate("t1"))
    
    edge = Edge(source="chall_1", target="tech_1", weight=1.0)
    store.add_edge(edge)
    
    retrieved_edge = store.get_edge("chall_1", "tech_1")
    assert retrieved_edge is not None
    assert retrieved_edge.weight == 1.0
    
    store.update_edge_weight("chall_1", "tech_1", 2.5)
    updated_edge = store.get_edge("chall_1", "tech_1")
    assert updated_edge.weight == 2.5


def test_missing_node_and_edge(store):
    assert store.get_node("missing") is None
    assert store.get_embedding("missing") is None
    assert store.get_edge("missing1", "missing2") is None
    
    with pytest.raises(KeyError):
        store.add_edge(Edge(source="missing1", target="missing2"))


def test_serialization(tmp_path, store, mock_embedder):
    # Add nodes and edges
    node1 = Node(id="n1", type=NodeType.CHALLENGE)
    node2 = Node(id="n2", type=NodeType.TECHNIQUE)
    
    vec1 = mock_embedder.generate("n1")
    vec2 = mock_embedder.generate("n2")
    
    store.add_node(node1, vec1)
    store.add_node(node2, vec2)
    store.add_edge(Edge(source="n1", target="n2", weight=1.5))
    
    save_path = tmp_path / "kg_test"
    store.save(save_path)
    
    # Verify files created
    assert (tmp_path / "kg_test.json").exists()
    assert (tmp_path / "kg_test.npy").exists()
    
    # Load into a new instance
    loaded_store = KnowledgeGraphStore.load(save_path)
    
    # Verify topology
    assert loaded_store.get_node("n1").type == NodeType.CHALLENGE
    assert loaded_store.get_edge("n1", "n2").weight == 1.5
    
    # Verify embeddings
    l_vec1 = loaded_store.get_embedding("n1")
    l_vec2 = loaded_store.get_embedding("n2")
    
    assert np.allclose(l_vec1, vec1)
    assert np.allclose(l_vec2, vec2)
    
    # Verify matrix shape after load
    ids, matrix = loaded_store.get_all_embeddings()
    assert len(ids) == 2
    assert matrix.shape == (2, 8)


def test_mock_embedder_determinism():
    embedder = MockEmbeddingGenerator(dim=16, seed=123)
    vec1 = embedder.generate("hello world")
    vec2 = embedder.generate("hello world")
    vec3 = embedder.generate("different")
    
    assert np.array_equal(vec1, vec2)
    assert not np.array_equal(vec1, vec3)
    assert np.isclose(np.linalg.norm(vec1), 1.0)
