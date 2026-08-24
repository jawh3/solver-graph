import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from .models import Edge, Node, NodeType

log = logging.getLogger(__name__)


class KnowledgeGraphStore:
    """
    Storage backend for the CTF Knowledge Graph.
    
    Uses NetworkX for graph topology and attributes, and NumPy for embedding storage.
    Optimized for determinism, reproducibility, and zero external service dependencies.
    """

    def __init__(self):
        self._graph = nx.DiGraph()
        
        # Mapping from node_id to row index in the embedding matrix
        self._node_to_idx: Dict[str, int] = {}
        
        # Dense matrix of embeddings for fast similarity search
        # Shape: (N, embedding_dim)
        self._embedding_matrix: Optional[np.ndarray] = None
        self._embedding_dim: int = 0
        
        # New embeddings added since last matrix rebuild
        self._unmerged_embeddings: List[Tuple[str, np.ndarray]] = []

    def _sync_embeddings(self) -> None:
        """Merge unmerged embeddings into the dense embedding matrix."""
        if not self._unmerged_embeddings:
            return

        new_vecs = []
        for node_id, vec in self._unmerged_embeddings:
            if self._embedding_dim == 0:
                self._embedding_dim = vec.shape[0]
            elif vec.shape[0] != self._embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch for node {node_id}. "
                    f"Expected {self._embedding_dim}, got {vec.shape[0]}"
                )
            new_vecs.append(vec)
            self._node_to_idx[node_id] = len(self._node_to_idx)

        new_matrix = np.vstack(new_vecs)
        
        if self._embedding_matrix is None:
            self._embedding_matrix = new_matrix
        else:
            self._embedding_matrix = np.vstack([self._embedding_matrix, new_matrix])
            
        self._unmerged_embeddings.clear()

    def add_node(self, node: Node, embedding: np.ndarray) -> None:
        """
        Add a new node to the knowledge graph with its vector embedding.

        If the node already exists, its attributes and embedding are UPDATED in
        place (no duplicate embedding row). This is required by the write-back
        (C7) and promotion (§9.4) loops, which re-add an existing node to persist
        a tier/attribute change.

        Args:
            node: The Node object to add.
            embedding: A 1D numpy array representing the node's vector.
        """
        # Ensure 1D array
        embedding = np.asarray(embedding, dtype=np.float32).flatten()

        # Always refresh graph attributes.
        self._graph.add_node(node.id, **node.to_dict())

        # --- embedding bookkeeping (avoid duplicate rows on re-add) ---
        # 1. pending (unmerged) row: replace it
        for i, (nid, _vec) in enumerate(self._unmerged_embeddings):
            if nid == node.id:
                self._unmerged_embeddings[i] = (node.id, embedding)
                return
        # 2. already merged into the matrix: update that row in place
        if node.id in self._node_to_idx and self._embedding_matrix is not None:
            idx = self._node_to_idx[node.id]
            if 0 <= idx < self._embedding_matrix.shape[0]:
                self._embedding_matrix[idx] = embedding
                return
        # 3. brand-new node: queue for the next sync
        self._unmerged_embeddings.append((node.id, embedding))

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by ID."""
        if node_id not in self._graph:
            return None
        attrs = self._graph.nodes[node_id].copy()
        attrs["id"] = node_id
        return Node.from_dict(attrs)

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Retrieve the embedding for a specific node."""
        # Check unmerged first
        for nid, vec in self._unmerged_embeddings:
            if nid == node_id:
                return vec
        
        if node_id not in self._node_to_idx or self._embedding_matrix is None:
            return None
            
        idx = self._node_to_idx[node_id]
        return self._embedding_matrix[idx]

    def add_edge(self, edge: Edge) -> None:
        """
        Add a directed edge between two nodes.
        
        Args:
            edge: The Edge object to add.
        """
        if edge.source not in self._graph:
            raise KeyError(f"Source node '{edge.source}' does not exist.")
        if edge.target not in self._graph:
            raise KeyError(f"Target node '{edge.target}' does not exist.")
            
        self._graph.add_edge(
            edge.source, 
            edge.target, 
            weight=edge.weight, 
            **edge.attributes
        )

    def update_edge_weight(self, source_id: str, target_id: str, new_weight: float) -> None:
        """Update the contextual bandit weight on a specific edge."""
        if not self._graph.has_edge(source_id, target_id):
            raise KeyError(f"Edge from '{source_id}' to '{target_id}' does not exist.")
            
        self._graph[source_id][target_id]['weight'] = new_weight

    def get_edge(self, source_id: str, target_id: str) -> Optional[Edge]:
        """Retrieve an edge between two nodes."""
        if not self._graph.has_edge(source_id, target_id):
            return None
            
        edge_data = self._graph[source_id][target_id].copy()
        weight = edge_data.pop('weight', 0.0)
        return Edge(source=source_id, target=target_id, weight=weight, attributes=edge_data)

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        Get all nodes and their dense embedding matrix for vector retrieval.
        
        Returns:
            Tuple containing:
            - List of node IDs aligned with the matrix rows
            - 2D numpy array of embeddings (N, embedding_dim)
        """
        self._sync_embeddings()
        
        if self._embedding_matrix is None:
            return [], np.array([])
            
        # Reconstruct ordered node IDs based on idx
        ordered_ids = ["" for _ in range(len(self._node_to_idx))]
        for node_id, idx in self._node_to_idx.items():
            ordered_ids[idx] = node_id
            
        return ordered_ids, self._embedding_matrix

    def save(self, base_path: str | Path) -> None:
        """
        Serialize the knowledge graph to disk.
        
        Produces two files:
        - {base_path}.json: Graph topology and attributes
        - {base_path}.npy: Dense embedding matrix
        """
        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._sync_embeddings()
        
        # Save topology
        graph_data = json_graph.node_link_data(self._graph, edges="edges")
        
        # We need to save the node_to_idx mapping so embeddings align on load
        payload = {
            "graph": graph_data,
            "node_to_idx": self._node_to_idx
        }
        
        json_path = base_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(payload, f, indent=2)
            
        # Save embeddings
        npy_path = base_path.with_suffix('.npy')
        if self._embedding_matrix is not None:
            np.save(npy_path, self._embedding_matrix)
        else:
            # Save empty array to ensure file exists
            np.save(npy_path, np.array([]))
            
        log.info(f"Knowledge graph saved to {json_path} and {npy_path}")

    @classmethod
    def load(cls, base_path: str | Path) -> "KnowledgeGraphStore":
        """
        Deserialize the knowledge graph from disk.
        """
        base_path = Path(base_path)
        json_path = base_path.with_suffix('.json')
        npy_path = base_path.with_suffix('.npy')
        
        if not json_path.exists():
            raise FileNotFoundError(f"Graph file not found: {json_path}")
            
        store = cls()
        
        # Load topology
        with open(json_path, 'r') as f:
            payload = json.load(f)
            
        store._graph = json_graph.node_link_graph(payload["graph"], edges="edges")
        store._node_to_idx = payload.get("node_to_idx", {})
        
        # Load embeddings if available
        if npy_path.exists():
            matrix = np.load(npy_path)
            if matrix.size > 0:
                store._embedding_matrix = matrix
                store._embedding_dim = matrix.shape[1]
                
        log.info(f"Loaded knowledge graph with {len(store._graph)} nodes from {base_path}")
        return store
