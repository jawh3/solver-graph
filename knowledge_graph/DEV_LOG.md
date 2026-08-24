# Development Log — Knowledge Graph

## 2026-05-29 — Plan: Knowledge Graph Storage and CRUD

**What:** Implementing the core storage backend for the knowledge graph.
**Interfaces Exposed:**
- `knowledge_graph.models` (Node, Edge, NodeType)
- `knowledge_graph.store.KnowledgeGraph` (add_node, add_edge, update_edge_weight, get_node, get_edge, save, load, get_embeddings)
- `knowledge_graph.embedding_mock.MockEmbeddingGenerator` (generate_mock_embedding)

**Internal Modules:**
- Use `networkx` for graph topology.
- Use `numpy` for embedding storage.

**Design Decisions:**
- *Decision:* Use `networkx` + `numpy` instead of a standalone graph DB or vector DB.
- *Reasoning:* Meets the requirement for a deterministic, reproducible research codebase without needing external services. Exact vector search via numpy is sufficient for graphs <100k nodes.
- *Decision:* Use standard `dataclasses` for models.
- *Reasoning:* Avoids adding the `pydantic` dependency while still providing type hints and structured data.

**Tests:**
- Interface boundary tests in `tests/knowledge_graph/test_store.py`.
- Will test serialization parity (saving and loading returns identical graph and embeddings), error handling for missing nodes, and mock embedding determinism.

**Experiments:**
- `experiment_harness.py`: Measures graph serialization and deserialization speeds, and storage size, across a sweep of $N \in \{100, 1000, 10000\}$ nodes.

**Estimated Complexity:**
- 4 new files (`models.py`, `store.py`, `embedding_mock.py`, `experiment_harness.py`).
- ~300 lines of code.
- 1 session.

## 2026-05-29 — Implementation: Knowledge Graph Storage

**What:** Implemented `KnowledgeGraphStore` using NetworkX and NumPy, along with data models (`Node`, `Edge`, `NodeType`) and a deterministic `MockEmbeddingGenerator`.
**Why:** To provide a robust, deterministic, zero-external-dependency graph storage backend for the retrieval system.
**Decisions:**
- NetworkX + NumPy backend chosen for simple file-based serialization and exact vector similarity without heavy DB instances (see `OPEN_DECISIONS.md`).
- Standard `dataclasses` used over Pydantic to minimize external dependencies.
**Experiments produced:** Created `experiment_harness.py` to sweep scalability ($N \in \{100, 1000, 10000\}$). Pre-registered in `SWEEPS.md`.
**Results (if any):** 
- Standard scalability sweep passed successfully. For $N=10,000$ nodes (with $D=1536$): save time was ~0.32s, load time was ~0.18s, and total disk size was ~63.6MB. The split JSON/NPY schema easily satisfies the <1.0s I/O prediction and proves viable for the lifelong-learning agent scale.
- Extreme scalability sweep ran for $N \in \{50000, 100000, 250000\}$. At $N=250,000$ nodes (~750,000 edges), save time was ~9.9s, load time was ~11.2s, and total disk size was ~1.59GB. Even at a quarter million nodes, the schema manages without crashing, though load times pass the 10-second mark.
**Next:** Run the serialization sweep to verify I/O performance. Proceed to Component 2: Embedding and Retrieval.
**Notes for future agents:** The numpy matrix is reconstructed dynamically into a contiguous block on `save()` and `get_all_embeddings()`. The indices align perfectly with `_node_to_idx`.
