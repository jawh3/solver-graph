# Knowledge Graph Storage Sweeps

## Sweep: Serialization Scalability (2026-05-29)

**What this measures:** The time it takes to serialize and deserialize the Knowledge Graph (NetworkX topology to JSON, NumPy embeddings to NPY) and the resulting disk footprint as the graph grows.

**Why:** The `baseline-solver` used a simple JSON file. We need to ensure that the new split JSON/NPY schema does not introduce prohibitive I/O bottlenecks when the graph scales to a realistic size for a lifelong-learning agent.

**Pre-specified conditions:**
- Graph sizes ($N$ nodes): 100, 1,000, 10,000.
- Edges: Roughly $3 \times N$ random edges per graph.
- Embedding dimension: 1536 (typical vLLM dense vector size).

**Predictions / Success criteria:**
- Serialization/Deserialization of $N=10,000$ should take less than 1.0 second on a standard SSD.
- Storage size for $N=10,000$ with $D=1536$ should be around ~60MB for the embeddings and ~2MB for the topology.
- If $N=10,000$ load times exceed 2 seconds, we must rethink the storage backend and consider an embedded database like SQLite or Chroma.

**Command to run:**
```bash
conda run --no-capture-output -n ctf_solver python knowledge_graph/experiment_harness.py
```
