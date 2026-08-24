# knowledge_graph

Typed-node knowledge graph: construction, embedding, and management.

## Purpose

Manages the knowledge graph that underpins the retrieval system. Handles:

- **Seed graph construction** — LLM decomposition of writeups into typed nodes (challenges, binaries, methodologies, tools, categories)
- **Edge formation** — Links nodes via shared LLM-generated canonical names
- **Embedding** — Vector representations for all nodes via an embedding model
- **Write-back** — Runtime graph extension from solver trajectories (lifelong learning)

## Node Types

| Type | Description |
|---|---|
| Challenge | Text of challenge descriptions |
| Binary/Artifact | File-level artifacts |
| Methodology/Technique | Approaches used to solve |
| Tool | Specific software invoked |
| Category | Challenge type (crypto, pwn, web, reversing) |

## Provenance

- Architecture per a superseded design note (not included in this release) §Knowledge Graph Construction
- Storage backend originally prototyped in `the superseded baseline solver (not in this release)`

## Status
**Implemented:** Storage backend and CRUD operations are fully implemented.

## Key Files
- `models.py`: Data classes (`Node`, `Edge`, `NodeType`) representing graph entities.
- `store.py`: `KnowledgeGraphStore` interface wrapping NetworkX and NumPy.
- `embedding_mock.py`: Deterministic mock embedding generator for local testing without vLLM.
- `experiment_harness.py`: Scalability sweep script.
- `SWEEPS.md`: Pre-registered experiment sweeps and success criteria.

## Design Decisions
- **Storage Backend:** We use `networkx` for graph topology (serialized to JSON) and `numpy` for embedding vectors (serialized to NPY). This ensures zero external dependencies and determinism. See `OPEN_DECISIONS.md` for details.
- **Node Validation:** Standard Python `@dataclass` is used instead of Pydantic to minimize dependencies, with `NodeType` enforced via enums.

## Experiments

### Serialization Scalability Sweep
**What it measures:** Wall-clock time to build, save, and load graphs of $N \in \{100, 1000, 10000\}$ nodes, and resulting disk footprint.
**Invocation:**
```bash
conda run --no-capture-output -n ctf_solver python knowledge_graph/experiment_harness.py
```
To run the extreme scale test ($N \in \{50000, 100000, 250000\}$):
```bash
conda run --no-capture-output -n ctf_solver python knowledge_graph/experiment_harness.py --config knowledge_graph/experiments/extreme_sweep_config.json
```
**Expected Output:** Structured JSON in `evaluation/eval_results/kg_storage_sweep_<timestamp>.json` containing exact timings and sizes. See `SWEEPS.md` for pre-registered predictions.
