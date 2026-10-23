# retrieval

> Early component notes (May 2026). They describe the first design and may be out of date; see the root
> README and RESULTS.md for the current state.


Contextual-bandit retrieval system for solve-time knowledge injection.

## Purpose

Implements the inference-time retrieval pipeline:

1. **Query construction** - Embed challenge description, binaries, and diagnostic metrics
2. **Top-N candidate retrieval** - Embedding similarity search against the knowledge graph
3. **Reranking** (planned) - Refine candidate quality post-retrieval
4. **Weight-based sampling** - Stochastic selection of M nodes using learned edge weights
5. **Context assembly** - Concatenate sampled nodes with challenge description for the solver

Also implements the **update loop** (bandit weight adjustment):
- Reward signal from binary outcome + time-to-flag
- Per-(category × difficulty) z-score standardization
- Distance-scaled, asymmetric weight updates on (context → node) edges

## Key Design Parameters (Open)

- N (candidate pool size) - scales with graph size (Implemented in `retriever.py`)
- M (nodes entering context per run) - TBD (Implemented in `retriever.py`)
- Exploration rate - tunable, swept experimentally in `experiment_harness.py`
- Cold-start weight initialization - optimistic init + forced exploration budget
- Refresh policy - static by default; periodic/event-triggered as future ablations

## Provenance

- MCP client from `baseline-solver/memory_client.py`
- Architecture per `system_description.md` §Inference Loop, §Update Loop
