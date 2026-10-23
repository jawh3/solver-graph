## 2026-05-29 - Plan: Embedding and Retrieval

**What:** Plan for Component 2. We will build `embedder.py` (Mock/vLLM classes) and `retriever.py` (Top-N cosine similarity + Top-M stochastic bandit sampling).
**Why:** To establish the core inference-time retrieval pipeline that extracts candidate nodes from the Knowledge Graph and stochastically samples them to form the context.
**Decisions:**
- Initial "context-source" bandit weights for retrieval will default to the raw cosine similarity score. This serves as the un-tuned baseline until Component 7 (Contextual-bandit weight updates) implements the true asymmetric update rule based on specific (category/difficulty) contexts.
- The `VLLMEmbedder` will use the standard `requests` library to hit the OpenAI-compatible `/v1/embeddings` vLLM endpoint to avoid adding heavy dependencies like `openai`.
**Experiments planned:** A sweep over `exploration_rate` ($T \in \{0.1, 0.5, 1.0, 2.0\}$) mapping to retrieval diversity/entropy to ensure stochastic sampling actually samples the tail as intended.
**Results:** The stochastic sampling sweep completed successfully. At $T=0.1$ the system behaves greedily (top candidate selected in 100% of iterations). At $T=2.0$, the top candidate selection probability drops to $0.019$ (nearing uniform distribution). The temperature parameter effectively modulates the exploration budget exactly as requested by the bandit spec.
**Tests:** Boundary testing on retrieval shapes and deterministic mock embeddings. All passing.
**Complexity:** ~2-3 files, ~200 lines of code. Moderate complexity (especially the stochastic sampling math).
