# Open Decisions

Decisions made during implementation that defer to later experimentation or that override/specify open questions from system_description.md.

## 2026-05-29 - Knowledge Graph Storage - Graph and Vector Storage Backend
**Question:** The system spec defines a Knowledge Graph with embeddings but leaves the storage backend ambiguous.
**Choice:** Using NetworkX (`networkx`) for the graph topology/attributes and NumPy (`numpy`) for embedding storage and cosine similarity retrieval. Graph serializes to JSON, embeddings to NPY.
**Reasoning:** NetworkX + NumPy is the standard Python approach for research graphs of this size ($N < 100,000$). It avoids adding heavy external dependencies (like Neo4j or ChromaDB), ensures complete determinism, and keeps the workflow reproducible from config alone. NumPy exact matrix multiplication for cosine similarity is instantaneous at these scales.
**Alternatives considered:** 
- *SQLite + sqlite-vec*: Good for durability, but slightly harder to inspect and version natively compared to JSON/NPY.
- *ChromaDB / Qdrant*: Overkill for the current scale, requires managing client-server state or complex local DB files, and makes reproducibility harder across machines.
**Reversibility:** Moderate. The `KnowledgeGraphStore` interface abstracts the backend, but switching to a SQL/NoSQL DB later would require a data migration script.
**Revisit when:** Graph load times exceed 5-10 seconds, or the number of nodes approaches 1,000,000 (unlikely for CTF trajectories).
**Related sweep:** [knowledge_graph/SWEEPS.md](knowledge_graph/SWEEPS.md)

## 2026-05-29 - Embedding and Retrieval - Initial Context-Source Bandit Weights
**Question:** The spec defines stochastic sampling using bandit weights on (context-source → candidate-node) edges. At step 0 of the challenge (initial retrieval), what is the context-source, and what are the initial weights before Component 7 (Bandit update loop) is implemented?
**Choice:** We will use the raw cosine similarity score as the base weight for the initial retrieval pipeline. The context-source is implicitly the text query itself.
**Reasoning:** Until the self-reflection grader (Component 4) and bandit updater (Component 7) are built, we do not have a mechanism to standardize rewards or map challenges into strict (category x difficulty) context cells. Using similarity as the un-tuned base weight provides a functional baseline that respects the Top-N ranking while allowing stochastic exploration.
**Alternatives considered:** Uniform random sampling from Top-N. Rejected because it destroys the ranking signal entirely, whereas temperature-scaled softmax on similarity preserves it.
**Reversibility:** Easy. The `sample_top_m` function will eventually accept a generic list of weights. The caller will just pass the specific edge weights instead of similarities once Component 7 is live.
**Revisit when:** Implementing Component 7 (Contextual-bandit weight updates).
**Related sweep:** [retrieval/SWEEPS.md](retrieval/SWEEPS.md)

## 2026-05-31 - Full build (C2-C9) - Similarity prior on the reward scale
**Question:** §6.2 says `prior(v) = normalize_to_reward_scale(cosine_sim)`. What is the concrete mapping, given Q lives on the reward scale (R = o·(1+λe) ≈ [0, 1.5]) and the optimistic init Q0 = 0.5?
**Choice:** `prior = max(0.0, cosine)` (clamp negatives to 0). Implemented in `retrieval/retriever.cosine_to_reward_scale`.
**Reasoning:** Keeps the prior in ~[0,1], commensurate with observed Q and Q0, so the pseudo-count blend `(nQ+κ·prior)/(n+κ)` is dimensionally sane. Negative cosine = irrelevant ⇒ 0 prior (never a reason to sample).
**Alternatives considered:** `(cos+1)/2` (maps [-1,1]→[0,1]) - rejected: gives irrelevant nodes a 0.5 baseline prior, inflating their sampling odds. Min-max over the pool - rejected: non-stationary, breaks OPE re-scoring.
**Reversibility:** Easy (one function); OPE can re-score if changed since `prior` is logged per candidate.
**Revisit when:** real-data cosine distributions are known; may need temperature on the prior.
**Related sweep:** `kappa` (S in spec §15).

## 2026-05-31 - Full build - Online vs two-pass reward standardization
**Question:** §9.2 standardizes steps within (category×difficulty). Fit the standardizer in a pre-pass over the dataset, or update it online as runs complete?
**Choice:** Online - each run's steps feed the cell stats before the next run; partial pooling (τ0) keeps the first runs in a fresh cell sane (they standardize against the global pool).
**Reasoning:** Matches the lifelong-learning stream (the bandit learns in one pass); avoids a second pass over the scarce GPU. Determinism preserved (cells are a pure function of the seeded run order).
**Alternatives considered:** Two-pass (collect all steps, then reward) - cleaner z-scores but needs the whole dataset solved first; reserved as an eval-only option.
**Reversibility:** Easy - `RewardStandardizer` supports both (just call `observe` for the whole set first).
**Revisit when:** if early-run reward noise visibly destabilizes the bandit on real data.
**Related sweep:** `tau0`.

## 2026-05-31 - Full build - Bandit context = CATEGORY default in eval; CLUSTER is spec default
**Question:** spec §15 default `context_resolution = cluster(32)`. The synthetic eval uses `category`. Which is canonical?
**Choice:** Spec default stays `cluster(32)` in `config.py`. The synthetic eval overrides to `category` (6 nodes) because the synthetic world has 6 clean categories and few challenges, so category is the right granularity there. Real runs use the spec default unless the §6.1 budget calc says otherwise.
**Reasoning:** §6.1 budget: with ~2000 runs, cluster(32) clears the ≥5 updates/cell threshold; category(6) is even safer. Cluster generalizes across categories on real data; category is the conservative cut.
**Alternatives considered:** per-challenge (rejected by C-1), parametric LinUCB (built, ablation).
**Reversibility:** Trivial (`config.context_resolution`); it is an explicit ablation axis.
**Revisit when:** real `expected_solve_runs` is known - run `bandit.sample_efficiency_budget` and log it (C6 acceptance).
**Related sweep:** `context_resolution`, `n_clusters`.

## 2026-05-31 - Full build - KnowledgeGraphStore.add_node updates in place on re-add
**Question:** Write-back and promotion re-add an existing node (to persist a tier/attribute change). The original `add_node` appended a new embedding row each time, desyncing the matrix from `node_to_idx` (IndexError in retrieval).
**Choice:** `add_node` now updates the existing embedding row / pending entry in place and refreshes graph attributes; only brand-new ids append. (C1 store, fixed during C7 integration.)
**Reasoning:** Required for the lifelong loop; the prior behavior was an add-once assumption that C7 violated. No interface change.
**Alternatives considered:** a separate `update_node` method - rejected as redundant; callers already pass the full node.
**Reversibility:** N/A (bug fix); covered by `tests/knowledge_graph/test_store.py` + `tests/test_orchestrator.py`.
**Revisit when:** never, unless a true append-only semantics is needed for audit (provenance already covers audit).

## 2026-05-31 - Full build - Episodic-only (B3) ≈ B2 in the synthetic world
**Question:** B3 (raw/episodic-only, the Faulty-Memory null) should retrieve raw write-up chunks with no decomposition. The synthetic world only has decomposed Tool/Technique nodes.
**Choice:** In the synthetic smoke test, B3 mirrors B2 (same retrieval; write-back disabled). On the REAL seed graph, B3 retrieves raw `Challenge`/write-up text nodes (no decomposition), so the contrast is real. The harness still computes the B5-vs-B3 justification check; it is only *meaningful* on real data.
**Reasoning:** The synthetic world validates the machinery (paired stats, the justification check, the loud warning), not the episodic-vs-decomposed phenomenon, which is a real-data question.
**Alternatives considered:** synthesizing raw-chunk nodes in the mock world - deferred as over-engineering of a smoke-test instrument.
**Reversibility:** Easy - add raw-chunk nodes to `evaluation/synthetic.build_world` if a synthetic B3 contrast is wanted.
**Revisit when:** building the real seed graph (decomposition pipeline lands raw + decomposed nodes).
