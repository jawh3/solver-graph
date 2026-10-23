# Agentic CTF Solver - Architecture Specification v2 (Implementation Brief)

**Status:** supersedes the v1 "Complete System Description."
**Audience:** the implementation agent coding in the `ctf_solver` conda environment.
**Author of this revision:** design review pass, 2026-05-30 (James Hugglestone / Samuel, adv. Dr. Liu, FSU).
**Companion docs:** `research_synthesis.html` (literature + rationale), `papers/` (PDFs + `README.md`), `OPEN_DECISIONS.md` (decision log).

> **How to use this document.** This is the source of truth for *what to build and in what order*. Where a number is a tunable, it is marked `[SWEEP]` with a default and a range. Where a choice is a research question, it is marked `[ABLATION]`. Do not silently change a `[SWEEP]`/`[ABLATION]` default - log it in `OPEN_DECISIONS.md`. **Build in the order in §16.** Do not skip ahead to write-back/bandit before the evaluation harness exists.

---

## 0. Changelog from v1 (read this first)

These are the substantive changes from the prior description. Rationale for each is in `research_synthesis.html`; the one-line "why" is given here.

| # | Change | Why |
|---|--------|-----|
| C-1 | **Bandit context is coarsened** from per-challenge nearest-analog to a **context node = category or cluster** (resolution is `[ABLATION]`). Optional **parametric LinUCB** variant. | Per-challenge `(context→node)` edges are too sparse to learn given the run budget (<1 update/edge). See §6.1 sample-efficiency calc. **This is the change most likely to prevent another non-significant result.** |
| C-2 | **Similarity is kept as a decaying prior in the sampling score** (not discarded at stage 2). | Without it, data-poor nodes are sampled near-randomly early on; similarity *is* the warm-start prior. |
| C-3 | **"Grow N to fight lure-similarity" is reversed.** Keep N high for recall, add a **cross-encoder reranker + diversity (MMR)**, sample a **small M**. | Bigger N admits *more* lures (long-context RAG degradation). Precision is fixed downstream, not by pool size. |
| C-4 | **Credit goes to the action-grounded *cited* subset of M, not uniformly to all M.** | Uniform terminal credit is how recovery-masking sneaks in. A node in context but never used must not gain weight on a success. |
| C-5 | **Primary efficiency metric = step/token count, not wall-clock seconds.** | Wall-clock on a shared machine is irreproducible; steps are the real "time-to-flag" signal. |
| C-6 | **Difficulty/category use ground-truth labels first**; LLM grader is a calibrated fallback. Sparse `(category×difficulty)` cells use **partial pooling**. | The blind LLM grader is a documented weak point (self-judgment unreliable; Qwen-grades-Qwen self-preference). |
| C-7 | **Propensity logging + action-grounded citation logging from day one.** | Enables off-policy evaluation (sweep on logged data, no new GPU) and powers the false-lesson paper. Cannot be reconstructed later. |
| C-8 | **Write-back is gated** (provisional → trusted only after verified utility), nodes carry **applicability conditions** and **provenance**, failures decomposed-and-tagged `[ABLATION]`. | The LLM abstraction step is where false lessons are born; gating + applicability conditions are the cheapest filter. |
| C-9 | **Build order reordered:** evaluation harness + baselines come **before** bandit updates and write-back. | You cannot attribute gains without locked-in measurement. |
| C-10 | **Mandatory baseline matrix** incl. zero-shot, raw/episodic-only, and a FORGE-style competitor. | Controls pretraining contamination and proves write-back is justified. |

**Non-goals for this iteration (do NOT build yet):** mid-run context refresh (instrument only), periodic fine-tuning, multi-agent/master-worker, retraction/decay (defined in §10, built in C9), Neo4j migration (trigger in §4).

---

## 1. System overview

A single LLM agent solves CTF challenges. On each challenge it **retrieves** a small set of "lessons" (typed nodes) from a **self-extending knowledge graph**, solves with tool access in a container, then **updates** a contextual bandit over the graph and optionally **writes new lessons back**. The flag is a **verifiable reward**. The research object is **false-lesson propagation** - the system encoding causally-incorrect lessons - and the claim is that outcome+step-conditioned contextual-bandit retrieval with stochastic sampling **differentially suppresses** false lessons, measured, not eliminated.

Two framings are supported by the same instrumentation (decide with Dr. Liu; the build is identical):
- **F-A (safer):** "Optimal context retrieval for agentic CTF solving," suppression as a measured consequence.
- **F-B (higher-ceiling, recommended):** "A self-extending memory poisons itself; here is the first method to measure and suppress endogenous false-lesson propagation in a verifiable-reward domain."

---

## 2. Hardware, model, environment (unchanged)

- Inference: `the GPU server`, dual RTX A5000. Solver: **Qwen3.5-27B-FP8** via externally-managed **vLLM** (OpenAI-compatible API), available only as **coordinated events** → treat solve runs as the scarce resource; maximize information per run (§12, §13).
- Dev env: conda `ctf_solver`. **Mock + real duality** for every vLLM-touching component (mock embedder, mock solver, mock grader) so development never blocks on the shared GPU.

---

## 3. Core data model

### 3.1 Node types
Every node has: `node_id` (stable hash), `type`, `text`, `embedding` (vector), `created_at`, `source` (`seed_github` | `seed_pico` | `writeback`), `tier` (`trusted` | `provisional`), and type-specific fields.

| Type | Key fields |
|------|-----------|
| `Challenge` | `category`, `difficulty`, `difficulty_source` (`ground_truth`|`grader`), `ground_truth_labels`, `artifacts[]`, `contamination_flag` (bool - is this challenge public/likely in pretraining?) |
| `Artifact` | `path`, `file_type`, `hashes`, `static_metrics` (size, arch, protections) |
| `Technique` | `canonical_name`, **`applicability`** (structured; see §3.4), `failure_derived` (bool) |
| `Tool` | `canonical_name`, `invocation_signature` (regex/keywords used for citation matching, §9.3) |
| `Category` | one per CTF category (crypto/pwn/web/rev/forensics/misc) - **also serves as a coarse context node** (§6) |
| `Cluster` | k-means cluster over `Challenge` embeddings - **the default context node** (§6) |

> Other node types may exist; all must carry the common header fields above.

### 3.2 Edge types
| Edge | Meaning |
|------|---------|
| `shares_canonical(a,b)` | Two nodes linked via shared LLM-canonicalized name (seed-graph structure). **Audit for misgrouping (§5.3).** |
| `derived_from(node, run_id)` | **Provenance.** Which run created a write-back node. Mandatory on every write-back node. |
| `context_weight(context_node → candidate_node)` | **The bandit-bearing edge.** Carries `Q`, `n`, `last_updated` (§6). `context_node` is a `Category` or `Cluster`, **not** a `Challenge`. |
| `applies_in(technique → category/cluster)` | Optional hard/soft retrieval filter from applicability conditions. |

### 3.3 Identity & determinism
`node_id = sha1(type + canonical_text)`. Embeddings stored in a NumPy matrix indexed by a stable `node_id → row` map. All randomness (sampling, seeds) is seeded and the seed is logged per run.

### 3.4 Applicability conditions (counters "interference")
`Technique.applicability` is a structured record captured at decomposition time:
```
applicability = {
  categories: [str],            # where it applies
  preconditions: [str],         # e.g., "no PIE", "libc 2.31", "format string in printf"
  artifacts_required: [str],    # e.g., "ELF binary", "pcap"
  confidence: float             # extractor confidence
}
```
A lesson with empty/over-broad applicability is a false-lesson risk; the extractor must attempt to fill it, and retrieval may filter on it `[ABLATION: hard filter vs soft prior vs ignore]`.

---

## 4. Storage backend (keep - do not migrate)

- **NetworkX (JSON)** for graph structure + node/edge attributes; **NumPy `.npy`** for the embedding matrix. Deterministic, zero-dependency, offline-reproducible. Validated to 10k nodes (<1s I/O, ~63 MB).
- **Bandit tables** (`Q`, `n` per `(context_node, candidate_node)`): store as a sparse keyed store (dict-of-dicts or `scipy.sparse`/CSR keyed by `(context_id, node_id)`), serialized alongside the graph, **versioned per experiment**.
- **Logs** (§12): append-only **JSONL**, one file per record type per experiment run. Never overwrite; logs are paper data.
- **Neo4j migration trigger (unchanged):** graph > ~50k nodes **or** retrieval queries needing Cypher-level joins. Until then, do not migrate. Logged in `OPEN_DECISIONS.md`.

---

## 5. Seed graph construction

### 5.1 Sources
- **GitHub write-ups:** ~1,000 repos scraped → ~4,000 candidates → LLM-filtered to ~500 high-quality.
- **picoCTF:** all challenges, descriptions, binaries/artifacts.

### 5.2 Decomposition
LLM decomposes each source into typed nodes (§3.1). Edges via `shares_canonical` on LLM-canonicalized names. Every node embedded.

### 5.3 New requirements in v2
- **Capture ground-truth metadata:** category (CTF tag) and difficulty (picoCTF point value/tier → difficulty bucket) stored in `ground_truth_labels`; set `difficulty_source = ground_truth` where available. This is the standardization key (§9.2), not the LLM grader.
- **Set `contamination_flag = true`** for any challenge that is public (all picoCTF, all scraped write-ups). Used to gate the headline evaluation (§14).
- **Edge-quality audit (one-time + sampled):** sample N=`[SWEEP: 200]` `shares_canonical` edges and check for **misgrouping** (two unrelated uses of the same surface name linked). Report spurious-link rate in the seed-graph dev log. If high, tighten canonicalization (disambiguate by category/context).
- **Extract `applicability` (§3.4)** for every `Technique` node at decomposition time.

---

## 6. The contextual bandit (the core change - C-1)

### 6.1 Why coarsen the context - do this calculation before coding
Budget check the implementation agent must run and record:
```
expected_updates_per_cell = (expected_solve_runs * M) / (num_context_nodes * avg_candidates_credited_per_run)
```
If `expected_updates_per_cell < ~5-10`, the granularity is too fine to learn. Per-challenge context (hundreds-thousands of context nodes) fails this; **Category (~6)** and **Cluster (k≈20-50)** pass it. Choose `num_context_nodes` so the cell budget clears the threshold for your realistic `expected_solve_runs`.

### 6.2 Default design - cluster/category-conditioned value bandit
- **Context resolution** `[ABLATION: category | cluster(k) | parametric]`, default **`cluster(k=[SWEEP: 32])`**.
  - At inference, map the query to a context node: nearest cluster centroid (or known category). This replaces v1's top-1 nearest-*challenge* analog.
- **Per `(context c, candidate v)` statistics:** `Q[c][v]` (estimated standardized reward), `n[c][v]` (effective count). Stored sparsely.
- **Initialization** `[ABLATION: zero | seed_warm | optimistic]`:
  - `zero`: `Q=0, n=0`.
  - `seed_warm`: `Q` initialized from seed co-occurrence/quality prior **(treat as alignment-gated - log an alignment estimate; warm-start can hurt if seed lessons misalign with how Qwen solves).**
  - `optimistic`: `Q=Q0` (`[SWEEP: 0.5]`), forces early exploration.
- **Sampling score** (keeps similarity as a decaying prior - C-2):
  ```
  prior(v)      = normalize_to_reward_scale(cosine_sim(query, v))   # similarity prior
  score(c,v)    = (n[c][v]*Q[c][v] + kappa*prior(v)) / (n[c][v] + kappa)
  ```
  `kappa` `[SWEEP: 5]` = prior strength (how many reward observations before similarity is overridden). New/unseen nodes are scored by similarity (sensible cold start), not by an arbitrary uniform weight.
- **Stochastic selection of M** (with exploration floor - C-3 partner):
  - Sample `M` `[SWEEP: 8; range 4-16]` nodes **without replacement** via sequential softmax over `score/τ`, temperature `τ` `[SWEEP: 1.0]` with floor `τ_min` `[SWEEP: 0.3]` to guarantee exploration.
  - **Record, for OPE, the per-draw softmax probability of each selected node and the sequential set log-probability** (§12, §13). This is mandatory.

### 6.3 Optional parametric variant `[ABLATION]`
Hybrid **LinUCB**: context feature `x = [query_emb ⊕ candidate_emb ⊕ interactions]`, arm = candidate node, reward = standardized run reward attributed to the (cited) node. Maintains `A, b`; selects by UCB; generalizes smoothly with principled exploration. Build only after the tabular default works; it is a planned comparison, not the first cut.

---

## 7. Inference loop (solve-time retrieval) - corrected pipeline

```
1. QUERY CONSTRUCTION
   q = embed( challenge_description
              + artifact_static_metrics(run diagnostics: file, checksec, strings-summary, etc.)
              + optional category hint )
   # may produce multiple query vectors (desc / artifact); fuse by max or mean [ABLATION]

2. HIGH-RECALL RETRIEVAL
   pool_N = top_N_by_cosine(q, all_candidate_nodes)     # N = [SWEEP: 200] for recall, NOT grown to "fight lures"

3. RERANK (C-3)  -- precision, cheap
   pool_R = cross_encoder_rerank(query_text, pool_N)[:R]  # R = [SWEEP: 50]; use a CROSS-ENCODER, not an LLM reranker

4. DIVERSITY (C-3) -- guarantees "competing better lessons exist in the pool"
   pool_D = MMR(pool_R, lambda_mmr=[SWEEP: 0.5])          # or category-stratified top-k; ensures non-duplicate alternatives

5. CONTEXT RESOLUTION (C-1)
   c = resolve_context_node(q)   # nearest cluster centroid (default) or category; bootstrap node if graph empty

6. WEIGHT-BLENDED STOCHASTIC SAMPLING (C-2)
   M_set, propensities = sample_without_replacement(
       pool_D, score=score(c,·), temperature=max(τ, τ_min), size=M)

7. CONTEXT ASSEMBLY
   context = render(M_set)   # include applicability conditions + tier flag; mark provisional nodes as low-confidence
   prompt  = challenge_description + context + container_pointer + tool_instructions

8. SOLVE
   run solver (Qwen3.5) with MCP tools in container; record every step/action (§12); terminate on flag | timeout | max_steps
```
**Logging during the run:** at each step, also log the **counterfactual "would-have-retrieved" set** if a refresh had fired (free instrumentation; no actual refresh - §8).

---

## 8. Context refresh policy (static default; instrument only)

- **Default: static.** The M nodes are inserted once and held for the whole trajectory.
- **Free instrumentation (build now):** at each step, compute and log what a refresh *would* have retrieved (given current trajectory state), **without** refreshing. Produces post-hoc data on whether/when/which refresh would have mattered.
- **Future ablations (define, do not build):** periodic refresh every `L` steps; event-triggered (tool-failure / stagnation / self-reflection); **hybrid slots** = `K` fixed semantic (category-level) + `M-K` adaptive episodic. (Hybrid maps onto the semantic/episodic split - see Megagon paper.)

---

## 9. Update loop (after each run)

### 9.1 Reward (C-5)
- Outcome `o ∈ {0,1}` (flag verified by the checker).
- **Efficiency from step count, not wall-clock.** Let `steps` = number of agent steps/tool-calls to flag. Standardize within the challenge's `(category × difficulty)` cell (§9.2): `z_steps = (steps - μ_cell)/σ_cell`. Efficiency `e = -z_steps` (faster → higher).
- **Run reward:** `R = o * (1 + λ * e)`, `λ` `[SWEEP: 0.5]`.
  - This is the latent-poison lever: a **slow success** (`e<0`) earns `R<1`; a **fast success** earns `R>1`; a **failure** earns `0` `[ABLATION: small negative]`. A false lesson that wins via a longer route accrues lower `Q` and is down-weighted.
- Optionally center by a baseline `b` (running mean reward in context `c`) for variance reduction: the bandit target is `R - b`.

### 9.2 Standardization (C-6)
- **Difficulty/category source priority:** (1) `ground_truth_labels`; (2) calibrated LLM grader (blind to the solve attempt) only when no ground truth. Maintain a calibration map from grader output → empirical solve rate; **validate grader difficulty against observed solve rate** and report the agreement. `[ABLATION: include a non-Qwen grader to measure family/self-preference bias.]`
- **Partial pooling for sparse cells:** shrink cell mean/var toward global:
  ```
  μ_cell_hat = (n_cell * mean_cell + τ0 * μ_global) / (n_cell + τ0)
  ```
  `τ0` `[SWEEP: 5]`; same shrinkage for σ. Prevents garbage z-scores from 1-3-run cells.

### 9.3 Credit assignment (C-4 - the recovery-masking lever)
- **Action-grounded citation.** A sampled node `v` is `cited` iff its `invocation_signature` (tool name / technique keywords / produced artifact) appears in the agent's **executed actions** (tool calls, shell commands). Deterministic matcher first; optional LLM-judge fallback **flagged low-confidence**. **Do not** use LLM self-report as the primary signal.
- **Credit-eligible set** `S = M_sampled ∩ cited`. Nodes in context but never used get **no positive credit on success** (this is exactly how recovery-masking is denied reward). `[ABLATION: credit-all-M vs credit-cited-only]` - default cited-only.
- **Update rule** for each `v ∈ S`, context `c`:
  ```
  scale = g(d_v)                              # distance scaling, d_v = 1 - cosine_sim
  Q[c][v] += α * scale * ((R - b) - Q[c][v])  # α = [SWEEP: 0.1]
  n[c][v] += 1
  ```
  - **Distance scaling** `[ABLATION: symmetric | asymmetric]`. Asymmetric candidate: on **failure** scale by similarity (blame near nodes more), on **success** scale by inverse similarity (reward far nodes more) - intended to fight rich-get-richer. **Treat asymmetric as an ablation against a principled-exploration baseline (UCB/optimistic init); do not make it load-bearing without the comparison.**
- **Causal credit for suspicious nodes (build in C9):** for nodes with high `Q` **and** high time-to-flag variance, run **leave-one-out re-rollouts** from a shared pre-lesson state (with vs without the node) and credit the **marginal** flag-reaching difference. This is the oracle-free causal probe (Memory-R2 / Math-Shepherd style); gate it to suspicious nodes for cost.

### 9.4 Write-back (C-8 - gated)
- Decompose the trajectory (LLM) into new nodes using the §3 schema.
- **Decompose failures too, tagged `failure_derived=true`** `[ABLATION: successful-only vs both]` - default both (FORGE supports failure decomposition; the comparison is publishable).
- Every new node: `tier=provisional`, low/optimistic init weight, **`derived_from` provenance edge to the origin run**, and extracted **`applicability`** (§3.4).
- **Promotion gate:** `provisional → trusted` only after the node is **cited in ≥ `k_promote`** `[SWEEP: 3]` distinct **successful** runs across **≥ `c_promote`** `[SWEEP: 2]` distinct challenges. Provisional nodes are retrievable but flagged and weight-capped. (This is the birth-defect lever: an N=1 lesson is never trusted prematurely.)
- **Mandatory comparison:** the system must be evaluated against a **raw/episodic-only** baseline (no decomposition; retrieve raw write-up/trajectory chunks). If gated write-back does not beat episodic-only, the write-back loop is not justified (Faulty-Memory null).

### 9.5 Cold-start init for new nodes
Covered by §6.2 init `[ABLATION]`. For seed-warm specifically, **log an alignment estimate** (agreement between seed-lesson recommendations and observed Qwen successes) as a covariate; predict that seed-warm wins only where alignment is high.

---

## 10. Graph hygiene: retraction & decay (define now, build in C9)

- **Decay/retract** nodes whose `Q` stays below `θ_retract` `[SWEEP]` across ≥ `k_retract` `[SWEEP]` exposures, or whose cited-failure rate exceeds a threshold. Demote `trusted→provisional→archived`; never hard-delete (keep for provenance/audit).
- Bounds graph growth (addresses the "exploding KG" concern) **and** removes confirmed false lessons. Report **graph-quality-over-time** as a first-class metric.

---

## 11. False-lesson propagation - modes → mechanisms → signals

| Mode | Definition | Suppression mechanism in this design | Detection signal logged |
|------|------------|--------------------------------------|--------------------------|
| **Birth defect** | N=1 lesson trusted prematurely | Provisional tier + promotion gate (§9.4) | per-node exposure count before promotion |
| **Latent poison** | False premise that wins via a longer route | Step-count efficiency reward `R=o(1+λe)` (§9.1) | time-to-flag (steps) distribution per node |
| **Recovery masking** | Lesson in context during success but not causal | Action-grounded cited-only credit (§9.3) + LOO re-rollout (C9) | cited vs in-context rate; marginal LOO value |

**Scope of claim (state in the paper):** "reduces false-lesson *exposure* within the verifier-detectable subset." Out of scope: model-internal memorization shortcuts (Qwen can pass the verifier without the lesson's stated mechanism - Spurious-Rewards) - name this as a limitation and use family-eval (§14) to bound it.

---

## 12. Logging schema (build with C2/C3 - C-7; this is paper data)

Append-only JSONL per record type. **Propensities are mandatory and non-reconstructable.**

```jsonc
RunRecord {
  run_id, ts, experiment_id, policy_config_hash, git_commit,
  challenge_id, challenge_meta:{category, difficulty, difficulty_source, contamination_flag},
  context_node_id, context_resolution,
  candidate_pool:[{node_id, cosine, rerank_score, mmr_kept:bool}],
  sampled_set:[node_id], per_draw_propensity:[{node_id, p}], set_logprob, temperature,
  outcome, steps, tokens, wall_clock_s, time_to_flag_steps,
  R_raw, R_efficiency_e, R_final, baseline_b,
  seed, solver_model, embedder_model, reranker_model, container_image, tool_versions
}
StepRecord { run_id, step_idx, action_type, tool, command, observation_hash,
             would_have_retrieved:[node_id], stagnation_flag }   // counterfactual refresh logging
CitationRecord { run_id, node_id, cited:bool, match_evidence, confidence:("deterministic"|"llm") }
UpdateRecord { run_id, context_node_id, node_id, pre_Q, post_Q, pre_n, post_n,
               distance_scale, credit_eligible:bool, reason }
WriteBackRecord { run_id, new_node_id, type, derived_from_run, failure_derived:bool,
                  applicability, init_weight, tier }
GraphSnapshot { experiment_id, version, node_count, edge_count, path }   // per experiment, for reproducibility
```

---

## 13. Off-policy evaluation (OPE) support (build in C8)

Because per-draw propensities are logged, alternative policies π′ (different M, τ, κ, scaling, init) can be evaluated on **already-logged runs without new GPU**:
- **IPS:** `V(π′) = mean_runs[ (π′(M_set | c) / π_log(M_set | c)) * R ]`, using the sequential set probability.
- **Doubly-robust** with a fitted reward model for variance reduction.
This multiplies effective sample size - essential given scarce vLLM. The sweep runner (C8) consumes JSONL logs and emits OPE estimates with bootstrap CIs before committing any real-run sweep.

---

## 14. Evaluation protocol

### 14.1 Datasets (control pretraining contamination - C-10)
- **SEED-PUBLIC** (`contamination_flag=true`): picoCTF + scraped - diagnostic only, **not** the headline number.
- **HELD-OUT-NOVEL:** challenges not public / not in pretraining (sourced fresh) - headline.
- **TRANSFORMED-FAMILIES:** semantics-preserving transformations of challenges (Evolve-CTF style; see `papers/CaptureTheFlags_2602.05523...pdf`) - separates *method learned* from *write-up memorized*. **Headline robustness number.**

### 14.2 Baseline matrix (all on all three datasets)
| ID | System | Isolates |
|----|--------|----------|
| **B0** | Zero-shot, no graph | Pretraining memorization; the **lift denominator** |
| **B1** | Seed graph, similarity-only retrieval, fixed M, no bandit, no write-back | Static retrieval value |
| **B2** | B1 + reranker + diversity | Retrieval-quality contribution |
| **B3** | Raw/episodic-only (retrieve raw chunks, no decomposed lessons) | **Faulty-Memory null** - must be beaten to justify write-back |
| **B4** | B2 + contextual bandit (no write-back) | Bandit contribution |
| **B5** | B4 + gated write-back | **Full system** |
| **B6** | B5 with write-back **ungated** | Shows gating matters |
| **BC** | FORGE-style champion-broadcast (minimal re-impl) | Strongest external competitor |

### 14.3 Metrics
- Solve rate; **lift over B0 (paired, per-challenge)** - the real claim.
- Median **time-to-flag in steps**.
- **False-lesson exposure rate** (credited-but-non-causal, via LOO re-rollout on a sample).
- Recovery-masking rate (cited vs in-context on successes).
- Graph quality over time (§10).
- Offline **regret** (from OPE).

### 14.4 Statistical methodology (the prior result was non-significant - design for power)
- **Pre-specified power analysis** before spending GPU (what N challenges × seeds detects a `[SWEEP: 10pp]` effect at observed variance?).
- **Paired / within-challenge** comparisons; **common random seeds** across conditions; **multiple seeds per challenge**.
- **Bootstrap CIs**; report effect sizes + CIs, not just p-values.
- Use **OPE** (§13) to extend effective sample size.

---

## 15. Configuration & swept parameters (single source)

| Param | Default | Sweep range | Resolves |
|-------|---------|-------------|----------|
| `N` (recall pool) | 200 | 100-500 | recall vs cost |
| `R` (rerank keep) | 50 | 20-100 | precision |
| `lambda_mmr` | 0.5 | 0.3-0.8 | diversity |
| `M` (context size) | 8 | 4-16 | context budget |
| `tau` / `tau_min` | 1.0 / 0.3 | 0.5-2.0 / 0.1-0.5 | exploration |
| `kappa` (prior strength) | 5 | 1-20 | sim-prior decay |
| `alpha` (lr) | 0.1 | 0.02-0.3 | bandit learning rate |
| `lambda` (efficiency wt) | 0.5 | 0-1.0 | latent-poison sensitivity |
| `context_resolution` | cluster(32) | category / cluster(k) / parametric | **C-1 granularity** |
| `init_mode` | optimistic | zero / seed_warm / optimistic | cold-start |
| `distance_scaling` | asymmetric | symmetric / asymmetric | rich-get-richer |
| `writeback_mode` | both | off / successful / both | failure decomposition |
| `credit_mode` | cited_only | cited_only / all_M | recovery-masking |
| `k_promote` / `c_promote` | 3 / 2 | 2-5 / 1-3 | birth-defect gate |
| `refresh_mode` | static | static / periodic_L / hybrid_K | (future) |

All defaults logged in `OPEN_DECISIONS.md`; any change is a logged decision.

---

## 16. Component build order (revised - C-9) with acceptance criteria

| C | Component | Acceptance criteria |
|---|-----------|---------------------|
| **C1** ✅ | KG storage + CRUD | Done: NetworkX+NumPy, mock embedder, 10k-node sweep. |
| **C2** (in progress) | Embedding + retrieval | top-N + **cross-encoder rerank** + **MMR** + context resolution + **sim-prior-blended** sampling + **propensity logging**. Mock+real modes. Unit tests on retrieval determinism & propensity correctness. |
| **C3** | Solver harness | MCP tools in container; deterministic **StepRecord** logging; **action-grounded citation matcher** (deterministic). Mock solver. Flag-checker integration. |
| **C4** | **Evaluation harness + baselines B0-B3 + metrics + stats layer** ← before bandit/write-back | Reproduces B0/B1/B2/B3 on SEED-PUBLIC; paired-design + bootstrap CIs implemented; power-analysis script. |
| **C5** | Reward + standardization | Step-count reward `R=o(1+λe)`; ground-truth-first difficulty; **partial pooling**; grader calibration + validation report. |
| **C6** | Contextual-bandit update loop | cited-only credit; init/scaling ablations; **B4** runs; UpdateRecords emitted; sample-efficiency budget (§6.1) reported. |
| **C7** | Write-back (gated) | provisional tier + promotion gate + applicability + provenance + failure tagging; **B5/B6**; beats B3 or it's flagged. |
| **C8** | OPE + sweep runner | IPS/DR estimators from logs; offline sweep of M/τ/κ/scaling with CIs **before** any real-run sweep. |
| **C9** (later) | Hygiene + causal credit + competitor | retraction/decay; LOO re-rollout causal credit; FORGE-style **BC** baseline. |

Each component ships with: a dev log, a mock+real test, and a harness that emits paper-grade data. **Janitorial pass at end of every session** (classify ad-hoc scripts: delete / promote / archive).

---

## 17. Open decisions still genuinely unresolved (sweep-decide; log in OPEN_DECISIONS.md)
Context resolution level · M · N-graph-size relation · τ exploration rate · init mode · failure-decomposition on/off · symmetric vs asymmetric scaling · refresh policy (later) · periodic fine-tuning (out of scope now) · provenance-edge retrofit depth · parametric vs tabular bandit.

---

## 18. Methodological commitments (unchanged, restated)
Test-heavy, no shortcuts · pre-specified bounded sweeps with documented predictions · mock+real duality · per-component dev logs + `OPEN_DECISIONS.md` · no vibe-coding clutter (mandatory janitorial pass) · trace-level instrumentation (§12) as the minimum bar.

---

## 19. Key references justifying specific choices (in `papers/`)
- Coarsen bandit / parametric contextual bandit, OPE - `PURPLE`, LinUCB (Li et al. 2010/2011), `CreditAssignmentSurvey_2604.09459`.
- Credit only used items / fair credit under memory - `MemoryR2_2605.21768`.
- Step-count efficiency / verifiable≠correct - `SpuriousRewards_2601.11061`, `LLMsGamingVerifiers_2604.15149`.
- Gated abstraction / episodic-only baseline - `FaultyMemory_2605.12978`, `ExperienceFollowing_2505.16067`.
- Grader caution - `CannotSelfCorrect_2310.01798`, `JudgingTheJudges_2406.07791`.
- Cold-start init alignment - `JumpStartFalseStart_2604.02527`.
- Contamination-robust eval - `CaptureTheFlags_2602.05523`.
- Competitor baseline / failure decomposition - `FORGE_2605.16233`.
- Diversity vs lure-similarity - `LongContextMeetsRAG_2410.05983`, `DynamicContextSelection_2512.14313`.
- Semantic/episodic hybrid slots - `Megagon_2510.19897`.

*End of spec v2.*
