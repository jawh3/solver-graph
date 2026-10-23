# batch_test/ - coordinated REAL-GPU run harness (the GPU server)

> Early component notes (May 2026). They describe the first design and may be out of date; see the root
> README and RESULTS.md for the current state.


Everything else in this repo runs GPU-free in mock mode (`pytest`, `python -m
evaluation.run_eval --mock`). **This directory is the only place that touches the
real GPU.** It swaps the mock embedder / reranker / solver for the real ones and
runs the *same* `evaluation/` harness, so the logs (§12), metrics (§14.3) and OPE
(§13) are byte-for-byte the same schema as the mock smoke test.

vLLM on the GPU server is a **scarce, coordinated event** (spec §2). The golden
rule: **maximize information per GPU window, then do everything else offline.**
Because every run logs sampling propensities, the whole parameter sweep
(M, τ, κ, …) is done AFTER the window with **zero extra GPU** via `ope/`.

---

## 0. One-time setup (on the GPU machine, in `ctf_solver` conda env)

```bash
conda activate ctf_solver
cd solver-graph
export PYTHONPATH=$(pwd)
pip install sentence-transformers          # the real cross-encoder reranker
# .env (or export): point at the real endpoints + the right model
export VLLM_URL=http://localhost:8000/v1
export EMBED_URL=http://localhost:8001/v1/embeddings
export MODEL_NAME=Qwen/Qwen2.5-14B-Instruct-AWQ   # MUST match the served model
export EMBED_MODEL=<your-dedicated-embedding-model>   # NOT the solver model (spec §7)
```

Serve the solver model (separate terminal): `STAGE=server bash experiments/phase2_launch.sh`.
Serve a **dedicated embedding endpoint** on another port (the retrieval embedder is
not the solver, per spec §7 / BUILD_CHECKLIST C2).

You also need a **seed graph** on disk (the `.json`/`.npy` pair produced by
`KnowledgeGraphStore.save`) and a **challenges JSONL** (schema below).

---

## 1. Preflight - never burn a GPU window on a typo

```bash
python -m batch_test.smoke_real --graph data/seed_graph/kg
```

Checks, fail-fast: solver endpoint + model id, embedding endpoint + dim, reranker
loads, seed graph loads (+ prints the **§6.1 sample-efficiency budget** so you know
the bandit *can* learn before you spend the window), and ONE real end-to-end solve.
Exit 0 = safe to launch.

## 2. The coordinated run (the merge-time batch)

```bash
python -m batch_test.run_batch \
    --challenges data/challenges.jsonl \
    --graph data/seed_graph/kg \
    --systems B0,B1,B2,B3,B4,B5,B6,BC \
    --train-epochs 3 --n-seeds 3 \
    --out results/real_eval
```

Writes `results/real_eval/results_table.json` (per-system solve rate, **paired lift
over B0 with bootstrap CIs**, median step-to-flag, recovery-masking rate,
write-back justification vs B3) and append-only logs under `results/real_eval/logs/`.

## 3. Offline sweep - NO extra GPU

```bash
python -m ope.sweep --logs results/real_eval/logs --estimator dr
```

Ranks counterfactual policies (τ, κ, M′) by IPS/DR with bootstrap CIs and runs the
sanity check (DR of the logging policy ≈ its measured value). Only the top-ranked
policies are worth a *second* coordinated GPU window.

---

## Challenges JSONL schema

```jsonc
{"name":"...", "category":"crypto|pwn|web|rev|forensics|misc",
 "points":100,            // or "difficulty":"easy|medium|hard|expert" (ground-truth first, §9.2)
 "description":"...", "hints":[...], "code_blocks":[...],
 "local_files":[{"local_path":"...", "text":"..."}],
 "dataset":"seed|heldout|families",   // headline = heldout + families (§14.1)
 "contamination_flag":true}           // true for public/pretraining-likely challenges
```

> `_world` is a MOCK-only field (synthetic ground truth); real challenges omit it -
> the real Qwen solver actually solves them and the flag checker is the oracle.

## What maps to what (real vs mock)

| Stage | mock (dev/laptop) | real (this harness) |
|------|-------------------|---------------------|
| embedder | `MockEmbedder` / `FeatureMockEmbedder` | `VLLMEmbedder` (disk-cached), dedicated endpoint |
| reranker | `MockReranker` (lexical) | `CrossEncoderReranker` (sentence-transformers) |
| solver | `MockSolver` (synthetic world) | `solver.solver.solve_challenge` (Qwen3.5-27B-FP8) |
| flag check | `_world` ground truth | real flag checker (the verifiable oracle) |

Everything between (retrieval → bandit → reward → write-back → logging → OPE) is the
**same code**.
