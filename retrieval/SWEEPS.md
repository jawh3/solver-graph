# Retrieval Sweeps

Pre-registered parameters and expected outcomes for Component 2 experiments.

## Diversity vs Temperature (`retrieval_sweep.json`)

**Hypothesis:** As exploration rate ($T$) increases, the selection probability of the absolute best cosine-similarity candidate will decrease from 1.0 per iteration (or `1/M` in the flattened distribution) toward the uniform probability `1/N`.

**Parameters:**
- $N=100$
- $M=5$
- Iterations = 1000
- $T \in \{0.1, 0.5, 1.0, 2.0\}$

**Expected Outcomes:**
- **T = 0.1:** The model behaves greedily. The top 1 candidate should be selected in ~100% of iterations (yielding a flattened probability of ~0.20 since M=5).
- **T = 2.0:** The model behaves almost uniformly. The top 1 candidate should be selected at a rate near its softmax uniform expectation. Given $N=100$, uniform iteration probability is $0.05$, yielding a flattened probability of $\sim0.01$.

**Actual Results (2026-05-29):**
- **T = 0.1:** `top_1_selection_prob` = 0.200 (Selected 100% of the time).
- **T = 0.5:** `top_1_selection_prob` = 0.064.
- **T = 1.0:** `top_1_selection_prob` = 0.027.
- **T = 2.0:** `top_1_selection_prob` = 0.019.

The sampling distribution accurately reflects temperature-modulated bandit exploration.
