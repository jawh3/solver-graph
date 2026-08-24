# Lab analysis — `results/lab/campaign_3b_M3/merged`

headline condition **outcome**, 2208 runs, lessons/label = {'helpful': 12, 'poison': 12, 'distractor': 4}.

> ⚠ **<20 lessons/label** → label-level bootstrap CIs are anti-conservative (pilot scale). Per-lesson CIs are the trustworthy current evidence; the label-level generalization claim awaits the scaled corpus.

## H2 — observational ATE sign-flip

| estimator | label | mean ATE | 95% CI | p | excl 0 |
|---|---|---|---|---|---|
| SNIPW | helpful | +0.021 | [-0.006, +0.049] | 0.133 | False |
| SNIPW | poison | -0.025 | [-0.084, +0.041] | 0.442 | False |
| SNIPW | distractor | +0.015 | [-0.069, +0.109] | 0.727 | False |
| AIPW | helpful | +0.005 | [-0.022, +0.031] | 0.725 | False |
| AIPW | poison | -0.031 | [-0.094, +0.036] | 0.359 | False |
| AIPW | distractor | +0.143 | [-0.007, +0.293] | 0.119 | False |

**SNIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: False, non-overlapping: False).

**AIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: False, non-overlapping: False).

## H3 — promotion gap (promote-if value>0)

| label | outcome-Q rate | causal-ATE rate | gap | 95% CI | p |
|---|---|---|---|---|---|
| helpful | 1.00 | 0.50 | +0.50 | [+0.25, +0.75] | 0.000 |
| poison | 1.00 | 0.33 | +0.67 | [+0.42, +0.92] | 0.000 |

## H4 — power / MDE

observed paired SD = 0.232, n_pairs = 36 → **MDE = 10.8 pp** at 80% power. n for 5pp = 170, for 10pp = 43.

## Robustness (H2 sign-flip must survive)

- propensity MC seeds×n_mc: poison ATE<0 in **100%**, true>0 in **100%** of 20 cells.
- clip: 0.02: poison=-0.024999542941081542, helpful=0.02130428786140891, 0.05: poison=-0.0249994652369368, helpful=0.02130343635898317, 0.1: poison=-0.023928523228528103, helpful=0.02141601338272649
- ess_frac: 0.2: poison=-0.0249994652369368, helpful=0.02130343635898317, 0.3: poison=-0.0249994652369368, helpful=0.02130343635898317, 0.5: poison=-0.0249994652369368, helpful=0.02130343635898317
- propensity_source: confounded_Q: poison=-0.0249994652369368, helpful=0.02130343635898317, prior_only_Q0: poison=-0.023342741463561693, helpful=0.018443907784290527
- estimator: snipw: poison=-0.0249994652369368, helpful=0.02130343635898317, aipw: poison=-0.031357351229558926, helpful=0.004624315393038835
- θ-sweep: outcome-Q separating θ = `None`, causal-ATE separating θ = `None`.
