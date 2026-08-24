# Lab analysis — `results/campaign_v2`

headline condition **outcome**, 144 runs, lessons/label = {'helpful': 6, 'poison': 6, 'distractor': 4}.

> ⚠ **<20 lessons/label** → label-level bootstrap CIs are anti-conservative (pilot scale). Per-lesson CIs are the trustworthy current evidence; the label-level generalization claim awaits the scaled corpus.

## H2 — observational ATE sign-flip

| estimator | label | mean ATE | 95% CI | p | excl 0 |
|---|---|---|---|---|---|
| SNIPW | helpful | +0.158 | [+0.065, +0.270] | 0.000 | True |
| SNIPW | poison | -0.104 | [-0.186, -0.031] | 0.000 | True |
| SNIPW | distractor | -0.053 | [-0.189, +0.051] | 0.505 | False |
| AIPW | helpful | +0.109 | [+0.060, +0.140] | 0.000 | True |
| AIPW | poison | -0.111 | [-0.157, -0.074] | 0.000 | True |
| AIPW | distractor | +0.018 | [-0.033, +0.103] | 0.629 | False |

**SNIPW verdict:** sign-flip = `True` (poison CI<0: True, helpful CI>0: True, non-overlapping: True).

**AIPW verdict:** sign-flip = `True` (poison CI<0: True, helpful CI>0: True, non-overlapping: True).

## H3 — promotion gap (promote-if value>0)

| label | outcome-Q rate | causal-ATE rate | gap | 95% CI | p |
|---|---|---|---|---|---|
| helpful | 1.00 | 1.00 | +0.00 | [+0.00, +0.00] | 1.000 |
| poison | 0.25 | 0.00 | +0.25 | [+0.00, +0.75] | 0.641 |

## H4 — power / MDE

observed paired SD = 0.448, n_pairs = 24 → **MDE = 25.6 pp** at 80% power. n for 5pp = 632, for 10pp = 158.

## Robustness (H2 sign-flip must survive)

- propensity MC seeds×n_mc: poison ATE<0 in **100%**, true>0 in **100%** of 20 cells.
- clip: 0.02: poison=-0.1038212810061789, helpful=0.1575367969922353, 0.05: poison=-0.1038212810061789, helpful=0.1575367969922353, 0.1: poison=-0.1038212810061789, helpful=0.1575367969922353
- ess_frac: 0.2: poison=-0.1038212810061789, helpful=0.1575367969922353, 0.3: poison=-0.1038212810061789, helpful=0.1575367969922353, 0.5: poison=-0.1038212810061789, helpful=0.1575367969922353
- propensity_source: confounded_Q: poison=-0.1038212810061789, helpful=0.1575367969922353, prior_only_Q0: poison=-0.09425511910630235, helpful=0.15648169201842657
- estimator: snipw: poison=-0.1038212810061789, helpful=0.1575367969922353, aipw: poison=-0.11141420983929383, helpful=0.10900246183140122
- θ-sweep: outcome-Q separating θ = `-0.0`, causal-ATE separating θ = `-0.06`.
