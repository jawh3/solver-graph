# Lab analysis — `results/lab/campaign_7b35_M3_rep/merged`

headline condition **outcome**, 8050 runs, lessons/label = {'helpful': 35, 'poison': 35, 'distractor': 4}.

> ⚠ **<20 lessons/label** → label-level bootstrap CIs are anti-conservative (pilot scale). Per-lesson CIs are the trustworthy current evidence; the label-level generalization claim awaits the scaled corpus.

## H2 — observational ATE sign-flip

| estimator | label | mean ATE | 95% CI | p | excl 0 |
|---|---|---|---|---|---|
| SNIPW | helpful | +0.085 | [+0.014, +0.168] | 0.014 | True |
| SNIPW | poison | +0.034 | [-0.019, +0.097] | 0.243 | False |
| AIPW | helpful | +0.091 | [+0.019, +0.176] | 0.008 | True |
| AIPW | poison | +0.048 | [-0.011, +0.119] | 0.122 | False |

**SNIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: True, non-overlapping: False).

**AIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: True, non-overlapping: False).

## H3 — promotion gap (promote-if value>0)

| label | outcome-Q rate | causal-ATE rate | gap | 95% CI | p |
|---|---|---|---|---|---|
| helpful | 1.00 | 0.51 | +0.49 | [+0.31, +0.66] | 0.000 |
| poison | 1.00 | 0.48 | +0.52 | [+0.33, +0.70] | 0.000 |

## H4 — power / MDE

observed paired SD = 0.361, n_pairs = 105 → **MDE = 9.9 pp** at 80% power. n for 5pp = 410, for 10pp = 103.

## Robustness (H2 sign-flip must survive)

- propensity MC seeds×n_mc: poison ATE<0 in **0%**, true>0 in **100%** of 20 cells.
- clip: 0.02: poison=0.03413500137123658, helpful=0.08460078038680298, 0.05: poison=0.03414591281965735, helpful=0.08459900299464554, 0.1: poison=0.03321020510115525, helpful=0.08273935017872049
- ess_frac: 0.2: poison=0.03414591281965735, helpful=0.08459900299464554, 0.3: poison=0.03414591281965735, helpful=0.08459900299464554, 0.5: poison=0.03414591281965735, helpful=0.08459900299464554
- propensity_source: confounded_Q: poison=0.03414591281965735, helpful=0.08459900299464554, prior_only_Q0: poison=0.030209457041748156, helpful=0.07929459581749075
- estimator: snipw: poison=0.03414591281965735, helpful=0.08459900299464554, aipw: poison=0.0483117222151058, helpful=0.09056642278388966
- θ-sweep: outcome-Q separating θ = `None`, causal-ATE separating θ = `None`.
