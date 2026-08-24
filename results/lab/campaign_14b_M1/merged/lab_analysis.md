# Lab analysis — `results/lab/campaign_14b_M1/merged`

headline condition **outcome**, 4264 runs, lessons/label = {'helpful': 41, 'poison': 41, 'distractor': 4}.

> ⚠ **<20 lessons/label** → label-level bootstrap CIs are anti-conservative (pilot scale). Per-lesson CIs are the trustworthy current evidence; the label-level generalization claim awaits the scaled corpus.

## H2 — observational ATE sign-flip

| estimator | label | mean ATE | 95% CI | p | excl 0 |
|---|---|---|---|---|---|
| SNIPW | helpful | +0.184 | [+0.047, +0.352] | 0.003 | True |
| SNIPW | poison | -0.039 | [-0.114, +0.046] | 0.330 | False |
| AIPW | helpful | +0.122 | [+0.007, +0.257] | 0.034 | True |
| AIPW | poison | -0.027 | [-0.101, +0.057] | 0.491 | False |

**SNIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: True, non-overlapping: True).

**AIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: True, non-overlapping: False).

## H3 — promotion gap (promote-if value>0)

| label | outcome-Q rate | causal-ATE rate | gap | 95% CI | p |
|---|---|---|---|---|---|
| helpful | 1.00 | 0.54 | +0.46 | [+0.31, +0.62] | 0.000 |
| poison | 1.00 | 0.31 | +0.69 | [+0.53, +0.83] | 0.000 |
## Robustness (H2 sign-flip must survive)

- propensity MC seeds×n_mc: poison ATE<0 in **100%**, true>0 in **100%** of 20 cells.
- clip: 0.02: poison=-0.038635802228941224, helpful=0.17488741812423414, 0.05: poison=-0.03941399890889728, helpful=0.18441342626525442, 0.1: poison=-0.03959010906751081, helpful=0.1849172432090401
- ess_frac: 0.2: poison=-0.03941399890889728, helpful=0.18441342626525442, 0.3: poison=-0.03941399890889728, helpful=0.18441342626525442, 0.5: poison=-0.03941399890889728, helpful=0.18441342626525442
- propensity_source: confounded_Q: poison=-0.03941399890889728, helpful=0.18441342626525442, prior_only_Q0: poison=-0.03934936742602855, helpful=0.1844284568033392
- estimator: snipw: poison=-0.03941399890889728, helpful=0.18441342626525442, aipw: poison=-0.02678276308904909, helpful=0.12159305186617135
- θ-sweep: outcome-Q separating θ = `None`, causal-ATE separating θ = `None`.
