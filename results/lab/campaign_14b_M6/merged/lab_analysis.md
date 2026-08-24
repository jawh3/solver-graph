# Lab analysis — `results/lab/campaign_14b_M6/merged`

headline condition **outcome**, 4264 runs, lessons/label = {'helpful': 41, 'poison': 41, 'distractor': 4}.

> ⚠ **<20 lessons/label** → label-level bootstrap CIs are anti-conservative (pilot scale). Per-lesson CIs are the trustworthy current evidence; the label-level generalization claim awaits the scaled corpus.

## H2 — observational ATE sign-flip

| estimator | label | mean ATE | 95% CI | p | excl 0 |
|---|---|---|---|---|---|
| SNIPW | helpful | +0.086 | [+0.040, +0.140] | 0.000 | True |
| SNIPW | poison | -0.008 | [-0.045, +0.031] | 0.667 | False |
| AIPW | helpful | +0.085 | [+0.037, +0.139] | 0.000 | True |
| AIPW | poison | -0.011 | [-0.048, +0.026] | 0.536 | False |

**SNIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: True, non-overlapping: True).

**AIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: True, non-overlapping: True).

## H3 — promotion gap (promote-if value>0)

| label | outcome-Q rate | causal-ATE rate | gap | 95% CI | p |
|---|---|---|---|---|---|
| helpful | 1.00 | 0.68 | +0.33 | [+0.17, +0.47] | 0.000 |
| poison | 1.00 | 0.38 | +0.62 | [+0.46, +0.77] | 0.000 |
## Robustness (H2 sign-flip must survive)

- propensity MC seeds×n_mc: poison ATE<0 in **100%**, true>0 in **100%** of 20 cells.
- clip: 0.02: poison=-0.007558626674273516, helpful=0.08635630032872348, 0.05: poison=-0.007558626674273516, helpful=0.08635630032872348, 0.1: poison=-0.007558022127835771, helpful=0.08635627660319542
- ess_frac: 0.2: poison=-0.007558626674273516, helpful=0.08635630032872348, 0.3: poison=-0.007558626674273516, helpful=0.08635630032872348, 0.5: poison=-0.007558626674273516, helpful=0.08635630032872348
- propensity_source: confounded_Q: poison=-0.007558626674273516, helpful=0.08635630032872348, prior_only_Q0: poison=-0.010490660463270369, helpful=0.08352588181584518
- estimator: snipw: poison=-0.007558626674273516, helpful=0.08635630032872348, aipw: poison=-0.011113341215516909, helpful=0.08487679938075307
- θ-sweep: outcome-Q separating θ = `None`, causal-ATE separating θ = `None`.
