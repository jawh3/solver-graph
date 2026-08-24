# Lab analysis — `results/lab/campaign_14b_M3_powered/merged`

headline condition **outcome**, 15622 runs, lessons/label = {'helpful': 41, 'poison': 41, 'distractor': 4}.

> ⚠ **<20 lessons/label** → label-level bootstrap CIs are anti-conservative (pilot scale). Per-lesson CIs are the trustworthy current evidence; the label-level generalization claim awaits the scaled corpus.

## H2 — observational ATE sign-flip

| estimator | label | mean ATE | 95% CI | p | excl 0 |
|---|---|---|---|---|---|
| SNIPW | helpful | +0.128 | [+0.071, +0.195] | 0.000 | True |
| SNIPW | poison | +0.002 | [-0.040, +0.059] | 0.981 | False |
| AIPW | helpful | +0.128 | [+0.073, +0.193] | 0.000 | True |
| AIPW | poison | -0.010 | [-0.042, +0.026] | 0.538 | False |

**SNIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: True, non-overlapping: True).

**AIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: True, non-overlapping: True).

## H3 — promotion gap (promote-if value>0)

| label | outcome-Q rate | causal-ATE rate | gap | 95% CI | p |
|---|---|---|---|---|---|
| helpful | 1.00 | 0.72 | +0.28 | [+0.15, +0.42] | 0.000 |
| poison | 0.92 | 0.33 | +0.59 | [+0.44, +0.74] | 0.000 |
## Robustness (H2 sign-flip must survive)

- propensity MC seeds×n_mc: poison ATE<0 in **45%**, true>0 in **100%** of 20 cells.
- clip: 0.02: poison=0.0021159370071485903, helpful=0.12827556670967272, 0.05: poison=0.00212308148096259, helpful=0.1283027507633811, 0.1: poison=0.0010434104032208253, helpful=0.12690440512192147
- ess_frac: 0.2: poison=0.00212308148096259, helpful=0.1283027507633811, 0.3: poison=0.00212308148096259, helpful=0.1283027507633811, 0.5: poison=0.00212308148096259, helpful=0.1283027507633811
- propensity_source: confounded_Q: poison=0.00212308148096259, helpful=0.1283027507633811, prior_only_Q0: poison=2.9256643711896228e-05, helpful=0.12570284253126288
- estimator: snipw: poison=0.00212308148096259, helpful=0.1283027507633811, aipw: poison=-0.010079327532809488, helpful=0.12827053308196387
- θ-sweep: outcome-Q separating θ = `None`, causal-ATE separating θ = `None`.
