# Lab analysis — `results/lab/campaign_14b_malig/merged`

headline condition **outcome**, 4264 runs, lessons/label = {'helpful': 41, 'poison': 41, 'distractor': 4}.

> ⚠ **<20 lessons/label** → label-level bootstrap CIs are anti-conservative (pilot scale). Per-lesson CIs are the trustworthy current evidence; the label-level generalization claim awaits the scaled corpus.

## H2 — observational ATE sign-flip

| estimator | label | mean ATE | 95% CI | p | excl 0 |
|---|---|---|---|---|---|
| SNIPW | helpful | +0.476 | [+0.242, +0.725] | 0.000 | True |
| SNIPW | poison | +0.142 | [+0.003, +0.336] | 0.041 | True |
| AIPW | helpful | +0.467 | [+0.232, +0.717] | 0.000 | True |
| AIPW | poison | +0.155 | [+0.006, +0.354] | 0.037 | True |

**SNIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: True, non-overlapping: False).

**AIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: True, non-overlapping: False).

## H3 — promotion gap (promote-if value>0)

| label | outcome-Q rate | causal-ATE rate | gap | 95% CI | p |
|---|---|---|---|---|---|
| helpful | 1.00 | 0.86 | +0.14 | [+0.00, +0.36] | 0.224 |
| poison | 1.00 | 0.43 | +0.57 | [+0.35, +0.78] | 0.000 |
## Robustness (H2 sign-flip must survive)

- propensity MC seeds×n_mc: poison ATE<0 in **0%**, true>0 in **100%** of 20 cells.
- clip: 0.02: poison=0.1415060888426815, helpful=0.47624333117463163, 0.05: poison=0.14150611940162613, helpful=0.47624335283022745, 0.1: poison=0.14172277067653968, helpful=0.48281036837924113
- ess_frac: 0.2: poison=0.14150611940162613, helpful=0.47624335283022745, 0.3: poison=0.14150611940162613, helpful=0.47624335283022745, 0.5: poison=0.14150611940162613, helpful=0.47624335283022745
- propensity_source: confounded_Q: poison=0.14150611940162613, helpful=0.47624335283022745, prior_only_Q0: poison=0.13739651060273297, helpful=0.4784244859533426
- estimator: snipw: poison=0.14150611940162613, helpful=0.47624335283022745, aipw: poison=0.15495410841813137, helpful=0.4673884978846566
- θ-sweep: outcome-Q separating θ = `None`, causal-ATE separating θ = `None`.
