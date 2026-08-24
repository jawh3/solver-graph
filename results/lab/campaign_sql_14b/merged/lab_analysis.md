# Lab analysis — `results/lab/campaign_sql_14b/merged`

headline condition **outcome**, 1664 runs, lessons/label = {'helpful': 16, 'poison': 16}.

> ⚠ **<20 lessons/label** → label-level bootstrap CIs are anti-conservative (pilot scale). Per-lesson CIs are the trustworthy current evidence; the label-level generalization claim awaits the scaled corpus.

## H2 — observational ATE sign-flip

| estimator | label | mean ATE | 95% CI | p | excl 0 |
|---|---|---|---|---|---|
| SNIPW | helpful | +0.111 | [+0.025, +0.204] | 0.009 | True |
| SNIPW | poison | -0.081 | [-0.144, -0.016] | 0.015 | True |
| AIPW | helpful | +0.123 | [+0.033, +0.218] | 0.004 | True |
| AIPW | poison | -0.085 | [-0.150, -0.019] | 0.010 | True |

**SNIPW verdict:** sign-flip = `True` (poison CI<0: True, helpful CI>0: True, non-overlapping: True).

**AIPW verdict:** sign-flip = `True` (poison CI<0: True, helpful CI>0: True, non-overlapping: True).

## H3 — promotion gap (promote-if value>0)

| label | outcome-Q rate | causal-ATE rate | gap | 95% CI | p |
|---|---|---|---|---|---|
| helpful | 1.00 | 0.69 | +0.31 | [+0.12, +0.56] | 0.004 |
| poison | 1.00 | 0.31 | +0.69 | [+0.44, +0.88] | 0.000 |
## Robustness (H2 sign-flip must survive)

- propensity MC seeds×n_mc: poison ATE<0 in **100%**, true>0 in **100%** of 20 cells.
- clip: 0.02: poison=-0.08143223048376166, helpful=0.11064074909075014, 0.05: poison=-0.0813984768376975, helpful=0.11064115175358755, 0.1: poison=-0.08223602416837447, helpful=0.11135474234454348
- ess_frac: 0.2: poison=-0.0813984768376975, helpful=0.11064115175358755, 0.3: poison=-0.0813984768376975, helpful=0.11064115175358755, 0.5: poison=-0.0813984768376975, helpful=0.11064115175358755
- propensity_source: confounded_Q: poison=-0.0813984768376975, helpful=0.11064115175358755, prior_only_Q0: poison=-0.0825565353609033, helpful=0.1104370791362495
- estimator: snipw: poison=-0.0813984768376975, helpful=0.11064115175358755, aipw: poison=-0.08527583302553912, helpful=0.12278049263692101
- θ-sweep: outcome-Q separating θ = `None`, causal-ATE separating θ = `None`.
