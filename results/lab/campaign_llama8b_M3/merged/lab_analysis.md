# Lab analysis — `results/lab/campaign_llama8b_M3/merged`

headline condition **outcome**, 3960 runs, lessons/label = {'helpful': 11, 'poison': 11, 'distractor': 4}.

> ⚠ **<20 lessons/label** → label-level bootstrap CIs are anti-conservative (pilot scale). Per-lesson CIs are the trustworthy current evidence; the label-level generalization claim awaits the scaled corpus.

## H2 — observational ATE sign-flip

| estimator | label | mean ATE | 95% CI | p | excl 0 |
|---|---|---|---|---|---|
| SNIPW | helpful | +0.018 | [-0.011, +0.061] | 0.419 | False |
| SNIPW | poison | -0.020 | [-0.058, +0.021] | 0.320 | False |
| SNIPW | distractor | -0.001 | [-0.039, +0.023] | 0.950 | False |
| AIPW | helpful | +0.021 | [-0.008, +0.064] | 0.249 | False |
| AIPW | poison | -0.022 | [-0.062, +0.018] | 0.284 | False |
| AIPW | distractor | +0.041 | [-0.033, +0.097] | 0.283 | False |

**SNIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: False, non-overlapping: False).

**AIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: False, non-overlapping: False).

## H3 — promotion gap (promote-if value>0)

| label | outcome-Q rate | causal-ATE rate | gap | 95% CI | p |
|---|---|---|---|---|---|
| helpful | 1.00 | 0.45 | +0.55 | [+0.27, +0.82] | 0.000 |
| poison | 1.00 | 0.27 | +0.73 | [+0.45, +1.00] | 0.000 |
## Robustness (H2 sign-flip must survive)

- propensity MC seeds×n_mc: poison ATE<0 in **100%**, true>0 in **100%** of 20 cells.
- clip: 0.02: poison=-0.020307571445278436, helpful=0.017769531191878532, 0.05: poison=-0.020307571445278436, helpful=0.01776954047023938, 0.1: poison=-0.01992180774750787, helpful=0.018211474949994377
- ess_frac: 0.2: poison=-0.020307571445278436, helpful=0.01776954047023938, 0.3: poison=-0.020307571445278436, helpful=0.01776954047023938, 0.5: poison=-0.020307571445278436, helpful=0.01776954047023938
- propensity_source: confounded_Q: poison=-0.020307571445278436, helpful=0.01776954047023938, prior_only_Q0: poison=-0.019508526929146574, helpful=0.016741390918853592
- estimator: snipw: poison=-0.020307571445278436, helpful=0.01776954047023938, aipw: poison=-0.022324316582465555, helpful=0.02148975567657334
- θ-sweep: outcome-Q separating θ = `None`, causal-ATE separating θ = `None`.
