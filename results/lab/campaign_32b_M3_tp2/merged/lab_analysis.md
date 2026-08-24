# Lab analysis — `results/lab/campaign_32b_M3_tp2/merged`

headline condition **outcome**, 15480 runs, lessons/label = {'helpful': 43, 'poison': 43, 'distractor': 4}.

> ⚠ **<20 lessons/label** → label-level bootstrap CIs are anti-conservative (pilot scale). Per-lesson CIs are the trustworthy current evidence; the label-level generalization claim awaits the scaled corpus.

## H2 — observational ATE sign-flip

| estimator | label | mean ATE | 95% CI | p | excl 0 |
|---|---|---|---|---|---|
| SNIPW | helpful | +0.107 | [+0.051, +0.165] | 0.000 | True |
| SNIPW | poison | -0.005 | [-0.045, +0.039] | 0.804 | False |
| AIPW | helpful | +0.108 | [+0.053, +0.166] | 0.000 | True |
| AIPW | poison | -0.005 | [-0.044, +0.037] | 0.824 | False |

**SNIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: True, non-overlapping: True).

**AIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: True, non-overlapping: True).

## H3 — promotion gap (promote-if value>0)

| label | outcome-Q rate | causal-ATE rate | gap | 95% CI | p |
|---|---|---|---|---|---|
| helpful | 0.98 | 0.69 | +0.29 | [+0.14, +0.43] | 0.000 |
| poison | 0.90 | 0.36 | +0.55 | [+0.40, +0.69] | 0.000 |
## Robustness (H2 sign-flip must survive)

- propensity MC seeds×n_mc: poison ATE<0 in **100%**, true>0 in **100%** of 20 cells.
- clip: 0.02: poison=-0.005290073794853631, helpful=0.10675278336475691, 0.05: poison=-0.005283277634355236, helpful=0.10672745310621606, 0.1: poison=-0.005719062886955991, helpful=0.10834814632703522
- ess_frac: 0.2: poison=-0.005283277634355236, helpful=0.10672745310621606, 0.3: poison=-0.005283277634355236, helpful=0.10672745310621606, 0.5: poison=-0.005283277634355236, helpful=0.10672745310621606
- propensity_source: confounded_Q: poison=-0.005283277634355236, helpful=0.10672745310621606, prior_only_Q0: poison=-0.0058423407943085925, helpful=0.10519332255759684
- estimator: snipw: poison=-0.005283277634355236, helpful=0.10672745310621606, aipw: poison=-0.004566403086591826, helpful=0.10797305015985381
- θ-sweep: outcome-Q separating θ = `None`, causal-ATE separating θ = `None`.
