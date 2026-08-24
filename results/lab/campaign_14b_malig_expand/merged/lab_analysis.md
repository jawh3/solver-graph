# Lab analysis — `results/lab/campaign_14b_malig_expand/merged`

headline condition **outcome**, 5096 runs, lessons/label = {'helpful': 49, 'poison': 49, 'distractor': 4}.

> ⚠ **<20 lessons/label** → label-level bootstrap CIs are anti-conservative (pilot scale). Per-lesson CIs are the trustworthy current evidence; the label-level generalization claim awaits the scaled corpus.

## H2 — observational ATE sign-flip

| estimator | label | mean ATE | 95% CI | p | excl 0 |
|---|---|---|---|---|---|
| SNIPW | helpful | +0.242 | [+0.128, +0.374] | 0.000 | True |
| SNIPW | poison | +0.026 | [-0.022, +0.083] | 0.329 | False |
| AIPW | helpful | +0.256 | [+0.129, +0.414] | 0.000 | True |
| AIPW | poison | +0.046 | [-0.014, +0.118] | 0.149 | False |

**SNIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: True, non-overlapping: True).

**AIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: True, non-overlapping: True).

## H3 — promotion gap (promote-if value>0)

| label | outcome-Q rate | causal-ATE rate | gap | 95% CI | p |
|---|---|---|---|---|---|
| helpful | 1.00 | 0.78 | +0.22 | [+0.06, +0.44] | 0.021 |
| poison | 1.00 | 0.47 | +0.53 | [+0.29, +0.76] | 0.000 |
## Robustness (H2 sign-flip must survive)

- propensity MC seeds×n_mc: poison ATE<0 in **0%**, true>0 in **100%** of 20 cells.
- clip: 0.02: poison=0.02629205782469531, helpful=0.2417810772261038, 0.05: poison=0.026292059502086424, helpful=0.2417810995971687, 0.1: poison=0.027641356056273464, helpful=0.24292068907232522
- ess_frac: 0.2: poison=0.026292059502086424, helpful=0.2417810995971687, 0.3: poison=0.026292059502086424, helpful=0.2417810995971687, 0.5: poison=0.026292059502086424, helpful=0.2417810995971687
- propensity_source: confounded_Q: poison=0.026292059502086424, helpful=0.2417810995971687, prior_only_Q0: poison=0.02871446437621075, helpful=0.24262496643961098
- estimator: snipw: poison=0.026292059502086424, helpful=0.2417810995971687, aipw: poison=0.04581868544378337, helpful=0.25561393905281843
- θ-sweep: outcome-Q separating θ = `None`, causal-ATE separating θ = `None`.
