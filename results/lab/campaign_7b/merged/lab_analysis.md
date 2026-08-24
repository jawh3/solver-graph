# Lab analysis — `results/lab/campaign_7b/merged`

headline condition **outcome**, 4400 runs, lessons/label = {'helpful': 11, 'poison': 11, 'distractor': 4}.

> ⚠ **<20 lessons/label** → label-level bootstrap CIs are anti-conservative (pilot scale). Per-lesson CIs are the trustworthy current evidence; the label-level generalization claim awaits the scaled corpus.

## H2 — observational ATE sign-flip

| estimator | label | mean ATE | 95% CI | p | excl 0 |
|---|---|---|---|---|---|
| SNIPW | helpful | +0.014 | [-0.017, +0.045] | 0.378 | False |
| SNIPW | poison | -0.011 | [-0.055, +0.030] | 0.649 | False |
| SNIPW | distractor | -0.037 | [-0.121, +0.048] | 0.416 | False |
| AIPW | helpful | +0.016 | [-0.015, +0.047] | 0.343 | False |
| AIPW | poison | -0.012 | [-0.057, +0.026] | 0.616 | False |
| AIPW | distractor | +0.003 | [-0.110, +0.108] | 0.863 | False |

**SNIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: False, non-overlapping: False).

**AIPW verdict:** sign-flip = `False` (poison CI<0: False, helpful CI>0: False, non-overlapping: False).

## H3 — promotion gap (promote-if value>0)

| label | outcome-Q rate | causal-ATE rate | gap | 95% CI | p |
|---|---|---|---|---|---|
| helpful | 1.00 | 0.64 | +0.36 | [+0.09, +0.64] | 0.014 |
| poison | 0.91 | 0.55 | +0.36 | [+0.09, +0.64] | 0.015 |

## H4 — power / MDE

observed paired SD = 0.468, n_pairs = 110 → **MDE = 12.5 pp** at 80% power. n for 5pp = 689, for 10pp = 173.

## Robustness (H2 sign-flip must survive)

- propensity MC seeds×n_mc: poison ATE<0 in **100%**, true>0 in **100%** of 20 cells.
- clip: 0.02: poison=-0.010897401203541107, helpful=0.014354459361078991, 0.05: poison=-0.010897401203541107, helpful=0.014354459361078991, 0.1: poison=-0.010897401203541107, helpful=0.014354459361078991
- ess_frac: 0.2: poison=-0.010897401203541107, helpful=0.014354459361078991, 0.3: poison=-0.010897401203541107, helpful=0.014354459361078991, 0.5: poison=-0.010897401203541107, helpful=0.014354459361078991
- propensity_source: confounded_Q: poison=-0.010897401203541107, helpful=0.014354459361078991, prior_only_Q0: poison=-0.012802668116398075, helpful=0.01577258063268173
- estimator: snipw: poison=-0.010897401203541107, helpful=0.014354459361078991, aipw: poison=-0.012018957157467173, helpful=0.01596336878029735
- θ-sweep: outcome-Q separating θ = `None`, causal-ATE separating θ = `None`.
