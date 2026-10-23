# Phase-2 held-out ladder - results/lab/phase2_14b - 21 complete arms

Paired held-out items: 1188 = 22 families × 27 items × 2 LLM seeds; unit = family; family-cluster bootstrap 95% CI (4,000 reps), family-level sign-permutation p (20,000), run-level exact McNemar (secondary).

| arm | solve | vs B2det: family-mean diff [CI] | up/down | p_sign | McNemar (a/b, p) |
|---|---|---|---|---|---|
| B0 | 0.015 | -0.039 [-0.073, -0.008] | 3/12 | 0.0324 | 17/63, 2.3e-07 |
| B2det | 0.054 | reference | | | |
| deployed_alphabet_tau01 | 0.186 | +0.132 [+0.061, +0.209] | 16/3 | 0.0014 | 209/52, 1.8e-23 |
| deployed_alphabet_tau01_rep1 | 0.166 | +0.112 [+0.053, +0.176] | 18/2 | 0.0014 | 183/50, 5.2e-19 |
| deployed_alphabet_tau01_rep2 | 0.191 | +0.137 [+0.072, +0.208] | 19/2 | 0.0003 | 215/52, 1.0e-24 |
| deployed_alphabet_tau02 | 0.086 | +0.032 [-0.004, +0.067] | 15/5 | 0.1187 | 96/58, 2.7e-03 |
| deployed_causal_alphabet | 0.177 | +0.123 [+0.050, +0.198] | 17/2 | 0.0049 | 196/50, 1.3e-21 |
| deployed_cluster_tau01 | 0.162 | +0.109 [+0.035, +0.186] | 15/3 | 0.0127 | 181/52, 6.7e-18 |
| deployed_cluster_tau02 | 0.086 | +0.032 [-0.004, +0.066] | 15/4 | 0.1200 | 97/59, 2.9e-03 |
| deployed_family_tau01 | 0.302 | +0.248 [+0.163, +0.343] | 20/1 | 0.0001 | 336/41, 9.9e-59 |
| deployed_family_tau02 | 0.205 | +0.151 [+0.088, +0.220] | 17/3 | 0.0001 | 219/40, 4.6e-31 |
| deployed_global_tau01 | 0.035 | -0.019 [-0.055, +0.011] | 7/8 | 0.3664 | 39/61, 3.5e-02 |
| deployed_global_tau02 | 0.039 | -0.015 [-0.054, +0.020] | 8/9 | 0.4897 | 44/62, 9.8e-02 |
| deployed_obs_alphabet | 0.042 | -0.012 [-0.043, +0.014] | 7/8 | 0.5101 | 45/59, 2.0e-01 |
| oracle_static | 0.274 | +0.220 [+0.165, +0.284] | 22/0 | 0.0000 | 279/18, 2.5e-61 |
| outcome_legacy_alphabet | 0.032 | -0.022 [-0.054, +0.004] | 9/8 | 0.2351 | 36/62, 1.1e-02 |
| s0_lure_alphabet | 0.213 | +0.159 [+0.086, +0.233] | 16/2 | 0.0001 | 237/48, 3.3e-31 |
| s1_cross_outer_alphabet | 0.148 | +0.094 [+0.038, +0.158] | 16/2 | 0.0038 | 160/48, 2.8e-15 |
| s4_malignant_clean_alphabet | 0.068 | +0.014 [-0.023, +0.056] | 9/5 | 0.5371 | 74/57, 1.6e-01 |
| s4_malignant_legacy_alphabet | 0.056 | +0.002 [-0.046, +0.050] | 11/8 | 0.9740 | 64/62, 9.3e-01 |
| worst_static | 0.034 | -0.020 [-0.040, -0.003] | 4/10 | 0.0549 | 31/55, 1.3e-02 |

## Contrasts - 22 families and a nested 12-family subset (the first three families per alphabet)

| contrast | 22 families: diff [CI], up/down, p_sign | nested 12: diff [CI], up/down, p_sign |
|---|---|---|
| deployed_alphabet_tau01 - B2det | +0.132 [+0.061, +0.209], 16/3, p=0.0014 | +0.117 [+0.022, +0.231], 7/2, p=0.0735 |
| deployed_alphabet_tau01 - deployed_global_tau01 | +0.151 [+0.087, +0.222], 17/2, p=0.0000 | +0.125 [+0.045, +0.219], 8/1, p=0.0115 |
| deployed_family_tau01 - deployed_alphabet_tau01 | +0.116 [+0.060, +0.176], 17/2, p=0.0005 | +0.159 [+0.080, +0.242], 10/0, p=0.0019 |
| deployed_alphabet_tau01 - outcome_legacy_alphabet | +0.154 [+0.090, +0.226], 17/2, p=0.0000 | +0.128 [+0.045, +0.228], 8/2, p=0.0129 |
| deployed_alphabet_tau01 - deployed_alphabet_tau02 | +0.100 [+0.042, +0.167], 13/4, p=0.0024 | +0.069 [+0.002, +0.159], 6/3, p=0.1340 |
| deployed_cluster_tau01 - deployed_alphabet_tau01 | -0.024 [-0.050, -0.001], 5/8, p=0.0969 | -0.012 [-0.048, +0.019], 3/3, p=0.5234 |
| s0_lure_alphabet - deployed_alphabet_tau01 | +0.027 [-0.005, +0.061], 11/7, p=0.1518 | +0.034 [-0.011, +0.086], 6/3, p=0.2655 |
| s1_cross_outer_alphabet - deployed_alphabet_tau01 | -0.038 [-0.109, +0.024], 10/9, p=0.2913 | +0.011 [-0.074, +0.071], 7/2, p=0.8504 |
| deployed_global_tau01 - B2det | -0.019 [-0.055, +0.011], 7/8, p=0.3664 | -0.008 [-0.065, +0.034], 4/3, p=0.9371 |
| deployed_obs_alphabet - B2det | -0.012 [-0.043, +0.014], 7/8, p=0.5101 | -0.003 [-0.052, +0.031], 4/2, p=0.9697 |
| deployed_causal_alphabet - deployed_alphabet_tau01 | -0.009 [-0.052, +0.032], 10/10, p=0.7008 | +0.000 [-0.063, +0.054], 7/3, p=1.0000 |

Multiplicity: the four context arms form one ordering (global < alphabet ≤ family); the poison kinds are a dose-response, not separate tests; other rows are descriptive. Error-flagged runs (`llm_bad_request`) are counted as failures in every arm, as in progress.json.
