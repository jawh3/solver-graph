# Phase-2 held-out ladder - results/lab/phase2_32b - 24 complete arms

Paired held-out items: 1188 = 22 families × 27 items × 2 LLM seeds; unit = family; family-cluster bootstrap 95% CI (4,000 reps), family-level sign-permutation p (20,000), run-level exact McNemar (secondary).

| arm | solve | vs B2det: family-mean diff [CI] | up/down | p_sign | McNemar (a/b, p) |
|---|---|---|---|---|---|
| B0 | 0.027 | -0.086 [-0.171, +0.004] | 1/17 | 0.0948 | 32/134, 4.8e-16 |
| B2det | 0.113 | reference | | | |
| deployed_alphabet_tau01 | 0.364 | +0.252 [+0.165, +0.336] | 19/3 | 0.0000 | 373/74, 4.5e-49 |
| deployed_alphabet_tau01_rep1 | 0.355 | +0.242 [+0.162, +0.321] | 18/3 | 0.0001 | 360/72, 3.9e-47 |
| deployed_alphabet_tau01_rep2 | 0.348 | +0.236 [+0.146, +0.321] | 19/2 | 0.0001 | 354/74, 7.2e-45 |
| deployed_alphabet_tau02 | 0.235 | +0.122 [+0.050, +0.184] | 19/2 | 0.0037 | 242/97, 1.9e-15 |
| deployed_causal_alphabet | 0.455 | +0.343 [+0.247, +0.436] | 20/0 | 0.0000 | 460/53, 5.3e-82 |
| deployed_causal_alphabet_rep1 | 0.471 | +0.359 [+0.254, +0.462] | 18/2 | 0.0000 | 482/56, 1.5e-85 |
| deployed_causal_alphabet_rep2 | 0.459 | +0.346 [+0.233, +0.457] | 19/1 | 0.0000 | 469/58, 6.2e-81 |
| deployed_causal_family | 0.727 | +0.614 [+0.507, +0.711] | 20/0 | 0.0000 | 753/23, 4.2e-190 |
| deployed_cluster_tau01 | 0.347 | +0.234 [+0.141, +0.325] | 19/2 | 0.0001 | 359/81, 8.0e-43 |
| deployed_cluster_tau02 | 0.241 | +0.128 [+0.047, +0.200] | 19/2 | 0.0060 | 250/98, 1.9e-16 |
| deployed_family_tau01 | 0.661 | +0.548 [+0.438, +0.646] | 21/0 | 0.0000 | 690/39, 5.8e-155 |
| deployed_family_tau02 | 0.508 | +0.396 [+0.304, +0.478] | 18/0 | 0.0000 | 529/59, 1.9e-95 |
| deployed_global_tau01 | 0.106 | -0.007 [-0.077, +0.056] | 13/6 | 0.8669 | 114/122, 6.5e-01 |
| deployed_global_tau02 | 0.092 | -0.021 [-0.089, +0.040] | 12/8 | 0.5739 | 95/120, 1.0e-01 |
| deployed_obs_alphabet | 0.391 | +0.279 [+0.201, +0.365] | 21/0 | 0.0000 | 389/58, 3.0e-61 |
| oracle_static | 0.568 | +0.455 [+0.372, +0.534] | 21/0 | 0.0000 | 557/16, 3.5e-142 |
| outcome_legacy_alphabet | 0.093 | -0.019 [-0.088, +0.041] | 10/9 | 0.6122 | 96/119, 1.3e-01 |
| s0_lure_alphabet | 0.390 | +0.277 [+0.186, +0.370] | 18/2 | 0.0000 | 406/77, 5.2e-55 |
| s1_cross_outer_alphabet | 0.386 | +0.273 [+0.182, +0.361] | 18/2 | 0.0000 | 398/74, 1.1e-54 |
| s4_malignant_clean_alphabet | 0.259 | +0.146 [+0.055, +0.236] | 18/2 | 0.0060 | 268/94, 1.7e-20 |
| s4_malignant_legacy_alphabet | 0.269 | +0.156 [+0.061, +0.256] | 14/6 | 0.0056 | 267/82, 5.4e-24 |
| worst_static | 0.075 | -0.038 [-0.088, +0.002] | 4/13 | 0.1290 | 43/88, 1.0e-04 |

## Contrasts - 22 families and a nested 12-family subset (the first three families per alphabet)

| contrast | 22 families: diff [CI], up/down, p_sign | nested 12: diff [CI], up/down, p_sign |
|---|---|---|
| deployed_alphabet_tau01 - B2det | +0.252 [+0.165, +0.336], 19/3, p=0.0000 | +0.248 [+0.120, +0.372], 10/2, p=0.0060 |
| deployed_alphabet_tau01 - deployed_global_tau01 | +0.258 [+0.199, +0.311], 21/1, p=0.0000 | +0.238 [+0.144, +0.323], 11/1, p=0.0019 |
| deployed_family_tau01 - deployed_alphabet_tau01 | +0.296 [+0.217, +0.373], 19/3, p=0.0000 | +0.247 [+0.128, +0.367], 9/3, p=0.0048 |
| deployed_alphabet_tau01 - outcome_legacy_alphabet | +0.271 [+0.205, +0.331], 21/1, p=0.0000 | +0.247 [+0.139, +0.347], 11/1, p=0.0024 |
| deployed_alphabet_tau01 - deployed_alphabet_tau02 | +0.130 [+0.088, +0.169], 19/2, p=0.0001 | +0.114 [+0.054, +0.170], 11/1, p=0.0045 |
| deployed_cluster_tau01 - deployed_alphabet_tau01 | -0.018 [-0.055, +0.017], 10/12, p=0.3957 | +0.005 [-0.029, +0.039], 6/6, p=0.8640 |
| s0_lure_alphabet - deployed_alphabet_tau01 | +0.025 [-0.010, +0.063], 12/9, p=0.2097 | +0.029 [-0.020, +0.086], 5/6, p=0.3575 |
| s1_cross_outer_alphabet - deployed_alphabet_tau01 | +0.021 [-0.026, +0.064], 12/9, p=0.4012 | +0.026 [-0.022, +0.074], 8/4, p=0.3446 |
| deployed_global_tau01 - B2det | -0.007 [-0.077, +0.056], 13/6, p=0.8669 | +0.011 [-0.096, +0.085], 8/2, p=0.9033 |
| deployed_obs_alphabet - B2det | +0.279 [+0.201, +0.365], 21/0, p=0.0000 | +0.267 [+0.151, +0.397], 11/0, p=0.0010 |
| deployed_causal_alphabet - deployed_alphabet_tau01 | +0.091 [+0.024, +0.158], 14/8, p=0.0217 | +0.088 [+0.017, +0.167], 8/4, p=0.0466 |

Multiplicity: the four context arms form one ordering (global < alphabet ≤ family); the poison kinds are a dose-response, not separate tests; other rows are descriptive. Error-flagged runs (`llm_bad_request`) are counted as failures in every arm, as in progress.json.
