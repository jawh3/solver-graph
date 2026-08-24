# ON-TARGET (family-conditioned) causal credit — results/lab/campaign_14b_malig_expand/merged  cond=outcome  n_runs=5096

poison in-context solve: on-target=0.03968253968253968 (n=126) vs off-target=0.036390101892285295 (n=687)

## response=outcome
  MARGINAL  helpful          ATE=+0.1040 CI[+0.0536,+0.1602] p=0.000 n=18 excl0=True
  ON-TARGET helpful          ATE=+0.1442 CI[+0.0806,+0.2051] p=0.000 n=15 excl0=True
  MARGINAL  poison           ATE=+0.0152 CI[-0.0128,+0.0461] p=0.317 n=17 excl0=False
  ON-TARGET poison           ATE=-0.0015 CI[-0.0448,+0.0554] p=0.876 n=11 excl0=False
  MARGINAL  malignant_destroy ATE=-0.0037 CI[-0.0164,+0.0101] p=0.575 n=47 excl0=False
  ON-TARGET malignant_destroy ATE=-0.0122 CI[-0.0242,-0.0002] p=0.043 n=40 excl0=True
  MARGINAL  malignant_exhaust ATE=-0.0042 CI[-0.0091,+0.0008] p=0.095 n=49 excl0=False
  ON-TARGET malignant_exhaust ATE=-0.0156 CI[-0.0286,-0.0032] p=0.016 n=39 excl0=True
  MARGINAL  malignant_wrongflag ATE=+0.0047 CI[-0.0072,+0.0201] p=0.517 n=44 excl0=False
  ON-TARGET malignant_wrongflag ATE=+0.0237 CI[-0.0048,+0.0582] p=0.118 n=33 excl0=False

## response=R_final
  MARGINAL  helpful          ATE=+0.2418 CI[+0.1312,+0.3741] p=0.000 n=18 excl0=True
  ON-TARGET helpful          ATE=+0.3156 CI[+0.1809,+0.4676] p=0.000 n=15 excl0=True
  MARGINAL  poison           ATE=+0.0263 CI[-0.0220,+0.0815] p=0.341 n=17 excl0=False
  ON-TARGET poison           ATE=+0.0424 CI[-0.0428,+0.1691] p=0.558 n=11 excl0=False
  MARGINAL  malignant_destroy ATE=-0.0152 CI[-0.0364,+0.0066] p=0.176 n=47 excl0=False
  ON-TARGET malignant_destroy ATE=-0.0209 CI[-0.0443,+0.0004] p=0.055 n=40 excl0=False
  MARGINAL  malignant_exhaust ATE=-0.0073 CI[-0.0169,+0.0030] p=0.160 n=49 excl0=False
  ON-TARGET malignant_exhaust ATE=-0.0298 CI[-0.0563,-0.0051] p=0.018 n=39 excl0=True
  MARGINAL  malignant_wrongflag ATE=+0.0108 CI[-0.0110,+0.0366] p=0.371 n=44 excl0=False
  ON-TARGET malignant_wrongflag ATE=+0.0366 CI[-0.0111,+0.0982] p=0.150 n=33 excl0=False
