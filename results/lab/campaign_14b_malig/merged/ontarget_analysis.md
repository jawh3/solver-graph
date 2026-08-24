# ON-TARGET (family-conditioned) causal credit — results/lab/campaign_14b_malig/merged  cond=outcome  n_runs=4264

poison in-context solve: on-target=0.11176470588235295 (n=170) vs off-target=0.06270096463022508 (n=622)

## response=outcome
  MARGINAL  helpful          ATE=+0.1851 CI[+0.1056,+0.2707] p=0.000 n=14 excl0=True
  ON-TARGET helpful          ATE=+0.2436 CI[+0.1459,+0.3510] p=0.000 n=14 excl0=True
  MARGINAL  poison           ATE=+0.0556 CI[+0.0003,+0.1350] p=0.050 n=23 excl0=True
  ON-TARGET poison           ATE=+0.0490 CI[+0.0029,+0.1007] p=0.037 n=16 excl0=True
  MARGINAL  malignant_destroy ATE=-0.0069 CI[-0.0182,+0.0045] p=0.247 n=38 excl0=False
  ON-TARGET malignant_destroy ATE=-0.0208 CI[-0.0416,+0.0002] p=0.053 n=34 excl0=False
  MARGINAL  malignant_exhaust ATE=-0.0081 CI[-0.0169,+0.0011] p=0.079 n=41 excl0=False
  ON-TARGET malignant_exhaust ATE=-0.0178 CI[-0.0441,+0.0068] p=0.154 n=32 excl0=False
  MARGINAL  malignant_wrongflag ATE=+0.0002 CI[-0.0175,+0.0196] p=0.967 n=37 excl0=False
  ON-TARGET malignant_wrongflag ATE=-0.0161 CI[-0.0398,+0.0073] p=0.189 n=31 excl0=False

## response=R_final
  MARGINAL  helpful          ATE=+0.4762 CI[+0.2439,+0.7322] p=0.000 n=14 excl0=True
  ON-TARGET helpful          ATE=+0.6087 CI[+0.3231,+0.9275] p=0.000 n=14 excl0=True
  MARGINAL  poison           ATE=+0.1415 CI[+0.0031,+0.3352] p=0.043 n=23 excl0=True
  ON-TARGET poison           ATE=+0.1187 CI[+0.0032,+0.2495] p=0.039 n=16 excl0=True
  MARGINAL  malignant_destroy ATE=-0.0204 CI[-0.0401,-0.0006] p=0.043 n=38 excl0=True
  ON-TARGET malignant_destroy ATE=-0.0507 CI[-0.0821,-0.0187] p=0.001 n=34 excl0=True
  MARGINAL  malignant_exhaust ATE=-0.0193 CI[-0.0362,-0.0027] p=0.023 n=41 excl0=True
  ON-TARGET malignant_exhaust ATE=-0.0403 CI[-0.0935,+0.0099] p=0.114 n=32 excl0=False
  MARGINAL  malignant_wrongflag ATE=-0.0021 CI[-0.0311,+0.0323] p=0.882 n=37 excl0=False
  ON-TARGET malignant_wrongflag ATE=-0.0262 CI[-0.0720,+0.0212] p=0.263 n=31 excl0=False
