# ON-TARGET (family-conditioned) causal credit — results/lab/campaign_14b_M6/merged  cond=outcome  n_runs=4264

poison in-context solve: on-target=0.33746130030959753 (n=646) vs off-target=0.35129068462401797 (n=3564)

## response=outcome
  MARGINAL  helpful          ATE=+0.0441 CI[+0.0166,+0.0760] p=0.000 n=40 excl0=True
  ON-TARGET helpful          ATE=+0.3738 CI[+0.2991,+0.4447] p=0.000 n=40 excl0=True
  MARGINAL  poison           ATE=-0.0019 CI[-0.0261,+0.0230] p=0.860 n=39 excl0=False
  ON-TARGET poison           ATE=-0.0537 CI[-0.1158,+0.0110] p=0.108 n=34 excl0=False

## response=R_final
  MARGINAL  helpful          ATE=+0.0864 CI[+0.0404,+0.1415] p=0.000 n=40 excl0=True
  ON-TARGET helpful          ATE=+0.7064 CI[+0.5694,+0.8390] p=0.000 n=40 excl0=True
  MARGINAL  poison           ATE=-0.0076 CI[-0.0436,+0.0309] p=0.693 n=39 excl0=False
  ON-TARGET poison           ATE=-0.1207 CI[-0.2169,-0.0161] p=0.021 n=34 excl0=True
