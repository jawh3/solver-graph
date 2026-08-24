# ON-TARGET (family-conditioned) causal credit — results/lab/campaign_sql_14b/merged  cond=outcome  n_runs=1664

poison in-context solve: on-target=0.2642857142857143 (n=140) vs off-target=0.4599250936329588 (n=1335)

## response=outcome
  MARGINAL  helpful          ATE=+0.0925 CI[+0.0185,+0.1774] p=0.009 n=16 excl0=True
  ON-TARGET helpful          ATE=+0.5810 CI[+0.4647,+0.6994] p=0.000 n=12 excl0=True
  MARGINAL  poison           ATE=-0.0646 CI[-0.1165,-0.0132] p=0.011 n=16 excl0=True
  ON-TARGET poison           ATE=-0.2513 CI[-0.4094,-0.0949] p=0.003 n=8 excl0=True

## response=R_final
  MARGINAL  helpful          ATE=+0.1106 CI[+0.0281,+0.2062] p=0.007 n=16 excl0=True
  ON-TARGET helpful          ATE=+0.7646 CI[+0.6090,+0.9180] p=0.000 n=12 excl0=True
  MARGINAL  poison           ATE=-0.0814 CI[-0.1432,-0.0176] p=0.011 n=16 excl0=True
  ON-TARGET poison           ATE=-0.2883 CI[-0.4954,-0.0824] p=0.007 n=8 excl0=True
