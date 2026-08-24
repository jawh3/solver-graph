# ON-TARGET (family-conditioned) causal credit — results/lab/campaign_32b_M3_tp2/merged  cond=outcome  n_runs=15480

poison in-context solve: on-target=0.2676923076923077 (n=1300) vs off-target=0.31856087681677386 (n=12591)

## response=outcome
  MARGINAL  helpful  ATE=+0.0667 CI[+0.0333,+0.1036] p=0.000 n=42 excl0=True
  ON-TARGET helpful  ATE=+0.5695 CI[+0.4913,+0.6462] p=0.000 n=42 excl0=True
  MARGINAL  poison   ATE=-0.0022 CI[-0.0250,+0.0233] p=0.857 n=42 excl0=False
  ON-TARGET poison   ATE=-0.0642 CI[-0.1094,-0.0180] p=0.005 n=36 excl0=True

## response=R_final
  MARGINAL  helpful  ATE=+0.1067 CI[+0.0513,+0.1675] p=0.000 n=42 excl0=True
  ON-TARGET helpful  ATE=+1.1302 CI[+0.9506,+1.3013] p=0.000 n=42 excl0=True
  MARGINAL  poison   ATE=-0.0053 CI[-0.0455,+0.0385] p=0.813 n=42 excl0=False
  ON-TARGET poison   ATE=-0.1063 CI[-0.1851,-0.0246] p=0.011 n=36 excl0=True
