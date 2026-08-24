# ON-TARGET (family-conditioned) causal credit — results/lab/campaign_7b35_M3_rep/merged  cond=outcome  n_runs=8050

poison in-context solve: on-target=0.08221476510067115 (n=596) vs off-target=0.10025824092359106 (n=6583)

## response=outcome
  MARGINAL  helpful  ATE=+0.0383 CI[+0.0063,+0.0774] p=0.011 n=35 excl0=True
  ON-TARGET helpful  ATE=+0.1951 CI[+0.1253,+0.2683] p=0.000 n=31 excl0=True
  MARGINAL  poison   ATE=+0.0103 CI[-0.0126,+0.0383] p=0.417 n=33 excl0=False
  ON-TARGET poison   ATE=-0.0050 CI[-0.0442,+0.0345] p=0.817 n=28 excl0=False

## response=R_final
  MARGINAL  helpful  ATE=+0.0846 CI[+0.0148,+0.1696] p=0.012 n=35 excl0=True
  ON-TARGET helpful  ATE=+0.4191 CI[+0.2627,+0.5823] p=0.000 n=31 excl0=True
  MARGINAL  poison   ATE=+0.0341 CI[-0.0191,+0.1003] p=0.242 n=33 excl0=False
  ON-TARGET poison   ATE=-0.0039 CI[-0.0907,+0.0870] p=0.938 n=28 excl0=False
