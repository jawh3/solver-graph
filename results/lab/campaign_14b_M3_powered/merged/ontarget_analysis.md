# ON-TARGET (family-conditioned) causal credit — results/lab/campaign_14b_M3_powered/merged  cond=outcome  n_runs=15622

poison in-context solve: on-target=0.18774243599689683 (n=1289) vs off-target=0.21841440303221732 (n=12664)

## response=outcome
  MARGINAL  helpful  ATE=+0.0602 CI[+0.0338,+0.0914] p=0.000 n=40 excl0=True
  ON-TARGET helpful  ATE=+0.5260 CI[+0.4641,+0.5819] p=0.000 n=40 excl0=True
  MARGINAL  poison   ATE=+0.0044 CI[-0.0186,+0.0321] p=0.786 n=39 excl0=False
  ON-TARGET poison   ATE=-0.0569 CI[-0.0873,-0.0216] p=0.001 n=36 excl0=True

## response=R_final
  MARGINAL  helpful  ATE=+0.1283 CI[+0.0706,+0.1962] p=0.000 n=40 excl0=True
  ON-TARGET helpful  ATE=+1.1175 CI[+0.9749,+1.2455] p=0.000 n=40 excl0=True
  MARGINAL  poison   ATE=+0.0021 CI[-0.0398,+0.0553] p=0.980 n=39 excl0=False
  ON-TARGET poison   ATE=-0.1195 CI[-0.1796,-0.0457] p=0.001 n=36 excl0=True
