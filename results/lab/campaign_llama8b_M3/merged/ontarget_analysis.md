# ON-TARGET (family-conditioned) causal credit — results/lab/campaign_llama8b_M3/merged  cond=outcome  n_runs=3960

poison in-context solve: on-target=0.07125307125307126 (n=407) vs off-target=0.0720136518771331 (n=2930)

## response=outcome
  MARGINAL  helpful  ATE=+0.0067 CI[-0.0072,+0.0269] p=0.545 n=11 excl0=False
  ON-TARGET helpful  ATE=+0.2275 CI[+0.1286,+0.3451] p=0.000 n=11 excl0=True
  MARGINAL  poison   ATE=-0.0076 CI[-0.0285,+0.0154] p=0.490 n=11 excl0=False
  ON-TARGET poison   ATE=-0.0033 CI[-0.0257,+0.0205] p=0.747 n=10 excl0=False

## response=R_final
  MARGINAL  helpful  ATE=+0.0178 CI[-0.0106,+0.0600] p=0.401 n=11 excl0=False
  ON-TARGET helpful  ATE=+0.5024 CI[+0.2839,+0.7644] p=0.000 n=11 excl0=True
  MARGINAL  poison   ATE=-0.0203 CI[-0.0585,+0.0213] p=0.321 n=11 excl0=False
  ON-TARGET poison   ATE=-0.0236 CI[-0.0656,+0.0226] p=0.286 n=10 excl0=False
