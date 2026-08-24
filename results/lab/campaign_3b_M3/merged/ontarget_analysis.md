# ON-TARGET (family-conditioned) causal credit — results/lab/campaign_3b_M3/merged  cond=outcome  n_runs=2208

poison in-context solve: on-target=0.07335907335907337 (n=259) vs off-target=0.07108433734939759 (n=1660)

## response=outcome
  MARGINAL  helpful  ATE=+0.0094 CI[-0.0058,+0.0254] p=0.255 n=12 excl0=False
  ON-TARGET helpful  ATE=+0.2474 CI[+0.1182,+0.3896] p=0.000 n=11 excl0=True
  MARGINAL  poison   ATE=-0.0112 CI[-0.0355,+0.0166] p=0.447 n=12 excl0=False
  ON-TARGET poison   ATE=+0.0210 CI[-0.0577,+0.1140] p=0.682 n=12 excl0=False

## response=R_final
  MARGINAL  helpful  ATE=+0.0213 CI[-0.0065,+0.0494] p=0.146 n=12 excl0=False
  ON-TARGET helpful  ATE=+0.6041 CI[+0.2996,+0.9538] p=0.000 n=11 excl0=True
  MARGINAL  poison   ATE=-0.0250 CI[-0.0824,+0.0385] p=0.457 n=12 excl0=False
  ON-TARGET poison   ATE=+0.0852 CI[-0.1264,+0.3423] p=0.506 n=12 excl0=False
