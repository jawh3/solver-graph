# ON-TARGET (family-conditioned) causal credit — results/lab/campaign_14b_M1/merged  cond=outcome  n_runs=4264

poison in-context solve: on-target=0.008928571428571428 (n=112) vs off-target=0.09955643173977329 (n=2029)

## response=outcome
  MARGINAL  helpful          ATE=+0.0686 CI[+0.0174,+0.1333] p=0.007 n=39 excl0=True
  ON-TARGET helpful          ATE=+0.7468 CI[+0.6155,+0.8600] p=0.000 n=15 excl0=True
  MARGINAL  poison           ATE=-0.0213 CI[-0.0533,+0.0156] p=0.253 n=36 excl0=False
  ON-TARGET poison           ATE=-0.0745 CI[-0.1254,-0.0333] p=0.000 n=9 excl0=True

## response=R_final
  MARGINAL  helpful          ATE=+0.1844 CI[+0.0486,+0.3546] p=0.004 n=39 excl0=True
  ON-TARGET helpful          ATE=+1.8942 CI[+1.4403,+2.3108] p=0.000 n=15 excl0=True
  MARGINAL  poison           ATE=-0.0394 CI[-0.1127,+0.0427] p=0.334 n=36 excl0=False
  ON-TARGET poison           ATE=-0.1794 CI[-0.3237,-0.0704] p=0.000 n=9 excl0=True
