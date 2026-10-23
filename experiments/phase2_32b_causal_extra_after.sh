#!/bin/bash
# experiments/phase2_32b_causal_extra_after.sh - P2-15: when the three added 32B causal arms (family context; alphabet
# replications 1/2 with independent draws) are done, run the standard analyses and commit the outputs. No stdout is needed;
# everything goes to logs/phase2_32b_causal_extra.log. Never edit while running.
# Record of how the reported runs were sequenced (it uses the shipped result folders); to run the campaign again use
# phase2_launch.sh with a fresh ROOT (RESULTS.md §6).
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} PYTHONPATH="$REPO" TOKENIZERS_PARALLELISM=false
ROOT=results/lab/phase2_32b; ARMS="deployed_causal_family deployed_causal_alphabet_rep1 deployed_causal_alphabet_rep2"
say(){ echo "[extra32b $(date '+%m-%d %H:%M:%S')] $*" >> logs/phase2_32b_causal_extra.log; }
say "waiting for: $ARMS"
while true; do
  n=0; for a in $ARMS; do $PYBIN -c "import json,sys; p=json.load(open('$ROOT/arms/$a/shard_00/progress.json')); sys.exit(0 if all(c.get('done') for c in p['conditions'].values()) else 1)" 2>/dev/null && n=$((n+1)); done
  [ $n -ge 3 ] && break; sleep 600
done
say "all three arms done; analyses"
D=$(date '+%Y-%m-%d')
$PYBIN -m experiments.phase2_ladder --root $ROOT --out docs/phase2_health/LADDER_32B_${D}_all.md >> logs/phase2_32b_causal_extra.log 2>&1 || say "ladder failed"
for a in $ARMS; do $PYBIN -m experiments.learning_curve --root $ROOT/arms/$a/shard_00 --cond deployed_causal --out docs/phase2_analysis/lc_32b/learning_curve_$a.json >> logs/phase2_32b_causal_extra.log 2>&1 || say "lc $a failed"; done
$PYBIN -m experiments.phase2_channels --root $ROOT --arms deployed_alphabet_tau01,deployed_family_tau01,deployed_causal_alphabet,deployed_causal_family,deployed_causal_alphabet_rep1,deployed_causal_alphabet_rep2 --out docs/phase2_analysis/CHANNELS_32B_causal_extra_${D}.md >> logs/phase2_32b_causal_extra.log 2>&1 || say "channels failed"
$PYBIN -m experiments.phase2_estimands --root $ROOT --arms deployed_alphabet_tau01,deployed_causal_alphabet,deployed_causal_alphabet_rep1,deployed_causal_alphabet_rep2,deployed_causal_family --primary deployed_alphabet_tau01 --pool deployed_causal_alphabet,deployed_causal_alphabet_rep1,deployed_causal_alphabet_rep2 --out docs/phase2_analysis/ESTIMANDS_P1_32B_causal_extra_${D}.md >> logs/phase2_32b_causal_extra.log 2>&1 || say "estimands failed"
$PYBIN -m experiments.compute_report --root $ROOT --snapshots "$ROOT/metrics/*.json" >> logs/phase2_32b_causal_extra.log 2>&1 && cp $ROOT/COMPUTE_REPORT.md docs/phase2_health/COMPUTE_32B_${D}_final.md
git add docs/phase2_health/LADDER_32B_${D}_all.md docs/phase2_health/LADDER_32B_${D}_all.json docs/phase2_analysis/lc_32b docs/phase2_analysis/CHANNELS_32B_causal_extra_${D}.md docs/phase2_analysis/CHANNELS_32B_causal_extra_${D}.json docs/phase2_analysis/ESTIMANDS_P1_32B_causal_extra_${D}.md docs/phase2_analysis/ESTIMANDS_P1_32B_causal_extra_${D}.json docs/phase2_health/COMPUTE_32B_${D}_final.md >> logs/phase2_32b_causal_extra.log 2>&1
git commit -q -m "Phase 2 (P2-15): 32B causal credit under the family context + two causal replications - automatic analyses" >> logs/phase2_32b_causal_extra.log 2>&1 || say "nothing committed"
say "DONE - outputs: LADDER_32B_${D}_all, CHANNELS_32B_causal_extra_${D}, ESTIMANDS_P1_32B_causal_extra_${D}, lc_32b/, COMPUTE_32B_${D}_final"
