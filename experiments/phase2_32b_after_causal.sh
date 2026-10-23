#!/bin/bash
# experiments/phase2_32b_after_causal.sh - after the 32B causal arm finishes: the leave-one-out factorial at 32B on the same
# 22 families (campaign solver config, filler + cross-outer cells, run-level log), then the within-poison rank test at 32B.
# Post-freeze addition (recorded as a deviation); the 32B server on :8004 stays up (the keeper keeps it alive).
# Record of how the reported runs were sequenced (it uses the shipped result folders); to run the campaign again use
# phase2_launch.sh with a fresh ROOT (RESULTS.md §6).
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} PYTHONPATH="$REPO" TOKENIZERS_PARALLELISM=false
M=Qwen/Qwen2.5-32B-Instruct-AWQ; U=http://127.0.0.1:8004/v1; ROOT=results/lab/phase2_32b
say(){ echo "[after32b $(date '+%m-%d %H:%M:%S')] $*" | tee -a logs/phase2_32b_after_causal.log; }
say "waiting for the causal arm"
until $PYBIN -c "import json,sys; p=json.load(open('$ROOT/arms/deployed_causal_alphabet/shard_00/progress.json')); sys.exit(0 if all(c.get('done') for c in p['conditions'].values()) else 1)" 2>/dev/null; do sleep 600; done
say "causal arm done; LOO factorial at 32B (campaign config, 14 cells)"
curl -s --max-time 20 "$U/models" | grep -q '"id"' || { say "32B server not up - abort"; exit 1; }
MODEL_NAME=$M VLLM_URL=$U $PYBIN -m experiments.constructed_loo --families-file $ROOT/families.json --n-inst 8 --seeds 1 --slack 2 \
  --workers 24 --workroot $ROOT/loo_work --out $ROOT/constructed_loo_campaigncfg.json --resume \
  --max-output-tokens 512 --context-window 8192 --trim-fraction 0.7 >> logs/phase2_32b_after_causal.log 2>&1 && say "LOO done" || say "LOO returned nonzero"
$PYBIN -m experiments.loo_report --families $ROOT/families.json --loo $ROOT/constructed_loo_campaigncfg.json --out docs/phase2_health/LOO_32B_PER_FAMILY.md >> logs/phase2_32b_after_causal.log 2>&1
$PYBIN -m experiments.phase2_within_poison --root $ROOT --arms deployed_alphabet_tau01,deployed_alphabet_tau01_rep1,deployed_alphabet_tau01_rep2 --loo $ROOT/constructed_loo_campaigncfg.json --out docs/phase2_analysis/WITHIN_POISON_32B.md >> logs/phase2_32b_after_causal.log 2>&1
say "32B post-causal analyses complete"
