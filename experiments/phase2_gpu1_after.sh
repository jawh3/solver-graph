#!/bin/bash
# experiments/phase2_gpu1_after.sh - GPU1 work while the causal arm runs on GPU0:
#   (1) the 2x2 LOO factorial on the 22 frozen families (+ malignant arms) - constructed_loo.py, 8 inst x 1 seed
#   (2) the slack sign-replication on the EVAL-ONLY arms: B0, B2det, oracle_static, worst_static at slack 1 and 3
#       (slack 2 = the main campaign), same families/items/seeds, 8 processes on GPU1's server.
# Requires the 14B server on GPU1:8003 (STAGE=server GPU=1 PORT=8003 of phase2_launch.sh).
# Record of how the reported runs were sequenced (it uses the shipped result folders); to run the campaign again use
# phase2_launch.sh with a fresh ROOT (RESULTS.md §6).
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} PYTHONPATH="$REPO" TOKENIZERS_PARALLELISM=false
M=Qwen/Qwen2.5-14B-Instruct-AWQ; U=http://127.0.0.1:8003/v1; ROOT=results/lab/phase2_14b
say(){ echo "[gpu1 $(date '+%m-%d %H:%M:%S')] $*" | tee -a logs/phase2_gpu1_after.log; }
curl -s --max-time 20 "$U/models" | grep -q '"id"' || { say "GPU1 server not up"; exit 1; }
FAMS=$(cat $ROOT/families.txt)
COMMON="--families $FAMS --epochs 2 --eval-seeds 2 --n-per-family 108 --M 3 --n-clusters 4 --tau 0.1 --epsilon-explore 0.25 \
  --tau-explore 1.0 --exploration-design stratified --citation-mode distinctive --query-diagnostics charset --recall-all \
  --prior-source none --credit-mode cited_only --init-mode zero --alpha-schedule mean --distance-scaling none \
  --context-window 8192 --max-output-tokens 512 --trim-fraction 0.7 --world-seed 424242"
say "LOO 2x2 factorial (+ malignant arms) on the 22 families"
MODEL_NAME=$M VLLM_URL=$U $PYBIN -m experiments.constructed_loo --families-file $ROOT/families.json --n-inst 8 --seeds 1 --slack 2 \
  --workers 24 --workroot $ROOT/loo_work --out $ROOT/constructed_loo.json >> logs/phase2_gpu1_after.log 2>&1 && say "LOO done" || say "LOO returned nonzero"
say "slack sign replication: eval-only arms at slack 1 and 3"
for SL in 1 3; do for cond in B0 B2det oracle_static worst_static; do
  out=$ROOT/slack$SL/$cond/shard_00; mkdir -p "$out"
  MODEL_NAME=$M VLLM_URL=$U nohup $PYBIN -m experiments.run_campaign --out "$out" --conditions $cond --context-key alphabet --slack $SL $COMMON \
    > $ROOT/slack$SL/$cond.log 2>&1 &
done; done
say "8 eval-only processes launched; waiting"
while pgrep -f "[r]un_campaign --out $ROOT/slack" >/dev/null; do sleep 120; done
say "slack sweep done -> $ROOT/slack{1,3}/<arm>/shard_00"
