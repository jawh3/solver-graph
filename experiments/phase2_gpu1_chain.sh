#!/bin/bash
# experiments/phase2_gpu1_chain.sh - review 2026-09-08 F4/F5/F18: re-run the 14B leave-one-out factorial at the CAMPAIGN
# solver configuration (512 / 8192 / 0.7) with the run-level log and the new cells ([H, X_random], [H, X, X'], [P_cross],
# [H, P_cross]) on GPU0's server (idle since the 17 regular arms finished; the causal arm and the slack replication share
# GPU1); then hand over to the 32B starter (which itself waits for every 14B job). Never edit while running.
# Record of how the reported runs were sequenced (it uses the shipped result folders); to run the campaign again use
# phase2_launch.sh with a fresh ROOT (RESULTS.md §6).
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} PYTHONPATH="$REPO" TOKENIZERS_PARALLELISM=false
M=Qwen/Qwen2.5-14B-Instruct-AWQ; U=http://127.0.0.1:8002/v1; ROOT=results/lab/phase2_14b
say(){ echo "[chain $(date '+%m-%d %H:%M:%S')] $*" | tee -a logs/phase2_gpu1_chain.log; }
curl -s --max-time 20 "$U/models" | grep -q '"id"' || { say "GPU0 server not up; starting the 32B chain anyway"; exec bash experiments/phase2_start_32b.sh; }
say "LOO factorial at the campaign config (+ filler and cross-outer cells) on the 22 families"
MODEL_NAME=$M VLLM_URL=$U $PYBIN -m experiments.constructed_loo --families-file $ROOT/families.json --n-inst 8 --seeds 1 --slack 2 \
  --workers 24 --workroot $ROOT/loo_work_cfg --out $ROOT/constructed_loo_campaigncfg.json \
  --max-output-tokens 512 --context-window 8192 --trim-fraction 0.7 >> logs/phase2_gpu1_chain.log 2>&1 && say "LOO (campaign config) done" || say "LOO returned nonzero"
say "handing over to the 32B starter"
exec bash experiments/phase2_start_32b.sh
