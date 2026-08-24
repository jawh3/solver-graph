#!/bin/bash
# power_14b_v2.sh — powered 14B boundary, OUTCOME-ONLY (skip the wasteful B0 phase). GPU1 only.
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
M="Qwen/Qwen2.5-14B-Instruct-AWQ"; PORT=8002; OUT=results/lab/campaign_14b_M3_powered
say(){ echo "[pow14b2 $(date +%H:%M:%S)] $*" | tee -a logs/pow14b.log; }
say "serving $M on GPU1:$PORT"
CUDA_VISIBLE_DEVICES=1 setsid nohup "$VLLM" serve "$M" --max-model-len 8192 \
  --gpu-memory-utilization 0.80 --max-num-seqs 64 --enable-prefix-caching --port $PORT \
  > logs/serve_14b_powered.log 2>&1 &
echo $! > /tmp/pow14b.pid
ok=0; for i in $(seq 1 180); do curl -s http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q "14B" && { ok=1; break; }; sleep 5; done
[ "$ok" = 1 ] || { say "serve FAILED; abort"; exit 1; }
say "healthy; outcome-only campaign (n16, fresh seed 999999)"
MODEL_NAME="$M" VLLM_URL=http://127.0.0.1:$PORT/v1 $PYBIN -m experiments.lab_pipeline \
  --families-file results/lab/probe_14b/selection.json --out "$OUT" --vllm-urls http://127.0.0.1:$PORT/v1 \
  --shards 8 --n-per-family 16 --epochs 4 --eval-seeds 3 --slack 2 --conditions outcome --n-clusters 6 \
  --M 3 --replicate --cond outcome --world-seed 999999 --deadline-hours 12 >> logs/pow14b.log 2>&1
say "campaign done; CRT + cite-rate cross-checks"
MODEL_NAME="$M" $PYBIN -m experiments.crt_test --dir "$OUT/merged" --key R_final >> logs/pow14b.log 2>&1 || true
MODEL_NAME="$M" $PYBIN -m experiments.cite_rate --dir "$OUT/merged" >> logs/pow14b.log 2>&1 || true
p=$(cat /tmp/pow14b.pid 2>/dev/null); { [ -n "${p:-}" ] && [ "$(ps -o user= -p $p|tr -d ' ')" = "$USER" ] && { say "stopping 14B server $p"; kill $p; sleep 8; kill -9 $p 2>/dev/null; }; }
say "POWERED 14B (outcome-only) DONE"
