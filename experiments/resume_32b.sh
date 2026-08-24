#!/bin/bash
# resume_32b.sh — RESUME the paused 32B-AWQ TP=2 campaign (checkpointed at ~4983 runs) + then 72B.
# Skips the probe (43 in-band already selected). Courtesy: aborts if another user is on a GPU.
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
M="Qwen/Qwen2.5-32B-Instruct-AWQ"; PORT=8002; OUT=results/lab/campaign_32b_M3_tp2; PROBE=results/lab/probe_32b_tp2
say(){ echo "[resume32b $(date +%H:%M:%S)] $*" | tee -a logs/resume32b.log; }

other_user_on_gpu(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '|while read p;do [ -n "$p" ]&&ps -o user= -p $p 2>/dev/null|tr -d ' ';done|grep -vc "^${USER}$"; }
if [ "$(other_user_on_gpu)" != 0 ]; then say "another user on a GPU — not resuming."; exit 0; fi

say "serving 32B TP=2 to RESUME (campaign checkpointed at $(cat $OUT/shard_*/logs/campaign_outcome/runs.jsonl 2>/dev/null|wc -l) outcome runs)"
CUDA_VISIBLE_DEVICES=0,1 setsid nohup "$VLLM" serve "$M" --tensor-parallel-size 2 --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --max-num-seqs 48 --enable-prefix-caching --port $PORT \
  > logs/serve_32b_tp2.log 2>&1 &
echo $! > /tmp/resume32b.pid
ok=0; for i in $(seq 1 240); do
  grep -qiE "OutOfMemoryError|error initializing" logs/serve_32b_tp2.log 2>/dev/null && { say "serve error. ABORT."; break; }
  curl -s http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q "32B" && { ok=1; break; }
  sleep 5
done
[ "$ok" = 1 ] || { say "32B serve FAILED."; p=$(cat /tmp/resume32b.pid); { [ -n "${p:-}" ] && [ "$(ps -o user= -p $p|tr -d ' ')" = "$USER" ] && kill -9 $p 2>/dev/null; }; exit 1; }

say "resuming campaign (same out/seed -> checkpointed shards continue)"
MODEL_NAME="$M" VLLM_URL=http://127.0.0.1:$PORT/v1 $PYBIN -m experiments.lab_pipeline \
  --families-file "$PROBE/selection.json" --out "$OUT" --vllm-urls http://127.0.0.1:$PORT/v1 \
  --shards 8 --n-per-family 12 --epochs 4 --eval-seeds 3 --slack 2 --conditions outcome --n-clusters 6 \
  --M 3 --replicate --cond outcome --world-seed 444444 --deadline-hours 14 >> logs/resume32b.log 2>&1
say "32B campaign done; CRT + cite-rate"
MODEL_NAME="$M" $PYBIN -m experiments.crt_test --dir "$OUT/merged" --key R_final >> logs/resume32b.log 2>&1 || true
MODEL_NAME="$M" $PYBIN -m experiments.cite_rate --dir "$OUT/merged" >> logs/resume32b.log 2>&1 || true
p=$(cat /tmp/resume32b.pid); { [ -n "${p:-}" ] && [ "$(ps -o user= -p $p|tr -d ' ')" = "$USER" ] && { say "stopping 32B server $p"; kill $p; sleep 8; kill -9 $p 2>/dev/null; }; }
say "RESUME 32B DONE"
