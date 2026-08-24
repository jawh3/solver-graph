#!/bin/bash
# power_72b.sh — 72B-AWQ across BOTH GPUs (TP=2): the max-capability rung. ~40GB / 2 cards.
# Probes the HARDER depth-2/3 families (72B is too capable for easy ones). Staged + courtesy-guarded.
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
M="Qwen/Qwen2.5-72B-Instruct-AWQ"; PORT=8002; OUT=results/lab/campaign_72b_M3; PROBE=results/lab/probe_72b
say(){ echo "[72b $(date +%H:%M:%S)] $*" | tee -a logs/pow72b.log; }
other_user_on_gpu(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '|while read p;do [ -n "$p" ]&&ps -o user= -p $p 2>/dev/null|tr -d ' ';done|grep -vc "^${USER}$"; }
if [ "$(other_user_on_gpu)" != 0 ]; then say "another user on a GPU — not launching."; exit 0; fi

say "serving $M TP=2 (~19.4GB/card weights on 24GB cards — barely fits)"
# WORKING CONFIG (2026-06-10): 72B-AWQ fits 2x24GB ONLY with the tight-memory squeeze:
#   --enforce-eager (no cudagraph capture) + expandable_segments (anti-fragmentation) +
#   max-model-len 3072 + max-num-seqs 6 + gpu-util 0.93 -> 19.38GiB/card weights, 2.11GiB KV pool.
# Earlier attempts (len 4096, 12 seqs, gpu-util 0.90, NO enforce-eager) OOM'd at engine-core init.
# Trade-off: tiny KV + len 3072 -> low throughput (~6 concurrent) + truncation risk on deep solves.
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup "$VLLM" serve "$M" \
  --tensor-parallel-size 2 --max-model-len 3072 \
  --gpu-memory-utilization 0.93 --max-num-seqs 6 --enforce-eager --port $PORT \
  > logs/serve_72b.log 2>&1 &
echo $! > /tmp/pow72b.pid
ok=0; for i in $(seq 1 300); do
  grep -qiE "OutOfMemoryError|No available memory|CUDA out of memory|error initializing" logs/serve_72b.log 2>/dev/null && { say "72B serve OOM/error -> does not fit even TP=2 here. ABORT."; break; }
  curl -s http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q "72B" && { ok=1; break; }
  sleep 5
done
if [ "$ok" != 1 ]; then say "72B serve FAILED."; p=$(cat /tmp/pow72b.pid); { [ -n "${p:-}" ] && [ "$(ps -o user= -p $p|tr -d ' ')" = "$USER" ] && kill -9 $p 2>/dev/null; }; exit 1; fi
say "72B SERVES (TP=2). Probing depth-2/3 (hard) families."

MODEL_NAME="$M" VLLM_URL=http://127.0.0.1:$PORT/v1 $PYBIN -m experiments.lab_probe \
  --difficulty hard --max-families 100 --n-inst 2 --seeds 3 --slack 2 --workers 16 \
  --max-none 0.5 --min-helpful 0.3 --min-lift 0.25 --out "$PROBE" --max-output-tokens 384 >> logs/pow72b.log 2>&1
NB=$($PYBIN -c "import json;print(json.load(open('$PROBE/selection.json')).get('n_in_band',0))" 2>/dev/null || echo 0)
say "72B in-band (depth-2/3) = $NB"
if [ "${NB:-0}" -lt 8 ]; then
  say "72B has <8 in-band even on depth-2/3 — too capable for available families. Recording probe only, ABORT campaign."
  p=$(cat /tmp/pow72b.pid); { [ -n "${p:-}" ] && [ "$(ps -o user= -p $p|tr -d ' ')" = "$USER" ] && { kill $p; sleep 8; kill -9 $p 2>/dev/null; }; }; exit 0; fi

say "running 72B outcome-only campaign on $NB families (n12, fresh seed)"
MODEL_NAME="$M" VLLM_URL=http://127.0.0.1:$PORT/v1 $PYBIN -m experiments.lab_pipeline \
  --families-file "$PROBE/selection.json" --out "$OUT" --vllm-urls http://127.0.0.1:$PORT/v1 \
  --shards 8 --n-per-family 12 --epochs 4 --eval-seeds 3 --slack 2 --conditions outcome --n-clusters 6 \
  --M 3 --replicate --cond outcome --world-seed 333333 --deadline-hours 18 >> logs/pow72b.log 2>&1
say "72B campaign done; CRT + cite-rate"
MODEL_NAME="$M" $PYBIN -m experiments.crt_test --dir "$OUT/merged" --key R_final >> logs/pow72b.log 2>&1 || true
MODEL_NAME="$M" $PYBIN -m experiments.cite_rate --dir "$OUT/merged" >> logs/pow72b.log 2>&1 || true
p=$(cat /tmp/pow72b.pid); { [ -n "${p:-}" ] && [ "$(ps -o user= -p $p|tr -d ' ')" = "$USER" ] && { say "stopping 72B server $p"; kill $p; sleep 8; kill -9 $p 2>/dev/null; }; }
say "POWER 72B DONE"
