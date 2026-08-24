#!/bin/bash
# power_32b_tp2.sh — 32B-AWQ across BOTH GPUs (tensor-parallel) so we get full context + batching
# (the single-GPU attempt truncated -> 0% solve). Staged: serve TP=2 -> probe standard grid ->
# campaign if >=8 in-band, else report "too capable, need deeper families". Yields if another user returns.
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
M="Qwen/Qwen2.5-32B-Instruct-AWQ"; PORT=8002; OUT=results/lab/campaign_32b_M3_tp2; PROBE=results/lab/probe_32b_tp2
say(){ echo "[32b-tp2 $(date +%H:%M:%S)] $*" | tee -a logs/pow32b_tp2.log; }

# courtesy: abort if another user has a GPU compute job (don't co-occupy if they came back)
if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | while read p; do [ -n "$p" ] && ps -o user= -p $p 2>/dev/null|tr -d ' '; done | grep -vc "^${USER}$")" != 0 ]; then
  say "another user has a GPU job — yielding, not launching."; exit 0; fi

say "serving $M tensor-parallel=2 across GPU0+GPU1 (full context 8192)"
CUDA_VISIBLE_DEVICES=0,1 setsid nohup "$VLLM" serve "$M" --tensor-parallel-size 2 --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --max-num-seqs 48 --enable-prefix-caching --port $PORT \
  > logs/serve_32b_tp2.log 2>&1 &
echo $! > /tmp/pow32b_tp2.pid
ok=0; for i in $(seq 1 240); do
  grep -qiE "OutOfMemoryError|No available memory|CUDA out of memory|error initializing" logs/serve_32b_tp2.log 2>/dev/null && { say "32B TP2 serve error/OOM. ABORT."; break; }
  curl -s http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q "32B" && { ok=1; break; }
  sleep 5
done
if [ "$ok" != 1 ]; then say "32B TP2 serve FAILED."; p=$(cat /tmp/pow32b_tp2.pid); { [ -n "${p:-}" ] && [ "$(ps -o user= -p $p|tr -d ' ')" = "$USER" ] && kill -9 $p 2>/dev/null; }; exit 1; fi
say "32B TP2 SERVES (full context). Probing STANDARD grid for in-band families."

MODEL_NAME="$M" VLLM_URL=http://127.0.0.1:$PORT/v1 $PYBIN -m experiments.lab_probe \
  --max-families 80 --n-inst 2 --seeds 3 --slack 2 --workers 16 \
  --max-none 0.5 --min-helpful 0.3 --min-lift 0.25 --out "$PROBE" --max-output-tokens 384 >> logs/pow32b_tp2.log 2>&1
NB=$($PYBIN -c "import json;print(json.load(open('$PROBE/selection.json')).get('n_in_band',0))" 2>/dev/null || echo 0)
say "32B in-band families (standard grid) = $NB"
if [ "${NB:-0}" -lt 8 ]; then
  say "32B has <8 in-band on standard grid (too capable -> need deeper families). Recording probe only; will generate depth-3 families next."
  p=$(cat /tmp/pow32b_tp2.pid); { [ -n "${p:-}" ] && [ "$(ps -o user= -p $p|tr -d ' ')" = "$USER" ] && { kill $p; sleep 8; kill -9 $p 2>/dev/null; }; }; exit 0; fi

say "running 32B outcome-only campaign on $NB families (n12, fresh seed)"
MODEL_NAME="$M" VLLM_URL=http://127.0.0.1:$PORT/v1 $PYBIN -m experiments.lab_pipeline \
  --families-file "$PROBE/selection.json" --out "$OUT" --vllm-urls http://127.0.0.1:$PORT/v1 \
  --shards 8 --n-per-family 12 --epochs 4 --eval-seeds 3 --slack 2 --conditions outcome --n-clusters 6 \
  --M 3 --replicate --cond outcome --world-seed 444444 --deadline-hours 14 >> logs/pow32b_tp2.log 2>&1
say "32B campaign done; CRT + cite-rate"
MODEL_NAME="$M" $PYBIN -m experiments.crt_test --dir "$OUT/merged" --key R_final >> logs/pow32b_tp2.log 2>&1 || true
MODEL_NAME="$M" $PYBIN -m experiments.cite_rate --dir "$OUT/merged" >> logs/pow32b_tp2.log 2>&1 || true
p=$(cat /tmp/pow32b_tp2.pid); { [ -n "${p:-}" ] && [ "$(ps -o user= -p $p|tr -d ' ')" = "$USER" ] && { say "stopping 32B TP2 server $p"; kill $p; sleep 8; kill -9 $p 2>/dev/null; }; }
say "POWER 32B TP2 DONE"
