#!/bin/bash
# run_direct_loo.sh — gold-standard DIRECT malignant harm on GPU0 (GPU1 = other user).
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
M="Qwen/Qwen2.5-14B-Instruct-AWQ"; PORT=8002; U="http://127.0.0.1:$PORT/v1"
OUT="results/lab/malignant_direct_loo.json"
mkdir -p logs
say(){ echo "[dloo $(date +%H:%M:%S)] $*" | tee -a logs/run_direct_loo.log; }
own(){ [ "$(ps -o user= -p "$1" 2>/dev/null|tr -d ' ')" = "$USER" ]; }
genok(){ curl -s --max-time 40 "$U/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"reply OK\"}],\"max_tokens\":5}" 2>/dev/null | grep -q '"content"'; }
gpu0_clear(){ local bad=0; for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '); do [ -n "$p" ] && [ "$(ps -o user= -p $p 2>/dev/null|tr -d ' ')" != "$USER" ] && bad=1; done; [ $bad = 0 ]; }
stop_server(){ p=$(cat /tmp/dloo.pid 2>/dev/null); { [ -n "${p:-}" ] && own "$p" && { say "stop server $p"; kill "$p" 2>/dev/null; sleep 8; kill -9 "$p" 2>/dev/null; }; }; }
[ -f "$OUT" ] && { say "$OUT exists — done"; exit 0; }
gpu0_clear || { say "GPU0 busy with another user — abort"; exit 0; }
if ! genok; then
  say "serving $M GPU0:$PORT"; : > logs/serve_dloo.log
  CUDA_VISIBLE_DEVICES=0 setsid nohup "$VLLM" serve "$M" --max-model-len 8192 --gpu-memory-utilization 0.82 \
    --max-num-seqs 48 --enable-prefix-caching --port $PORT > logs/serve_dloo.log 2>&1 &
  echo $! > /tmp/dloo.pid
  ok=0; for i in $(seq 1 240); do grep -qiE "OutOfMemory|error initializing|No available memory" logs/serve_dloo.log 2>/dev/null && break; genok && { ok=1; break; }; sleep 5; done
  [ "$ok" = 1 ] || { say "serve FAILED"; stop_server; exit 1; }
fi
say "server up; running constructed-context direct-harm LOO (41 families)"
MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.malignant_direct_loo \
  --families-file results/lab/probe_14b/selection.json --n-inst 4 --seeds 2 --workers 24 --out "$OUT" >> logs/run_direct_loo.log 2>&1
stop_server
say "DIRECT-LOO DONE -> $OUT"
