#!/bin/bash
# run_active_loo.sh — T1-12/Task#10 gold-standard validation on GPU0 (GPU1 is another user's).
# Serve 14B, run active_loo on the powered 14B campaign (validate cheap SNIPW vs LOO re-rollout for
# helpful/poison), then stop. Resumable (skips if active_loo.json exists).
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
M="Qwen/Qwen2.5-14B-Instruct-AWQ"; GPU=0; PORT=8002; U="http://127.0.0.1:$PORT/v1"
DIR="${1:-results/lab/campaign_14b_M3_powered/merged}"     # arg1 = merged dir (default: powered)
MAXNODES="${2:-60}"
mkdir -p logs
say(){ echo "[loo $(date +%H:%M:%S)] $*" | tee -a logs/run_active_loo.log; }
own(){ [ "$(ps -o user= -p "$1" 2>/dev/null|tr -d ' ')" = "$USER" ]; }
genok(){ curl -s --max-time 40 "$U/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"reply OK\"}],\"max_tokens\":5}" 2>/dev/null | grep -q '"content"'; }
gpu0_clear(){ local bad=0; for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '); do [ -n "$p" ] && [ "$(ps -o user= -p $p 2>/dev/null|tr -d ' ')" != "$USER" ] && bad=1; done; [ $bad = 0 ]; }
stop_server(){ p=$(cat /tmp/loo.pid 2>/dev/null); { [ -n "${p:-}" ] && own "$p" && { say "stopping server $p"; kill "$p" 2>/dev/null; sleep 8; kill -9 "$p" 2>/dev/null; }; }; }

[ -f "$DIR/active_loo.json" ] && { say "active_loo.json exists — done."; exit 0; }
gpu0_clear || { say "GPU0 has another user's process — NOT launching."; exit 0; }

if ! genok; then
  say "serving $M on GPU$GPU:$PORT"; : > logs/serve_loo.log
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup "$VLLM" serve "$M" --max-model-len 8192 \
    --gpu-memory-utilization 0.82 --max-num-seqs 48 --enable-prefix-caching --port $PORT \
    > logs/serve_loo.log 2>&1 &
  echo $! > /tmp/loo.pid
  ok=0; for i in $(seq 1 240); do
    grep -qiE "OutOfMemoryError|error initializing|No available memory" logs/serve_loo.log 2>/dev/null && { say "serve error; abort"; break; }
    genok && { ok=1; break; }; sleep 5
  done
  [ "$ok" = 1 ] || { say "serve FAILED"; stop_server; exit 1; }
fi
say "server healthy; running active_loo (gold-standard re-rollout)"
MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.active_loo --dir "$DIR" \
  --n-inst 3 --seeds 2 --contexts 2 --max-nodes 60 --slack 2 --workers 24 >> logs/run_active_loo.log 2>&1
say "active_loo done -> $DIR/active_loo.json"
stop_server
say "ACTIVE-LOO DONE"
