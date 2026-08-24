#!/bin/bash
# run_boundary_loo.sh — bulletproof gold-standard boundary battery on GPU0 (one 14B model load):
#   1) crypto constructed-context LOO (none/helpful/poison/helpful+poison/helpful+malignant)
#   2) SQL    constructed-context LOO (none/helpful/poison/helpful+poison)
#   3) SQL    realized-context active_loo (Spearman vs cheap SNIPW) — estimator validation in SQL
# Makes the boundary claim interventional + symmetric across both domains. Resumable.
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
M="Qwen/Qwen2.5-14B-Instruct-AWQ"; PORT=8002; U="http://127.0.0.1:$PORT/v1"
say(){ echo "[bloo $(date +%H:%M:%S)] $*" | tee -a logs/run_boundary_loo.log; }
own(){ [ "$(ps -o user= -p "$1" 2>/dev/null|tr -d ' ')" = "$USER" ]; }
genok(){ curl -s --max-time 40 "$U/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"OK\"}],\"max_tokens\":5}" 2>/dev/null | grep -q '"content"'; }
gpu0_clear(){ local bad=0; for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '); do [ -n "$p" ] && [ "$(ps -o user= -p $p 2>/dev/null|tr -d ' ')" != "$USER" ] && bad=1; done; [ $bad = 0 ]; }
stop_server(){ p=$(cat /tmp/bloo.pid 2>/dev/null); { [ -n "${p:-}" ] && own "$p" && { say "stop $p"; kill "$p" 2>/dev/null; sleep 8; kill -9 "$p" 2>/dev/null; }; }; }
mkdir -p logs
gpu0_clear || { say "GPU0 busy (other user) — abort"; exit 0; }
if ! genok; then
  say "serving $M GPU0:$PORT"; : > logs/serve_bloo.log
  CUDA_VISIBLE_DEVICES=0 setsid nohup "$VLLM" serve "$M" --max-model-len 8192 --gpu-memory-utilization 0.82 \
    --max-num-seqs 48 --enable-prefix-caching --port $PORT > logs/serve_bloo.log 2>&1 &
  echo $! > /tmp/bloo.pid
  ok=0; for i in $(seq 1 240); do grep -qiE "OutOfMemory|error initializing|No available memory" logs/serve_bloo.log 2>/dev/null && break; genok && { ok=1; break; }; sleep 5; done
  [ "$ok" = 1 ] || { say "serve FAILED"; stop_server; exit 1; }
fi
say "server up"
if [ ! -f results/lab/constructed_loo_crypto.json ]; then
  say "1/3 CRYPTO constructed-context LOO (41 families)"
  CTF_DOMAIN=benchmark.hard_ctf MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.constructed_loo \
    --families-file results/lab/probe_14b/selection.json --n-inst 4 --seeds 2 --workers 24 \
    --workroot results/lab/cloo_crypto_work --out results/lab/constructed_loo_crypto.json >> logs/run_boundary_loo.log 2>&1
fi
if [ ! -f results/lab/constructed_loo_sql.json ]; then
  say "2/3 SQL constructed-context LOO (16 families)"
  CTF_DOMAIN=benchmark.sql_domain MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.constructed_loo \
    --families-file results/lab/probe_sql/selection.json --n-inst 4 --seeds 2 --workers 24 \
    --workroot results/lab/cloo_sql_work --out results/lab/constructed_loo_sql.json >> logs/run_boundary_loo.log 2>&1
fi
if [ ! -f results/lab/campaign_sql_14b/merged/active_loo.json ]; then
  say "3/3 SQL realized-context active_loo (Spearman vs cheap SNIPW)"
  CTF_DOMAIN=benchmark.sql_domain MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.active_loo \
    --dir results/lab/campaign_sql_14b/merged --n-inst 3 --seeds 2 --contexts 2 --max-nodes 40 \
    --workers 24 --workroot results/lab/aloo_sql_work >> logs/run_boundary_loo.log 2>&1 || true
fi
stop_server
say "BOUNDARY LOO BATTERY DONE"
