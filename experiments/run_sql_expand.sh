#!/bin/bash
# run_sql_expand.sh — harden the SQL poison thin spot (n=8 -> ~30) on GPU0. Probe the expanded SQL grid
# -> in-band selection -> bigger campaign -> re-run ontarget + active_loo + constructed_loo on robust n.
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
export CTF_DOMAIN=benchmark.sql_domain
M="Qwen/Qwen2.5-14B-Instruct-AWQ"; PORT=8002; U="http://127.0.0.1:$PORT/v1"
SQLP="results/lab/probe_sql_big"; SQLC="results/lab/campaign_sql_14b_big"
say(){ echo "[sqlx $(date +%H:%M:%S)] $*" | tee -a logs/run_sql_expand.log; }
own(){ [ "$(ps -o user= -p "$1" 2>/dev/null|tr -d ' ')" = "$USER" ]; }
genok(){ curl -s --max-time 40 "$U/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"OK\"}],\"max_tokens\":5}" 2>/dev/null | grep -q '"content"'; }
gpu0_clear(){ local bad=0; for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '); do [ -n "$p" ] && [ "$(ps -o user= -p $p 2>/dev/null|tr -d ' ')" != "$USER" ] && bad=1; done; [ $bad = 0 ]; }
stop_server(){ p=$(cat /tmp/sqlx.pid 2>/dev/null); { [ -n "${p:-}" ] && own "$p" && { say "stop $p"; kill "$p" 2>/dev/null; sleep 8; kill -9 "$p" 2>/dev/null; }; }; }
mkdir -p logs
gpu0_clear || { say "GPU0 busy (other user) — abort"; exit 0; }
if ! genok; then
  say "serving $M GPU0:$PORT"; : > logs/serve_sqlx.log
  CUDA_VISIBLE_DEVICES=0 setsid nohup "$VLLM" serve "$M" --max-model-len 8192 --gpu-memory-utilization 0.82 \
    --max-num-seqs 48 --enable-prefix-caching --port $PORT > logs/serve_sqlx.log 2>&1 &
  echo $! > /tmp/sqlx.pid
  ok=0; for i in $(seq 1 240); do grep -qiE "OutOfMemory|error initializing|No available memory" logs/serve_sqlx.log 2>/dev/null && break; genok && { ok=1; break; }; sleep 5; done
  [ "$ok" = 1 ] || { say "serve FAILED"; stop_server; exit 1; }
fi
say "server up"
if [ ! -f "$SQLP/selection.json" ]; then
  say "probe expanded SQL grid (48 families)"
  MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.lab_probe --max-families 48 --n-inst 2 --seeds 3 \
    --workers 24 --out "$SQLP" >> logs/run_sql_expand.log 2>&1 || true
fi
NB=$($PYBIN -c "import json;print(len(json.load(open('$SQLP/selection.json')).get('in_band_families') or []))" 2>/dev/null||echo 0)
say "SQL in-band (expanded): $NB"
if [ ! -f "$SQLC/merged/ontarget_analysis.json" ] && [ "${NB:-0}" -ge 20 ]; then
  say "big SQL campaign ($NB families)"
  MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.lab_pipeline --families-file "$SQLP/selection.json" \
    --out "$SQLC" --vllm-urls "$U" --shards 8 --n-per-family 4 --epochs 4 --eval-seeds 1 --slack 2 \
    --conditions outcome --n-clusters 6 --M 3 --replicate --cond outcome --world-seed 575757 --deadline-hours 7 >> logs/run_sql_expand.log 2>&1
  MODEL_NAME="$M" $PYBIN -m experiments.ontarget_analysis --dir "$SQLC/merged" --cond outcome >> logs/run_sql_expand.log 2>&1 || true
  MODEL_NAME="$M" $PYBIN -m experiments.cite_rate --dir "$SQLC/merged" --cond outcome >> logs/run_sql_expand.log 2>&1 || true
  MODEL_NAME="$M" $PYBIN -m experiments.active_loo --dir "$SQLC/merged" --n-inst 3 --seeds 2 --contexts 2 --max-nodes 60 --workers 24 --workroot results/lab/aloo_sqlbig_work >> logs/run_sql_expand.log 2>&1 || true
fi
say "constructed_loo on expanded SQL in-band (helpful+poison contrast, robust n)"
MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.constructed_loo --families-file "$SQLP/selection.json" \
  --n-inst 4 --seeds 2 --workers 24 --workroot results/lab/cloo_sqlbig_work --out an artifact not retained >> logs/run_sql_expand.log 2>&1 || true
stop_server
say "SQL EXPANSION DONE"
