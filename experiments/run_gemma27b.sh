#!/bin/bash
# run_gemma27b.sh — HIGHER-CAP non-Qwen rung: Gemma-2-27B-it (AWQ, non-gated, mbley). TP=2 across both GPUs.
# Distinct architecture (Google Gemma2, SentencePiece tokenizer -> no byte-BPE bug), 8192 native context (sound for
# the multi-round solver), 27B = high-cap. Tests whether the null->weak poison-detection transition (Qwen 14B/32B)
# is cross-architecture. Replaces Mistral-Small-24B (unservable AWQ tokenizer) and Yi-1.5-34B (4K context too short).
# 27B-AWQ ~16GB -> TP=2 (~8GB/card + ample KV), single instance port 8002, like the 32B-AWQ TP=2 rung.
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
M="mbley/google-gemma-2-27b-it-AWQ"; TAG="gemma27b"; PORT=8002
OUT="results/lab/campaign_${TAG}_M3"; PROBE="results/lab/probe_${TAG}"; U="http://127.0.0.1:$PORT/v1"
say(){ echo "[gemma27b $(date +%H:%M:%S)] $*" | tee -a logs/run_gemma27b.log; }
own(){ [ "$(ps -o user= -p "$1" 2>/dev/null|tr -d ' ')" = "$USER" ]; }
other_user_on_gpu(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '|while read p;do [ -n "$p" ]&&ps -o user= -p $p 2>/dev/null|tr -d ' ';done|grep -vc "^${USER}$"; }
genok(){ curl -s --max-time 40 "$U/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"reply OK\"}],\"max_tokens\":5}" 2>/dev/null | grep -q '"content"'; }

[ "$(other_user_on_gpu)" != 0 ] && { say "another user on a GPU — not launching."; exit 0; }

say "serving $M TP=2 (both GPUs), AWQ ~8GB/card, len8192"
: > logs/serve_gemma27b.log
# --chat-template: Gemma-2's stock template raises on a system role; this one merges system->first user turn.
CUDA_VISIBLE_DEVICES=0,1 setsid nohup "$VLLM" serve "$M" --tensor-parallel-size 2 --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --max-num-seqs 32 --chat-template infrastructure/gemma2_system.jinja \
  --enable-prefix-caching --port $PORT \
  > logs/serve_gemma27b.log 2>&1 &
echo $! > /tmp/gemma27b.pid
ok=0; for i in $(seq 1 240); do
  grep -qiE "OutOfMemoryError|No available memory|CUDA out of memory|error initializing" logs/serve_gemma27b.log 2>/dev/null && { say "serve OOM/error. ABORT."; break; }
  genok && { ok=1; break; }
  sleep 5
done
if [ "$ok" != 1 ]; then say "Gemma-27B serve FAILED (no generation)."; p=$(cat /tmp/gemma27b.pid); { [ -n "${p:-}" ] && own "$p" && kill -9 "$p" 2>/dev/null; }; exit 1; fi
say "Gemma-27B SERVES (TP=2). Probing standard grid for in-band families."

MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.lab_probe \
  --max-families 80 --n-inst 2 --seeds 3 --slack 2 --workers 16 \
  --max-none 0.5 --min-helpful 0.3 --min-lift 0.25 --out "$PROBE" --max-output-tokens 384 >> logs/run_gemma27b.log 2>&1
NB=$($PYBIN -c "import json;print(json.load(open('$PROBE/selection.json')).get('n_in_band',0))" 2>/dev/null || echo 0)
say "$TAG in-band families (standard grid) = $NB"
if [ "${NB:-0}" -lt 8 ]; then
  say "$TAG has <8 in-band (floored OR too-capable). Recording probe only, skip campaign."
  p=$(cat /tmp/gemma27b.pid); { [ -n "${p:-}" ] && own "$p" && { kill "$p"; sleep 8; kill -9 "$p" 2>/dev/null; }; }; exit 0; fi

say "running $TAG outcome campaign on $NB families (n12, powered like 14B/32B)"
MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.lab_pipeline \
  --families-file "$PROBE/selection.json" --out "$OUT" --vllm-urls "$U" \
  --shards 8 --n-per-family 12 --epochs 4 --eval-seeds 3 --slack 2 --conditions outcome --n-clusters 6 \
  --M 3 --replicate --cond outcome --world-seed 565656 --deadline-hours 16 >> logs/run_gemma27b.log 2>&1
say "$TAG campaign done; CRT + cite-rate"
MODEL_NAME="$M" $PYBIN -m experiments.crt_test --dir "$OUT/merged" --key R_final >> logs/run_gemma27b.log 2>&1 || true
MODEL_NAME="$M" $PYBIN -m experiments.cite_rate --dir "$OUT/merged" >> logs/run_gemma27b.log 2>&1 || true
p=$(cat /tmp/gemma27b.pid); { [ -n "${p:-}" ] && own "$p" && { say "stopping Gemma-27B server $p"; kill "$p"; sleep 8; kill -9 "$p" 2>/dev/null; }; }
say "GEMMA27B RUNG DONE (n_in_band=$NB)"
