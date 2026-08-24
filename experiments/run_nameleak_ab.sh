#!/bin/bash
# run_nameleak_ab.sh — T2-2 name-leak A/B at 3B on GPU0 (GPU1 = other user). Two campaigns on the SAME
# 12 in-band families/params: FIXED (neutral data.bin) vs LEAKED (family name in filename/desc). Compare
# the on-target poison ATE: prediction = leaked makes poison spuriously harmful, fixed reads ~null (the
# pilot's poison effect was the leak, not the lesson). Converts the cautionary tale to a controlled result.
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
M="Qwen/Qwen2.5-3B-Instruct"; PORT=8002; U="http://127.0.0.1:$PORT/v1"
FAMS="results/lab/probe_3b/selection.json"
say(){ echo "[nleak $(date +%H:%M:%S)] $*" | tee -a logs/run_nameleak_ab.log; }
own(){ [ "$(ps -o user= -p "$1" 2>/dev/null|tr -d ' ')" = "$USER" ]; }
genok(){ curl -s --max-time 40 "$U/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"OK\"}],\"max_tokens\":5}" 2>/dev/null | grep -q '"content"'; }
gpu0_clear(){ local bad=0; for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '); do [ -n "$p" ] && [ "$(ps -o user= -p $p 2>/dev/null|tr -d ' ')" != "$USER" ] && bad=1; done; [ $bad = 0 ]; }
stop_server(){ p=$(cat /tmp/nleak.pid 2>/dev/null); { [ -n "${p:-}" ] && own "$p" && { say "stop server $p"; kill "$p" 2>/dev/null; sleep 6; kill -9 "$p" 2>/dev/null; }; }; }
mkdir -p logs
gpu0_clear || { say "GPU0 busy with another user — abort"; exit 0; }
if ! genok; then
  say "serving $M GPU0:$PORT"; : > logs/serve_nleak.log
  CUDA_VISIBLE_DEVICES=0 setsid nohup "$VLLM" serve "$M" --max-model-len 8192 --gpu-memory-utilization 0.55 \
    --max-num-seqs 64 --enable-prefix-caching --port $PORT > logs/serve_nleak.log 2>&1 &
  echo $! > /tmp/nleak.pid
  ok=0; for i in $(seq 1 240); do grep -qiE "OutOfMemory|error initializing|No available memory" logs/serve_nleak.log 2>/dev/null && break; genok && { ok=1; break; }; sleep 5; done
  [ "$ok" = 1 ] || { say "serve FAILED"; stop_server; exit 1; }
fi
say "server up"
# arm: TAG | OUT | extra-flag
for spec in "fixed|results/lab/campaign_3b_nameab_fixed|" "leak|results/lab/campaign_3b_nameab_leak|--leak-names"; do
  IFS='|' read -r TAG OUT FLAG <<< "$spec"
  if [ -f "$OUT/merged/ontarget_analysis.json" ]; then say "$TAG already done, skip"; continue; fi
  say "=== $TAG campaign (8 shards n6 e4 s1 replicate) $FLAG ==="
  MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.lab_pipeline \
    --families-file "$FAMS" --out "$OUT" --vllm-urls "$U" \
    --shards 8 --n-per-family 6 --epochs 4 --eval-seeds 1 --slack 2 \
    --conditions outcome --n-clusters 4 --M 3 --replicate --cond outcome \
    --world-seed 525252 $FLAG --deadline-hours 6 >> logs/run_nameleak_ab.log 2>&1
  say "$TAG done -> $OUT/merged/ontarget_analysis.json"
done
stop_server
say "NAMELEAK A/B DONE"
