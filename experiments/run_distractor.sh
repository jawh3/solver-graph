#!/bin/bash
# run_distractor.sh — T2-3 on-target distractor arm at 14B on GPU0 (GPU1 = other user). One outcome
# campaign on the 41 in-band families with --distractor-arm; the on-target distractor SNIPW ATE should
# read ~null (retrieved-but-irrelevant => neither credited nor blamed) — the specificity control.
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
M="Qwen/Qwen2.5-14B-Instruct-AWQ"; PORT=8002; U="http://127.0.0.1:$PORT/v1"
FAMS="results/lab/probe_14b/selection.json"; OUT="results/lab/campaign_14b_distractor"
say(){ echo "[distract $(date +%H:%M:%S)] $*" | tee -a logs/run_distractor.log; }
own(){ [ "$(ps -o user= -p "$1" 2>/dev/null|tr -d ' ')" = "$USER" ]; }
genok(){ curl -s --max-time 40 "$U/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"OK\"}],\"max_tokens\":5}" 2>/dev/null | grep -q '"content"'; }
gpu0_clear(){ local bad=0; for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '); do [ -n "$p" ] && [ "$(ps -o user= -p $p 2>/dev/null|tr -d ' ')" != "$USER" ] && bad=1; done; [ $bad = 0 ]; }
stop_server(){ p=$(cat /tmp/distract.pid 2>/dev/null); { [ -n "${p:-}" ] && own "$p" && { say "stop server $p"; kill "$p" 2>/dev/null; sleep 8; kill -9 "$p" 2>/dev/null; }; }; }
mkdir -p logs
[ -f "$OUT/merged/ontarget_analysis.json" ] && { say "already done"; exit 0; }
gpu0_clear || { say "GPU0 busy with another user — abort"; exit 0; }
if ! genok; then
  say "serving $M GPU0:$PORT"; : > logs/serve_distract.log
  CUDA_VISIBLE_DEVICES=0 setsid nohup "$VLLM" serve "$M" --max-model-len 8192 --gpu-memory-utilization 0.82 \
    --max-num-seqs 48 --enable-prefix-caching --port $PORT > logs/serve_distract.log 2>&1 &
  echo $! > /tmp/distract.pid
  ok=0; for i in $(seq 1 240); do grep -qiE "OutOfMemory|error initializing|No available memory" logs/serve_distract.log 2>/dev/null && break; genok && { ok=1; break; }; sleep 5; done
  [ "$ok" = 1 ] || { say "serve FAILED"; stop_server; exit 1; }
fi
say "server up; distractor campaign (8 shards n4 e4 s1 replicate)"
MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.lab_pipeline \
  --families-file "$FAMS" --out "$OUT" --vllm-urls "$U" \
  --shards 8 --n-per-family 4 --epochs 4 --eval-seeds 1 --slack 2 \
  --conditions outcome --n-clusters 6 --M 3 --replicate --cond outcome \
  --world-seed 636363 --distractor-arm --deadline-hours 8 >> logs/run_distractor.log 2>&1
say "campaign done; analyses"
for a in "ontarget_analysis --cond outcome" "cite_rate --cond outcome" "malignant_pooled_snipw --key outcome --out distractor_pooled.json"; do
  MODEL_NAME="$M" $PYBIN -m experiments.$a --dir "$OUT/merged" >> logs/run_distractor.log 2>&1 || true
done
stop_server
say "T2-3 DISTRACTOR DONE -> $OUT/merged/ontarget_analysis.json"
