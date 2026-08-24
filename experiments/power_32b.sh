#!/bin/bash
# power_32b.sh — staged, abort-gated attempt at a 32B rung on ONE GPU (GPU1).
# Answers: (a) does 32B-AWQ fit a 24GB card? (b) does it strengthen the weak 14B poison signal?
# Aborts cleanly if 32B OOMs on serve OR floors (too capable for the depth-2 grid -> too few in-band).
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
M="Qwen/Qwen2.5-32B-Instruct-AWQ"; PORT=8002; OUT=results/lab/campaign_32b_M3; PROBE=results/lab/probe_32b
say(){ echo "[pow32b $(date +%H:%M:%S)] $*" | tee -a logs/pow32b.log; }

say "serving $M (feasibility: ~19GB AWQ on a 24GB card, max-model-len 6144 to spare KV)"
CUDA_VISIBLE_DEVICES=1 setsid nohup "$VLLM" serve "$M" --max-model-len 6144 \
  --gpu-memory-utilization 0.95 --max-num-seqs 32 --enable-prefix-caching --port $PORT \
  > logs/serve_32b.log 2>&1 &
echo $! > /tmp/pow32b.pid
ok=0; for i in $(seq 1 200); do
  grep -qiE "OutOfMemoryError|No available memory|CUDA out of memory" logs/serve_32b.log 2>/dev/null && { say "32B OOM on serve -> does NOT fit one GPU. ABORT."; break; }
  curl -s http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q "32B" && { ok=1; break; }
  sleep 5
done
if [ "$ok" != 1 ]; then say "32B serve failed/OOM. Conclusion: 32B-AWQ does not run on one 24GB GPU here."; p=$(cat /tmp/pow32b.pid); { [ -n "${p:-}" ] && [ "$(ps -o user= -p $p|tr -d ' ')" = "$USER" ] && kill -9 $p 2>/dev/null; }; exit 1; fi
say "32B SERVES on one GPU. Probing depth-2 (hard) families for in-band selection."

MODEL_NAME="$M" VLLM_URL=http://127.0.0.1:$PORT/v1 $PYBIN -m experiments.lab_probe \
  --difficulty hard --max-families 60 --n-inst 2 --seeds 3 --slack 2 --workers 12 \
  --max-none 0.5 --min-helpful 0.3 --min-lift 0.25 --out "$PROBE" --max-output-tokens 384 >> logs/pow32b.log 2>&1
NB=$($PYBIN -c "import json;print(json.load(open('$PROBE/selection.json')).get('n_in_band',0))" 2>/dev/null || echo 0)
say "32B in-band families = $NB (depth-2 grid)"
if [ "${NB:-0}" -lt 8 ]; then
  say "32B FLOORED at $NB in-band (too capable for depth-2 families; a 32B rung needs DEEPER 3-4 layer families = a benchmark extension). Recording probe only, ABORT campaign."
  p=$(cat /tmp/pow32b.pid); { [ -n "${p:-}" ] && [ "$(ps -o user= -p $p|tr -d ' ')" = "$USER" ] && { kill $p; sleep 8; kill -9 $p 2>/dev/null; }; }; exit 0
fi

say "running 32B outcome-only campaign on $NB families (n12, fresh seed)"
MODEL_NAME="$M" VLLM_URL=http://127.0.0.1:$PORT/v1 $PYBIN -m experiments.lab_pipeline \
  --families-file "$PROBE/selection.json" --out "$OUT" --vllm-urls http://127.0.0.1:$PORT/v1 \
  --shards 8 --n-per-family 12 --epochs 4 --eval-seeds 3 --slack 2 --conditions outcome --n-clusters 6 \
  --M 3 --replicate --cond outcome --world-seed 555555 --deadline-hours 12 >> logs/pow32b.log 2>&1
say "32B campaign done; CRT + cite-rate"
MODEL_NAME="$M" $PYBIN -m experiments.crt_test --dir "$OUT/merged" --key R_final >> logs/pow32b.log 2>&1 || true
MODEL_NAME="$M" $PYBIN -m experiments.cite_rate --dir "$OUT/merged" >> logs/pow32b.log 2>&1 || true
p=$(cat /tmp/pow32b.pid); { [ -n "${p:-}" ] && [ "$(ps -o user= -p $p|tr -d ' ')" = "$USER" ] && { say "stopping 32B server $p"; kill $p; sleep 8; kill -9 $p 2>/dev/null; }; }
say "POWER 32B DONE"
