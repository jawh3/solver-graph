#!/bin/bash
# experiments/lab_autorun.sh — AUTONOMOUS master for the 2026-06-03 replication + capability ladder.
# Flow: wait for the 7B replication -> GATE on whether on-target poison REPLICATED -> free GPU1 ->
# run the capability ladder (per-model in-band probe -> clean causal campaign -> on-target analysis).
#
# SAFETY (shared machine): GPU1 ONLY (CUDA_VISIBLE_DEVICES=1). Kills ONLY hugglest-owned vllm
# servers that THIS script started (pid in /tmp/ladder.pid) or the idle 7B. NEVER touches GPU0
# or other users' processes. Resumable: skips a model whose ontarget_analysis.md already exists.
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
LOG=logs/autorun.log; mkdir -p logs
say(){ echo "[autorun $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
own(){ [ "$(ps -o user= -p "$1" 2>/dev/null | tr -d ' ')" = "hugglest" ]; }   # is pid hugglest-owned?

if [ "${LADDER_FORCE:-0}" = "1" ]; then
  say "LADDER_FORCE=1: bypassing rep-wait + replication gate (deliberate NEGATIVE-RESULT data-gathering across the capability ladder)."
else
REP=results/lab/campaign_7b35_M3_rep/merged
REPROOT=results/lab/campaign_7b35_M3_rep
say "waiting for 7B replication ontarget result (up to ~7.5h) ..."
for i in $(seq 1 600); do [ -f "$REP/ontarget_analysis.json" ] && break; sleep 45; done
if [ ! -f "$REP/ontarget_analysis.json" ]; then say "rep not done in time; abort"; exit 0; fi
# SAFETY: never free the GPU / proceed while rep shards are still running (up to 2h grace)
for i in $(seq 1 240); do pgrep -f "run_campaign --out $REPROOT" >/dev/null 2>&1 || break; sleep 30; done
if pgrep -f "run_campaign --out $REPROOT" >/dev/null 2>&1; then say "rep shards still running; abort to protect them"; exit 0; fi
say "7B replication finished + shards exited."

# GATE: only run the ladder if on-target poison REPLICATED on fresh data (reward CI excludes 0, mean<0)
GATE=$($PYBIN - <<'PY'
import json
try:
    d=json.load(open("results/lab/campaign_7b35_M3_rep/merged/ontarget_analysis.json"))
    p=d["by_label"]["R_final"]["on_target"].get("poison")
    print("PASS" if (p and p.get("excl0") and p.get("mean",0)<0) else "HOLD")
except Exception as e:
    print("HOLD")
PY
)
say "on-target replication gate => $GATE"
if [ "$GATE" != "PASS" ]; then
  say "HOLD: on-target poison did NOT replicate; NOT launching ladder. Flagging for user review."
  touch results/lab/LADDER_HELD_review_needed; exit 0
fi
fi   # end LADDER_FORCE bypass

# free GPU1: stop the now-idle 7B server (ONLY if hugglest-owned)
for pid in $(pgrep -f "vllm serve Qwen/Qwen2.5-7B-Instruct" 2>/dev/null); do
  if own "$pid"; then say "stopping idle 7B server pid $pid to free GPU1"; kill "$pid" 2>/dev/null; fi
done
sleep 25

PORT=8002
MODELS=("Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-14B-Instruct-AWQ" "mistralai/Mistral-7B-Instruct-v0.2")
TAGS=("3b" "1p5b" "0p5b" "14b" "mistral7b")
for idx in "${!MODELS[@]}"; do
  M="${MODELS[$idx]}"; TAG="${TAGS[$idx]}"
  OUT="results/lab/campaign_${TAG}_M3"; PROBE="results/lab/probe_${TAG}"
  if [ -f "$OUT/merged/ontarget_analysis.md" ]; then say "$TAG already done, skip"; continue; fi
  say "=== ladder rung: $M ($TAG) ==="
  CUDA_VISIBLE_DEVICES=1 setsid nohup "$VLLM" serve "$M" --max-model-len 8192 \
     --gpu-memory-utilization 0.80 --max-num-seqs 64 --enable-prefix-caching --port $PORT \
     > "logs/ladder_serve_${TAG}.log" 2>&1 &
  echo $! > /tmp/ladder.pid
  ok=0; for i in $(seq 1 144); do curl -s http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q "$M" && { ok=1; break; }; sleep 5; done
  if [ "$ok" != 1 ]; then say "serve FAILED for $M; skipping"; p=$(cat /tmp/ladder.pid 2>/dev/null); { [ -n "${p:-}" ] && own "$p" && kill -9 "$p" 2>/dev/null; }; continue; fi
  say "probing in-band families for $TAG"
  MODEL_NAME="$M" VLLM_URL="http://127.0.0.1:$PORT/v1" $PYBIN -m experiments.lab_probe \
     --max-families 80 --n-inst 2 --seeds 3 --slack 2 --workers 16 \
     --max-none 0.5 --min-helpful 0.3 --min-lift 0.25 --out "$PROBE" --max-output-tokens 384 >> "$LOG" 2>&1
  NB=$($PYBIN -c "import json;print(json.load(open('$PROBE/selection.json')).get('n_in_band',0))" 2>/dev/null || echo 0)
  say "$TAG in-band families = $NB"
  if [ "${NB:-0}" -lt 5 ]; then say "$TAG has <5 in-band families (floored); recording probe only, skipping campaign"; p=$(cat /tmp/ladder.pid 2>/dev/null); { [ -n "${p:-}" ] && own "$p" && { kill "$p"; sleep 8; kill -9 "$p" 2>/dev/null; }; }; continue; fi
  say "running campaign for $TAG on $NB families"
  MODEL_NAME="$M" VLLM_URL="http://127.0.0.1:$PORT/v1" $PYBIN -m experiments.lab_pipeline \
     --families-file "$PROBE/selection.json" --out "$OUT" --vllm-urls "http://127.0.0.1:$PORT/v1" \
     --shards 8 --n-per-family 6 --epochs 4 --eval-seeds 3 --slack 2 --conditions B0,outcome \
     --n-clusters 6 --M 3 --replicate --cond outcome --world-seed 777777 --deadline-hours 2.0 >> "$LOG" 2>&1
  p=$(cat /tmp/ladder.pid 2>/dev/null); { [ -n "${p:-}" ] && own "$p" && { say "stopping $TAG server pid $p"; kill "$p"; sleep 8; kill -9 "$p" 2>/dev/null; }; }
done
say "LADDER ALL DONE"
