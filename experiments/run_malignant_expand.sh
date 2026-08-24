#!/bin/bash
# run_malignant_expand.sh — T1-1 EXPANSION: add more in-band families so the equal-weight lesson-level
# SNIPW clears .05 for M-exhaust too. The label-level SE is between-family-heterogeneity-limited, so the
# ONLY lever is more FAMILIES (reduces between/sqrt(n_lessons)); more runs-per-family is futile. Runs on
# GPU1 ONLY (GPU0 is occupied by another user — never contended). serve -> probe candidates for in-band
# (same headroom criteria) -> malignant campaign (same 3 arms, same params) on the NEW in-band families.
# Per-family ATEs are pooled with results/lab/campaign_14b_malig afterwards (malignant_pooled_snipw --dirs).
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
M="Qwen/Qwen2.5-14B-Instruct-AWQ"; GPU=1; PORT=8003; U="http://127.0.0.1:$PORT/v1"
ARMS="destroy,exhaust,wrongflag"
PROBE_OUT="results/lab/probe_14b_expand"
FULL="results/lab/campaign_14b_malig_expand"
CANDS=$(cat /tmp/malig_candidates.txt)
mkdir -p logs
say(){ echo "[expand $(date +%H:%M:%S)] $*" | tee -a logs/run_malignant_expand.log; }
own(){ [ "$(ps -o user= -p "$1" 2>/dev/null|tr -d ' ')" = "$USER" ]; }
genok(){ curl -s --max-time 40 "$U/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"reply OK\"}],\"max_tokens\":5}" 2>/dev/null | grep -q '"content"'; }
# GPU1-specific guard: refuse to serve if a NON-hugglest compute proc is on GPU1
gpu1_clear(){ local bad=0; for p in $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '); do [ -n "$p" ] && [ "$(ps -o user= -p $p 2>/dev/null|tr -d ' ')" != "$USER" ] && bad=1; done; [ $bad = 0 ]; }
stop_server(){ p=$(cat /tmp/malig_expand.pid 2>/dev/null); { [ -n "${p:-}" ] && own "$p" && { say "stopping server $p"; kill "$p" 2>/dev/null; sleep 8; kill -9 "$p" 2>/dev/null; }; }; }

gpu1_clear || { say "GPU1 has another user's process — NOT launching."; exit 0; }

# ---- serve on GPU1 (skip if healthy) ----
if ! genok; then
  say "serving $M on GPU$GPU:$PORT (TP=1)"; : > logs/serve_malig_expand.log
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup "$VLLM" serve "$M" --max-model-len 8192 \
    --gpu-memory-utilization 0.82 --max-num-seqs 48 --enable-prefix-caching --port $PORT \
    > logs/serve_malig_expand.log 2>&1 &
  echo $! > /tmp/malig_expand.pid
  ok=0; for i in $(seq 1 240); do
    grep -qiE "OutOfMemoryError|error initializing|No available memory" logs/serve_malig_expand.log 2>/dev/null && { say "serve OOM/error; abort"; break; }
    genok && { ok=1; break; }; sleep 5
  done
  [ "$ok" = 1 ] || { say "serve FAILED; abort"; stop_server; exit 1; }
fi
say "server healthy at $U"

# ---- PROBE candidates -> new in-band selection ----
if [ ! -f "$PROBE_OUT/selection.json" ]; then
  say "probing $(echo $CANDS|tr ',' '\n'|wc -l) candidate families for in-band (headroom criteria)"
  MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.lab_probe \
    --families "$CANDS" --n-inst 2 --seeds 3 --slack 2 --workers 24 \
    --out "$PROBE_OUT" --max-output-tokens 512 >> logs/run_malignant_expand.log 2>&1
fi
NB=$($PYBIN -c "import json;d=json.load(open('$PROBE_OUT/selection.json'));print(len(d.get('in_band_families') or []))" 2>/dev/null || echo 0)
say "probe done: $NB NEW in-band families -> $PROBE_OUT/selection.json"
[ "${NB:-0}" -lt 8 ] && { say "too few new in-band ($NB); stopping (investigate)."; stop_server; exit 2; }

# ---- malignant campaign on the NEW in-band families (same 3 arms, same params, distinct seed) ----
if [ ! -f "$FULL/merged/malignant_pooled_snipw.json" ]; then
  say "FULL expansion campaign on $NB new families (shards8 n4 e4 s1 M3 replicate, arms=$ARMS)"
  MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.lab_pipeline \
    --families-file "$PROBE_OUT/selection.json" --out "$FULL" --vllm-urls "$U" \
    --shards 8 --n-per-family 4 --epochs 4 --eval-seeds 1 --slack 2 \
    --conditions outcome --n-clusters 6 --M 3 --replicate --cond outcome \
    --world-seed 818181 --malignant-arms "$ARMS" --deadline-hours 8 >> logs/run_malignant_expand.log 2>&1
  say "expansion campaign done; per-arm analyses"
  for a in "ontarget_analysis --cond outcome" "cite_rate --cond outcome" "malignant_probe_check --min-cited 10 --out malignant_result.json" "malignant_pooled_snipw --key outcome --out malignant_pooled_snipw.json" "malignant_pooled_snipw --key R_final --out malignant_pooled_snipw_rfinal.json"; do
    MODEL_NAME="$M" $PYBIN -m experiments.$a --dir "$FULL/merged" >> logs/run_malignant_expand.log 2>&1 || true
  done
fi

stop_server
say "T1-1 EXPANSION DONE -> $FULL/merged/ (pool with campaign_14b_malig via malignant_pooled_snipw --dirs)"
