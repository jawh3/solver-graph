#!/bin/bash
# run_h5.sh — H5 downstream-harm TRANSFER experiment on the powered high-signal rungs (14B, 32B).
# The pre-registered PRIMARY outcome that was built but never run. Measures whether a poison lesson
# PROMOTED into trusted memory by a credit gate causes real downstream solve-rate harm, on a transfer
# split where the family's TRUE lesson is withheld. Arms (force-injected): empty / clean(true) /
# poison_forced(max harm) / outcome_mem(poison iff outcome_Q>0) / causal_mem(poison iff causal_ATE>0)
# / shuffle_mem(matched-budget control). Headline = paired Δsolve(causal_mem - outcome_mem) + the
# clean vs poison_forced bracket, with bootstrap CI + exact McNemar. PREREG verdict: HARM if CI
# excludes 0 & lower bound >=5pp; bounded-NULL if within +-2.5pp. Each rung >=600 paired items.
# 14B-AWQ + 32B-AWQ are cached; TP=2 single endpoint (real_solve_fn uses one VLLM_URL). Resumable.
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
PORT=8002; U="http://127.0.0.1:$PORT/v1"
say(){ echo "[h5 $(date +%H:%M:%S)] $*" | tee -a logs/run_h5.log; }
own(){ [ "$(ps -o user= -p "$1" 2>/dev/null|tr -d ' ')" = "$USER" ]; }
other_user_on_gpu(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '|while read p;do [ -n "$p" ]&&ps -o user= -p $p 2>/dev/null|tr -d ' ';done|grep -vc "^${USER}$"; }
genok(){ curl -s --max-time 40 "$U/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"reply OK\"}],\"max_tokens\":5}" 2>/dev/null | grep -q '"content"'; }

# rung: TAG | MODEL | values-file | families-file
RUNGS=(
  "14b|Qwen/Qwen2.5-14B-Instruct-AWQ|results/lab/campaign_14b_M3_powered/merged/node_values.json|results/lab/probe_14b/selection.json"
  "32b|Qwen/Qwen2.5-32B-Instruct-AWQ|results/lab/campaign_32b_M3_tp2/merged/node_values.json|results/lab/probe_32b_tp2/selection.json"
)

for spec in "${RUNGS[@]}"; do
  IFS='|' read -r TAG M VALS FAMS <<< "$spec"
  OUT="results/lab/h5_${TAG}"
  if [ -f "$OUT/h5_summary.json" ]; then say "$TAG already has h5_summary.json, skip"; continue; fi
  [ "$(other_user_on_gpu)" != 0 ] && { say "another user on a GPU — stopping before $TAG."; exit 0; }
  say "=== H5 rung $TAG ($M) ==="
  : > logs/serve_h5_${TAG}.log
  CUDA_VISIBLE_DEVICES=0,1 setsid nohup "$VLLM" serve "$M" --tensor-parallel-size 2 --max-model-len 8192 \
    --gpu-memory-utilization 0.90 --max-num-seqs 48 --enable-prefix-caching --port $PORT \
    > logs/serve_h5_${TAG}.log 2>&1 &
  echo $! > /tmp/h5_${TAG}.pid
  ok=0; for i in $(seq 1 240); do
    grep -qiE "OutOfMemoryError|error initializing|No available memory" logs/serve_h5_${TAG}.log 2>/dev/null && { say "$TAG serve OOM/error. skip rung."; break; }
    genok "$M" && { ok=1; break; }; sleep 5
  done
  if [ "$ok" != 1 ]; then say "$TAG serve FAILED; skip."; p=$(cat /tmp/h5_${TAG}.pid); { [ -n "${p:-}" ] && own "$p" && kill -9 "$p" 2>/dev/null; }; continue; fi
  say "$TAG serves; running H5 transfer (n_inst=4 seeds=4 -> >=600 paired; arms empty/clean/poison_forced/outcome_mem/causal_mem/shuffle_mem)"
  MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.lab_h5 \
    --values-file "$VALS" --families-file "$FAMS" --theta 0.0 \
    --n-inst 4 --seeds 4 --slack 2 --workers 24 --out "$OUT" --max-output-tokens 384 >> logs/run_h5.log 2>&1
  say "$TAG H5 done -> $OUT/h5_summary.json"
  p=$(cat /tmp/h5_${TAG}.pid); { [ -n "${p:-}" ] && own "$p" && { say "stopping $TAG server $p"; kill "$p"; sleep 8; kill -9 "$p" 2>/dev/null; }; }
done
say "H5 ALL RUNGS DONE"
