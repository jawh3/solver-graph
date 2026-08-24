#!/bin/bash
# run_internlm.sh — THIRD architecture (after Qwen + Llama): InternLM2.5-7B-chat (non-gated, agentic-tuned).
# Breadth of cross-architecture generality. Same harness as run_llama.sh: dual TP=1 (GPU0:8000 + GPU1:8001),
# GENERATION health check, probe standard grid, campaign on both urls if >=8 in-band else probe-only.
# InternLM ships custom modeling code -> --trust-remote-code. Courtesy: abort if another user on a GPU.
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
M="internlm/internlm2_5-7b-chat"; TAG="internlm7b"
OUT="results/lab/campaign_${TAG}_M3"; PROBE="results/lab/probe_${TAG}"
U0="http://127.0.0.1:8000/v1"; U1="http://127.0.0.1:8001/v1"
say(){ echo "[internlm $(date +%H:%M:%S)] $*" | tee -a logs/run_internlm.log; }
own(){ [ "$(ps -o user= -p "$1" 2>/dev/null|tr -d ' ')" = "$USER" ]; }
other_user_on_gpu(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '|while read p;do [ -n "$p" ]&&ps -o user= -p $p 2>/dev/null|tr -d ' ';done|grep -vc "^${USER}$"; }
genok(){ curl -s --max-time 40 "$1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"reply OK\"}],\"max_tokens\":5}" 2>/dev/null | grep -q '"content"'; }

[ "$(other_user_on_gpu)" != 0 ] && { say "another user on a GPU — not launching."; exit 0; }

say "serving $M dual TP=1 (GPU0:8000 + GPU1:8001), fp16 len8192, trust-remote-code"
: > logs/serve_internlm_g0.log; : > logs/serve_internlm_g1.log
CUDA_VISIBLE_DEVICES=0 setsid nohup "$VLLM" serve "$M" --max-model-len 8192 --trust-remote-code \
  --gpu-memory-utilization 0.80 --max-num-seqs 64 --enable-prefix-caching --port 8000 \
  > logs/serve_internlm_g0.log 2>&1 &
echo $! > /tmp/internlm_g0.pid
CUDA_VISIBLE_DEVICES=1 setsid nohup "$VLLM" serve "$M" --max-model-len 8192 --trust-remote-code \
  --gpu-memory-utilization 0.80 --max-num-seqs 64 --enable-prefix-caching --port 8001 \
  > logs/serve_internlm_g1.log 2>&1 &
echo $! > /tmp/internlm_g1.pid

ok=0; for i in $(seq 1 180); do
  grep -qiE "OutOfMemoryError|error initializing|Engine core initialization failed" logs/serve_internlm_g0.log logs/serve_internlm_g1.log 2>/dev/null && { say "serve error/OOM. ABORT."; break; }
  if genok "$U0" && genok "$U1"; then ok=1; break; fi
  sleep 5
done
if [ "$ok" != 1 ]; then
  say "dual serve FAILED (no generation). cleanup + abort."
  for f in /tmp/internlm_g0.pid /tmp/internlm_g1.pid; do p=$(cat $f 2>/dev/null); { [ -n "${p:-}" ] && own "$p" && kill -9 "$p" 2>/dev/null; }; done
  exit 1
fi
say "BOTH InternLM servers GENERATE. Probing standard grid for in-band families."

MODEL_NAME="$M" VLLM_URL="$U0" $PYBIN -m experiments.lab_probe \
  --max-families 80 --n-inst 2 --seeds 3 --slack 2 --workers 24 \
  --max-none 0.5 --min-helpful 0.3 --min-lift 0.25 --out "$PROBE" --max-output-tokens 384 >> logs/run_internlm.log 2>&1
NB=$($PYBIN -c "import json;print(json.load(open('$PROBE/selection.json')).get('n_in_band',0))" 2>/dev/null || echo 0)
say "$TAG in-band families (standard grid) = $NB"
if [ "${NB:-0}" -lt 8 ]; then
  say "$TAG has <8 in-band (floored OR too-capable). Recording probe only, skip campaign."
  for f in /tmp/internlm_g0.pid /tmp/internlm_g1.pid; do p=$(cat $f 2>/dev/null); { [ -n "${p:-}" ] && own "$p" && { kill "$p"; sleep 6; kill -9 "$p" 2>/dev/null; }; }; done
  exit 0
fi

say "running $TAG outcome campaign on $NB families (n12, both GPUs)"
MODEL_NAME="$M" VLLM_URL="$U0" $PYBIN -m experiments.lab_pipeline \
  --families-file "$PROBE/selection.json" --out "$OUT" --vllm-urls "$U0,$U1" \
  --shards 8 --n-per-family 12 --epochs 4 --eval-seeds 3 --slack 2 --conditions outcome --n-clusters 6 \
  --M 3 --replicate --cond outcome --world-seed 929292 --deadline-hours 10 >> logs/run_internlm.log 2>&1
say "$TAG campaign done; CRT + cite-rate"
MODEL_NAME="$M" $PYBIN -m experiments.crt_test --dir "$OUT/merged" --key R_final >> logs/run_internlm.log 2>&1 || true
MODEL_NAME="$M" $PYBIN -m experiments.cite_rate --dir "$OUT/merged" >> logs/run_internlm.log 2>&1 || true
for f in /tmp/internlm_g0.pid /tmp/internlm_g1.pid; do p=$(cat $f 2>/dev/null); { [ -n "${p:-}" ] && own "$p" && { say "stopping server $p"; kill "$p"; sleep 6; kill -9 "$p" 2>/dev/null; }; }; done
say "INTERNLM RUNG DONE (n_in_band=$NB)"
