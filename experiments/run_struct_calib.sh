#!/bin/bash
# run_struct_calib.sh — CALIBRATION (design gate, NOT the headline): does struct_domain hit the
# non-ceiling helpful band [0.20,0.90] AND show interference (helpful+poison < helpful)? Serve 14B GPU0,
# run constructed_loo on 12 families. Inspect BEFORE freezing the prereg / running the 75-family headline.
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
export CTF_DOMAIN=benchmark.struct_domain
M="Qwen/Qwen2.5-14B-Instruct-AWQ"; PORT=8002; U="http://127.0.0.1:$PORT/v1"
say(){ echo "[scal $(date +%H:%M:%S)] $*" | tee -a logs/run_struct_calib.log; }
own(){ [ "$(ps -o user= -p "$1" 2>/dev/null|tr -d ' ')" = "$USER" ]; }
genok(){ curl -s --max-time 40 "$U/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"OK\"}],\"max_tokens\":5}" 2>/dev/null|grep -q '"content"'; }
g0(){ local b=0; for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '); do [ -n "$p" ]&&[ "$(ps -o user= -p $p 2>/dev/null|tr -d ' ')" != "$USER" ]&&b=1;done;[ $b = 0 ];}
stopsrv(){ p=$(cat /tmp/scal.pid 2>/dev/null);{ [ -n "${p:-}" ]&&own "$p"&&{ kill "$p" 2>/dev/null;sleep 6;kill -9 "$p" 2>/dev/null;};};}
mkdir -p logs
g0 || { say "GPU0 busy (other user) abort"; exit 0; }
if ! genok; then
  say "serving $M GPU0:$PORT"; : > logs/serve_scal.log
  CUDA_VISIBLE_DEVICES=0 setsid nohup "$VLLM" serve "$M" --max-model-len 8192 --gpu-memory-utilization 0.82 --max-num-seqs 48 --enable-prefix-caching --port $PORT > logs/serve_scal.log 2>&1 &
  echo $! > /tmp/scal.pid
  ok=0; for i in $(seq 1 240);do grep -qiE "OutOfMemory|error initializing|No available memory" logs/serve_scal.log 2>/dev/null&&break; genok&&{ ok=1;break;};sleep 5;done
  [ "$ok" = 1 ]||{ say "serve FAILED";stopsrv;exit 1;}
fi
say "server up; constructed_loo calibration (12 struct families, n_inst4 seeds2)"
MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.constructed_loo --families-file results/lab/probe_struct_calib/selection.json \
  --n-inst 4 --seeds 2 --workers 24 --workroot results/lab/scal_work --out results/lab/constructed_loo_struct_calib.json >> logs/run_struct_calib.log 2>&1
stopsrv
say "STRUCT CALIB DONE"
