#!/bin/bash
# experiments/nyu_rerun.sh - the NYU CTF Bench run behind the reported results: OFF and ON-static side by side.
# One server (Qwen3.8-27B-FP8); two run_cybench processes at once, one per arm, each parallel PAR (4; the first hour ran at 3) /
# service-parallel PAR, the usable test tasks x 3 trials (seeds 1000-1002, trial-major). Egress blocked; per-solve service
# networks. The main pass resumes from existing records (--retry-errors keeps clean records, re-runs errored + missing).
# Then one --retry-errors pass per arm. Records: results/realbench_nyu_v2/<slug>/{off,static}.jsonl. Standalone (never edit
# while a copy runs). Stops at DEADLINE_AT if set (the reported run: 2026-10-22 16:00 EDT) or early on
# touch $ROOT/STOP (the arms are killed, the server stopped).
#   setsid nohup bash experiments/nyu_rerun.sh > logs/nyu_rerun.out 2>&1 &
# To run it again: ROOT=<fresh dir> (the default is the shipped results, which it would resume into); no stop time unless
# DEADLINE_AT is set (the reported run used DEADLINE_AT='2026-10-22 16:00 EDT'); needs 2 GPUs (TP=2), the
# cybench/cybench image and $NYUCTF_ROOT.
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}; VLLM=${VLLM:-vllm}
export PYTHONPATH="$REPO" HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} CTF_SOLVE_ISOLATE=0 TOKENIZERS_PARALLELISM=false PATH=${CONDA_ENVS:-$HOME/miniconda3/envs}/grader/bin:$PATH
M=Qwen/Qwen3.8-27B-FP8; S=qwen_qwen3.8-27b-fp8; PORT=8010; CTX=32768; UTIL=0.92; MAXSEQ=8; PAR=${PAR:-4}; READY_WAIT_S=1500
DEADLINE_AT=${DEADLINE_AT:-}; DEADLINE=${DEADLINE_AT:+$(date -d "$DEADLINE_AT" +%s)}
ROOT=${ROOT:-results/realbench_nyu_v2}; OUT=$ROOT/$S; mkdir -p "$OUT" logs
LESSONS=results/realbench_nyu_mem/$S/lessons.jsonl
TEST_TASKS=$(cut -f1 docs/phase3/NYU_USABLE_test.txt | paste -sd,)
LIMITS="--max-rounds 10 --max-output-tokens 8192 --trim-fraction 0.7 --image cybench/cybench:latest --command-timeout 300 --container-memory 8g --container-cpus 4 --container-user root --context-window $CTX --egress blocked"
RUN="--bench nyuctf --split test --tasks $TEST_TASKS --trials 3 --seed0 1000 --parallel $PAR --service-parallel $PAR --degenerate-after 0 $LIMITS"
say(){ echo "[rerun $(date '+%m-%d %H:%M:%S')] $*" | tee -a logs/nyu_rerun.log; }
stop_server(){ p=$(cat /tmp/nyu_rerun_server.pid 2>/dev/null); [ -n "$p" ] && kill $p 2>/dev/null; sleep 15
  for q in $(pgrep -u "$(id -un)" -f "[v]llm serve"); do kill -9 $q 2>/dev/null; done
  for q in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do [ "$(ps -o user= -p $q 2>/dev/null | tr -d ' ')" = "$(id -un)" ] && kill -9 $q 2>/dev/null; done; }
others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | while read q; do u=$(ps -o user= -p $q 2>/dev/null | tr -d ' '); [ -n "$u" ] && [ "$u" != "$(id -un)" ] && echo "$u"; done | sort -u)
[ -n "$others" ] && { say "GPU in use by: $others - not starting"; exit 1; }
[ -f "$LESSONS" ] || { say "lesson pool missing: $LESSONS"; exit 1; }
say "harness $(git rev-parse --short HEAD); serving $M"
CUDA_VISIBLE_DEVICES=0,1 setsid nohup $VLLM serve "$M" --tensor-parallel-size 2 --max-model-len $CTX --gpu-memory-utilization $UTIL --max-num-seqs $MAXSEQ --enable-prefix-caching --port $PORT > logs/serve_nyu_rerun.log 2>&1 &
echo $! > /tmp/nyu_rerun_server.pid; t0=$(date +%s)
until curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" | grep -q '"id"'; do
  pgrep -f "[v]llm serve" >/dev/null || { say "server died"; exit 1; }; [ $(( $(date +%s) - t0 )) -gt $READY_WAIT_S ] && { say "server not ready"; stop_server; exit 1; }; sleep 15; done
export MODEL_NAME="$M" VLLM_URL="http://127.0.0.1:$PORT/v1"
arms(){  # $1 = extra flag for both (e.g. --retry-errors), $2 = log suffix
  $PYBIN -m realbench.run_cybench $RUN --out "$OUT/off.jsonl" $1 > "logs/nyu_rerun_off$2.log" 2>&1 & a=$!
  $PYBIN -m realbench.run_cybench $RUN --lessons "$LESSONS" --lessons-m 3 --out "$OUT/static.jsonl" $1 > "logs/nyu_rerun_static$2.log" 2>&1 & b=$!
  while kill -0 $a 2>/dev/null || kill -0 $b 2>/dev/null; do
    [ -n "$DEADLINE" ] && [ "$(date +%s)" -ge "$DEADLINE" ] && touch "$ROOT/STOP" && say "freeze time reached"
    if [ -f "$ROOT/STOP" ]; then say "STOP file: killing the arms"; kill $a $b 2>/dev/null; sleep 30; break; fi; sleep 30; done
  wait $a; ra=$?; wait $b; rb=$?; say "arms$2 exit: off $ra, static $rb; records off $(grep -c '' "$OUT/off.jsonl" 2>/dev/null), static $(grep -c '' "$OUT/static.jsonl" 2>/dev/null)"; }
say "OFF + ON-static: $(echo "$TEST_TASKS" | tr , "\n" | grep -c .) test tasks x 3 seeds each, side by side, parallel $PAR per arm"
RESUME=""; { [ -s "$OUT/off.jsonl" ] || [ -s "$OUT/static.jsonl" ]; } && RESUME="--retry-errors"   # keep finished records
arms "$RESUME" ""
[ -f "$ROOT/STOP" ] || { say "retry pass"; arms "--retry-errors" "_retry"; }
stop_server
say "NYU RERUN DONE"
