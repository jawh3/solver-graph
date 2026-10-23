#!/bin/bash
# experiments/nyu_rerun_finish.sh - how the reported NYU run finished, after the OFF arm had completed all its pairs. Use
# only once off.jsonl holds every (task, trial) pair and the main driver (experiments/nyu_rerun.sh) has been stopped
# (touch $ROOT/STOP, wait for "NYU RERUN DONE").
#   1. serve (same settings as nyu_rerun.sh), 2. ON-static resumes with --retry-errors at parallel PAR_STATIC (8: the slots
#   the finished OFF arm left idle), 3. one retry pass for BOTH arms (parallel 4 each), 4. stop.
# Same harness. No stop time unless DEADLINE_AT is set (the reported run used '2026-10-22 16:00 EDT'). To run it again, use
# the same ROOT as the nyu_rerun.sh run it finishes (the default is the shipped results). Never edit while a copy runs.
#   setsid nohup bash experiments/nyu_rerun_finish.sh > logs/nyu_rerun_finish.out 2>&1 &
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}; VLLM=${VLLM:-vllm}
export PYTHONPATH="$REPO" HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} CTF_SOLVE_ISOLATE=0 TOKENIZERS_PARALLELISM=false PATH=${CONDA_ENVS:-$HOME/miniconda3/envs}/grader/bin:$PATH
M=Qwen/Qwen3.8-27B-FP8; S=qwen_qwen3.8-27b-fp8; PORT=8010; CTX=32768; UTIL=0.92; MAXSEQ=8; PAR=4; PAR_STATIC=${PAR_STATIC:-8}; READY_WAIT_S=1500
DEADLINE_AT=${DEADLINE_AT:-}; DEADLINE=${DEADLINE_AT:+$(date -d "$DEADLINE_AT" +%s)}
ROOT=${ROOT:-results/realbench_nyu_v2}; OUT=$ROOT/$S; mkdir -p logs
LESSONS=results/realbench_nyu_mem/$S/lessons.jsonl
TEST_TASKS=$(cut -f1 docs/phase3/NYU_USABLE_test.txt | paste -sd,); N=$(echo "$TEST_TASKS" | tr ',' '\n' | wc -l)
LIMITS="--max-rounds 10 --max-output-tokens 8192 --trim-fraction 0.7 --image cybench/cybench:latest --command-timeout 300 --container-memory 8g --container-cpus 4 --container-user root --context-window $CTX --egress blocked"
BASE="--bench nyuctf --split test --tasks $TEST_TASKS --trials 3 --seed0 1000 --degenerate-after 0 $LIMITS --retry-errors"
say(){ echo "[finish $(date '+%m-%d %H:%M:%S')] $*" | tee -a logs/nyu_rerun.log; }
stop_server(){ p=$(cat /tmp/nyu_rerun_server.pid 2>/dev/null); [ -n "$p" ] && kill $p 2>/dev/null; sleep 15
  for q in $(pgrep -u "$(id -un)" -f "[v]llm serve"); do kill -9 $q 2>/dev/null; done
  for q in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do [ "$(ps -o user= -p $q 2>/dev/null | tr -d ' ')" = "$(id -un)" ] && kill -9 $q 2>/dev/null; done; }
wait_arms(){  # wait for the given PIDs; honour the freeze and the STOP file
  while ( for p in "$@"; do kill -0 $p 2>/dev/null && exit 0; done; exit 1 ); do
    [ -n "$DEADLINE" ] && [ "$(date +%s)" -ge "$DEADLINE" ] && touch "$ROOT/STOP" && say "freeze time reached"
    if [ -f "$ROOT/STOP" ]; then say "STOP file: killing $*"; kill "$@" 2>/dev/null; sleep 30; return 1; fi; sleep 30; done; }
n_off=$(grep -c '' "$OUT/off.jsonl" 2>/dev/null || echo 0)
[ "$n_off" -ge $((3 * N)) ] || { say "OFF has $n_off of $((3 * N)) records - not finished, not starting"; exit 1; }
pgrep -f "[r]ealbench.run_cybench" >/dev/null && { say "a run_cybench process is still running - stop the main driver first"; exit 1; }
others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | while read q; do u=$(ps -o user= -p $q 2>/dev/null | tr -d ' '); [ -n "$u" ] && [ "$u" != "$(id -un)" ] && echo "$u"; done | sort -u)
[ -n "$others" ] && { say "GPU in use by: $others - not starting"; exit 1; }
rm -f "$ROOT/STOP"
say "harness $(git rev-parse --short HEAD); serving $M (OFF complete: $n_off records)"
CUDA_VISIBLE_DEVICES=0,1 setsid nohup $VLLM serve "$M" --tensor-parallel-size 2 --max-model-len $CTX --gpu-memory-utilization $UTIL --max-num-seqs $MAXSEQ --enable-prefix-caching --port $PORT > logs/serve_nyu_rerun_finish.log 2>&1 &
echo $! > /tmp/nyu_rerun_server.pid; t0=$(date +%s)
until curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" | grep -q '"id"'; do
  pgrep -f "[v]llm serve" >/dev/null || { say "server died"; exit 1; }; [ $(( $(date +%s) - t0 )) -gt $READY_WAIT_S ] && { say "server not ready"; stop_server; exit 1; }; sleep 15; done
export MODEL_NAME="$M" VLLM_URL="http://127.0.0.1:$PORT/v1"
say "ON-static resumes at parallel $PAR_STATIC"
$PYBIN -m realbench.run_cybench $BASE --parallel $PAR_STATIC --service-parallel $PAR_STATIC --lessons "$LESSONS" --lessons-m 3 --out "$OUT/static.jsonl" > logs/nyu_rerun_static_finish.log 2>&1 & b=$!
wait_arms $b; wait $b; say "static exit $?; records $(grep -c '' "$OUT/static.jsonl")"
if [ ! -f "$ROOT/STOP" ]; then
  say "retry pass (both arms, parallel $PAR each)"
  $PYBIN -m realbench.run_cybench $BASE --parallel $PAR --service-parallel $PAR --out "$OUT/off.jsonl" > logs/nyu_rerun_off_retry.log 2>&1 & a=$!
  $PYBIN -m realbench.run_cybench $BASE --parallel $PAR --service-parallel $PAR --lessons "$LESSONS" --lessons-m 3 --out "$OUT/static.jsonl" > logs/nyu_rerun_static_retry.log 2>&1 & b=$!
  wait_arms $a $b; wait $a; ra=$?; wait $b; rb=$?; say "retry exit: off $ra, static $rb"
fi
stop_server
say "NYU RERUN DONE"
