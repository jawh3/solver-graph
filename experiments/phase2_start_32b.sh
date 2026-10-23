#!/bin/bash
# experiments/phase2_start_32b.sh - starts the 32B rung once every
# 14B job is finished: waits for all run_campaign / constructed_loo processes to exit, stops both 14B servers,
# serves Qwen2.5-32B-Instruct-AWQ with tensor parallelism over both GPUs, launches the full arm table
# (all_arms incl. the causal arm) under results/lab/phase2_32b, and starts a keeper for that root.
# Record of how the reported runs were sequenced (it uses the shipped result folders); to run the campaign again use
# phase2_launch.sh with a fresh ROOT (RESULTS.md §6).
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
say(){ echo "[start32b $(date '+%m-%d %H:%M:%S')] $*" | tee -a logs/phase2_start_32b.log; }
say "waiting for all 14B jobs to finish"
while pgrep -f "[r]un_campaign --out results/lab/phase2_14b" >/dev/null || pgrep -f "[c]onstructed_loo" >/dev/null || pgrep -f "[p]hase2_gpu1_after.sh" >/dev/null; do sleep 300; done
say "14B jobs done; stopping the 14B keeper and servers"
touch results/lab/phase2_14b/KEEPER_STOP; sleep 65
for f in /tmp/phase2_gpu0.pid /tmp/phase2_gpu1.pid /tmp/smoke14b.pid; do p=$(cat $f 2>/dev/null); [ -n "$p" ] && [ "$(ps -o user= -p $p 2>/dev/null|tr -d ' ')" = "$USER" ] && { say "kill server $p"; kill $p; }; done
sleep 20; for p in $(pgrep -f "[v]llm serve Qwen/Qwen2.5-14B"); do kill -9 $p 2>/dev/null; done; sleep 10
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a logs/phase2_start_32b.log
export MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ GPU=0,1 TP=2 PORT=8004 ROOT=results/lab/phase2_32b MAXSEQ=48 GPU_UTIL=0.88
export PRIOR=none CREDIT=cited_only INIT=zero ALPHA=mean SCALING=none FAMS_FILE=results/lab/phase2_14b/families.txt
mkdir -p $ROOT; cp results/lab/phase2_14b/families.txt results/lab/phase2_14b/families.json $ROOT/
STAGE=server bash experiments/phase2_launch.sh >> logs/phase2_start_32b.log 2>&1 || { say "32B server failed"; exit 1; }
STAGE=arms bash experiments/phase2_launch.sh >> logs/phase2_start_32b.log 2>&1
say "32B arms launched: $(pgrep -f '[r]un_campaign --out results/lab/phase2_32b' | wc -l)"
ROOT=results/lab/phase2_32b MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ KEEPER_SERVERS="0,1 8004" KEEPER_URLS=http://127.0.0.1:8004/v1 TP=2 GPU_UTIL=0.88 MAXSEQ=48 \
  setsid nohup bash experiments/phase2_keeper.sh > /dev/null 2>&1 &
say "32B keeper started"
