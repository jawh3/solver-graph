#!/bin/bash
# experiments/phase2_keeper.sh - watchdog for the Phase-2 arms. Every 60 s: any arm from the launcher's table whose
# process has exited and whose campaign is not `done` is relaunched via STAGE=restart (resumes from its checkpoint).
# Also keeps both 14B servers alive. Logs to logs/phase2_keeper.log. Stop with: touch $ROOT/KEEPER_STOP
# Record of how the reported runs were sequenced (it uses the shipped result folders); to run the campaign again use
# phase2_launch.sh with a fresh ROOT (RESULTS.md §6).
ROOT=${ROOT:-results/lab/phase2_14b}; REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
export MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct-AWQ} PRIOR=${PRIOR:-none} CREDIT=${CREDIT:-cited_only} INIT=${INIT:-zero} ALPHA=${ALPHA:-mean} SCALING=${SCALING:-none}
IFS=',' read -r -a URLS <<< "${KEEPER_URLS:-http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1}"; i=0
# "GPU PORT" pairs separated by ';' (2026-09-09: a comma separator split the 32B rung's "0,1 8004" into "0" and "1 8004",
# so the keeper tried every ~30 min to start a second server on GPU0 with an empty port - 16 doomed attempts, no harm)
SERVERS=${KEEPER_SERVERS:-"0 8002;1 8003"}
export TP=${TP:-1} GPU_UTIL=${GPU_UTIL:-0.85} MAXSEQ=${MAXSEQ:-64}
say(){ echo "[keeper $(date '+%m-%d %H:%M:%S')] $*" >> logs/phase2_keeper.log; }
say "started (root=$ROOT)"
while [ ! -f "$ROOT/KEEPER_STOP" ]; do
  # servers: e.g. GPU0:8002 and GPU1:8003 (14B) or GPU0,1:8004 (32B, TP=2)
  IFS=';' read -r -a SRV <<< "$SERVERS"
  for gp in "${SRV[@]}"; do set -- $gp
    if ! curl -s --max-time 20 "http://127.0.0.1:$2/v1/models" | grep -q '"id"'; then
      say "server GPU$1:$2 down -> restarting"; STAGE=server GPU=$1 PORT=$2 ROOT=$ROOT bash experiments/phase2_launch.sh >> logs/phase2_keeper.log 2>&1
    fi
  done
  for a in "$ROOT"/arms/*/; do
    n=$(basename "$a"); [ -f "$a/shard_00/progress.json" ] || continue
    if ! pgrep -f "[r]un_campaign --out $a" >/dev/null; then
      done_=$($PYBIN -c "import json; p=json.load(open('$a/shard_00/progress.json')); print(all(c.get('done') for c in p['conditions'].values()))" 2>/dev/null)
      if [ "$done_" != "True" ]; then
        url=${URLS[$((i % ${#URLS[@]}))]}; i=$((i + 1))
        set -- ${SRV[0]}
        say "arm $n is dead and not done -> restart on $url"
        STAGE=restart ARM=$n ROOT=$ROOT GPU=$1 PORT=$2 ARMS_URLS=$url bash experiments/phase2_launch.sh >> logs/phase2_keeper.log 2>&1
      fi
    fi
  done
  sleep 60
done
say "stopped (KEEPER_STOP)"
