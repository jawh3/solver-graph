#!/bin/bash
# experiments/phase2_launch.sh - Phase-2 launcher. Every design parameter is an environment variable; the
# defaults are the 14B rung's values (the 32B rung set MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ, GPU=0,1, TP=2). To run it
# again set ROOT to a fresh directory (the default is the shipped results/lab/phase2_14b, which it would resume into)
# and FAMS_FILE=results/lab/phase2_14b/families.txt.
#
#   STAGE=probe   : family probe (label-blind in-band rule) -> $ROOT/probe/selection.json -> balanced list
#                   via hard_ctf.select_balanced_families -> $ROOT/families.txt   (~1.3-2 h at 14B)
#   STAGE=arms    : one sequential learner per (arm, context key) as separate processes on one server:
#                   deployed@{global,cluster,alphabet,family}, deployed_obs@alphabet, outcome@alphabet,
#                   S0 lure (deployed@alphabet --distractor-arm), S1 (deployed@alphabet --poison-kind cross_outer);
#                   eval-only arms B0, B2det, oracle_static, worst_static (no learner).
#   STAGE=causal  : the main causal-credit (leave-one-out) arm alone (the reported runs started it after the 14B arms;
#                   STAGE=arms now lists every arm, causal ones included; CAUSAL_EXTRA=0 leaves out the extra causal arms).
#   STAGE=checks  : smoke_checks per key + learning_curve per deployed arm.
#   STAGE=server  : only bring up the vLLM server for $MODEL on $GPU:$PORT (a second server for ARMS_URLS).
# Env: ROOT (out dir), MODEL, PORT, GPU, PER_OUTER (3 -> 12 families; 6 -> 24), N_PER_FAMILY, EPOCHS, EVAL_SEEDS,
#      PRIOR (cosine|rerank), FAMS_FILE (override), DEADLINE_H.
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
STAGE=${STAGE:?set STAGE=probe|arms|causal|checks|server|restart}
ROOT=${ROOT:-results/lab/phase2_14b}
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct-AWQ}; PORT=${PORT:-8002}; GPU=${GPU:-0}; U="http://127.0.0.1:$PORT/v1"
PER_OUTER=${PER_OUTER:-6}; N_PER_FAMILY=${N_PER_FAMILY:-108}; EPOCHS=${EPOCHS:-2}; EVAL_SEEDS=${EVAL_SEEDS:-2}   # D8: 24 families
MAXSEQ=${MAXSEQ:-64}; TP=${TP:-1}; GPU_UTIL=${GPU_UTIL:-0.85}   # 32B rung: GPU=0,1 TP=2 (does not fit one card at 8k)
PRIOR=${PRIOR:-none}; CREDIT=${CREDIT:-cited_only}; WORLD_SEED=${WORLD_SEED:-424242}; SLACK=${SLACK:-2}
INIT=${INIT:-zero}; ALPHA=${ALPHA:-mean}; SCALING=${SCALING:-none}   # the learner the reported runs used is the DEFAULT
mkdir -p logs "$ROOT"
say(){ echo "[phase2 $(date +%H:%M:%S)] $*" | tee -a logs/phase2_launch.log; }
own(){ [ "$(ps -o user= -p "$1" 2>/dev/null|tr -d ' ')" = "$USER" ]; }
genok(){ curl -s --max-time 40 "$U/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"OK\"}],\"max_tokens\":5}" 2>/dev/null | grep -q '"content"'; }
ensure_server(){
  for p in $(nvidia-smi -i ${GPU%%,*} --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '); do
    [ -n "$p" ] && ! own "$p" && { say "GPU$GPU busy (other user pid $p) - abort"; exit 0; }
  done
  if ! genok; then
    say "serving $MODEL on GPU$GPU:$PORT"
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup "$VLLM" serve "$MODEL" --max-model-len 8192 --gpu-memory-utilization $GPU_UTIL \
      --tensor-parallel-size $TP --max-num-seqs $MAXSEQ --enable-prefix-caching --port $PORT > "logs/serve_phase2_gpu${GPU//,/_}.log" 2>&1 &
    echo $! > "/tmp/phase2_gpu${GPU//,/_}.pid"
    ok=0; for i in $(seq 1 360); do genok && { ok=1; break; }; sleep 5; done
    [ "$ok" = 1 ] || { say "serve FAILED"; exit 1; }
  fi
  say "server up"
}
COMMON="--epochs $EPOCHS --eval-seeds $EVAL_SEEDS --n-per-family $N_PER_FAMILY --slack $SLACK --M 3 --n-clusters 4 \
  --tau 0.1 --epsilon-explore 0.25 --tau-explore 1.0 --exploration-design stratified --citation-mode distinctive \
  --query-diagnostics charset --recall-all --prior-source $PRIOR --credit-mode $CREDIT --init-mode $INIT --alpha-schedule $ALPHA --distance-scaling $SCALING \
  --context-window 8192 --max-output-tokens 512 --trim-fraction 0.7 --world-seed $WORLD_SEED"

all_arms(){ # name key cond [extra flags...]  - the single source of truth for the arm table
  for key in global cluster alphabet family; do            # D1 all rungs x D5 both temperatures
    echo "deployed_${key}_tau01 $key deployed --tau 0.1"
    echo "deployed_${key}_tau02 $key deployed --tau 0.2"
  done
  echo "deployed_obs_alphabet alphabet deployed_obs"
  # D10: the legacy learner runs ONLY as the tau=1 replication arm, with its legacy settings
  echo "outcome_legacy_alphabet alphabet outcome --prior-source cosine --init-mode optimistic --alpha-schedule ema --distance-scaling asymmetric"
  echo "s0_lure_alphabet alphabet deployed --distractor-arm"
  echo "s1_cross_outer_alphabet alphabet deployed --poison-kind cross_outer"
  # S4 (F20 re-run with isolation): the legacy malignant texts, plus their clean variants
  echo "s4_malignant_legacy_alphabet alphabet deployed --malignant-arms destroy,exhaust,wrongflag"
  echo "s4_malignant_clean_alphabet alphabet deployed --malignant-arms destroy_clean,exhaust_clean,wrongflag_clean"
  echo "B0 alphabet B0"; echo "B2det alphabet B2det"; echo "oracle_static alphabet oracle_static"; echo "worst_static alphabet worst_static"
  # the causal-credit arm: launched with the others by STAGE=arms (the reported runs started it alone, with STAGE=causal)
  echo "deployed_causal_alphabet alphabet deployed_causal --tau 0.1"
  # review 2026-09-08 F3 (P2-7): replication arms of the primary contrast with INDEPENDENT train draws (REPS=0 disables)
  for i in $(seq 1 ${REPS:-2}); do echo "deployed_alphabet_tau01_rep$i alphabet deployed --tau 0.1 --draw-seed-offset $i"; done
  # P2-15 (2026-09-12): causal credit under the FAMILY context and two causal replications with independent draws (CAUSAL_EXTRA=0 disables)
  if [ "${CAUSAL_EXTRA:-1}" = 1 ]; then
    echo "deployed_causal_family family deployed_causal --tau 0.1"
    for i in 1 2; do echo "deployed_causal_alphabet_rep$i alphabet deployed_causal --tau 0.1 --draw-seed-offset $i"; done
  fi
}

case "$STAGE" in
server)
  ensure_server
  ;;
probe)
  ensure_server
  # candidates: 10 grid families per outer alphabet, stratified over depth, fixed seed
  CANDS=$($PYBIN - <<'PY'
from benchmark import hard_ctf as h
import numpy as np
rng = np.random.RandomState(20260905); out = []
for outer in ("b64", "hex", "b32", "b85"):
    names = sorted(n for n in h._BY_NAME if n.startswith(outer + "_"))
    by_depth = {}
    for n in names: by_depth.setdefault(len(h._inner_tags_of(n)), []).append(n)
    for d, grp in sorted(by_depth.items()):
        k = {1: 5, 2: 6, 3: 5}.get(d, 5)          # 16 candidates per alphabet -> 64 probed (D8 needs 6 in-band per alphabet)
        out += list(rng.choice(grp, size=min(k, len(grp)), replace=False))
print(",".join(out))
PY
)
  say "probing $(echo $CANDS | tr ',' '\n' | wc -l) candidate families"
  MODEL_NAME="$MODEL" VLLM_URL="$U" $PYBIN -m experiments.lab_probe --families "$CANDS" --n-inst 4 --seeds 4 \
    --workers 24 --out "$ROOT/probe" --max-output-tokens 512 >> logs/phase2_launch.log 2>&1
  $PYBIN - <<PY
import json
from benchmark import hard_ctf as h
sel = json.load(open("$ROOT/probe/selection.json"))
inband = sel.get("in_band_families") or []
chosen = h.select_balanced_families(inband, per_outer=$PER_OUTER, seed=$WORLD_SEED, poison_kinds=("sibling", "cross_outer"))
open("$ROOT/families.txt", "w").write(",".join(chosen))
print("in-band:", len(inband), "chosen:", len(chosen), chosen)
PY
  say "families -> $ROOT/families.txt"
  ;;
arms)
  ensure_server
  FAMS=$(cat "${FAMS_FILE:-$ROOT/families.txt}")
  # ARMS_URLS: comma list of OpenAI-compatible base URLs serving $MODEL (e.g. two 14B servers, one per GPU);
  # learner processes are assigned round-robin so the aggregate throughput scales with the servers.
  IFS=',' read -r -a URLS <<< "${ARMS_URLS:-$U}"; NURL=${#URLS[@]}; ARM_I=0
  launch(){ # name key cond extra...
    local name=$1 key=$2 cond=$3; shift 3; local out="$ROOT/arms/$name/shard_00"; mkdir -p "$out"
    local url="${URLS[$((ARM_I % NURL))]}"; ARM_I=$((ARM_I + 1))
    MODEL_NAME="$MODEL" VLLM_URL="$url" nohup $PYBIN -m experiments.run_campaign --out "$out" --conditions "$cond" \
      --context-key "$key" --families "$FAMS" $COMMON "$@" > "$ROOT/arms/$name.log" 2>&1 &
    echo "$name:$! -> $url"
  }
  say "launching arms for families: $FAMS"
  all_arms | while read -r name key cond extra; do launch "$name" "$key" "$cond" $extra; done
  say "arms launched; monitor with: ls $ROOT/arms/*/shard_00/progress.json ; STOP file: touch $ROOT/arms/<name>/STOP"
  ;;
causal)
  # user decision 2026-09-07: the causal-credit arm runs AFTER the 14B block (one learner, GPU0 server)
  ensure_server
  FAMS=$(cat "${FAMS_FILE:-$ROOT/families.txt}")
  out="$ROOT/arms/deployed_causal_alphabet/shard_00"; mkdir -p "$out"
  MODEL_NAME="$MODEL" VLLM_URL="$U" nohup $PYBIN -m experiments.run_campaign --out "$out" --conditions deployed_causal \
    --context-key alphabet --families "$FAMS" $COMMON --tau 0.1 >> "$ROOT/arms/deployed_causal_alphabet.log" 2>&1 &
  say "launched deployed_causal_alphabet:$! -> $U"
  ;;
restart)
  # STAGE=restart ARM=<name>: re-launch ONE arm with its exact construction (resumes from its checkpoint)
  ensure_server
  FAMS=$(cat "${FAMS_FILE:-$ROOT/families.txt}")
  IFS=',' read -r -a URLS <<< "${ARMS_URLS:-$U}"; NURL=${#URLS[@]}; ARM_I=${ARM_I:-0}
  launch(){ local name=$1 key=$2 cond=$3; shift 3; local out="$ROOT/arms/$name/shard_00"; mkdir -p "$out"
    local url="${URLS[$((ARM_I % NURL))]}"
    MODEL_NAME="$MODEL" VLLM_URL="$url" nohup $PYBIN -m experiments.run_campaign --out "$out" --conditions "$cond" \
      --context-key "$key" --families "$FAMS" $COMMON "$@" >> "$ROOT/arms/$name.log" 2>&1 &
    say "restarted $name:$! -> $url"; }
  line=$(all_arms | awk -v a="${ARM:?set ARM}" '$1==a'); [ -n "$line" ] || { say "unknown ARM $ARM"; exit 2; }
  read -r name key cond extra <<< "$line"; launch "$name" "$key" "$cond" $extra
  ;;
checks)
  if pgrep -f "[r]un_campaign --out $ROOT/arms/" >/dev/null; then say "arms still running - refusing to run checks"; exit 3; fi
  for a in "$ROOT"/arms/*/; do
    n=$(basename "$a")
    $PYBIN -m experiments.smoke_checks --root "$a" > "$a/CHECKS.md" 2>&1; echo "$n checks exit=$?"
    case "$n" in deployed_*|s0_*|s1_*|s4_*|outcome_*) $PYBIN -m experiments.learning_curve --root "$a" --cond "$( [ "$n" = outcome_legacy_alphabet ] && echo outcome || ([ "$n" = deployed_obs_alphabet ] && echo deployed_obs || echo deployed) )" > "$a/LEARNING_CURVE.txt" 2>&1;; esac
  done
  ;;
*) echo "unknown STAGE $STAGE"; exit 2;;
esac
