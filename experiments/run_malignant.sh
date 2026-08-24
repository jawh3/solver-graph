#!/bin/bash
# run_malignant.sh — T1-1 MALIGNANT positive control (RIGOR_LEDGER, pre-registered 2026-06-11, b51a184).
#
# WHY: our headline is a NEGATIVE result — a poison (benign-wrong) lesson is UNDETECTABLE from a
# capable agent's outcomes because following it == the family base (it recovers for free; the
# identifiability theorem says outcome-ATE(poison)=0 EXACTLY). A reviewer's first objection: "your
# estimator is just insensitive — it would miss ANY harm." This experiment refutes that. We inject
# FLOOR-VIOLATING malignant lessons (destroy: corrupt the artifact; exhaust: blow the step budget;
# wrongflag: submit a fixed wrong flag) that drop solve BELOW base, and show the SAME on-target
# SNIPW + exact CRT estimators DO flag them (ATE<0, CI excludes 0), within the SAME campaign where
# poison reads null. teeth proven mechanically in tests/test_malignant_teeth.py (10 pass); here we
# prove the REAL 14B solver FOLLOWS them and the estimator catches the harm.
#
# PRE-REGISTERED KILL: if on-target SNIPW fails to flag M-destroy AND M-exhaust (CI excludes 0, ATE<0)
# OR the exact CRT frac_sig<0.5 for them, the identifiability claim is UNSUPPORTED as scoped.
#
# Single GPU0, TP=1 (leaves GPU1 free for other users). Resumable (lab_pipeline skips done shards).
set -u
REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}
VLLM=${VLLM:-vllm}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO"
M="Qwen/Qwen2.5-14B-Instruct-AWQ"; PORT=8002; U="http://127.0.0.1:$PORT/v1"
GPU=0                                   # GPU0 only; GPU1 stays free
ARMS="destroy,exhaust,wrongflag"
FAMS="results/lab/probe_14b/selection.json"
PROBE="results/lab/campaign_14b_malig_probe"
FULL="results/lab/campaign_14b_malig"
mkdir -p logs
say(){ echo "[malig $(date +%H:%M:%S)] $*" | tee -a logs/run_malignant.log; }
own(){ [ "$(ps -o user= -p "$1" 2>/dev/null|tr -d ' ')" = "$USER" ]; }
# count GPU compute procs owned by another user (NEVER contend with them)
other_user_on_gpu(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '|while read p;do [ -n "$p" ]&&ps -o user= -p $p 2>/dev/null|tr -d ' ';done|grep -vc "^${USER}$"; }
genok(){ curl -s --max-time 40 "$U/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"reply OK\"}],\"max_tokens\":5}" 2>/dev/null | grep -q '"content"'; }
stop_server(){ p=$(cat /tmp/malig.pid 2>/dev/null); { [ -n "${p:-}" ] && own "$p" && { say "stopping 14B server $p"; kill "$p" 2>/dev/null; sleep 8; kill -9 "$p" 2>/dev/null; }; }; }

[ "$(other_user_on_gpu)" != 0 ] && { say "another user is on a GPU — NOT launching. Try later."; exit 0; }

# ---- serve (skip if already healthy) ----
if ! genok; then
  say "serving $M on GPU$GPU:$PORT (TP=1)"
  : > logs/serve_malig.log
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup "$VLLM" serve "$M" --max-model-len 8192 \
    --gpu-memory-utilization 0.82 --max-num-seqs 48 --enable-prefix-caching --port $PORT \
    > logs/serve_malig.log 2>&1 &
  echo $! > /tmp/malig.pid
  ok=0; for i in $(seq 1 240); do
    grep -qiE "OutOfMemoryError|error initializing|No available memory|raise " logs/serve_malig.log 2>/dev/null && { say "serve error in log; aborting"; break; }
    genok && { ok=1; break; }; sleep 5
  done
  [ "$ok" = 1 ] || { say "serve FAILED; abort"; stop_server; exit 1; }
fi
say "server healthy at $U"

# ---- PROBE: small campaign -> gate (does the REAL solver FOLLOW malignant recipes?) ----
if [ ! -f "$PROBE/merged/malignant_probe_check.json" ]; then
  say "PROBE campaign (shards6 n2 e2 s1 M3 replicate, arms=$ARMS)"
  MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.lab_pipeline \
    --families-file "$FAMS" --out "$PROBE" --vllm-urls "$U" \
    --shards 6 --n-per-family 2 --epochs 2 --eval-seeds 1 --slack 2 \
    --conditions outcome --n-clusters 6 --M 3 --replicate --cond outcome \
    --world-seed 313131 --malignant-arms "$ARMS" --deadline-hours 3 >> logs/run_malignant.log 2>&1
  MODEL_NAME="$M" $PYBIN -m experiments.malignant_probe_check --dir "$PROBE/merged" --min-cited 3 >> logs/run_malignant.log 2>&1 || true
fi
# decision: ABORT only if malignant lessons were essentially never followed (wiring/compliance
# failure). The scientific KILL call is made on the FULL result with CIs, not the noisy probe.
DEC=$(MODEL_NAME="$M" $PYBIN - "$PROBE/merged/malignant_probe_check.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception as e:
    print("ABORT no-probe-json"); sys.exit(0)
arms=d.get("arms",{})
tot=sum((arms.get(a,{}).get("n_cited") or 0) for a in ("destroy","exhaust","wrongflag"))
de=arms.get("destroy",{}); ex=arms.get("exhaust",{})
print(f"probe: base={d.get('base'):.3f} destroy(n={de.get('n_cited')},delta={de.get('delta')}) "
      f"exhaust(n={ex.get('n_cited')},delta={ex.get('delta')}) total_cited={tot}")
print("ABORT malignant-never-followed" if tot==0 else "PROCEED")
PY
)
say "$DEC"
echo "$DEC" | grep -q ABORT && { say "probe gate ABORT -> not launching full. Investigate wiring/compliance."; stop_server; exit 3; }

# ---- FULL campaign ----
if [ ! -f "$FULL/merged/ontarget_analysis.json" ]; then
  # ~3.9k runs (~4h on one 14B-TP1 server, ~17 runs/min) -> ~180 malignant-cited/arm: the
  # SNIPW label-level ATE (the headline; solve|cited~0 vs base~0.07) is well-powered. CRT is
  # exploratory (that pillar was retracted, T1-3c). Resumable: re-run the driver to add more n.
  say "FULL campaign (shards8 n4 e4 s1 M3 replicate, arms=$ARMS)"
  MODEL_NAME="$M" VLLM_URL="$U" $PYBIN -m experiments.lab_pipeline \
    --families-file "$FAMS" --out "$FULL" --vllm-urls "$U" \
    --shards 8 --n-per-family 4 --epochs 4 --eval-seeds 1 --slack 2 \
    --conditions outcome --n-clusters 6 --M 3 --replicate --cond outcome \
    --world-seed 727272 --malignant-arms "$ARMS" --deadline-hours 6 >> logs/run_malignant.log 2>&1
  say "FULL campaign done; malignant-specific analyses"
  MODEL_NAME="$M" $PYBIN -m experiments.ontarget_analysis --dir "$FULL/merged" --cond outcome >> logs/run_malignant.log 2>&1 || true
  MODEL_NAME="$M" $PYBIN -m experiments.crt_test --dir "$FULL/merged" --cond outcome --key R_final --out crt_test.json >> logs/run_malignant.log 2>&1 || true
  MODEL_NAME="$M" $PYBIN -m experiments.crt_test --dir "$FULL/merged" --cond outcome --key outcome --out crt_test_outcome.json >> logs/run_malignant.log 2>&1 || true
  MODEL_NAME="$M" $PYBIN -m experiments.cite_rate --dir "$FULL/merged" --cond outcome >> logs/run_malignant.log 2>&1 || true
  MODEL_NAME="$M" $PYBIN -m experiments.malignant_probe_check --dir "$FULL/merged" --min-cited 20 --out malignant_result.json >> logs/run_malignant.log 2>&1 || true
fi

stop_server
say "T1-1 MALIGNANT positive control DONE -> $FULL/merged/{ontarget_analysis,crt_test,cite_rate,malignant_result}.json"
