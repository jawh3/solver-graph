#!/bin/bash
# experiments/phase2_health.sh - Phase-2 health check on LIVE arms.
# Reports per arm: progress (phase/epoch/runs), errors, throughput; runs the smoke gates and the learning-curve
# estimator on the partial logs of every learner arm (read-only; never stops anything).
ROOT=${ROOT:-results/lab/phase2_14b}; REPO=${REPO:-$(pwd)}; cd "$REPO" || exit 1
PYBIN=${PYBIN:-python3}; export PYTHONPATH="$REPO"
echo "# Phase-2 health - $ROOT - $(date '+%Y-%m-%d %H:%M')"
echo; echo "## arms"; printf "%-32s %-14s %6s %6s %5s %s\n" arm phase/epoch runs solves errs alive
for a in "$ROOT"/arms/*/; do
  n=$(basename "$a"); pj="$a/shard_00/progress.json"; rl="$a/shard_00/logs"
  runs=$(cat "$rl"/campaign_*/runs.jsonl 2>/dev/null | wc -l); solves=$(cat "$rl"/campaign_*/runs.jsonl 2>/dev/null | grep -c '"outcome": 1')
  errs=$(grep -c '"error": true' "$rl"/campaign_*/runs.jsonl 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
  ph=$($PYBIN -c "import json,sys; p=json.load(open('$pj')); c=list(p['conditions'].values())[0]; print(f\"{c.get('phase')}/ep{c.get('epoch')}\")" 2>/dev/null || echo "-")
  alive=$(pgrep -f "[r]un_campaign --out $a" >/dev/null && echo yes || echo NO)
  printf "%-32s %-14s %6s %6s %5s %s\n" "$n" "$ph" "$runs" "$solves" "$errs" "$alive"
  grep -a -q "Traceback" "$ROOT/arms/$n.log" 2>/dev/null && { echo "   !! Traceback in $ROOT/arms/$n.log ($(grep -a -c Traceback "$ROOT/arms/$n.log")x):"; grep -a -A22 Traceback "$ROOT/arms/$n.log" | grep -a -E "^[A-Za-z]*Error" | tail -n 2; }
done
# server counters snapshot (prompt/generation tokens, finish reasons) so compute survives server shutdown
mkdir -p "$ROOT/metrics"; for port in 8002 8003; do curl -s --max-time 10 "http://127.0.0.1:$port/metrics" > "$ROOT/metrics/metrics_${port}_$(date +%m%d_%H%M).prom" 2>/dev/null || true; done
echo; echo "## throughput (runs in the last 60 min, all arms)"
$PYBIN - <<PY
import json, glob, datetime as dt
now = dt.datetime.now(dt.timezone.utc); n = 0; tot = 0
for f in glob.glob("$ROOT/arms/*/shard_00/logs/campaign_*/runs.jsonl"):
    for l in open(f):
        if not l.strip(): continue
        tot += 1
        ts = json.loads(l).get("ts", "")
        try:
            if (now - dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds() < 3600: n += 1
        except Exception: pass
print(f"{n} runs in the last hour ({n/60:.1f}/min); {tot} total")
PY
echo; echo "## replay check (runs per challenge item: train items once per completed epoch, eval items once per eval seed; >2 = replayed after a restart)"
$PYBIN - <<PY
import json, glob, collections
for a in sorted(glob.glob("$ROOT/arms/*/")):
    n = a.rstrip("/").split("/")[-1]
    runs = [json.loads(l) for f in glob.glob(f"{a}/shard_00/logs/campaign_*/runs.jsonl") for l in open(f) if l.strip()]
    hist = collections.Counter(collections.Counter(r["challenge_id"] for r in runs).values())
    print(f"  {n:30s} {dict(sorted(hist.items()))} {'<-- REPLAY?' if any(k > 2 for k in hist) else 'ok'}")
PY
echo; echo "## gates + learning curves (partial logs; learning gate applies only at >= 300 train runs)"
for a in "$ROOT"/arms/deployed_*/ "$ROOT"/arms/s0_*/ "$ROOT"/arms/s1_*/ "$ROOT"/arms/s4_*/ "$ROOT"/arms/outcome_*/ "$ROOT"/arms/deployed_obs_*/; do
  [ -d "$a" ] || continue; n=$(basename "$a")
  case "$n" in outcome_*) cond=outcome;; deployed_obs_*) cond=deployed_obs;; deployed_causal_*) cond=deployed_causal;; *) cond=deployed;; esac
  echo "### $n"
  $PYBIN -m experiments.smoke_checks --root "$a" 2>/dev/null | grep -E "^\[$cond\] (pi_deploy|learning|contexts|on-target helpful in M-set \(train)|F3-|^- (PASS|FAIL)" | cut -c1-200
  $PYBIN -m experiments.learning_curve --root "$a" --cond "$cond" --n-mc 100 2>/dev/null | grep -E "tercile|pi_helpful|pi_poison" | cut -c1-160
done
