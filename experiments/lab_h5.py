#!/usr/bin/env python3
"""
experiments/lab_h5.py — H5 transfer / value-gated write-back propagation experiment.

The promotion proxy (outcome credit promotes poison, causal does not) is converted into a
MEASURED downstream solve-rate harm. On a TRANSFER split the family's TRUE lesson is WITHHELD
from memory, so a *propagated* poison (one a credit signal promoted into trusted memory) can be
retrieved-and-FOLLOWED and lower solve — the only regime where false-lesson harm is visible
(co-present-with-true gives distraction≈0, per the pilot). The arms differ ONLY in the
credit signal's promotion DECISION, so any Δsolve is attributable to the gate.

Arms (forced retrieval set; true lesson ABSENT except CLEAN):
  empty          (no lesson)                         -> base rate (headroom check)
  clean          (true lesson present)               -> positive control (lesson lifts)
  poison_forced  (poison always present)             -> max harm bracket
  outcome_mem    (poison present IFF outcome_Q>theta)-> memory built under outcome credit
  causal_mem     (poison present IFF causal_ATE>theta)-> memory built under causal credit
  shuffle_mem    (poison present at the OUTCOME promotion RATE, families shuffled) -> matched-budget control

Headline: Δsolve = solve(causal_mem) - solve(outcome_mem), paired over (challenge,seed), with
bootstrap CI + exact McNemar. PREREG H5 verdict: harm if CI excludes 0 and lower bound >= 5pp;
null if CI within +/-2.5pp of 0; else inconclusive.

Needs a values file {node_id: {label, family, outcome_Q, causal_ATE}} from the campaign analysis
(experiments.lab_analysis writes node_values.json). Concurrent; isolated workdirs; resumable.
"""
from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import benchmark.hard_ctf as hard
try:
    from batch_test.real_adapters import real_solve_fn
except ModuleNotFoundError:  # not shipped in the public release
    def real_solve_fn(*a, **k):
        raise RuntimeError(
            "real-GPU backend adapters (batch_test/) are not included in "
            "this release; supply your own solve_fn")
from evaluation.stats import paired_bootstrap_ci, mcnemar_exact
from experiments.lab_probe import _nodes_for, _isolate
from ope.causal_value import _stable_seed

_LOCK = threading.Lock()


def _promotion(values, theta):
    """{family: {'outcome': bool, 'causal': bool}} for each family's POISON lesson."""
    promo = {}
    for nid, d in values.items():
        if d.get("label") != "poison":
            continue
        promo[d["family"]] = {
            "outcome": (d.get("outcome_Q") is not None and d["outcome_Q"] > theta),
            "causal": (d.get("causal_ATE") is not None and d["causal_ATE"] > theta),
        }
    return promo


def h5_family(name, promo_fam, shuffle_inject, n_inst, seeds, slack, solve, workroot, corpus_root):
    helpful, poison = _nodes_for(name)
    budget = hard._BY_NAME[name].solution_length + slack
    corpus = hard.generate_corpus(Path(corpus_root) / name, n_per_family=n_inst,
                                  seed=(_stable_seed(name) % 9000) + 1, families=[name])
    arms = {
        "empty": [],
        "clean": [helpful],
        "poison_forced": [poison],
        "outcome_mem": [poison] if promo_fam.get("outcome") else [],
        "causal_mem": [poison] if promo_fam.get("causal") else [],
        "shuffle_mem": [poison] if shuffle_inject else [],
    }
    rec = {a: [] for a in arms}
    for ch in corpus:
        for a, nodes in arms.items():
            for sd in range(seeds):
                cj = _isolate(ch, a, sd, workroot)
                res = solve(cj, [n for n in nodes if n is not None], budget, seed=sd)
                rec[a].append((ch["name"], sd, 1 if hard.check_flag(ch, res.get("flag")) else 0))
    return {"family": name, "budget": budget, "promo": promo_fam, "rec": rec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values-file", default="results/lab/campaign_7b/merged/node_values.json")
    ap.add_argument("--families-file", default="results/lab/probe_7b/selection.json")
    ap.add_argument("--families", default=None)
    ap.add_argument("--theta", type=float, default=0.0)
    ap.add_argument("--n-inst", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--slack", type=int, default=2)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--out", default="results/lab/h5")
    ap.add_argument("--max-output-tokens", type=int, default=512)
    args = ap.parse_args()

    values = json.loads(Path(args.values_file).read_text())
    promo = _promotion(values, args.theta)
    if args.families:
        fams = [f.strip() for f in args.families.split(",")]
    else:
        sel = json.loads(Path(args.families_file).read_text())
        fams = sel.get("in_band_families") or sel.get("teeth_families") or []
    fams = [f for f in fams if f in promo]      # need a poison value for the family
    # matched-budget shuffle control: inject poison in a RANDOM subset of families at the
    # SAME rate outcome credit promotes, decoupled from which families (controls for "more
    # injected lessons" vs "the right promotion decision").
    rng = np.random.RandomState(0)
    out_rate = np.mean([promo[f]["outcome"] for f in fams]) if fams else 0.0
    shuffled = set(rng.choice(fams, size=int(round(out_rate * len(fams))), replace=False)) if fams else set()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "h5.json"
    done = {}
    if res_path.exists():
        done = {r["family"]: r for r in json.loads(res_path.read_text()).get("families", [])}
    todo = [f for f in fams if f not in done]
    print(f"[h5] {len(fams)} transfer families; outcome-promote rate={out_rate:.2f}; "
          f"{len(done)} done, {len(todo)} to run; theta={args.theta}", flush=True)

    solve = real_solve_fn(max_output_tokens=args.max_output_tokens, context_window=6144)
    workroot = out_dir / "work"; corpus_root = out_dir / "corpus"
    results = list(done.values())
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(h5_family, f, promo[f], f in shuffled, args.n_inst, args.seeds,
                          args.slack, solve, workroot, corpus_root): f for f in todo}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:  # noqa
                results.append({"family": f, "error": str(e)})
            with _LOCK:
                res_path.write_text(json.dumps({"families": results, "theta": args.theta}, indent=2))
            print(f"  done {f}", flush=True)

    # aggregate per-arm solve + paired Δ(causal_mem - outcome_mem)
    by_arm = {}
    paired = {"causal_mem": [], "outcome_mem": []}
    for r in results:
        for a, rows in r.get("rec", {}).items():
            by_arm.setdefault(a, []).extend(x[2] for x in rows)
    # align paired arms on (family,challenge,seed)
    keymap = {a: {} for a in ("causal_mem", "outcome_mem")}
    for r in results:
        for a in keymap:
            for (nm, sd, ok) in r.get("rec", {}).get(a, []):
                keymap[a][(r["family"], nm, sd)] = ok
    common = sorted(set(keymap["causal_mem"]) & set(keymap["outcome_mem"]))
    ca = [keymap["causal_mem"][k] for k in common]
    oa = [keymap["outcome_mem"][k] for k in common]
    summary = {"arms_solve": {a: float(np.mean(v)) if v else None for a, v in by_arm.items()},
               "n_paired": len(common)}
    if common:
        pr = paired_bootstrap_ci(ca, oa, n_boot=10000)
        mc = mcnemar_exact(ca, oa)
        d, lo, hi = pr.mean_diff, pr.ci_low, pr.ci_high
        # PREREG H5: Δ = solve(causal_mem) − solve(outcome_mem). "harm_prevented" if the CI
        # excludes 0 with lower bound ≥5pp (propagation harm was real and causal credit stopped
        # it); "null" if CI within ±2.5pp; "reverse" if causal is significantly WORSE (falsifier);
        # else inconclusive.
        verdict = ("harm_prevented" if (lo > 0 and lo >= 0.05) else
                   "reverse" if (hi < 0 and hi <= -0.05) else
                   "null" if (abs(lo) <= 0.025 and abs(hi) <= 0.025) else "inconclusive")
        summary["transfer_delta_causal_minus_outcome"] = {
            "delta": d, "ci_lo": lo, "ci_hi": hi, "boot_p": pr.p_value_sign,
            "mcnemar_p": mc["p_value"], "n": len(common), "verdict_PREREG": verdict}
    (out_dir / "h5_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[h5] arm solve rates:", {k: round(v, 3) for k, v in summary["arms_solve"].items() if v is not None})
    td = summary.get("transfer_delta_causal_minus_outcome")
    if td:
        print(f"[h5] Δsolve(causal_mem - outcome_mem) = {100*td['delta']:+.1f}pp "
              f"CI[{100*td['ci_lo']:+.1f},{100*td['ci_hi']:+.1f}] McNemar p={td['mcnemar_p']:.3f} "
              f"-> {td['verdict_PREREG']} (n={td['n']})")
    print(f"[h5] wrote {out_dir/'h5_summary.json'}")


if __name__ == "__main__":
    main()
