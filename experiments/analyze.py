#!/usr/bin/env python3
"""
experiments/analyze.py — turn results/pilot/*.json + logs into paper tables + figures.

Reads the phase outputs written by run_pilot.py and emits:
  results/pilot/figures/*.pdf   — matplotlib figures for the paper
  results/pilot/tables.tex      — LaTeX tables
  results/pilot/analysis.json   — a flat summary of every number the paper cites

Robust to partial data (any phase may be missing). Also runs the offline OPE
sweep on the Phase-B (full-method) run logs — the "extend sample size with no GPU"
result — including the DR-≈-measured sanity check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from instrumentation.jsonl_logger import load_runs
from ope.sweep import offline_sweep, sanity_check, measured_value

RESULTS = Path("results/pilot")
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 140, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"true": "#2c7fb8", "poison": "#d7301f", "distractor": "#999999",
     "B0": "#888888", "B2": "#fdae61", "B4": "#2c7fb8"}


def _load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_solve_rates(A, B):
    if not A:
        return
    labels, vals, errs = [], [], []
    b0 = A["B0_solve"]; b2 = A["B2_solve"]
    labels += ["B0\n(no graph)", "B2\n(static retrieval)"]; vals += [b0, b2]; errs += [0, 0]
    if B and "cited_step" in B and "B4_solve" in B["cited_step"]:
        labels.append("B4\n(bandit, full)"); vals.append(B["cited_step"]["B4_solve"]); errs.append(0)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    colors = [C["B0"], C["B2"], C["B4"]][:len(vals)]
    ax.bar(range(len(vals)), vals, color=colors, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel("solve rate"); ax.set_ylim(0, max(vals) * 1.25)
    ax.axhline(b0, ls="--", c="#bbbbbb", lw=1, zorder=0)
    lift = A.get("paired_lift", 0)
    ax.set_title(f"Retrieval significantly improves solve rate\n(+{lift*100:.0f}pp over no memory, paired)")
    fig.savefig(FIG / "fig_solve_rates.pdf"); plt.close(fig)


def fig_q_trajectory(B):
    if not B or "cited_step" not in B:
        return
    traj = B["cited_step"]["trajectory"]
    eps = [t["epoch"] for t in traj]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    series = [("helpful", "true", "true lessons"), ("poison", "poison", "poison lessons"),
              ("distractor", "distractor", "distractor lessons")]
    for key, ckey, lab in series:
        ys = [t["q_by_truth"].get(key, [0, 0, 0])[0] for t in traj]
        ax.plot([0] + eps, [0] + ys, "-o", color=C[ckey], label=lab)
    ax.set_xlabel("training epoch"); ax.set_ylabel("mean learned Q")
    ax.legend(fontsize=8)
    ax.set_title("Outcome credit cannot separate executed poison\nfrom true lessons (both rise); distractors stay low")
    ax.axhline(0, c="#dddddd", lw=1, zorder=0)
    fig.savefig(FIG / "fig_q_trajectory.pdf"); plt.close(fig)


def fig_q_by_config(B):
    if not B:
        return
    names = [n for n in ["cited_step", "allM_step", "cited_outcome"] if n in B]
    pretty = {"cited_step": "cited-only\n+ step (full)", "allM_step": "all-M\ncredit",
              "cited_outcome": "outcome-only\nreward"}
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    qt = [B[n]["final_q_by_truth"].get("helpful", [0])[0] for n in names]
    qp = [B[n]["final_q_by_truth"].get("poison", [0])[0] for n in names]
    qd = [B[n]["final_q_by_truth"].get("distractor", [0])[0] for n in names]
    ax.bar(x - w, qt, w, color=C["true"], label="true")
    ax.bar(x, qp, w, color=C["poison"], label="poison")
    ax.bar(x + w, qd, w, color=C["distractor"], label="distractor (unused)")
    ax.set_xticks(x); ax.set_xticklabels([pretty[n] for n in names])
    ax.set_ylabel("final mean learned Q"); ax.legend(fontsize=8)
    ax.set_title("Cited-only credit zeroes out unused (distractor) lessons;\nexecuted poison still evades it")
    ax.axhline(0, c="#dddddd", lw=1, zorder=0)
    fig.savefig(FIG / "fig_q_by_config.pdf"); plt.close(fig)


def fig_learning_curve(B):
    if not B:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for n in [x for x in ["cited_step", "allM_step", "cited_outcome"] if x in B]:
        es = B[n].get("epoch_solve", [])
        ax.plot(range(1, len(es) + 1), es, "-o", label=n)
    ax.set_xlabel("epoch"); ax.set_ylabel("streaming solve rate"); ax.legend(fontsize=8)
    ax.set_title("Learning curve by configuration")
    fig.savefig(FIG / "fig_learning_curve.pdf"); plt.close(fig)


def fig_loo(D):
    if not D or not D.get("marginals"):
        return
    pois = [m["marginal_solve"] for m in D["marginals"] if m["node_kind"] == "poison"]
    tru = [m["marginal_solve"] for m in D["marginals"] if m["node_kind"] == "true"]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    data = [tru, pois]
    bp = ax.boxplot(data, labels=["true", "poison"], patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], [C["true"], C["poison"]]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.axhline(0, c="#888888", lw=1, ls="--")
    ax.set_ylabel("LOO marginal Δ solve rate")
    ax.set_title("Oracle-free causal probe:\npoison lessons have ~0 marginal value")
    fig.savefig(FIG / "fig_loo.pdf"); plt.close(fig)


def fig_ope_sweep():
    runs = load_runs("B_cited_step", root=str(RESULTS / "logs"))
    if not runs:
        return None
    sane = sanity_check(runs)
    rows = offline_sweep(runs, taus=(0.3, 0.5, 1.0, 1.5, 2.0), kappas=(5.0,), estimator="dr")
    taus = [r["tau"] for r in sorted(rows, key=lambda r: r["tau"])]
    vals = [r["value"] for r in sorted(rows, key=lambda r: r["tau"])]
    los = [r["ci_low"] for r in sorted(rows, key=lambda r: r["tau"])]
    his = [r["ci_high"] for r in sorted(rows, key=lambda r: r["tau"])]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(taus, vals, "-o", color=C["B4"])
    ax.fill_between(taus, los, his, alpha=0.2, color=C["B4"])
    ax.axhline(measured_value(runs), ls="--", c="#888", label="logged-policy value")
    ax.set_xlabel("temperature τ (counterfactual)"); ax.set_ylabel("DR off-policy value")
    ax.legend(fontsize=8); ax.set_title("Offline policy sweep (no extra GPU)")
    fig.savefig(FIG / "fig_ope_sweep.pdf"); plt.close(fig)
    best = max(rows, key=lambda r: r["value"])
    return {"sanity": sane, "best": best}


def emit_tables(A, B, C_, D, ope):
    lines = []
    # main table
    lines.append("% main results (Phase A + B)")
    lines.append(r"\begin{tabular}{lccc}\toprule")
    lines.append(r"System & Solve rate & Lift vs B0 & 95\% CI \\\midrule")
    if A:
        lines.append(rf"B0 (no graph) & {A['B0_solve']:.2f} & --- & --- \\")
        b2lift = A["paired_lift"]; ci = A["ci"]
        lines.append(rf"B2 (static retrieval) & {A['B2_solve']:.2f} & {b2lift:+.2f} & [{ci[0]:+.2f},{ci[1]:+.2f}] \\")
    if B and "cited_step" in B and "lift_vs_B0" in B["cited_step"]:
        cs = B["cited_step"]; lv = cs["lift_vs_B0"]
        lines.append(rf"B4 (bandit, full) & {cs['B4_solve']:.2f} & {lv['lift']:+.2f} & [{lv['ci'][0]:+.2f},{lv['ci'][1]:+.2f}] \\")
    lines.append(r"\bottomrule\end{tabular}")
    lines.append("")
    # ablation table
    if B:
        pretty = {"cited_step": "cited-only + step (full)", "allM_step": "all-$M$ credit",
                  "cited_outcome": "outcome-only reward"}
        lines.append("% ablation table (Phase B)")
        lines.append(r"\begin{tabular}{lccc}\toprule")
        lines.append(r"Config & Q(true) & Q(poison) & Q(distractor) \\\midrule")
        for n in [x for x in ["cited_step", "allM_step", "cited_outcome"] if x in B]:
            e = B[n]["final_q_by_truth"]
            qt = e.get("helpful", [0])[0]; qp = e.get("poison", [0])[0]
            qd = e.get("distractor", [0])[0]
            lines.append(rf"{pretty.get(n,n)} & {qt:+.2f} & {qp:+.2f} & {qd:+.2f} \\")
        lines.append(r"\bottomrule\end{tabular}")
    (RESULTS / "tables.tex").write_text("\n".join(lines))


def main():
    A, B, C_, D = _load("phaseA"), _load("phaseB"), _load("phaseC"), _load("phaseD")
    fig_solve_rates(A, B)
    fig_q_trajectory(B)
    fig_q_by_config(B)
    fig_learning_curve(B)
    fig_loo(D)
    ope = fig_ope_sweep()
    emit_tables(A, B, C_, D, ope)

    summary = {"phaseA": A, "phaseB_keys": (list(B.keys()) if B else None),
               "phaseC": C_, "phaseD": D, "ope": ope}
    # flatten the headline numbers
    head = {}
    if A:
        head.update(B0_solve=A["B0_solve"], B2_solve=A["B2_solve"],
                    B2_lift=A["paired_lift"], B2_sig=A["significant"])
    if B and "cited_step" in B:
        cs = B["cited_step"]
        head.update(B4_solve=cs.get("B4_solve"),
                    B4_lift_vs_B0=cs.get("lift_vs_B0", {}).get("lift"),
                    B4_lift_vs_B2=cs.get("lift_vs_B2", {}).get("lift"),
                    Qtrue=cs["final_q_by_truth"].get("helpful", [None])[0],
                    Qpoison=cs["final_q_by_truth"].get("poison", [None])[0])
        if "allM_step" in B:
            head["Qpoison_allM"] = B["allM_step"]["final_q_by_truth"].get("poison", [None])[0]
        if "cited_outcome" in B:
            head["Qpoison_outcome"] = B["cited_outcome"]["final_q_by_truth"].get("poison", [None])[0]
    if C_:
        head["writeback_tiers"] = {k: v.get("tiers") for k, v in C_.items()}
    if D:
        head.update(loo_poison=D["poison_mean_marginal_solve"],
                    loo_true=D["true_mean_marginal_solve"],
                    exposure=D["false_lesson_exposure_rate"])
    if ope:
        head["ope_sanity"] = ope["sanity"]; head["ope_best"] = ope["best"]
    summary["headline"] = head
    (RESULTS / "analysis.json").write_text(json.dumps(summary, indent=2, default=str))

    # emit numbers.tex so the paper prose references macros, not hard-coded numbers
    # (every number then traces back to results/pilot/analysis.json -> the logs).
    def _pct(x):
        return f"{100*x:.0f}\\%" if isinstance(x, (int, float)) else "??"

    def _f2(x):
        return f"{x:+.2f}" if isinstance(x, (int, float)) else "??"
    nums = []
    # provide-then-renew so it works whether or not main.tex pre-defines the macro
    def cmd(name, val):
        nums.append(rf"\providecommand{{\{name}}}{{}}\renewcommand{{\{name}}}{{{val}}}")
    if A:
        cmd("BzeroSolve", _pct(A["B0_solve"])); cmd("BtwoSolve", _pct(A["B2_solve"]))
        cmd("BtwoLift", _f2(A["paired_lift"]))
        cmd("BtwoCIlo", f"{A['ci'][0]:+.2f}"); cmd("BtwoCIhi", f"{A['ci'][1]:+.2f}")
        cmd("Npairs", str(A["n_pairs"]))
        # challenge-level re-analysis (independent units; addresses correlated seeds)
        from collections import defaultdict as _dd
        from evaluation.stats import paired_bootstrap_ci as _pb, mcnemar_exact as _mc
        def _cl(d):
            agg = _dd(list)
            for k, v in d.items():
                agg[k.split("#")[0]].append(v["solved"])
            return {n: sum(x) / len(x) for n, x in agg.items()}
        b0c, b2c = _cl(A["per_run"]["B0"]), _cl(A["per_run"]["B2"])
        names = sorted(set(b0c) & set(b2c))
        pr = _pb([b2c[n] for n in names], [b0c[n] for n in names], seed=1)
        mc = _mc([1 if b2c[n] >= .5 else 0 for n in names], [1 if b0c[n] >= .5 else 0 for n in names])
        ident = sum(1 for n in names
                    if abs(A["per_run"]["B0"][f"{n}#s0"]["solved"] - A["per_run"]["B0"][f"{n}#s1"]["solved"]) < .01)
        cmd("NchalPairs", str(len(names)))
        cmd("BtwoLiftChalP", f"{mc['p_value']:.3f}")
        cmd("SeedAgree", str(ident))
        # per-family ceiling/floor/lift census
        fam = _dd(lambda: [0, 0, 0, 0])
        for v in A["per_run"]["B0"].values():
            fam[v["family"]][0] += v["solved"]; fam[v["family"]][1] += 1
        for v in A["per_run"]["B2"].values():
            fam[v["family"]][2] += v["solved"]; fam[v["family"]][3] += 1
        ceil = sum(1 for s in fam.values() if s[0] == s[1] and s[2] == s[3])
        flr = sum(1 for s in fam.values() if s[0] == 0 and s[2] == 0)
        cmd("NCeiling", str(ceil)); cmd("NFloor", str(flr)); cmd("NLift", str(len(fam) - ceil - flr))
    if B and "cited_step" in B:
        cs = B["cited_step"]
        if cs.get("B4_solve") is not None:
            cmd("BfourSolve", _pct(cs["B4_solve"]))
            cmd("BfourLiftBzero", _f2(cs.get("lift_vs_B0", {}).get("lift")))
            cmd("BfourLiftBtwo", _f2(cs.get("lift_vs_B2", {}).get("lift")))
        cmd("Qtrue", f"{cs['final_q_by_truth'].get('helpful',[0])[0]:+.2f}")
        cmd("Qpoison", f"{cs['final_q_by_truth'].get('poison',[0])[0]:+.2f}")
        cmd("PoisonCite", _pct(cs.get("poison_citation_rate", 0)))
        # the clean lever result: distractor (unused) Q under cited-only vs all-M
        cmd("QdistractCited", f"{cs['final_q_by_truth'].get('distractor',[0])[0]:+.2f}")
        cmd("RecovMask", _pct(cs.get("recovery_masking_rate", 0)))
        if "allM_step" in B:
            cmd("QpoisonAllM", f"{B['allM_step']['final_q_by_truth'].get('poison',[0])[0]:+.2f}")
            cmd("QtrueAllM", f"{B['allM_step']['final_q_by_truth'].get('helpful',[0])[0]:+.2f}")
            cmd("QdistractAllM", f"{B['allM_step']['final_q_by_truth'].get('distractor',[0])[0]:+.2f}")
        if "cited_outcome" in B:
            cmd("QpoisonOutcome", f"{B['cited_outcome']['final_q_by_truth'].get('poison',[0])[0]:+.2f}")
    if C_ and "B5_gated" in C_ and "B6_ungated" in C_:
        g = C_["B5_gated"]["tiers"]; u = C_["B6_ungated"]["tiers"]
        gt, gp = g.get("trusted", 0), g.get("provisional", 0)
        ut, up = u.get("trusted", 0), u.get("provisional", 0)
        cmd("GatedProv", str(gp)); cmd("GatedTrust", str(gt)); cmd("GatedTotal", str(gp + gt))
        cmd("GatedProvPct", _pct(gp / max(gp + gt, 1)))
        cmd("UngatedTrust", str(ut)); cmd("UngatedTotal", str(ut + up))
        cmd("UngatedTrustPct", _pct(ut / max(ut + up, 1)))
    if D:
        cmd("LooPoison", f"{D['poison_mean_marginal_solve']:+.2f}")
        cmd("LooTrue", f"{D['true_mean_marginal_solve']:+.2f}")
        cmd("Exposure", _pct(D["false_lesson_exposure_rate"]))
    if ope and ope.get("sanity"):
        s = ope["sanity"]
        cmd("OpeMeasured", f"{s['measured']:.2f}"); cmd("OpeDR", f"{s['dr_estimate']:.2f}")
        cmd("OpeBestTau", str(ope["best"]["tau"]))
    Path("paper/numbers.tex").write_text("\n".join(nums) + "\n")

    print(json.dumps(head, indent=2, default=str))
    print(f"\nfigures -> {FIG}/  tables -> {RESULTS}/tables.tex  numbers -> paper/numbers.tex")


if __name__ == "__main__":
    main()
