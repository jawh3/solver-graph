#!/usr/bin/env python3
"""
experiments/fig_solution.py — figures for the causal-credit solution paper.

Reads results/campaign_v2/analysis.json (emitted by analyze_campaign.py) and writes
two PDFs into paper/:

  fig_separation.pdf  — THE money figure. Left: the OUTCOME bandit's learned Q by
    ground-truth label (helpful ≈ poison: the fooled signal). Right: the
    OBSERVATIONAL causal ATE computed from the SAME logs (helpful > poison ≈ 0: the
    separation outcome credit cannot produce). Same data, two credit signals.

  fig_solve_rates_v2.pdf — held-out solve rate by condition (B0 / B2 / outcome /
    observational / causal), with the paired observational-vs-outcome and
    observational-vs-B2 contrasts annotated.

Number-independent: it plots whatever analysis.json contains, so it runs on partial
data too. Re-run after each analyze_campaign.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = ["helpful", "poison", "distractor"]
COLORS = {"helpful": "#2a9d8f", "poison": "#e76f51", "distractor": "#9aa0a6"}
PRETTY = {"helpful": "true", "poison": "poison", "distractor": "distractor"}


def _bars(ax, by_label, key, title, ylabel):
    xs = [l for l in LABELS if l in by_label]
    vals = [by_label[l][key] for l in xs]
    ax.bar([PRETTY[l] for l in xs], vals, color=[COLORS[l] for l in xs],
           edgecolor="black", linewidth=0.6, width=0.62)
    ax.axhline(0, color="black", lw=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.01 if v >= 0 else -0.03), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    # headroom so the bar-value labels and the axes-fraction annotation never clip
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - (0.18 * span if lo < 0 else 0.0), hi + 0.45 * span)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.25)


def fig_separation(an, out: Path):
    outq = an.get("learned_Q", {}).get("outcome", {})
    obs = an.get("observational", {}).get("outcome", {})
    if not outq and not obs:
        print("  [separation] no data yet — skipping")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    if outq:
        _bars(axes[0], outq, "mean_Q", "Outcome credit (the bandit's signal)",
              "mean learned $Q$ by label")
    if obs:
        _bars(axes[1], {l: {"v": d["mean_ate"]} for l, d in obs.items()}, "v",
              "Observational causal credit (same logs)", "mean IPW ATE on reward")
    axes[0].annotate("helpful $\\approx$ poison\n(cannot separate)", (0.5, 0.92),
                     xycoords="axes fraction", ha="center", va="top", fontsize=8.5,
                     color="#b00020")
    axes[1].annotate("helpful $>$ poison\n(separated, free)", (0.5, 0.92),
                     xycoords="axes fraction", ha="center", va="top", fontsize=8.5,
                     color="#0a7d2c")
    fig.tight_layout()
    fig.savefig(out / "fig_separation.pdf", bbox_inches="tight")
    print(f"  wrote {out/'fig_separation.pdf'}")


def fig_solve_rates(an, out: Path):
    sr = an.get("solve_rates", {})
    order = [c for c in ("B0", "B2", "outcome", "observational", "causal") if sr.get(c) is not None]
    if not order:
        print("  [solve_rates] no data yet — skipping")
        return
    pretty = {"B0": "B0\nno mem", "B2": "B2\nstatic", "outcome": "outcome\ncredit",
              "observational": "observ.\ncausal", "causal": "active\ncausal"}
    cols = {"B0": "#9aa0a6", "B2": "#577590", "outcome": "#e76f51",
            "observational": "#2a9d8f", "causal": "#264653"}
    vals = [100 * sr[c] for c in order]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.bar([pretty[c] for c in order], vals, color=[cols[c] for c in order],
           edgecolor="black", linewidth=0.6, width=0.66)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("held-out solve rate (%)", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25 if vals else 100)
    ax.grid(axis="y", alpha=0.25)

    # annotate the two headline paired contrasts
    pc = an.get("paired", {})
    notes = []
    for key, lab in (("obs_vs_outcome", "obs$-$outcome"), ("obs_vs_B2", "obs$-$B2")):
        r = pc.get(key)
        if r:
            star = "*" if r.get("significant") else ""
            notes.append(f"{lab}: {100*r['diff']:+.1f}pp (p={r['mcnemar_p']:.2f}){star}")
    if notes:
        ax.set_title("   ".join(notes), fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_solve_rates_v2.pdf", bbox_inches="tight")
    print(f"  wrote {out/'fig_solve_rates_v2.pdf'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/campaign_v2")
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    an_path = Path(args.dir) / "analysis.json"
    if not an_path.exists():
        print(f"no analysis.json at {an_path} — run analyze_campaign.py first")
        return
    an = json.loads(an_path.read_text())
    out = Path(args.out)
    fig_separation(an, out)
    fig_solve_rates(an, out)


if __name__ == "__main__":
    main()
