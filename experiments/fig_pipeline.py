#!/usr/bin/env python3
"""experiments/fig_pipeline.py — regenerate the method pipeline figure (Fig 1).

Layout: seven stage boxes with uniform 0.2-unit gaps; the gated-write-back loop
is routed as a dashed elbow BELOW the boxes (down, across, up) so it never
crosses the diagram; box text is sized/reworded to fit inside the boxes.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = Path("results/pilot/figures"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(7.2, 2.4)); ax.axis("off")
ax.set_xlim(0, 15.7); ax.set_ylim(0.5, 5.5)

w = 1.9; h = 1.5; y = 2.7           # boxes span y = 2.7 .. 4.2, gaps 0.3
xs = [0.30, 2.50, 4.70, 6.90, 9.10, 11.30, 13.50]
stages = [("Query\n(desc + artifact\ndiagnostics)", "#eef3f8"),
          ("Retrieve\nN→rerank→MMR\n→ sample M", "#e7f5e7"),
          ("Assemble\n(applicability\n+ tier flags)", "#e7f5e7"),
          ("Solve\n(LLM + shell,\nflag check)", "#eef3f8"),
          ("Cite\n(executed\nactions)", "#fde7e7"),
          ("Reward\nR = o(1+λe)", "#fdf0e3"),
          ("Bandit update\n(cited∩sampled)", "#f3e9fb")]
for (txt, c), x in zip(stages, xs):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc=c, ec="#555", lw=1.0))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=6.3)
for x0, x1 in zip(xs[:-1], xs[1:]):
    ax.add_patch(FancyArrowPatch((x0 + w, y + h/2), (x1, y + h/2),
                                 arrowstyle="-|>", mutation_scale=8, color="#555", lw=1.0))

# gated write-back: elbow routed below the boxes (down, across, up-arrow)
wb = dict(color="#7a4fb0", lw=1.0, ls="--")
x_from = xs[-1] + w/2   # bottom center of "Bandit update"
x_to   = xs[1] + w/2    # bottom center of "Retrieve"
y_run  = 1.45
ax.plot([x_from, x_from], [y, y_run], **wb)
ax.plot([x_from, x_to], [y_run, y_run], **wb)
ax.add_patch(FancyArrowPatch((x_to, y_run), (x_to, y - 0.05), arrowstyle="-|>",
                             mutation_scale=9, color="#7a4fb0", lw=1.0, linestyle="--"))
ax.text((x_from + x_to) / 2, y_run - 0.38, "gated write-back (provisional→trusted)",
        ha="center", va="top", color="#7a4fb0", fontsize=7.2)

# false-lesson-lever labels above their stages
for t, x, c in [("recovery\nmasking", xs[4] + w/2, "#c0392b"),
                ("latent\npoison", xs[5] + w/2, "#ca7a1e"),
                ("birth\ndefect", xs[6] + w/2, "#7a4fb0")]:
    ax.text(x, y + h + 0.22, t, ha="center", va="bottom", fontsize=6.8, color=c, style="italic")
ax.text(xs[1] + w/2, y + h + 0.22, "sim. prior +\ncoarse context", ha="center", va="bottom",
        fontsize=6.8, color="#2e7d32", style="italic")
fig.savefig(FIG / "fig_pipeline.pdf", bbox_inches="tight")
print("wrote", FIG / "fig_pipeline.pdf")
