"""T0-5c (RIGOR_LEDGER): per-lesson on-target ATE vs family base-solve (Spearman).

Tests the 'no base-rate gradient' claim from a scripted artifact (the RUN_LOG quotes
Spearman -0.089 p=.60 at 14B from an ad-hoc computation). For each label:
  x_f = within-family base b_f (solve among no-recipe runs of family f, from logs)
  y_f = node_values.json causal_ATE_ontarget_o of the family's lesson (ON-TARGET binary-outcome
        SNIPW). F8 (FIX_CATALOG 2026-09): the previous version read `causal_ATE`, which is the
        MARGINAL R_final estimate (diluted ~10x by off-target runs), so the "no base-rate gradient"
        test was a correlation of b_f with noise. Falls back to the legacy column only when the
        on-target column is absent, and records which column it used.
Spearman rho + p per label. Writes <merged>/base_rate_sweep.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments._loglib import load_nodes, load_runs_with_flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    merged = Path(args.dir)

    runs = load_runs_with_flags(merged)
    base: dict[str, list] = {}
    for r in runs:
        if not r["HC"] and not r["PC"]:
            base.setdefault(r["family"], []).append(r["outcome"])
    b_f = {f: sum(v) / len(v) for f, v in base.items() if v}

    nodes = load_nodes(merged)
    col = "causal_ATE_ontarget_o" if any(d.get("causal_ATE_ontarget_o") is not None for d in nodes.values()) else "causal_ATE"
    out = {"n_families_with_base": len(b_f), "value_column": col}
    try:
        from scipy.stats import spearmanr
    except ImportError:
        spearmanr = None
    for label in ("poison", "helpful"):
        pairs = [(b_f[d["family"]], d[col]) for d in nodes.values()
                 if d.get("label") == label and d.get(col) is not None
                 and d.get("family") in b_f]
        row = {"n": len(pairs),
               "pairs": [[round(a, 4), round(b, 4)] for a, b in sorted(pairs)]}
        if spearmanr and len(pairs) >= 3:
            rho, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
            row.update({"spearman_rho": float(rho), "p": float(p)})
        out[label] = row
    (merged / "base_rate_sweep.json").write_text(json.dumps(out, indent=2))
    pz = out["poison"]
    print(f"[base_rate_sweep] {merged}: poison rho={pz.get('spearman_rho'):.3f} "
          f"p={pz.get('p'):.3f} (n={pz['n']})" if pz.get("spearman_rho") is not None
          else f"[base_rate_sweep] {merged}: n={pz['n']} (scipy missing?)")


if __name__ == "__main__":
    main()
