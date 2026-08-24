"""T1-6 (RIGOR_LEDGER): persist the two free figures' data + summary stats from committed JSON.

(a) theta-sweep: promotion rate of poison vs helpful under promote-if-(value>theta), for outcome
    credit and causal credit. Headline: outcome credit has NO separating theta (un-tunable to safety);
    causal credit holds a helpful>poison gap (imperfect — also drops helpful, keeps ~1/3 poison).
(b) per-lesson forest: causal_ATE per on-target lesson by label (poison = mean-zero scatter vs
    helpful positive). + Cochran's Q heterogeneity proxy on the poison point estimates.

Writes <merged>/figures_data.json per rung. Pure extraction, no GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results" / "lab"
RUNGS = ["campaign_3b_M3", "campaign_7b35_M3_rep", "campaign_14b_M3_powered",
         "campaign_32b_M3_tp2", "campaign_llama8b_M3"]


def main():
    for camp in RUNGS:
        d = ROOT / camp / "merged"
        la = json.loads((d / "lab_analysis.json").read_text())
        nv = json.loads((d / "node_values.json").read_text())
        ts = la.get("theta_sweep", {})
        th = ts.get("thetas", [])
        oq = ts.get("outcome_Q_rates", [])
        ca = ts.get("causal_ATE_rates", [])

        def gap(rates, i):
            r = rates[i] if isinstance(rates, list) else rates.get(str(th[i]), {})
            return (r.get("helpful"), r.get("poison"))

        sweep = []
        for i, t in enumerate(th):
            oh, op = gap(oq, i)
            ch, cp = gap(ca, i)
            sweep.append({"theta": t, "outcome_helpful": oh, "outcome_poison": op,
                          "outcome_gap": (oh - op) if (oh is not None and op is not None) else None,
                          "causal_helpful": ch, "causal_poison": cp,
                          "causal_gap": (ch - cp) if (ch is not None and cp is not None) else None})

        # per-lesson forest (point estimates from node_values causal_ATE)
        forest = {"poison": [], "helpful": []}
        for v in nv.values():
            lab = v.get("label")
            a = v.get("causal_ATE")
            if lab in forest and a is not None:
                forest[lab].append(round(a, 4))

        def hetero(vals):
            if len(vals) < 3:
                return None
            v = np.array(vals)
            return {"n": len(v), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "frac_below_0": float((v < 0).mean()), "frac_above_0": float((v > 0).mean()),
                    "min": float(v.min()), "max": float(v.max())}

        out = {"campaign": camp,
               "theta_sweep": sweep,
               "outcome_Q_separating_theta": ts.get("outcome_Q_separating_theta"),
               "causal_ATE_separating_theta": ts.get("causal_ATE_separating_theta"),
               "forest_poison_ATE": sorted(forest["poison"]),
               "forest_helpful_ATE": sorted(forest["helpful"]),
               "poison_heterogeneity": hetero(forest["poison"]),
               "helpful_heterogeneity": hetero(forest["helpful"])}
        (d / "figures_data.json").write_text(json.dumps(out, indent=2))
        ph = out["poison_heterogeneity"] or {}
        print(f"{camp}: outcome_sep_theta={out['outcome_Q_separating_theta']} | "
              f"poison ATE mean={ph.get('mean')} sd={ph.get('sd')} "
              f"below0={ph.get('frac_below_0')} above0={ph.get('frac_above_0')} (n={ph.get('n')})")


if __name__ == "__main__":
    main()
