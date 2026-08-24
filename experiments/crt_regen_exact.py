"""T1-3a (RIGOR_LEDGER): regenerate every CRT artifact at the EXACT setting (clip=1e-6,
n_mc=1000) and record the clipped-vs-exact robustness comparison.

The shipped crt_test*.json used clip=0.05 (an IPW positivity guard) which makes the exact
CRT null draw T' from the WRONG law. This rewrites crt_test.json (R_final) + crt_test_outcome.json
(outcome) at clip~0, and writes crt_robustness.json = {clipped, exact} per key so the paper can
state the conclusions are clip-invariant.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.crt_test import run

RUNGS = ["campaign_3b_M3", "campaign_7b35_M3_rep", "campaign_14b_M3_powered",
         "campaign_32b_M3_tp2", "campaign_llama8b_M3"]
ROOT = Path(__file__).resolve().parents[1] / "results" / "lab"


def _slim(r):
    return {s: {l: {k: r[s][l][k] for k in ("n", "frac_sig_p05", "median_p", "min_p")}
                for l in r[s] if l in r[s]} for s in ("on_target", "marginal")}


def main():
    for camp in RUNGS:
        d = ROOT / camp / "merged"
        if not d.exists():
            print(f"skip {camp} (no merged)")
            continue
        rob = {"campaign": camp}
        for key, out in (("R_final", "crt_test.json"), ("outcome", "crt_test_outcome.json")):
            clipped = run(d, key=key, clip=0.05, n_mc=400)
            exact = run(d, key=key, clip=1e-6, n_mc=1000)
            exact["clip"] = 1e-6
            exact["n_mc"] = 1000
            (d / out).write_text(json.dumps(exact, indent=2))   # canonical = exact
            rob[key] = {"clipped_0.05_nmc400": _slim(clipped), "exact_1e-6_nmc1000": _slim(exact)}
            po, pe = clipped["on_target"].get("poison", {}), exact["on_target"].get("poison", {})
            print(f"{camp} [{key}] poison on-tgt frac {po.get('frac_sig_p05')}->{pe.get('frac_sig_p05')} "
                  f"med {po.get('median_p',0):.3f}->{pe.get('median_p',0):.3f}")
        (d / "crt_robustness.json").write_text(json.dumps(rob, indent=2))
    print("DONE crt_regen_exact")


if __name__ == "__main__":
    main()
