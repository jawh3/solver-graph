"""T0-5a (RIGOR_LEDGER): measure A2's co-citation + displacement directly from logs.

Replaces the theorem's circular back-fitted delta_f with measurement:
  cocite      = P(HC | PC)            on-target helpful cited GIVEN on-target poison cited
  displacement delta_hat = P(HC | !PC) - P(HC | PC)
both pooled over runs of families that have BOTH labels, with a within-family
stratified bootstrap CI. Also the sampled-set (ITT) versions as the upper proxy.

Writes <merged>/cocitation.json.   Usage: python -m experiments.cocitation --dir <merged> [--seed 7]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments._loglib import load_runs_with_flags, stratified_boot


def _stats(runs, cited=True):
    a, b = ("HC", "PC") if cited else ("HS", "PS")
    pc = [r for r in runs if r[b]]
    npc = [r for r in runs if not r[b]]
    if not pc or not npc:
        return None
    p_hc_pc = sum(r[a] for r in pc) / len(pc)
    p_hc_npc = sum(r[a] for r in npc) / len(npc)
    return {"cocite": p_hc_pc, "p_hc_given_not_pc": p_hc_npc,
            "displacement": p_hc_npc - p_hc_pc, "n_pc": len(pc), "n_not_pc": len(npc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    merged = Path(args.dir)

    runs = load_runs_with_flags(merged)
    # restrict to families that actually have both labels among nodes seen in runs
    fams_with_both = {r["family"] for r in runs if r["HC"] or r["HS"]} & \
                     {r["family"] for r in runs if r["PC"] or r["PS"]}
    rr = [r for r in runs if r["family"] in fams_with_both]

    out = {"n_runs": len(runs), "n_runs_fams_with_both": len(rr),
           "n_families_with_both": len(fams_with_both)}
    for kind, cited in (("cited", True), ("sampled", False)):
        s = _stats(rr, cited)
        if s is None:
            out[kind] = None
            continue
        lo, hi = stratified_boot(
            rr, lambda x, c=cited: (_stats(x, c) or {}).get("displacement"),
            n_boot=args.n_boot, seed=args.seed)
        s["displacement_ci95"] = [lo, hi]
        out[kind] = s

    # per-family table (cited)
    per_fam = {}
    for f in sorted(fams_with_both):
        s = _stats([r for r in rr if r["family"] == f], True)
        if s:
            per_fam[f] = {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}
    out["per_family_cited"] = per_fam

    (merged / "cocitation.json").write_text(json.dumps(out, indent=2))
    c = out["cited"]
    print(f"[cocitation] {merged}: P(HC|PC)={c['cocite']:.3f} (n_pc={c['n_pc']}), "
          f"delta_hat={c['displacement']:+.3f} CI{c['displacement_ci95']}")


if __name__ == "__main__":
    main()
