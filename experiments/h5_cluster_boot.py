"""T1-4 (RIGOR_LEDGER): family-CLUSTER bootstrap for the H5 transfer Delta.

The committed H5 Delta(causal_mem - outcome_mem) uses an item-level paired bootstrap over ~672 paired
(family,challenge,seed) items, but those items cluster into ~42 families (16 each) -> items within a
family are correlated, so the item-level CI understates uncertainty. This recomputes Delta with a
FAMILY-cluster bootstrap (resample families as blocks) and reports both, for 14B + 32B.

Verdict-relevant: if the 32B family-cluster CI now INCLUDES 0, the marginal item-level p=.04 was a
clustering artifact and the 32B verdict becomes a cleaner null. Writes <h5dir>/h5_cluster_boot.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

H5 = {"14b": "results/lab/h5_14b/h5.json", "32b": "results/lab/h5_32b/h5.json"}
ROOT = Path(__file__).resolve().parents[1]


def paired_diffs_by_family(h5):
    """Return {family: np.array(diffs)} for d = causal_mem - outcome_mem on common (challenge,seed)."""
    out = {}
    for f in h5["families"]:
        rec = f.get("rec", {})
        cm = {(c, s): o for c, s, o in rec.get("causal_mem", [])}
        om = {(c, s): o for c, s, o in rec.get("outcome_mem", [])}
        keys = sorted(set(cm) & set(om))
        if keys:
            out[f["family"]] = np.array([cm[k] - om[k] for k in keys], dtype=np.float64)
    return out


def boot_ci(diffs_flat, n_boot, rng):
    n = len(diffs_flat)
    means = [diffs_flat[rng.randint(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min(np.mean(np.array(means) <= 0), np.mean(np.array(means) >= 0))
    return float(diffs_flat.mean()), float(lo), float(hi), float(min(1.0, p))


def cluster_boot_ci(by_fam, n_boot, rng):
    fams = list(by_fam)
    means = []
    for _ in range(n_boot):
        pick = [by_fam[fams[i]] for i in rng.randint(0, len(fams), len(fams))]
        means.append(np.concatenate(pick).mean())
    means = np.array(means)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min(np.mean(means <= 0), np.mean(means >= 0))
    flat = np.concatenate(list(by_fam.values()))
    return float(flat.mean()), float(lo), float(hi), float(min(1.0, p))


def main():
    out = {}
    for tag, path in H5.items():
        h5 = json.loads((ROOT / path).read_text())
        by_fam = paired_diffs_by_family(h5)
        flat = np.concatenate(list(by_fam.values()))
        rng = np.random.RandomState(0)
        im = boot_ci(flat, 10000, rng)
        rng = np.random.RandomState(0)
        cl = cluster_boot_ci(by_fam, 10000, rng)
        out[tag] = {
            "n_families": len(by_fam), "n_pairs": int(len(flat)),
            "item_boot": {"delta_pp": 100 * im[0], "ci95_pp": [100 * im[1], 100 * im[2]], "p": im[3]},
            "family_cluster_boot": {"delta_pp": 100 * cl[0], "ci95_pp": [100 * cl[1], 100 * cl[2]], "p": cl[3]},
            "verdict_item": _verdict(im[1], im[2]), "verdict_cluster": _verdict(cl[1], cl[2])}
        (ROOT / path).with_name("h5_cluster_boot.json").write_text(json.dumps(out[tag], indent=2))
        print(f"[{tag}] Δ={100*im[0]:+.2f}pp | item CI[{100*im[1]:+.2f},{100*im[2]:+.2f}] p={im[3]:.3f} ({out[tag]['verdict_item']}) "
              f"-> CLUSTER CI[{100*cl[1]:+.2f},{100*cl[2]:+.2f}] p={cl[3]:.3f} ({out[tag]['verdict_cluster']})  "
              f"[{len(by_fam)} families, {len(flat)} pairs]")
    (ROOT / "results/lab/h5_cluster_boot_summary.json").write_text(json.dumps(out, indent=2))


def _verdict(lo, hi):
    if lo > 0 and lo >= 0.05: return "harm_prevented"
    if hi < 0 and hi <= -0.05: return "reverse"
    if abs(lo) <= 0.025 and abs(hi) <= 0.025: return "null"
    return "inconclusive"


if __name__ == "__main__":
    main()
