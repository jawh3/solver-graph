"""T1-2 (RIGOR_LEDGER): power / MDE of the exact CRT (and SNIPW) at the OBSERVED n.

Makes "CRT does not corroborate poison" interpretable: is it "no effect" or "the test
cannot see a -0.06 inclusion effect at n_treat~30-40"? We GRAFT a known inclusion harm of
size Delta onto each on-target poison lesson and measure the CRT rejection rate.

Graft (per lesson v, key=outcome, binary solve): among rows with T=1 (v sampled) and R=1,
flip R->0 w.p. q = Delta / E[R|T=1], so the grafted inclusion effect E[R|T=1]-E[R|T=0] shifts
by exactly -Delta. T and pi (the design) are untouched -> a clean injected causal effect.
power(Delta) = mean over {poison lessons x graft seeds} of 1[CRT p < 0.05]. Delta=0 row is the
Type-I rate (should be ~0.05). MDE = smallest Delta with power >= 0.8.

  python -m experiments.crt_power --dir <merged> [--deltas 0,0.02,0.04,0.057,0.08,0.10]
                                  [--n-graft 100 --B 1500 --label poison]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import benchmark.hard_ctf as hard
from instrumentation.jsonl_logger import load_runs
from ope.causal_value import _collect_obs
from retrieval.embedder import MockEmbedder

_fam_of = lambda cid: cid.rsplit("_", 1)[0]


def _crt_p_arrays(R, T, pi, rng, B):
    if len(R) < 10 or T.sum() < 3:
        return float("nan")
    s_obs = abs(float(np.sum((T - pi) * R)))
    draws = (rng.random((B, len(pi))) < pi).astype(np.float64)
    s_null = np.abs((draws - pi) @ R)
    return float((1 + np.sum(s_null >= s_obs)) / (1 + B))


def _snipw_ate(R, T, pi):
    w1 = T / pi
    w0 = (1 - T) / (1 - pi)
    if w1.sum() <= 0 or w0.sum() <= 0:
        return None
    return float((w1 * R).sum() / w1.sum() - (w0 * R).sum() / w0.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--deltas", default="0,0.02,0.04,0.057,0.08,0.10")
    ap.add_argument("--n-graft", type=int, default=100)
    ap.add_argument("--B", type=int, default=1500)
    ap.add_argument("--label", default="poison")
    ap.add_argument("--clip", type=float, default=1e-6)
    ap.add_argument("--n-mc", type=int, default=1000)
    ap.add_argument("--calibrate", action="store_true",
                    help="Type-I mode: permute R within each lesson's rows (kills any T-R link) and "
                         "report the CRT rejection rate (should be ~0.05 if exact). Writes crt_calib.json.")
    args = ap.parse_args()
    d = Path(args.dir)
    deltas = [float(x) for x in args.deltas.split(",")]

    summ = json.loads((d / "summary.json").read_text())
    families = summ.get("families") or json.loads((d / "progress.json").read_text()).get("families")
    _, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=families)
    runs = load_runs("campaign_outcome", root=str(d / "logs"))

    # per on-target lesson of the target label: arrays (R binary solve, T sampled, pi)
    by_fam = defaultdict(list)
    for r in runs:
        by_fam[_fam_of(r["challenge_id"])].append(r)
    lessons = {}
    for f, fr in by_fam.items():
        obs = _collect_obs(fr, clip=args.clip, n_mc=args.n_mc, response_key="outcome")
        for v, rows in obs.items():
            if truth.get(v, ("?", "?")) == (args.label, f):
                R = np.array([x[0] for x in rows], dtype=np.float64)
                T = np.array([x[1] for x in rows], dtype=np.float64)
                pi = np.array([x[2] for x in rows], dtype=np.float64)
                if len(R) >= 10 and T.sum() >= 3:
                    lessons[v] = (R, T, pi)

    rng = np.random.RandomState(0)

    if args.calibrate:
        # Type-I: permute R within each lesson's own rows (preserves R marginal, destroys T-R link)
        rej = []
        per_lesson = {}
        for v, (R, T, pi) in lessons.items():
            lr = []
            for _ in range(args.n_graft):
                Rp = rng.permutation(R)
                p = _crt_p_arrays(Rp, T, pi, rng, args.B)
                if not np.isnan(p):
                    lr.append(int(p < 0.05))
            if lr:
                per_lesson[v] = float(np.mean(lr))
                rej.extend(lr)
        out = {"dir": str(d), "label": args.label, "mode": "type_I_permute_R",
               "n_lessons": len(lessons), "n_perm": args.n_graft, "B": args.B,
               "typeI_rejection_rate": float(np.mean(rej)) if rej else None,
               "per_lesson_typeI": per_lesson}
        (d / "crt_calib.json").write_text(json.dumps(out, indent=2))
        print(f"[crt_calib] {d.name} {args.label}: Type-I rejection rate = "
              f"{out['typeI_rejection_rate']:.3f} (nominal 0.05; n_lessons={len(lessons)}, n_perm={args.n_graft})")
        return

    out = {"dir": str(d), "label": args.label, "n_lessons": len(lessons),
           "n_graft": args.n_graft, "B": args.B, "deltas": deltas, "power_crt": {}, "power_snipw": {},
           "per_lesson_n_treat": {v: int(T.sum()) for v, (R, T, pi) in lessons.items()}}

    for delta in deltas:
        crt_rej = []
        snipw_rej = []
        for v, (R, T, pi) in lessons.items():
            treated_solved = np.where((T == 1) & (R == 1))[0]
            m1 = R[T == 1].mean() if T.sum() else 0.0
            q = 0.0 if m1 <= 0 else min(1.0, delta / m1)
            for g in range(args.n_graft):
                Rg = R.copy()
                if q > 0 and len(treated_solved):
                    flip = rng.random(len(treated_solved)) < q
                    Rg[treated_solved[flip]] = 0.0
                p = _crt_p_arrays(Rg, T, pi, rng, args.B)
                crt_rej.append(0 if np.isnan(p) else int(p < 0.05))
                ate = _snipw_ate(Rg, T, pi)
                snipw_rej.append(int(ate is not None and ate < 0 and abs(ate) > delta * 0.4))  # crude one-sided proxy
        out["power_crt"][f"{delta}"] = float(np.mean(crt_rej)) if crt_rej else None
        out["power_snipw"][f"{delta}"] = float(np.mean(snipw_rej)) if snipw_rej else None

    # MDE = smallest delta>0 with CRT power >= 0.8
    mde = next((dl for dl in deltas if dl > 0 and (out["power_crt"].get(f"{dl}") or 0) >= 0.8), None)
    out["crt_MDE_pow80"] = mde
    out["crt_typeI_delta0"] = out["power_crt"].get("0.0") or out["power_crt"].get("0")
    (d / "crt_power.json").write_text(json.dumps(out, indent=2))
    print(f"[crt_power] {d.name} {args.label}: typeI={out['crt_typeI_delta0']:.3f} | "
          + " ".join(f"Δ{dl}:{out['power_crt'][f'{dl}']:.2f}" for dl in deltas if dl > 0)
          + f" | MDE@.8={mde}")


if __name__ == "__main__":
    main()
