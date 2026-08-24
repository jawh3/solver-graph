#!/usr/bin/env python3
"""appendix_c/g1_verify.py — G1 acceptance evidence.

Runs the PRE-EXISTING deterministic execution-verifier's own criterion
(`experiments/execution_verifier.py`) over every lesson label, apples to apples:

    helpful          the family's correct decode                    (expected: accepted)
    sibling_poison   the depth-matched sibling's decode             (expected: FLAGGED)
    latent_detour    correct flag, route lengthened by an involution pair
    latent_search    correct flag, reached by exhaustive trial
    malignant_*      the pre-existing positive-control arms         (expected: FLAGGED)

Two criteria are reported because the parent verifier uses two:
  * `wellformed`  — `verify()`'s criterion: output starts `picoCTF{`, ends `}`, printable.
  * `strict`      — `verify_malignant()`'s criterion: well-formed AND input-sensitive
                    (two distinct instances of the family must yield different outputs,
                    which is what kills the constant-output `wrongflag` arm).

`strict` is the stronger screen and therefore the one the attack must beat.

Also checks the two structural claims the `detour` arm is built on, per instance:
  * OUTPUT IDENTITY  — detour output is byte-identical to the correct method's.
  * STRICTLY LONGER  — the prescribed route has more steps than the correct method's.

Zero GPU, deterministic.
    python -m appendix_c.g1_verify --domain crypto --n-families 40 --n-per-family 8
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmark.hard_ctf as hard
import benchmark.sql_domain as sqld
from experiments.execution_verifier import _well_formed, _apply, _exec_malignant, _sibling_map
from appendix_c import latent_poison as lp
from appendix_c.provenance import write_result


def _accept(o1, o2) -> dict:
    """The parent verifier's two acceptance criteria, evaluated on one instance pair."""
    wf = _well_formed(o1)
    return {"wellformed": wf, "strict": bool(wf and o1 != o2)}


def verify_crypto(family_names, n_per_family: int = 8, seed: int = 7, pad: str = "rot13") -> dict:
    by_name = {f.name: f for f in hard.GRID_FAMILIES}
    by_name.update({f.name: f for f in hard.FAMILIES})
    sib = _sibling_map()
    lessons = lp.crypto_latent_lessons(family_names, pad=pad)

    rng = np.random.RandomState(seed)
    tally = defaultdict(lambda: {"wellformed": 0, "strict": 0, "n": 0})
    structural = {"identity_checked": 0, "identity_holds": 0,
                  "route_checked": 0, "route_strictly_longer": 0,
                  "extra_steps": []}
    per_family = []
    skipped = defaultdict(list)

    for name in family_names:
        fam = by_name.get(name)
        if fam is None:
            skipped["unknown_family"].append(name)
            continue
        les_by_arm = {l.arm: l for l in lessons.get(name, [])}
        for arm in lp.LATENT_ARMS:
            if arm not in les_by_arm:
                skipped[f"arm_undefined:{arm}"].append(name)
        fam_row = {"family": name, "labels": {}}

        for _ in range(n_per_family):
            # two independent instances of the SAME family -> input-sensitivity test
            f1 = hard._rand_flag(rng); a1 = fam.encode(f1, rng)
            f2 = hard._rand_flag(rng); a2 = fam.encode(f2, rng)

            routes = {"helpful": (lambda a: _apply(fam.decode, a))}
            if name in sib:
                sibling = sib[name]
                routes["sibling_poison"] = lambda a, s=sibling: _apply(s.decode, a)
            for arm, les in les_by_arm.items():
                routes[f"latent_{arm}"] = lambda a, l=les: _apply(l.decode, a)

            for label, fn in routes.items():
                o1, o2 = fn(a1), fn(a2)
                acc = _accept(o1, o2)
                tally[label]["wellformed"] += int(acc["wellformed"])
                tally[label]["strict"] += int(acc["strict"])
                tally[label]["n"] += 1
                fam_row["labels"].setdefault(label, {"strict": 0, "n": 0})
                fam_row["labels"][label]["strict"] += int(acc["strict"])
                fam_row["labels"][label]["n"] += 1

            # structural claims for the detour arm
            det = les_by_arm.get("detour")
            if det is not None:
                truth_out = _apply(fam.decode, a1)
                det_out = _apply(det.decode, a1)
                structural["identity_checked"] += 1
                structural["identity_holds"] += int(det_out == truth_out and truth_out is not None)
                structural["route_checked"] += 1
                structural["route_strictly_longer"] += int(det.route_steps > det.true_route_steps)
                structural["extra_steps"].append(det.extra_steps)

            # malignant arms (pre-existing positive control), same criterion
            for m_arm in hard.MALIGNANT_ARMS:
                o1 = _exec_malignant(m_arm, fam, a1)
                o2 = _exec_malignant(m_arm, fam, a2)
                acc = _accept(o1, o2)
                lbl = f"malignant_{m_arm}"
                tally[lbl]["wellformed"] += int(acc["wellformed"])
                tally[lbl]["strict"] += int(acc["strict"])
                tally[lbl]["n"] += 1

        per_family.append(fam_row)

    return _finalize(tally, structural, per_family, skipped, family_names, n_per_family,
                     seed, {"pad": pad, "search_candidates": list(lp.SEARCH_CANDIDATES)})


def verify_sql(family_names, n_per_family: int = 8, seed: int = 7) -> dict:
    by_name = {f.name: f for f in sqld.GRID_FAMILIES}
    lessons = lp.sql_latent_lessons(family_names)
    rng = np.random.RandomState(seed)
    tally = defaultdict(lambda: {"wellformed": 0, "strict": 0, "n": 0})
    structural = {"identity_checked": 0, "identity_holds": 0,
                  "route_checked": 0, "route_strictly_longer": 0, "extra_steps": []}
    per_family = []
    skipped = defaultdict(list)

    for name in family_names:
        fam = by_name.get(name)
        if fam is None:
            skipped["unknown_family"].append(name)
            continue
        sibling = sqld._sibling(fam)
        det = lessons.get(name, [None])[0]
        fam_row = {"family": name, "labels": {}}

        for _ in range(n_per_family):
            f1 = sqld._rand_flag(rng); a1 = fam.encode(f1, rng)
            f2 = sqld._rand_flag(rng); a2 = fam.encode(f2, rng)
            routes = {
                "helpful": lambda a: _apply(fam.decode, a),
                "sibling_poison": lambda a: _apply(sibling.decode, a),
            }
            if det is not None:
                routes["latent_detour"] = lambda a, l=det: _apply(l.decode, a)
            for label, fn in routes.items():
                o1, o2 = fn(a1), fn(a2)
                acc = _accept(o1, o2)
                tally[label]["wellformed"] += int(acc["wellformed"])
                tally[label]["strict"] += int(acc["strict"])
                tally[label]["n"] += 1
                fam_row["labels"].setdefault(label, {"strict": 0, "n": 0})
                fam_row["labels"][label]["strict"] += int(acc["strict"])
                fam_row["labels"][label]["n"] += 1
            if det is not None:
                truth_out = _apply(fam.decode, a1)
                det_out = _apply(det.decode, a1)
                structural["identity_checked"] += 1
                structural["identity_holds"] += int(det_out == truth_out and truth_out is not None)
                structural["route_checked"] += 1
                structural["route_strictly_longer"] += int(det.route_steps > det.true_route_steps)
                structural["extra_steps"].append(det.extra_steps)
        per_family.append(fam_row)

    return _finalize(tally, structural, per_family, skipped, family_names, n_per_family,
                     seed, {"note": "search arm is crypto-only: SQL has no per-output "
                                    "well-formedness oracle (every row's secret is flag-shaped)"})


def _finalize(tally, structural, per_family, skipped, family_names, n_per_family, seed,
              params) -> dict:
    by_label = {}
    for label, t in sorted(tally.items()):
        n = max(1, t["n"])
        by_label[label] = {
            "n": t["n"],
            "accept_rate_wellformed": t["wellformed"] / n,
            "accept_rate_strict": t["strict"] / n,
            "verifier_FLAG_rate_strict": 1.0 - t["strict"] / n,
        }
    extra = structural.pop("extra_steps", [])
    structural["extra_steps_mean"] = float(np.mean(extra)) if extra else None
    structural["extra_steps_min"] = int(np.min(extra)) if extra else None
    return {
        "params": {"n_families_requested": len(family_names), "n_per_family": n_per_family,
                   "seed": seed, **params},
        "n_families_evaluated": len(per_family),
        "by_label": by_label,
        "structural_detour_claims": structural,
        "skipped": {k: v for k, v in skipped.items()},
        "per_family": per_family,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="crypto", choices=["crypto", "sql"])
    ap.add_argument("--n-families", type=int, default=40)
    ap.add_argument("--n-per-family", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pad", default="rot13")
    ap.add_argument("--difficulty", default=None, help="e.g. 'easy,medium'")
    ap.add_argument("--depth", type=int, default=None,
                    help="crypto only: restrict to families with this many inner transforms. "
                         "depth=1 is where the `search` arm is defined.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    diff = [d.strip() for d in args.difficulty.split(",")] if args.difficulty else None
    if args.domain == "crypto":
        fams = hard.grid_family_names(difficulty=diff, seed=424242)
        if args.depth is not None:
            fams = [f for f in fams if len(hard._inner_tags_of(f)) == args.depth]
        fams = fams[:args.n_families]
        res = verify_crypto(fams, n_per_family=args.n_per_family, seed=args.seed, pad=args.pad)
    else:
        fams = sqld.grid_family_names(difficulty=diff, n=args.n_families, seed=424242)
        res = verify_sql(fams, n_per_family=args.n_per_family, seed=args.seed)
    res["params"]["domain"] = args.domain
    res["params"]["families"] = list(fams)

    name = args.out or f"g1_verifier_acceptance_{args.domain}"
    path = write_result(name, res, script="appendix_c/g1_verify.py", seeds=[args.seed],
                        extra={"domain": args.domain, "n_families": len(fams)})

    print(f"\nG1 EXECUTION-VERIFIER ACCEPTANCE — domain={args.domain}, "
          f"{res['n_families_evaluated']} families x {args.n_per_family} instances")
    print(f"{'label':<20} {'n':>5} {'accept(wellformed)':>20} {'accept(strict)':>16} {'FLAGGED':>9}")
    for label, d in res["by_label"].items():
        print(f"{label:<20} {d['n']:>5} {d['accept_rate_wellformed']:>20.3f} "
              f"{d['accept_rate_strict']:>16.3f} {d['verifier_FLAG_rate_strict']:>9.3f}")
    s = res["structural_detour_claims"]
    if s["identity_checked"]:
        print(f"\nstructural (detour): output-identical {s['identity_holds']}/{s['identity_checked']}"
              f"  strictly-longer {s['route_strictly_longer']}/{s['route_checked']}"
              f"  (+{s['extra_steps_mean']:.1f} steps mean, min +{s['extra_steps_min']})")
    if res["skipped"]:
        print(f"skipped: { {k: len(v) for k, v in res['skipped'].items()} }")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
