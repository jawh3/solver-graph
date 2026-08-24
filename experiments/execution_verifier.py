#!/usr/bin/env python3
"""experiments/execution_verifier.py — the POSITIVE-CONTRAST deliverable.

Thesis of the negative result: a harmful (benign-wrong) lesson is undetectable from a capable
agent's OUTCOMES (it recovers for free), so per-item outcome-based causal credit cannot flag it.
But the lesson is still *false* — and in a VERIFIABLE domain you can flag it WITHOUT the agent, by
EXECUTING what the lesson prescribes and checking it produces a well-formed result. This is the
deployable answer ("verify the skill against its claim") and the paper's positive contrast.

Here a "lesson" is a decode recipe. We execute:
  - the HELPFUL lesson  = the family's own correct decode  -> recovers a picoCTF{...}-shaped flag.
  - the POISON lesson   = the depth-matched SIBLING's decode (wrong transforms) -> garbage / error.
Verification criterion is DEPLOYABLE (no secret answer needed): does applying the recipe to the
artifact yield a well-formed flag (startswith 'picoCTF{' and printable)? We report the verifier's
poison-flag rate vs the outcome estimator's (~0, null) on the SAME lessons.

Zero GPU. Deterministic.  python -m experiments.execution_verifier [--families-file ...]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import benchmark.hard_ctf as hard


def _sibling_map():
    """Reconstruct each grid family's depth-matched poison sibling (mirrors generate_family_grid)."""
    by_cell = defaultdict(list)
    for f in hard.GRID_FAMILIES:
        tags = hard._inner_tags_of(f.name)
        by_cell[(f.name.split("_")[0], len(tags))].append(f)
    sib = {}
    for _key, group in by_cell.items():
        for idx, f in enumerate(group):
            sib[f.name] = group[(idx + 1) % len(group)] if len(group) > 1 else f
    return sib


def _well_formed(s) -> bool:
    return isinstance(s, str) and s.startswith("picoCTF{") and s.endswith("}") and s.isprintable()


def _apply(decode, artifact):
    """Run a family's decode on an artifact; return the string or None on any failure (= flagged)."""
    try:
        out = decode(artifact)
        return out if isinstance(out, str) else None
    except Exception:
        return None


def _exec_malignant(arm: str, fam, art: bytes):
    """Deterministically EXECUTE a malignant recipe on an artifact (mirrors hard.malignant_recipe
    semantics; see tests/test_malignant_teeth.py). Returns the recipe's produced string or None."""
    if arm == "destroy":
        return _apply(fam.decode, art[:-16])               # strip the (fictional) 16-byte tag first
    if arm == "exhaust":
        budget = fam.solution_length + 2                   # the recipe burns one key per step
        for k in range(budget):                            # single-byte-XOR brute (the wrong model)
            cand = bytes(b ^ k for b in art)
            try:
                s = cand.decode("latin-1")
            except Exception:
                continue
            if _well_formed(s):
                return s
        return None
    if arm == "wrongflag":
        return hard._MALIGNANT_FLAG                         # a fixed, input-INDEPENDENT constant
    raise ValueError(arm)


def verify_malignant(families, arms=None, n_per_family: int = 8, seed: int = 7) -> dict:
    """Per malignant arm: the DEPLOYABLE flag rate (no secret needed). A recipe is 'verified' iff
    executing it yields a well-formed flag that is INPUT-SENSITIVE (differs across two distinct
    instances of the family) — a real decode of a random per-instance flag passes both; destroy/
    exhaust fail well-formedness; wrongflag fails input-sensitivity (constant output). flag = not
    verified. Expect ~1.0 flag rate for every arm, vs the outcome estimator's ~0 (null)."""
    arms = list(arms or hard.MALIGNANT_ARMS)
    by_name = {f.name: f for f in hard.GRID_FAMILIES}
    by_name.update({f.name: f for f in hard.FAMILIES})
    rng = np.random.RandomState(seed)
    per_arm = {a: {"verified": 0, "n": 0} for a in arms}
    for name in families:
        fam = by_name.get(name)
        if fam is None:
            continue
        for _ in range(n_per_family):
            f1 = hard._rand_flag(rng); art1 = fam.encode(f1, rng)
            f2 = hard._rand_flag(rng); art2 = fam.encode(f2, rng)
            for a in arms:
                o1 = _exec_malignant(a, fam, art1)
                o2 = _exec_malignant(a, fam, art2)
                verified = _well_formed(o1) and (o1 != o2)   # well-formed AND input-sensitive
                per_arm[a]["verified"] += int(verified); per_arm[a]["n"] += 1
    out = {"n_families": len([f for f in families if f in by_name]), "by_arm": {}}
    for a in arms:
        n = max(1, per_arm[a]["n"]); v = per_arm[a]["verified"] / n
        out["by_arm"][a] = {"n": per_arm[a]["n"], "verified_rate": v, "verifier_FLAG_rate": 1.0 - v}
    return out


def verify(families, n_per_family: int = 8, seed: int = 7) -> dict:
    by_name = {f.name: f for f in hard.GRID_FAMILIES}
    by_name.update({f.name: f for f in hard.FAMILIES})
    sib = _sibling_map()
    rng = np.random.RandomState(seed)

    helpful_ok = poison_ok = n = 0
    per_family = []
    for name in families:
        fam = by_name.get(name)
        if fam is None or name not in sib:
            continue
        sibling = sib[name]
        h_ok = p_ok = 0
        for _ in range(n_per_family):
            flag = hard._rand_flag(rng)
            art = fam.encode(flag, rng)
            if _well_formed(_apply(fam.decode, art)):       # helpful = own correct decode
                h_ok += 1
            if _well_formed(_apply(sibling.decode, art)):   # poison = sibling's (wrong) decode
                p_ok += 1
        helpful_ok += h_ok; poison_ok += p_ok; n += n_per_family
        per_family.append({"family": name, "sibling": sibling.name,
                            "helpful_verified": h_ok / n_per_family,
                            "poison_verified": p_ok / n_per_family})

    helpful_verify_rate = helpful_ok / max(1, n)
    poison_verify_rate = poison_ok / max(1, n)
    return {
        "n_families": len(per_family), "n_trials_per_label": n,
        "helpful_verified_rate": helpful_verify_rate,        # should be ~1.0
        "poison_verified_rate": poison_verify_rate,          # should be ~0.0
        "verifier_poison_FLAG_rate": 1.0 - poison_verify_rate,   # the headline: ~1.0
        "verifier_helpful_FALSE_FLAG_rate": 1.0 - helpful_verify_rate,  # should be ~0.0
        "per_family": per_family,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families-file", default="results/lab/probe_7b/selection_35.json")
    ap.add_argument("--n-per-family", type=int, default=8)
    ap.add_argument("--out", default="results/lab/execution_verifier.json")
    args = ap.parse_args()
    d = json.loads(Path(args.families_file).read_text())
    fams = d.get("in_band_families") or d.get("families") or []
    if fams and isinstance(fams[0], dict):
        fams = [f["family"] for f in fams]
    res = verify(fams, n_per_family=args.n_per_family)
    res["malignant"] = verify_malignant(fams, n_per_family=args.n_per_family)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"EXECUTION-VERIFIER on {res['n_families']} families ({args.families_file}):")
    print(f"  helpful verified : {res['helpful_verified_rate']:.3f}  (false-flag {res['verifier_helpful_FALSE_FLAG_rate']:.3f})")
    print(f"  poison  verified : {res['poison_verified_rate']:.3f}")
    print(f"  >> verifier FLAGS poison at rate {res['verifier_poison_FLAG_rate']:.3f}  "
          f"(vs outcome-estimator poison-flag ~0.0 / null)")
    print(f"  malignant arms (deployable flag rate; should be ~1.0):")
    for a, d in res["malignant"]["by_arm"].items():
        print(f"    M-{a:9s}: verifier FLAG rate {d['verifier_FLAG_rate']:.3f}  (verified {d['verified_rate']:.3f}, n={d['n']})")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
