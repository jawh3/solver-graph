"""Shared loaders for the T0-5 mechanism-artifact scripts (RIGOR_LEDGER).

Reads a campaign `merged/` dir (runs.jsonl + citations.jsonl + node_values.json) into
per-run flags used by cocitation.py / a1_contrast.py / step_detour.py / base_rate_sweep.py:

  family  : challenge family (challenge_id with the trailing instance index stripped)
  HC / PC : an ON-TARGET helpful / poison lesson (node family == run family) was CITED
  HS / PS : same, but SAMPLED (ITT version)
  no-recipe: neither on-target label cited (the within-family base arm b_f)

Determinism: all bootstraps take an explicit seed; no wall-clock anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

COND = "campaign_outcome"


def _family_of_challenge(challenge_id: str) -> str:
    """b64_xor2a_rev_15 -> b64_xor2a_rev (strip trailing integer instance index)."""
    head, _, tail = challenge_id.rpartition("_")
    return head if tail.isdigit() else challenge_id


def load_nodes(merged: Path) -> dict:
    """node_id -> {label, family, outcome_Q, causal_ATE}."""
    return json.loads((merged / "node_values.json").read_text())


def family_label_nodes(nodes: dict) -> dict:
    """family -> {'helpful': set(node_ids), 'poison': set(node_ids)}."""
    fam: dict = {}
    for nid, d in nodes.items():
        lab = d.get("label")
        if lab not in ("helpful", "poison"):
            continue
        fam.setdefault(d["family"], {"helpful": set(), "poison": set()})[lab].add(nid)
    return fam


def load_runs_with_flags(merged: Path, cond: str = COND) -> list[dict]:
    """One record per run: family, outcome, steps, max_steps, seed, HC, PC, HS, PS."""
    nodes = load_nodes(merged)
    fam_nodes = family_label_nodes(nodes)

    cited: dict[str, set] = {}
    sampled: dict[str, set] = {}
    with open(merged / "logs" / cond / "citations.jsonl") as f:
        for line in f:
            c = json.loads(line)
            rid = c["run_id"]
            if c.get("sampled"):
                sampled.setdefault(rid, set()).add(c["node_id"])
            if c.get("cited"):
                cited.setdefault(rid, set()).add(c["node_id"])

    out = []
    with open(merged / "logs" / cond / "runs.jsonl") as f:
        for line in f:
            r = json.loads(line)
            fam = _family_of_challenge(r["challenge_id"])
            fn = fam_nodes.get(fam)
            if fn is None:           # family without labelled lessons (shouldn't happen)
                continue
            cit = cited.get(r["run_id"], set())
            smp = sampled.get(r["run_id"], set()) or set(r.get("sampled_set") or [])
            out.append({
                "run_id": r["run_id"],
                "family": fam,
                "outcome": int(r.get("outcome") or 0),
                "steps": r.get("steps"),
                "max_steps": (r.get("challenge_meta") or {}).get("max_steps"),
                "seed": r.get("seed"),
                "HC": bool(fn["helpful"] & cit),
                "PC": bool(fn["poison"] & cit),
                "HS": bool(fn["helpful"] & smp),
                "PS": bool(fn["poison"] & smp),
            })
    return out


def stratified_boot(runs: list[dict], stat_fn, n_boot: int = 1000, seed: int = 7):
    """Bootstrap CI resampling runs WITHIN family (strata preserved). stat_fn(list)->float|None."""
    import numpy as np
    rng = np.random.RandomState(seed)
    by_fam: dict[str, list] = {}
    for r in runs:
        by_fam.setdefault(r["family"], []).append(r)
    fams = sorted(by_fam)
    stats = []
    for _ in range(n_boot):
        resampled = []
        for f in fams:
            rows = by_fam[f]
            idx = rng.randint(0, len(rows), size=len(rows))
            resampled.extend(rows[i] for i in idx)
        s = stat_fn(resampled)
        if s is not None:
            stats.append(s)
    if not stats:
        return None, None
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)
