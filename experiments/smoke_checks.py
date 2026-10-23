#!/usr/bin/env python3
"""experiments/smoke_checks.py - PASS/FAIL report for GPU smokes (FIX_CATALOG 2026-09; review 2026-09-05).

Reads a campaign root (shard_*/logs/campaign_<cond>/{runs,citations,steps}.jsonl + corpora) and checks:

  F1   no family string in any solver-visible corpus path
  F2   distinctive citation: no citation grounded on an outer-transport token; cite rates by label
  F3   deterministic arms: p=1 and sampled == pool top-M in the arm's logged pool order; agreement with
       the cross-encoder top-M reported. Stochastic arms: exploration fraction ~ epsilon, no exploration
       in frozen runs, per-draw p, and - the quantity the deployed arm actually samples from - the mean
       logged deploy propensity pi_deploy(on-target helpful / poison) recomputed from the candidate
       scores (review F-2), overall and by training tercile.
  F3-mechanics  the deployed arm's bandit contexts are consistent with ONE context key
  F3-prior      pi_deploy(on-target helpful) vs chance M/pool with a family-cluster bootstrap CI (informative)
  F3-learning   last-minus-first tercile of pi_deploy(on-target helpful) > 0 (CI excl. 0); GATED only when
                the arm has >= MIN_TRAIN deploy-stream train runs (a smoke cannot show learning - review F-2)
  F4b  no run logged as llm_error; F4a no query_memory steps; F9 the same eval items in every condition

Usage: python -m experiments.smoke_checks --root <campaign root> [--probe DIR]. Exit 1 if a BLOCKING check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import importlib as _il
hard = _il.import_module(os.environ.get("CTF_DOMAIN", "benchmark.hard_ctf"))
from retrieval.embedder import MockEmbedder
from ope.causal_value import marginal_inclusion_propensities

_fam_of = lambda cid: cid.rsplit("_", 1)[0]
MIN_TRAIN = 300                         # deploy-stream train runs needed before the learning gate applies
OUTER = {"base64", "xxd", "fromhex", "b32decode", "base32", "b85decode", "base85"}
ALPHABETS = {"hex", "base32", "base64", "base85"}
GLOBAL = {}                             # cond -> stats


def _load(root: Path, cond: str, kind: str):
    out = []
    files = sorted(root.glob(f"shard_*/logs/campaign_{cond}/{kind}.jsonl")) or sorted(root.glob(f"logs/campaign_{cond}/{kind}.jsonl"))
    for f in files:
        for line in open(f):
            if line.strip():
                out.append(json.loads(line))
    return out


def _conditions(root: Path):
    conds = set()
    for d in list(root.glob("shard_*/logs/campaign_*")) + list(root.glob("logs/campaign_*")):
        conds.add(d.name[len("campaign_"):])
    return sorted(conds)


def _cluster_boot(vals, fams, n_boot=2000, seed=0):
    """Family-cluster bootstrap 95% CI of the mean of `vals` (aligned with `fams`)."""
    vals = np.asarray(vals, dtype=float); fams = np.asarray(fams)
    ok = ~np.isnan(vals); vals, fams = vals[ok], fams[ok]
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    uf = sorted(set(fams)); by = {f: vals[fams == f] for f in uf}; rng = np.random.RandomState(seed)
    bs = [np.mean(np.concatenate([by[f] for f in rng.choice(uf, len(uf), replace=True)])) for _ in range(n_boot)]
    return float(np.mean(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def check_f1(root: Path, report, fails):
    bad, n = [], 0
    for cdir in list(root.glob("shard_*/corpus/*")) + list(root.glob("corpus/*")):
        n += 1; name = cdir.name.lower()
        if any(name.startswith(f.lower() + "_") or name == f.lower() for f in hard._BY_NAME):
            bad.append(str(cdir))
    report.append(f"F1 corpus dirs: {n} checked, {len(bad)} carry a family string")
    if bad:
        fails.append(("F1", f"family string in corpus dir names, e.g. {bad[:3]}"))


def check_f4_errors(runs, cond, report, fails):
    conn = [r for r in runs if r.get("gave_up_reason") == "llm_error"]
    bad = [r for r in runs if r.get("gave_up_reason") == "llm_bad_request"]
    reasons = Counter(r.get("gave_up_reason") or ("solved" if r.get("outcome") else "?") for r in runs)
    report.append(f"[{cond}] n={len(runs)} solve={np.mean([r['outcome'] for r in runs]):.3f} "
                  f"steps={np.mean([r['steps'] for r in runs]):.2f} reasons={dict(reasons)}")
    if conn:
        fails.append(("F4b", f"{cond}: {len(conn)} runs logged as llm_error (must be raised, never logged)"))
    if bad:
        report.append(f"[{cond}] NOTE {len(bad)} harness errors (llm_bad_request, error=True, excluded by merge)")


def _context_key_of(contexts):
    """Infer the context key from the logged context ids; None if inconsistent."""
    ids = set(contexts)
    if not ids:
        return None
    if all(c.startswith("ctx::cluster::") for c in ids):
        return "cluster"
    cats = {c[len("ctx::cat::"):] for c in ids if c.startswith("ctx::cat::")}
    if len(cats) != len(ids):
        return None
    if cats == {"global"}:
        return "global"
    if cats <= ALPHABETS:
        return "alphabet"
    if all(c in hard._BY_NAME for c in cats):
        return "family"
    return None


def check_f3(runs, cond, truth, hid, pid, report, fails, n_mc=200):
    if not runs or not runs[0].get("candidate_pool"):
        return
    pol = runs[0].get("policy") or {}
    M = int(np.median([len(r["sampled_set"]) for r in runs]))
    pd = np.array([d["p"] for r in runs for d in r.get("per_draw_propensity", [])])
    expl = [bool(r.get("exploration")) for r in runs]
    report.append(f"[{cond}] policy={pol} M={M} per-draw p: mean {pd.mean():.3f} p5 {np.percentile(pd,5):.3f} "
                  f"p95 {np.percentile(pd,95):.3f} max {pd.max():.3f}; exploration fraction {np.mean(expl):.2f}")
    st = GLOBAL.setdefault(cond, {})
    st["M"] = M; st["chance"] = float(np.mean([M / max(len(r["candidate_pool"]), 1) for r in runs]))
    st["contexts"] = Counter(str(r.get("context_node_id")) for r in runs)
    st["context_key"] = _context_key_of(st["contexts"])
    report.append(f"[{cond}] contexts ({st['context_key']}): {dict(st['contexts'].most_common(6))}")

    # empirical on-target inclusion by stream
    incl = defaultdict(list)
    for r in runs:
        f = _fam_of(r["challenge_id"]); h, p_ = hid.get(f), pid.get(f)
        if h is None:
            continue
        _pol = r.get("policy") or {}
        _train = (r.get("phase") == "train") if r.get("phase") else bool(_pol.get("explore_enabled", True))
        stream = "explore" if r.get("exploration") else ("train" if _train else "eval")
        incl[(stream, "helpful")].append(int(h in set(r["sampled_set"])))
        incl[(stream, "poison")].append(int(p_ in set(r["sampled_set"])))
    for (stream, lab), v in sorted(incl.items()):
        report.append(f"[{cond}] on-target {lab} in M-set ({stream} runs): {np.mean(v):.3f} (n={len(v)}; chance {st['chance']:.3f})")
    st["incl_deploy_helpful"] = float(np.mean(incl[("train", "helpful")] + incl[("eval", "helpful")])) \
        if (incl.get(("train", "helpful")) or incl.get(("eval", "helpful"))) else None

    if pol.get("deterministic"):
        ok_p = bool(np.all(pd == 1.0))
        ok_top = all(r["sampled_set"] == [c["node_id"] for c in r["candidate_pool"][:len(r["sampled_set"])]] for r in runs)
        agree = []
        for r in runs:
            by_rr = sorted(r["candidate_pool"], key=lambda c: -float(c.get("rerank_score", 0) or 0))
            agree.append(set(r["sampled_set"]) == {c["node_id"] for c in by_rr[:len(r["sampled_set"])]})
        report.append(f"[{cond}] deterministic (pool order {pol.get('pool_order', '?')}): all p==1 {ok_p}; "
                      f"sampled==pool top-M {ok_top}; sampled==cross-encoder top-M {np.mean(agree):.2f}")
        if not (ok_p and ok_top):
            fails.append(("F3", f"{cond}: deterministic arm did not log p=1 / top-M"))
        return

    # stochastic arm: pi_deploy(on-target helpful / poison) from the logged scores, deploy-stream TRAIN runs
    def _is_train(r):
        if r.get("phase"):
            return r["phase"] == "train"
        return bool((r.get("policy") or {}).get("explore_enabled", True))
    train = sorted([r for r in runs if not r.get("exploration") and _is_train(r)], key=lambda r: r.get("ts", ""))
    rows = []
    for k, r in enumerate(train):
        f = _fam_of(r["challenge_id"]); h, p_ = hid.get(f), pid.get(f)
        if h is None:
            continue
        tau = float(pol.get("tau_deploy", r.get("temperature", 1.0)) or 1.0)
        pis = marginal_inclusion_propensities(r["candidate_pool"], len(r["sampled_set"]), max(tau, 1e-3), n_mc=n_mc, seed=k)
        rows.append((k * 3 // max(len(train), 1), f, float(pis.get(h, 0.0)), float(pis.get(p_, 0.0)) if p_ else np.nan))
    st["n_train_deploy"] = len(rows)
    if rows:
        m_h, lo_h, hi_h = _cluster_boot([x[2] - st["chance"] for x in rows], [x[1] for x in rows])
        m_p, lo_p, hi_p = _cluster_boot([x[3] - st["chance"] for x in rows], [x[1] for x in rows], seed=1)
        st["pi_helpful_minus_chance"] = (m_h, lo_h, hi_h); st["pi_poison_minus_chance"] = (m_p, lo_p, hi_p)
        report.append(f"[{cond}] pi_deploy(on-target helpful) - chance: {m_h:+.3f} [{lo_h:+.3f},{hi_h:+.3f}]; "
                      f"poison: {m_p:+.3f} [{lo_p:+.3f},{hi_p:+.3f}]  (n={len(rows)} deploy-stream train runs)")
        terc = {t: [x for x in rows if x[0] == t] for t in range(3)}
        if terc[0] and terc[2]:
            fams = sorted({x[1] for x in rows}); rng = np.random.RandomState(2)
            def _c(sample):
                a = [x[2] for x in sample if x[0] == 2]; b = [x[2] for x in sample if x[0] == 0]
                return np.mean(a) - np.mean(b) if a and b else np.nan
            by = {f: [x for x in rows if x[1] == f] for f in fams}
            bs = [_c([x for f in rng.choice(fams, len(fams), replace=True) for x in by[f]]) for _ in range(1000)]
            bs = [b for b in bs if not np.isnan(b)]
            d = _c(rows); lo, hi = (np.percentile(bs, 2.5), np.percentile(bs, 97.5)) if bs else (np.nan, np.nan)
            st["learning"] = (float(d), float(lo), float(hi))
            report.append(f"[{cond}] learning: pi_deploy(helpful) last tercile - first: {d:+.3f} [{lo:+.3f},{hi:+.3f}] "
                          f"(terciles n={len(terc[0])}/{len(terc[1])}/{len(terc[2])})")
    # exploration-stream gates
    if pol.get("epsilon", 0) > 0:
        en = [bool(r.get("exploration")) for r in runs if (r.get("policy") or {}).get("explore_enabled")]
        e = np.mean(en) if en else 0.0
        if not (pol["epsilon"] * 0.4 <= e <= pol["epsilon"] * 1.8 + 0.05):
            fails.append(("F3", f"{cond}: exploration fraction {e:.2f} (explore-enabled runs) vs epsilon {pol['epsilon']}"))
        if any(r.get("exploration") for r in runs if not (r.get("policy") or {}).get("explore_enabled")):
            fails.append(("F3", f"{cond}: an eval (frozen) run was drawn from the exploration stream"))
        pex = [c.get("pi_explore") for r in runs if r.get("exploration") for c in r["candidate_pool"] if c.get("on_target")]
        if pex:
            report.append(f"[{cond}] exploration design: pi_explore(on-target) mean {np.mean(pex):.3f} (0.5 exact when k<=M)")


def check_f3_gates(report, fails):
    """Cross-condition gates (review 2026-09-05 F-2/F-25). Evaluated after every condition is read."""
    dep = GLOBAL.get("deployed")
    if not dep:
        return
    # mechanics: one consistent context key
    if dep.get("context_key") is None:
        fails.append(("F3-mechanics", f"deployed arm: bandit contexts are not consistent with one context key: "
                      f"{dict(dep['contexts'].most_common(8))}"))
    else:
        report.append(f"F3-mechanics: deployed arm context key = {dep['context_key']} ({len(dep['contexts'])} contexts)")
    legacy = GLOBAL.get("outcome") or GLOBAL.get("B2")
    if legacy and legacy.get("incl_deploy_helpful") is not None and dep.get("incl_deploy_helpful") is not None:
        report.append(f"F3 relative (informative): deployed on-target helpful inclusion {dep['incl_deploy_helpful']:.3f} "
                      f"vs tau=1 arm {legacy['incl_deploy_helpful']:.3f}")
    pr = dep.get("pi_helpful_minus_chance")
    if pr:
        report.append(f"F3-prior (informative): pi_deploy(on-target helpful) - chance = {pr[0]:+.3f} [{pr[1]:+.3f},{pr[2]:+.3f}]")
    lr = dep.get("learning")
    if dep.get("n_train_deploy", 0) >= MIN_TRAIN and lr:
        if not (lr[1] > 0):
            fails.append(("F3-learning", f"deployed arm: pi_deploy(on-target helpful) did not rise from the first to "
                          f"the last training tercile ({lr[0]:+.3f} [{lr[1]:+.3f},{lr[2]:+.3f}])"))
        else:
            report.append(f"F3-learning: PASS ({lr[0]:+.3f} [{lr[1]:+.3f},{lr[2]:+.3f}])")
    else:
        report.append(f"F3-learning: not gated ({dep.get('n_train_deploy', 0)} deploy-stream train runs < {MIN_TRAIN}; "
                      f"a smoke verifies mechanics, the mock ladder and Phase 2 verify learning)")


def check_f2(root: Path, cond, runs, truth, report, fails):
    cits = _load(root, cond, "citations")
    if not cits:
        return
    by_run = {r["run_id"]: r for r in runs}
    ev_tok = defaultdict(Counter); cited = defaultdict(lambda: [0, 0]); modes = Counter()
    for c in cits:
        r = by_run.get(c["run_id"])
        if r is None or c["node_id"] not in truth:
            continue
        lab, lf = truth[c["node_id"]]
        on = (lf == _fam_of(r["challenge_id"]))
        key = f"{lab}:{'on' if on else 'off'}"
        cited[key][1] += 1; modes[c.get("mode", "?")] += 1
        if c.get("cited"):
            cited[key][0] += 1
            ev_tok[key][c.get("match_evidence", "").split(" ~ ")[0].lower()] += 1
    report.append(f"[{cond}] citation modes: {dict(modes)}")
    for key in sorted(cited):
        a, n = cited[key]
        report.append(f"[{cond}] cite {key:22s} {a}/{n} = {a/n if n else 0:.2f}  evidence top: {ev_tok[key].most_common(4)}")
    outer_hits = sum(cnt for key in ev_tok for tok, cnt in ev_tok[key].items() if tok in OUTER)
    if modes.get("distinctive", 0) and outer_hits:
        fails.append(("F2", f"{cond}: {outer_hits} citations grounded on an outer-transport token under distinctive mode"))


def check_f9_pairing(root: Path, conds, report, fails):
    """The same (challenge, seed) eval items, with the same multiplicity, in every condition. Counted as a
    multiset of (name, seed) so both layouts work: replicate shards holding every condition, or one
    directory per arm (`shard_<arm>/`, v2 launch drivers)."""
    sets = {}
    for c in conds:
        items = Counter()
        for pf in list(root.glob("shard_*/progress.json")) + list(root.glob("progress.json")):
            p = json.loads(pf.read_text())
            for m in p["conditions"].get(c, {}).get("metrics", []):
                if m["phase"] == "eval":
                    items[(m["name"], m["seed"])] += 1
        sets[c] = items
    base = None
    for c, s in sets.items():
        if base is None:
            base = s
        elif s != base:
            fails.append(("F9", f"eval items differ between conditions ({c}: {sum(s.values())} vs {sum(base.values())})"))
    report.append(f"F9 pairing: eval items per condition {[(c, sum(s.values())) for c, s in sets.items()]}")


def wasted_steps(root: Path, cond, runs, report):
    steps = _load(root, cond, "steps")
    acts = defaultdict(Counter); fin = Counter()
    for s in steps:
        acts[s["run_id"]][s.get("action_type")] += 1
        if s.get("action_type") == "parse_error":
            fin[s.get("finish_reason") or "?"] += 1
    q = [acts[r["run_id"]].get("query_memory", 0) for r in runs]
    pe = [acts[r["run_id"]].get("parse_error", 0) for r in runs]
    report.append(f"[{cond}] wasted steps/run: query_memory {np.mean(q):.2f} (must be 0 after F4a), parse_error {np.mean(pe):.2f} "
                  f"(finish_reason of parse errors: {dict(fin)})")
    if np.mean(q) > 0:
        return ("F4a", f"{cond}: query_memory actions still occur ({np.mean(q):.2f}/run)")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--probe", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    report, fails = [f"# Smoke report - {root}"], []
    fams = set()
    for pf in list(root.glob("shard_*/progress.json")) + list(root.glob("progress.json")):
        p = json.loads(pf.read_text())
        for cp in p["conditions"].values():
            for m in cp.get("metrics", []):
                fams.add(m["family"])
    _, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=sorted(fams))
    hid = {f: n for n, (l, f) in truth.items() if l == "helpful"}
    pid = {f: n for n, (l, f) in truth.items() if l == "poison"}
    check_f1(root, report, fails)
    conds = _conditions(root)
    for cond in conds:
        runs = _load(root, cond, "runs")
        if not runs:
            continue
        check_f4_errors(runs, cond, report, fails)
        check_f3(runs, cond, truth, hid, pid, report, fails)
        check_f2(root, cond, runs, truth, report, fails)
        w = wasted_steps(root, cond, runs, report)
        if w:
            fails.append(w)
    check_f3_gates(report, fails)
    check_f9_pairing(root, conds, report, fails)
    if args.probe and Path(args.probe, "probe.json").exists():
        for r in json.loads(Path(args.probe, "probe.json").read_text()).get("families", []):
            report.append(f"probe {r.get('family'):20s} {r.get('solve')}")
    report.append(""); report.append("## Verdict")
    if fails:
        for tag, msg in fails:
            report.append(f"- FAIL {tag}: {msg}")
    else:
        report.append("- PASS: every blocking check passed")
    print("\n".join(report))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
