#!/usr/bin/env python3
"""experiments/phase2_estimands.py - the pre-specified PRIMARY estimands and the P3
compliance-adjusted effects on the EXPLORATION SLICE of a deployed arm.

On exploration runs the fixed stratified design includes each on-target lesson independently with probability
exactly 1/2 (logged `pi_explore`), so the on-target helpful (H) and poison (P) indicators are randomized,
balanced and independent of everything else: the estimands are cell contrasts.
  ITT(helpful)   = E[Y | H in] - E[Y | H out]              (marginal over P)
  ITT(poison)    = E[Y | P in] - E[Y | P out]              (marginal over H; floored when H is out)
  displacement   = E[Y | H in, P in] - E[Y | H in, P out]  (the identifying poison contrast)
  poison alone   = E[Y | P in, H out] - E[Y | H out, P out]
Per estimand (review 2026-09-08 F7): risk difference with the family-CLUSTER bootstrap CI (random-family target),
the FIXED-family (within-family) bootstrap CI, the DerSimonian-Laird random-effects pooled RD with I²,
Mantel-Haenszel RR and OR (family-stratified), SNIPW and AIPW on the logged design propensities (under the exact
1/2 design they coincide with the cell means up to the outcome model), and the exact-design randomization test
(within-family permutation of the indicator; F15: the observed statistic uses the same family set as the
permutations). P3: CACE = ITT / compliance, compliance = P(lesson cited | sampled) from citations.jsonl
(one-sided non-compliance: a lesson not in context cannot be executed from context), family-cluster bootstrap CI.
F3: the arms share the exploration draws (one rng per (epoch, item)), so P1 is reported ONCE on the primary arm;
the other arms are re-executions of the same draws and are tabulated as such with the fraction of identical
sampled sets and the outcome agreement. F12: S4 arms (k = 5 on-target lessons, π = 0.456, P-vs-malignant filler)
are a different design and are tabulated apart. F11: error-flagged runs count as failures (the campaign's rule);
`--exclude-errors` gives the sensitivity.
Usage: python -m experiments.phase2_estimands --root results/lab/phase2_14b --primary deployed_alphabet_tau01 --arms a,b,...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

import importlib as _il
hard = _il.import_module(os.environ.get("CTF_DOMAIN", "benchmark.hard_ctf"))
from retrieval.embedder import MockEmbedder

_fam_of = lambda cid: cid.rsplit("_", 1)[0]


def load(root, arm, kind="runs"):
    out = []
    for f in glob.glob(f"{root}/arms/{arm}/shard_00/logs/campaign_*/{kind}.jsonl"):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out


def mh(strata):
    """Mantel-Haenszel RR and OR over (a, n1, c, n0) strata."""
    nr = dr = no = do = 0.0
    for a, n1, c, n0 in strata:
        N = n1 + n0
        if N == 0 or n1 == 0 or n0 == 0:
            continue
        b, d = n1 - a, n0 - c
        nr += a * n0 / N; dr += c * n1 / N; no += a * d / N; do += b * c / N
    return (nr / dr if dr else float("nan")), (no / do if do else float("nan"))


def dl_pool(strata):
    """DerSimonian-Laird random-effects pooled risk difference over per-family (a, n1, c, n0) strata
    (continuity-corrected variances; also returns the fixed-effect pooled RD and I²)."""
    e, v = [], []
    for a, n1, c, n0 in strata:
        if n1 == 0 or n0 == 0:
            continue
        p1, p0 = (a + 0.5) / (n1 + 1), (c + 0.5) / (n0 + 1)
        e.append(a / n1 - c / n0); v.append(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    if not e:
        return None
    e, v = np.array(e), np.array(v); w = 1 / v; k = len(e)
    fe = float((w * e).sum() / w.sum()); Q = float((w * (e - fe) ** 2).sum())
    C = w.sum() - (w ** 2).sum() / w.sum(); tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    wr = 1 / (v + tau2); re = float((wr * e).sum() / wr.sum()); se = float(np.sqrt(1 / wr.sum()))
    return {"rd": re, "ci": [re - 1.96 * se, re + 1.96 * se], "tau2": float(tau2),
            "I2": float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0, "k": int(k), "fixed_rd": fe}


def _snipw(T, y, pi):
    wt = np.where(T == 1, 1 / pi, 0.0); wc = np.where(T == 0, 1 / (1 - pi), 0.0)
    if wt.sum() == 0 or wc.sum() == 0:
        return np.nan
    return float((wt * y).sum() / wt.sum() - (wc * y).sum() / wc.sum())


def _aipw(T, y, pi, fam):
    """AIPW with a saturated (family × arm) outcome model."""
    mu_t, mu_c = {}, {}
    gt, gc = (y[T == 1].mean() if (T == 1).any() else 0.0), (y[T == 0].mean() if (T == 0).any() else 0.0)
    for f in set(fam.tolist()):
        m = fam == f
        mu_t[f] = y[m & (T == 1)].mean() if (m & (T == 1)).any() else gt
        mu_c[f] = y[m & (T == 0)].mean() if (m & (T == 0)).any() else gc
    mt = np.array([mu_t[f] for f in fam]); mc = np.array([mu_c[f] for f in fam])
    psi = mt - mc + T * (y - mt) / pi - (1 - T) * (y - mc) / (1 - pi)
    return float(psi.mean())


def contrast(rows, cond_t, cond_c, pi_col, n_boot=4000, n_perm=4000, seed=0):
    """rows: list of (family, H, P, y, piH, piP). cond_*: predicate on (H, P); pi_col: 4 (H) or 5 (P)."""
    rng = np.random.RandomState(seed)
    sub = [r for r in rows if cond_t(r[1], r[2]) or cond_c(r[1], r[2])]
    if not sub:
        return None
    fam = np.array([r[0] for r in sub]); T = np.array([1 if cond_t(r[1], r[2]) else 0 for r in sub])
    y = np.array([r[3] for r in sub], dtype=float); pi = np.clip(np.array([r[pi_col] for r in sub], dtype=float), 0.02, 0.98)
    if T.sum() == 0 or T.sum() == len(T):
        return None
    fams = sorted(set(fam.tolist())); idx_by = {f: np.where(fam == f)[0] for f in fams}
    rd = float(y[T == 1].mean() - y[T == 0].mean())
    # family-cluster bootstrap (random-family target)
    bs_rd, bs_snipw, bs_aipw = [], [], []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[f] for f in rng.choice(fams, len(fams), replace=True)])
        Tb, yb, pb, fb = T[idx], y[idx], pi[idx], fam[idx]
        if Tb.sum() and Tb.sum() < len(Tb):
            bs_rd.append(yb[Tb == 1].mean() - yb[Tb == 0].mean()); bs_snipw.append(_snipw(Tb, yb, pb)); bs_aipw.append(_aipw(Tb, yb, pb, fb))
    pct = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))] if len(a) else [np.nan, np.nan]
    # fixed-family (within-family) bootstrap
    bs_fixed = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(idx_by[f], len(idx_by[f]), replace=True) for f in fams])
        Tb, yb = T[idx], y[idx]
        if Tb.sum() and Tb.sum() < len(Tb):
            bs_fixed.append(yb[Tb == 1].mean() - yb[Tb == 0].mean())
    strata = [(int(y[(fam == f) & (T == 1)].sum()), int(((fam == f) & (T == 1)).sum()),
               int(y[(fam == f) & (T == 0)].sum()), int(((fam == f) & (T == 0)).sum())) for f in fams]
    rr, or_ = mh(strata); dl = dl_pool(strata)
    # exact-design randomization test (F15: the same family set in the observed statistic and the permutations)
    ok = [f for f in fams if len(idx_by[f]) >= 2 and 0 < T[idx_by[f]].sum() < len(idx_by[f])]
    keep = np.isin(fam, ok); To, yo, fo = T[keep], y[keep], fam[keep]
    obs = float(yo[To == 1].mean() - yo[To == 0].mean()); cnt = 0
    perm_idx = {f: np.where(fo == f)[0] for f in ok}
    for _ in range(n_perm):
        Tp = To.copy()
        for f in ok:
            ii = perm_idx[f]; Tp[ii] = To[ii][rng.permutation(len(ii))]
        cnt += abs(yo[Tp == 1].mean() - yo[Tp == 0].mean()) >= abs(obs) - 1e-12
    return {"treated_rate": float(y[T == 1].mean()), "control_rate": float(y[T == 0].mean()),
            "n_treated": int(T.sum()), "n_control": int(len(T) - T.sum()), "rd": rd, "rd_ci": pct(bs_rd),
            "rd_ci_fixed_family": pct(bs_fixed), "dl": dl, "mh_rr": rr, "mh_or": or_,
            "snipw": _snipw(T, y, pi), "snipw_ci": pct(bs_snipw), "aipw": _aipw(T, y, pi, fam), "aipw_ci": pct(bs_aipw),
            "crt_p": float((cnt + 1) / (n_perm + 1)), "crt_stat": obs, "n_families": len(fams), "n_families_crt": len(ok)}


def cace(rows, cited, cond_t, cond_c, target_col, n_boot=4000, seed=0):
    """P3: ITT / compliance with a family-cluster bootstrap CI. rows carry the run id at index 6; cited[run_id] =
    set of cited node ids; target_col 7 (helpful node id) or 8 (poison node id)."""
    rng = np.random.RandomState(seed)
    sub = [r for r in rows if cond_t(r[1], r[2]) or cond_c(r[1], r[2])]
    if not sub:
        return None
    fam = np.array([r[0] for r in sub]); T = np.array([1 if cond_t(r[1], r[2]) else 0 for r in sub])
    y = np.array([r[3] for r in sub], dtype=float)
    comp = np.array([1.0 if (r[target_col] in cited.get(r[6], ())) else 0.0 for r in sub])
    fams = sorted(set(fam.tolist())); idx_by = {f: np.where(fam == f)[0] for f in fams}

    def est(Tb, yb, cb):
        if Tb.sum() == 0 or Tb.sum() == len(Tb):
            return np.nan, np.nan, np.nan
        itt = yb[Tb == 1].mean() - yb[Tb == 0].mean(); c1 = cb[Tb == 1].mean()
        return itt, c1, (itt / c1 if c1 > 0 else np.nan)
    itt, c1, ca = est(T, y, comp); bs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[f] for f in rng.choice(fams, len(fams), replace=True)])
        bs.append(est(T[idx], y[idx], comp[idx])[2])
    bs = np.array([b for b in bs if not np.isnan(b)])
    return {"itt": float(itt), "compliance": float(c1), "n_treated": int(T.sum()), "cace": float(ca),
            "cace_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) else [np.nan, np.nan]}


def rows_for(runs, hid, pid, exclude_errors=False):
    rows = []
    for r in runs:
        if not r.get("exploration"):
            continue
        if exclude_errors and r.get("error"):
            continue
        f = _fam_of(r["challenge_id"]); h, p = hid.get(f), pid.get(f)
        if h is None or p is None:
            continue
        S = set(r["sampled_set"]); pe = {c["node_id"]: c.get("pi_explore") for c in r["candidate_pool"]}
        rows.append((f, int(h in S), int(p in S), int(bool(r["outcome"])), float(pe.get(h) or 0.5), float(pe.get(p) or 0.5),
                     r["run_id"], h, p))
    return rows


def analyse(root, arm, hid, pid, exclude_errors=False, with_cace=True):
    runs = load(root, arm); rows = rows_for(runs, hid, pid, exclude_errors)
    expl = [r for r in runs if r.get("exploration")]
    pi = [c.get("pi_explore") for r in expl for c in r["candidate_pool"] if c.get("on_target")]
    out = {"arm": arm, "n_runs": len(runs), "n_exploration": len(expl), "n_rows": len(rows),
           "n_error_in_slice": sum(1 for r in expl if r.get("error")),
           "pi_explore_on_target_mean": float(np.mean(pi)) if pi else None,
           "cells": {k: {"rate": float(np.mean([r[3] for r in rows if (r[1], r[2]) == hp])) if any((r[1], r[2]) == hp for r in rows) else None,
                         "n": sum(1 for r in rows if (r[1], r[2]) == hp)} for k, hp in (("neither", (0, 0)), ("H_only", (1, 0)), ("P_only", (0, 1)), ("both", (1, 1)))},
           "ITT_helpful": contrast(rows, lambda H, P: H == 1, lambda H, P: H == 0, 4, seed=1),
           "ITT_poison": contrast(rows, lambda H, P: P == 1, lambda H, P: P == 0, 5, seed=2),
           "displacement": contrast(rows, lambda H, P: H == 1 and P == 1, lambda H, P: H == 1 and P == 0, 5, seed=3),
           "poison_alone": contrast(rows, lambda H, P: H == 0 and P == 1, lambda H, P: H == 0 and P == 0, 5, seed=4)}
    if with_cace:
        cits = load(root, arm, "citations"); cited = defaultdict(set)
        for c in cits:
            if c.get("cited"):
                cited[c["run_id"]].add(c["node_id"])
        out["P3"] = {"CACE_helpful": cace(rows, cited, lambda H, P: H == 1, lambda H, P: H == 0, 7, seed=11),
                     "CACE_poison": cace(rows, cited, lambda H, P: P == 1, lambda H, P: P == 0, 8, seed=12),
                     "CACE_displacement": cace(rows, cited, lambda H, P: H == 1 and P == 1, lambda H, P: H == 1 and P == 0, 8, seed=13)}
        # compliance over ALL train runs (deploy + exploration) and the lure's compliance (S0)
        train_ids = {r["run_id"]: r for r in runs if (r.get("policy") or {}).get("explore_enabled", True) and not (r.get("policy") or {}).get("deterministic")}
        samp = defaultdict(lambda: [0, 0])
        for c in cits:
            r = train_ids.get(c["run_id"])
            if r is None or not c.get("sampled", True):
                continue
            f = _fam_of(r["challenge_id"]); nid = c["node_id"]
            lab = "helpful_on" if nid == hid.get(f) else "poison_on" if nid == pid.get(f) else ("lure" if nid.startswith("distract::") else "off_target")
            samp[lab][0] += 1; samp[lab][1] += int(bool(c.get("cited")))
        out["P3"]["compliance_train_all"] = {k: {"sampled": v[0], "cited": v[1], "rate": v[1] / v[0] if v[0] else None} for k, v in samp.items()}
    return out


def shared_draws(root, primary, arm):
    """F3: fraction of the primary arm's exploration runs whose (challenge, occurrence) counterpart in `arm` drew
    the identical sampled set, and the outcome agreement (with Cohen's κ) on those runs."""
    def index(a):
        seen = defaultdict(int); d = {}
        for r in sorted(load(root, a), key=lambda r: r.get("ts", "")):
            pol = r.get("policy") or {}
            if not pol.get("explore_enabled", True) or pol.get("deterministic"):
                continue
            k = (r["challenge_id"], seen[r["challenge_id"]]); seen[r["challenge_id"]] += 1
            d[k] = (bool(r.get("exploration")), frozenset(r["sampled_set"]), int(bool(r["outcome"])))
        return d
    A, B = index(primary), index(arm)
    keys = [k for k, v in A.items() if v[0] and k in B]
    same_flag = sum(1 for k in keys if B[k][0]); same_set = [k for k in keys if B[k][0] and B[k][1] == A[k][1]]
    if not same_set:
        return {"n_expl": len(keys), "same_flag": same_flag, "same_set": 0}
    a = np.array([A[k][2] for k in same_set]); b = np.array([B[k][2] for k in same_set])
    po = float((a == b).mean()); pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return {"n_expl": len(keys), "same_flag": same_flag, "same_set": len(same_set), "outcome_agreement": po,
            "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan")}


def fmt_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci and not any(np.isnan(ci)) else "[-]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--primary", default="deployed_alphabet_tau01")
    ap.add_argument("--exclude-errors", action="store_true")
    ap.add_argument("--pool", default=None, help="comma list of arms with INDEPENDENT draws (P2-7) to pool as one stratified experiment")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    fams = open(f"{args.root}/families.txt").read().strip().split(",")
    _, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=fams)
    hid = {f: n for n, (l, f) in truth.items() if l == "helpful"}; pid = {f: n for n, (l, f) in truth.items() if l == "poison"}
    arms = args.arms.split(",")
    res = {a: analyse(args.root, a, hid, pid, args.exclude_errors, with_cace=True) for a in arms}
    if args.primary in res and not args.exclude_errors:
        res[args.primary + "__errors_excluded"] = analyse(args.root, args.primary, hid, pid, True, with_cace=False)
    shared = {a: shared_draws(args.root, args.primary, a) for a in arms if a != args.primary and a in res}
    P = res.get(args.primary)
    L = ["# Phase-2 primary estimands on the exploration slice - v2 (review 2026-09-08)", "",
         f"Root `{args.root}`; primary arm **{args.primary}**; error-flagged runs {'EXCLUDED' if args.exclude_errors else 'counted as failures (campaign rule; sensitivity below)'}.",
         "The exploration draws are shared across arms (one rng per (epoch, item) drives the ε draw and the design draw), so P1 is",
         "identified once; the other arms re-execute the same draws (table C) and are NOT independent replications.", ""]
    if P:
        c = P["cells"]
        L += [f"## A. Primary arm - {P['n_exploration']} exploration runs, π_explore(on-target) = {P['pi_explore_on_target_mean']:.3f}, "
              f"cells neither/H/P/both = {c['neither']['rate']:.3f} ({c['neither']['n']}) / {c['H_only']['rate']:.3f} ({c['H_only']['n']}) / "
              f"{c['P_only']['rate']:.3f} ({c['P_only']['n']}) / {c['both']['rate']:.3f} ({c['both']['n']}); {P['n_error_in_slice']} error-flagged runs in the slice", "",
              "| estimand | RD | cluster-family 95% CI | fixed-family 95% CI | DL random-effects RD [CI], I² | MH RR | MH OR | SNIPW [CI] | AIPW [CI] | CRT p |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k in ("ITT_helpful", "ITT_poison", "displacement", "poison_alone"):
            e = P[k]
            if not e:
                L.append(f"| {k} | - |"); continue
            dl = e["dl"]
            L.append(f"| {k} | {e['rd']:+.3f} | {fmt_ci(e['rd_ci'])} | {fmt_ci(e['rd_ci_fixed_family'])} | {dl['rd']:+.3f} {fmt_ci(dl['ci'])}, I²={dl['I2']:.2f} | "
                     f"{e['mh_rr']:.2f} | {e['mh_or']:.2f} | {e['snipw']:+.3f} {fmt_ci(e['snipw_ci'])} | {e['aipw']:+.3f} {fmt_ci(e['aipw_ci'])} | {e['crt_p']:.4f} |")
        p3 = P.get("P3", {})
        L += ["", "## B. P3 - compliance-adjusted effects (CACE = ITT / P(cited | sampled); one-sided non-compliance)", "",
              "| estimand | ITT | compliance (cited when in context) | CACE [cluster CI] |", "|---|---|---|---|"]
        for k in ("CACE_helpful", "CACE_poison", "CACE_displacement"):
            e = p3.get(k)
            if e:
                L.append(f"| {k} | {e['itt']:+.3f} | {e['compliance']:.3f} (n={e['n_treated']}) | {e['cace']:+.3f} {fmt_ci(e['cace_ci'])} |")
        ct = p3.get("compliance_train_all", {})
        L += ["", "Compliance over all train runs (deploy + exploration), P(cited | sampled): " +
              ", ".join(f"{k} {v['rate']:.3f} ({v['cited']}/{v['sampled']})" for k, v in sorted(ct.items()) if v['rate'] is not None)]
        Pe = res.get(args.primary + "__errors_excluded")
        if Pe:
            L += ["", "Sensitivity (F11), error-flagged runs excluded: " + "; ".join(
                f"{k} {Pe[k]['rd']:+.3f} {fmt_ci(Pe[k]['rd_ci'])}" for k in ("ITT_helpful", "ITT_poison", "displacement", "poison_alone") if Pe[k])]
    # table C: re-executions; table D: S4 arms
    def row(a, r, extra=""):
        f = lambda k: (f"{r[k]['rd']:+.3f} {fmt_ci(r[k]['rd_ci'])}, p={r[k]['crt_p']:.3f}" if r.get(k) else "-")
        return f"| {a} | {r['n_exploration']} | {f('ITT_helpful')} | {f('ITT_poison')} | {f('displacement')} | {f('poison_alone')} |{extra}"
    L += ["", "## C. Other sibling-poison arms - re-executions of the same design draws (not replications)", "",
          "| arm | expl. runs | ITT helpful RD [CI], p | ITT poison | displacement | poison alone | same sampled set as primary | outcome agreement (κ) |",
          "|---|---|---|---|---|---|---|---|"]
    s4 = []
    for a in arms:
        r = res[a]
        if a.startswith("s4_"):
            s4.append(a); continue
        if a == args.primary:
            continue
        sh = shared.get(a, {})
        extra = (f" {sh.get('same_set', 0)}/{sh.get('n_expl', 0)} | {sh.get('outcome_agreement', float('nan')):.3f} ({sh.get('kappa', float('nan')):.2f}) |")
        L.append(row(a, r, extra))
    if s4:
        L += ["", "## D. S4 malignant arms - a different design (k = 5 on-target lessons > M = 3: π = 0.456, H and P negatively dependent; "
              "'P out' leaves two slots to malignant lessons, so 'displacement' is P-versus-malignant-filler, not the sibling-poison estimand)", "",
              "| arm | expl. runs | ITT helpful | ITT poison | 'displacement' (vs malignant filler) | poison alone |", "|---|---|---|---|---|---|"]
        for a in s4:
            L.append(row(a, res[a]))
    pooled = None
    if args.pool:
        # P2-7: arms whose train draws are independent (draw_seed_offset) are pooled as ONE stratified experiment (family =
        # cluster; each run is a distinct solve). Compliance for the pooled CACE comes from every pooled arm's citations.
        parms = args.pool.split(","); rows = []; cited = defaultdict(set)
        for a in parms:
            rows += rows_for(load(args.root, a), hid, pid, args.exclude_errors)
            for c in load(args.root, a, "citations"):
                if c.get("cited"):
                    cited[c["run_id"]].add(c["node_id"])
        pooled = {"arms": parms, "n_rows": len(rows),
                  "ITT_helpful": contrast(rows, lambda H, P: H == 1, lambda H, P: H == 0, 4, seed=21),
                  "ITT_poison": contrast(rows, lambda H, P: P == 1, lambda H, P: P == 0, 5, seed=22),
                  "displacement": contrast(rows, lambda H, P: H == 1 and P == 1, lambda H, P: H == 1 and P == 0, 5, seed=23),
                  "poison_alone": contrast(rows, lambda H, P: H == 0 and P == 1, lambda H, P: H == 0 and P == 0, 5, seed=24),
                  "CACE_helpful": cace(rows, cited, lambda H, P: H == 1, lambda H, P: H == 0, 7, seed=31),
                  "CACE_poison": cace(rows, cited, lambda H, P: P == 1, lambda H, P: P == 0, 8, seed=32),
                  "CACE_displacement": cace(rows, cited, lambda H, P: H == 1 and P == 1, lambda H, P: H == 1 and P == 0, 8, seed=33)}
        L += ["", f"## E. Pooled over {len(parms)} arms with INDEPENDENT exploration draws ({', '.join(parms)}; {len(rows)} exploration runs)", "",
              "| estimand | RD | cluster-family CI | fixed-family CI | DL RD [CI], I² | MH RR | MH OR | CRT p | CACE [CI] (compliance) |", "|---|---|---|---|---|---|---|---|---|"]
        for k, ck in (("ITT_helpful", "CACE_helpful"), ("ITT_poison", "CACE_poison"), ("displacement", "CACE_displacement"), ("poison_alone", None)):
            e = pooled[k]; dl = e["dl"]; c = pooled.get(ck) if ck else None
            L.append(f"| {k} | {e['rd']:+.3f} | {fmt_ci(e['rd_ci'])} | {fmt_ci(e['rd_ci_fixed_family'])} | {dl['rd']:+.3f} {fmt_ci(dl['ci'])}, I²={dl['I2']:.2f} | {e['mh_rr']:.2f} | {e['mh_or']:.2f} | {e['crt_p']:.4f} | "
                     + (f"{c['cace']:+.3f} {fmt_ci(c['cace_ci'])} ({c['compliance']:.2f})" if c else "-") + " |")
    L += ["", "Multiplicity: the four context arms form one pre-specified ordering and the poison kinds a dose-response; the per-arm",
          "p-values in table C are re-executions of one experiment and are not to be read as independent tests."]
    out = args.out or f"{args.root}/ESTIMANDS_P1.md"
    Path(out).write_text("\n".join(L) + "\n"); Path(out).with_suffix(".json").write_text(json.dumps({"results": res, "shared_draws": shared, "pooled": pooled}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
