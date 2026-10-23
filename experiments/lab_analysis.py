#!/usr/bin/env python3
"""
experiments/lab_analysis.py - frozen, CI-bearing analysis for the lab-scale study.

Re-scores ALREADY-logged campaign runs (ZERO extra inference) to produce, with bootstrap
CIs + p-values on EVERY number, the pre-specified quantities:

  H1  outcome credit banks poison: learned Q by label, label-level CI over lessons.
  H2  the free experiment flips it: observational ATE sign-flip, BOTH (a) per-lesson CIs
      (resample over a lesson's runs - well-powered) and (b) label-level CI (resample over
      LESSONS - the generalization claim), under SNIPW and AIPW(DR).
  H3  propagation gap: promote-if-(value>0) rate, outcome-Q vs causal-ATE, paired over
      lessons, with CI.
  H4  solve-rate parity is structural: paired contrasts + a power/MDE analysis showing the
      minimum gap the study could have detected.
  the plan robustness ablations (each must leave the H2 sign-flip intact):
      propensity MC seeds {0..9} x n_mc {400,1000}; clip range; ESS frac; propensity source
      {confounded-Q, prior-only/Q=0}; estimator {SNIPW, AIPW}; promotion-theta sweep.
  A.6 ESS/positivity diagnostics table.

Emits results/<dir>/lab_analysis.json, results/<dir>/lab_analysis.md, paper/numbers_lab.tex.
This file is FROZEN before any headline (scaled) run - same code scores pilot and scaled
logs; only the corpus size changes. n=4 pilot label-level CIs are flagged anti-conservative.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from instrumentation.jsonl_logger import load_runs
from ope.causal_value import (
    observational_lesson_values, observational_lesson_values_aipw,
    summarize_by_label_ci, promotion_rate_ci, promotion_gap_ci,
)
from evaluation.stats import paired_bootstrap_ci, mcnemar_exact, power_analysis, _z
from retrieval.embedder import MockEmbedder
import os as _os, importlib as _il
hard = _il.import_module(_os.environ.get("CTF_DOMAIN", "benchmark.hard_ctf"))  # T2-7 domain switch

N_BOOT = 10000


# --------------------------------------------------------------------------- #
# inputs                                                                       #
# --------------------------------------------------------------------------- #

def _truth(families=None):
    _, truth = hard.build_seed_graph(MockEmbedder(dim=8, seed=1), families=families)
    return truth


def _label_map(truth):
    return {nid: t[0] for nid, t in truth.items()}


def _outcome_q_by_node(state_dir: Path, truth) -> dict:
    """Per-node MEAN learned Q across contexts (the value a write-back gate sees)."""
    bpath = state_dir / "bandit.json"
    if not bpath.exists():
        return {}
    cells = json.loads(bpath.read_text()).get("cells", {})
    per_node = defaultdict(list)
    for ctx, nodes in cells.items():
        for nid, cell in nodes.items():
            if nid in truth and cell["n"] > 0:
                per_node[nid].append(cell["Q"])
    return {nid: float(np.mean(qs)) for nid, qs in per_node.items()}


def _runs_prior_only(runs):
    """Counterfactual propensity source: set each candidate's sampling score to its
    similarity PRIOR (Q=0 / similarity-only), to test propensity-circularity (A.4)."""
    out = []
    for r in runs:
        r2 = copy.deepcopy(r)
        for c in r2.get("candidate_pool", []):
            c["score"] = float(c.get("prior", c.get("cosine", c.get("score", 0.0))))
        out.append(r2)
    return out


# --------------------------------------------------------------------------- #
# H1 / H2 / H3                                                                 #
# --------------------------------------------------------------------------- #

def _label_ci_block(per_lesson_vals):
    """Label-level mean ATE CI (resample over lessons) + per-lesson significance counts."""
    summ = summarize_by_label_ci(per_lesson_vals, n_boot=N_BOOT)
    sig = defaultdict(lambda: {"neg": 0, "pos": 0, "total": 0})
    for nid, d in per_lesson_vals.items():
        if d.get("ate") is None or not d.get("label"):
            continue
        s = sig[d["label"]]; s["total"] += 1
        if d.get("ate_excl0"):
            if d["ate"] < 0:
                s["neg"] += 1
            else:
                s["pos"] += 1
    return {"label_level": summ, "per_lesson_sig": dict(sig)}


def signflip(runs, truth, n_boot_lesson=2000, mc_seed=0, n_mc=400, clip=0.05, ess_frac=0.3,
             response_key="R_final"):
    """H2: SNIPW + AIPW per-lesson values with CIs, label-level CIs, and the verdict
    (poison CI upper < 0 < helpful CI lower, non-overlapping). response_key selects the
    estimand: "outcome" (binary solve - budget/step-rescaling INVARIANT, the headline) or
    "R_final" (full reward, secondary)."""
    snipw = observational_lesson_values(runs, truth=truth, n_boot=n_boot_lesson,
                                        mc_seed=mc_seed, n_mc=n_mc, clip=clip, ess_frac=ess_frac,
                                        response_key=response_key)
    aipw = observational_lesson_values_aipw(runs, truth=truth, n_boot=n_boot_lesson,
                                            mc_seed=mc_seed, n_mc=n_mc, clip=clip, ess_frac=ess_frac,
                                            response_key=response_key)
    out = {"snipw": _label_ci_block(snipw), "aipw": _label_ci_block(aipw),
           "per_lesson_snipw": {k: {kk: v[kk] for kk in
                ("ate", "ate_ci_lo", "ate_ci_hi", "ate_p", "ate_excl0", "label",
                 "n_treat", "n_control", "ess_treat", "ess_control")}
                for k, v in snipw.items()}}

    def verdict(block):
        ll = block["label_level"]
        if "poison" not in ll or "helpful" not in ll:
            return None
        p, h = ll["poison"], ll["helpful"]
        return {"poison_ci": [p["ci_lo"], p["ci_hi"]], "helpful_ci": [h["ci_lo"], h["ci_hi"]],
                "poison_neg": p["ci_hi"] < 0, "helpful_pos": h["ci_lo"] > 0,
                "non_overlapping": p["ci_hi"] < h["ci_lo"],
                "signflip": (p["ci_hi"] < 0 < h["ci_lo"])}
    out["verdict_snipw"] = verdict(out["snipw"])
    out["verdict_aipw"] = verdict(out["aipw"])
    out["_snipw_values"] = snipw  # for downstream promotion/ESS (not serialized raw)
    return out


def promotion(qval, ateval, label_map, theta=0.0):
    """H3: promote-if-(value>theta). Outcome-Q vs causal-ATE, paired over lessons."""
    q_only = {k: v for k, v in qval.items() if v is not None}
    a_only = {k: v for k, v in ateval.items() if v is not None}
    return {
        "outcome_Q": promotion_rate_ci(q_only, label_map, theta=theta, n_boot=N_BOOT),
        "causal_ATE": promotion_rate_ci(a_only, label_map, theta=theta, n_boot=N_BOOT),
        "gap_Q_minus_ATE": promotion_gap_ci(q_only, a_only, label_map, theta=theta, n_boot=N_BOOT),
    }


# --------------------------------------------------------------------------- #
# H4 - power / MDE behind solve-rate parity                                    #
# --------------------------------------------------------------------------- #

def _eval_vec(cond_prog):
    return {(m["name"], m["seed"]): int(m["solved"])
            for m in cond_prog.get("metrics", []) if m.get("phase") == "eval"}


def power_mde(prog, alpha=0.05, power=0.80):
    conds = prog.get("conditions", {})
    vecs = {n: _eval_vec(cp) for n, cp in conds.items()}
    out = {"contrasts": {}, "mde": {}}
    pairs = [("outcome", "B2"), ("outcome", "B0"), ("observational", "outcome"), ("B2", "B0")]
    sds = []
    n_common = None
    for a, b in pairs:
        if a not in vecs or b not in vecs:
            continue
        keys = sorted(set(vecs[a]) & set(vecs[b]))
        if not keys:
            continue
        va = [vecs[a][k] for k in keys]; vb = [vecs[b][k] for k in keys]
        pr = paired_bootstrap_ci(va, vb, n_boot=N_BOOT)
        mc = mcnemar_exact(va, vb)
        diffs = np.array(va) - np.array(vb)
        out["contrasts"][f"{a}_vs_{b}"] = {
            "n": len(keys), "diff": pr.mean_diff, "ci_lo": pr.ci_low, "ci_hi": pr.ci_high,
            "boot_p": pr.p_value_sign, "mcnemar_p": mc["p_value"], "significant": pr.significant}
        sds.append(float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0)
        n_common = len(keys)
    # MDE: smallest paired mean difference detectable at (alpha,power) for the observed n.
    if sds and n_common:
        sd = float(np.mean(sds))
        mde = (_z(1 - alpha / 2) + _z(power)) * sd / np.sqrt(n_common)
        out["mde"] = {"observed_paired_sd": sd, "n_pairs": n_common,
                      "mde_at_n": float(mde), "power": power, "alpha": alpha,
                      "n_for_5pp": power_analysis(0.4, 0.05, paired_sd=sd)["required_challenges"],
                      "n_for_10pp": power_analysis(0.4, 0.10, paired_sd=sd)["required_challenges"]}
    return out


# --------------------------------------------------------------------------- #
# robustness ablations                                                         #
# --------------------------------------------------------------------------- #

def ablations(runs, truth):
    lab = _label_map(truth)
    res = {}

    def label_means(vals):
        s = summarize_by_label_ci(vals, n_boot=2000)
        return {k: s[k]["mean"] for k in s}, {k: (s[k]["ci_lo"], s[k]["ci_hi"]) for k in s}

    # 1. propensity MC seeds x n_mc - sign stability of the LABEL-level poison/true ATE
    seed_grid = {}
    for n_mc in (400, 1000):
        for sd in range(10):
            v = observational_lesson_values(runs, truth=truth, mc_seed=sd, n_mc=n_mc)
            m, _ = label_means(v)
            seed_grid[f"nmc{n_mc}_seed{sd}"] = {k: m.get(k) for k in ("helpful", "poison", "distractor")}
    poison_signs = [x["poison"] < 0 for x in seed_grid.values() if x.get("poison") is not None]
    true_signs = [x["helpful"] > 0 for x in seed_grid.values() if x.get("helpful") is not None]
    res["mc_seed"] = {"grid": seed_grid,
                      "poison_neg_fraction": float(np.mean(poison_signs)) if poison_signs else None,
                      "true_pos_fraction": float(np.mean(true_signs)) if true_signs else None}

    # 2. clip range
    res["clip"] = {}
    for clip in (0.02, 0.05, 0.1):
        v = observational_lesson_values(runs, truth=truth, clip=clip)
        m, ci = label_means(v)
        res["clip"][str(clip)] = {"poison": m.get("poison"), "helpful": m.get("helpful")}

    # 3. ESS frac
    res["ess_frac"] = {}
    for ef in (0.2, 0.3, 0.5):
        v = observational_lesson_values(runs, truth=truth, ess_frac=ef)
        m, _ = label_means(v)
        res["ess_frac"][str(ef)] = {"poison": m.get("poison"), "helpful": m.get("helpful"),
                                    "n_poison": sum(1 for d in v.values()
                                                    if d["label"] == "poison" and d["ate"] is not None)}

    # 4. propensity source: confounded-Q (default) vs prior-only / Q=0
    res["propensity_source"] = {}
    for tag, rr in (("confounded_Q", runs), ("prior_only_Q0", _runs_prior_only(runs))):
        v = observational_lesson_values(rr, truth=truth)
        m, ci = label_means(v)
        res["propensity_source"][tag] = {"poison": m.get("poison"), "poison_ci": ci.get("poison"),
                                         "helpful": m.get("helpful"), "helpful_ci": ci.get("helpful")}

    # 5. estimator: SNIPW vs AIPW
    res["estimator"] = {}
    for tag, fn in (("snipw", observational_lesson_values),
                    ("aipw", observational_lesson_values_aipw)):
        v = fn(runs, truth=truth)
        m, ci = label_means(v)
        res["estimator"][tag] = {"poison": m.get("poison"), "poison_ci": ci.get("poison"),
                                 "helpful": m.get("helpful"), "helpful_ci": ci.get("helpful")}
    return res


def theta_sweep(qval, ateval, label_map, lo=-0.2, hi=0.2, step=0.02):
    """A.12c: does a promotion threshold theta exist that admits true but not poison?
    Show outcome-Q has none (poison & true promote together); causal-ATE has one."""
    thetas = list(np.round(np.arange(lo, hi + 1e-9, step), 3))

    def rates(value_by_node):
        by = defaultdict(list)
        for nid, v in value_by_node.items():
            if v is not None and nid in label_map:
                by[label_map[nid]].append(v)
        out = {}
        for th in thetas:
            out[float(th)] = {labk: float(np.mean([x > th for x in xs]))
                              for labk, xs in by.items()}
        return out

    def separating(rate_table):
        # exists theta with promote(helpful) >= 0.75 and promote(poison) <= 0.25
        for th, r in rate_table.items():
            if r.get("helpful", 0) >= 0.75 and r.get("poison", 1) <= 0.25:
                return th
        return None

    rq = rates({k: v for k, v in qval.items() if v is not None})
    ra = rates({k: v for k, v in ateval.items() if v is not None})
    return {"thetas": [float(t) for t in thetas], "outcome_Q_rates": rq, "causal_ATE_rates": ra,
            "outcome_Q_separating_theta": separating(rq),
            "causal_ATE_separating_theta": separating(ra)}


def ess_table(per_lesson_snipw):
    """A.6 ESS / positivity diagnostics by label."""
    by = defaultdict(lambda: {"ess_t": [], "ess_c": [], "rejected": 0, "total": 0})
    for nid, d in per_lesson_snipw.items():
        if not d.get("label"):
            continue
        b = by[d["label"]]; b["total"] += 1
        if d["ate"] is None:
            b["rejected"] += 1
        else:
            b["ess_t"].append(d.get("ess_treat", 0.0)); b["ess_c"].append(d.get("ess_control", 0.0))
    out = {}
    for labk, b in by.items():
        out[labk] = {"mean_ess_treat": float(np.mean(b["ess_t"])) if b["ess_t"] else None,
                     "mean_ess_control": float(np.mean(b["ess_c"])) if b["ess_c"] else None,
                     "min_ess_treat": float(np.min(b["ess_t"])) if b["ess_t"] else None,
                     "rejected": b["rejected"], "total": b["total"],
                     "rejection_rate": (b["rejected"] / b["total"]) if b["total"] else None}
    # DIFFERENTIAL-positivity check (OPE risk): if helpful & poison are rejected at very
    # different rates, the label-level mean is biased by selection, not signal.
    rr = {k: v["rejection_rate"] for k, v in out.items() if v["rejection_rate"] is not None}
    if "helpful" in rr and "poison" in rr:
        out["_differential_rejection"] = {"helpful": rr["helpful"], "poison": rr["poison"],
                                          "gap": abs(rr["helpful"] - rr["poison"]),
                                          "flag": abs(rr["helpful"] - rr["poison"]) > 0.1}
    return out


# --------------------------------------------------------------------------- #
# driver + emit                                                                #
# --------------------------------------------------------------------------- #

def _fmt_ci(d, scale=1.0, pct=False):
    if d is None:
        return "--"
    u = "\\%" if pct else ""
    s = 100 if pct else scale
    return f"{s*d['mean'] if 'mean' in d else s*d.get('rate', 0):+.3f}{u} CI[{s*d['ci_lo']:+.3f}, {s*d['ci_hi']:+.3f}]"


def run(dir_, headline_cond="outcome"):
    root = Path(dir_)
    summ = json.loads((root / "summary.json").read_text()) if (root / "summary.json").exists() else {}
    truth = _truth(summ.get("families")); lab = _label_map(truth)
    prog = json.loads((root / "progress.json").read_text()) if (root / "progress.json").exists() else {}
    n_lessons_per_label = defaultdict(int)
    for nid, (l, _) in truth.items():
        n_lessons_per_label[l] += 1

    out = {"meta": {"dir": str(root), "headline_cond": headline_cond,
                    "n_lessons_per_label": dict(n_lessons_per_label),
                    "anti_conservative_warning": (min(n_lessons_per_label.values()) < 20)}}

    runs = load_runs(f"campaign_{headline_cond}", root=str(root / "logs"))
    out["meta"]["n_runs_headline"] = len(runs)

    # Report BOTH estimands (which is "headline" depends on the run regime, per the design
    # panel + the plan): full reward R=o(1+λe) is valid for the FIXED-budget pilot (the harm
    # is the step-efficiency penalty - latent, surfaces via write-back); binary SOLVE is the
    # budget/step-rescaling-INVARIANT headline for the GRADED scaled run (where poison teeth
    # show on solve itself). Promotion/ESS/H1 use the reward estimand (pilot anchor).
    sf_reward = signflip(runs, truth, response_key="R_final")
    snipw_vals = sf_reward.pop("_snipw_values")
    sf_solve = signflip(runs, truth, response_key="outcome")
    sf_solve.pop("_snipw_values", None)
    out["H2_signflip"] = sf_reward
    out["H2_signflip_solve"] = sf_solve
    out["meta"]["estimands"] = {
        "H2_signflip": "R=o(1+lambda*e) full reward - fixed-budget pilot headline",
        "H2_signflip_solve": "binary solve - budget-invariant; headline for the graded scaled run",
    }

    # H1 = the OUTCOME bandit's learned Q (the deployed write-back signal). lab_merge
    # namespaces each shard's context ids so per-node Q POOLS across replicate shards here.
    qval = _outcome_q_by_node(root / "state" / headline_cond, truth)
    # cross-check: the log-derived outcome-credit advantage (E[R|cited&sampled] - E[R]).
    from ope.causal_value import outcome_credit_values
    cit_path = root / "logs" / f"campaign_{headline_cond}" / "citations.jsonl"
    citations = ([json.loads(l) for l in open(cit_path) if l.strip()]
                 if cit_path.exists() else None)
    ocv = outcome_credit_values(runs, truth=truth, citations=citations)
    global_R = float(np.mean([float(r.get("R_final", 0.0)) for r in runs])) if runs else 0.0
    out["H1_outcome_credit_logcrosscheck"] = {
        nid: {"adv": d["value"] - global_R, "label": d["label"], "n": d["n"]}
        for nid, d in ocv.items()}
    ateval = {k: d["ate"] for k, d in snipw_vals.items()}
    # F8 (FIX_CATALOG 2026-09): `ateval` above is the MARGINAL, R_final SNIPW (pooled over all
    # families' runs). The pre-specified primary estimand is the ON-TARGET binary-outcome
    # SNIPW; compute it here too so node_values carries BOTH, explicitly named.
    _fam_of = lambda cid: cid.rsplit("_", 1)[0]
    _by_fam = defaultdict(list)
    for r in runs:
        _by_fam[_fam_of(r["challenge_id"])].append(r)
    ontarget_o = {}
    for f, fr in _by_fam.items():
        v = observational_lesson_values(fr, truth=truth, response_key="outcome", n_boot=0)
        for nid, d in v.items():
            if truth.get(nid, ("?", "?"))[1] == f and d.get("ate") is not None:
                ontarget_o[nid] = d["ate"]
    # H1: outcome Q by label (label-level CI over lessons)
    out["H1_outcome_Q"] = {
        lab_: summarize_by_label_ci(
            {k: {"ate": v, "label": lab.get(k)} for k, v in qval.items()}, n_boot=N_BOOT).get(lab_)
        for lab_ in ("helpful", "poison", "distractor")}
    out["H3_promotion"] = promotion(qval, ateval, lab, theta=0.0)
    out["H3_promotion"]["_value_column"] = "causal_ATE_marginal_R (legacy)"
    out["H3_promotion_ontarget_o"] = promotion(qval, ontarget_o, lab, theta=0.0)
    out["H3_promotion_ontarget_o"]["_value_column"] = "causal_ATE_ontarget_o (pre-specified primary)"
    # per-node values for the H5 transfer experiment (promotion decisions). F8: three named
    # columns; `causal_ATE` is kept == the legacy marginal-R value for backward compatibility.
    node_values = {nid: {"label": lab.get(nid), "family": truth.get(nid, ("?", "?"))[1],
                         "outcome_Q": qval.get(nid),
                         "causal_ATE": ateval.get(nid),
                         "causal_ATE_marginal_R": ateval.get(nid),
                         "causal_ATE_ontarget_o": ontarget_o.get(nid)}
                   for nid in set(qval) | set(ateval) | set(ontarget_o)}
    (root / "node_values.json").write_text(json.dumps(node_values, indent=2, default=str))
    out["H4_power"] = power_mde(prog)
    out["ablations"] = ablations(runs, truth)
    out["theta_sweep"] = theta_sweep(qval, ateval, lab)
    out["ess_positivity"] = ess_table(snipw_vals)

    (root / "lab_analysis.json").write_text(json.dumps(out, indent=2, default=str))
    _emit_md(out, root / "lab_analysis.md")
    _emit_tex(out, Path("paper_final/numbers_lab.tex"))
    _print(out)
    return out


def _print(out):
    m = out["meta"]
    print("=" * 78)
    print(f"LAB ANALYSIS - {m['dir']}  (headline cond={m['headline_cond']}, "
          f"n_runs={m['n_runs_headline']}, lessons/label={m['n_lessons_per_label']})")
    if m["anti_conservative_warning"]:
        print("  ⚠ <20 lessons/label: label-level bootstrap CIs are ANTI-CONSERVATIVE "
              "(pilot). Per-lesson CIs are the trustworthy current evidence.")
    print("\nH2 SIGN-FLIP (observational ATE; label-level CI resamples over LESSONS):")
    for est in ("snipw", "aipw"):
        ll = out["H2_signflip"][est]["label_level"]; v = out["H2_signflip"][f"verdict_{est}"]
        print(f"  [{est.upper()}]")
        for labk in ("helpful", "poison", "distractor"):
            if labk in ll and ll[labk]:
                d = ll[labk]
                print(f"    {labk:10s} mean={d['mean']:+.3f} CI[{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}] "
                      f"p={d['p']:.3f} excl0={d['excl0']} (n={d['n']})")
        if v:
            print(f"    -> sign-flip={v['signflip']} (poison_neg={v['poison_neg']}, "
                  f"helpful_pos={v['helpful_pos']}, non_overlap={v['non_overlapping']})")
        sig = out["H2_signflip"][est]["per_lesson_sig"]
        for labk in ("helpful", "poison"):
            if labk in sig:
                s = sig[labk]
                print(f"    per-lesson sig [{labk}]: {s['neg']} neg / {s['pos']} pos of {s['total']} (CI excl 0)")
    sv = out.get("H2_signflip_solve", {}).get("snipw", {}).get("label_level", {})
    if sv:
        print("  [binary-SOLVE estimand - budget-invariant; the GRADED scaled-run headline]")
        for labk in ("helpful", "poison", "distractor"):
            d = sv.get(labk)
            if d:
                print(f"    {labk:10s} mean={d['mean']:+.3f} CI[{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}] "
                      f"p={d['p']:.3f} excl0={d['excl0']}  (pilot: ~0 = harm is latent in efficiency, not solve)")

    print("\nH1 OUTCOME LEARNED Q (label-level CI over lessons):")
    for labk in ("helpful", "poison", "distractor"):
        d = out["H1_outcome_Q"].get(labk)
        if d:
            print(f"    {labk:10s} mean={d['mean']:+.3f} CI[{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}] excl0={d['excl0']}")
    print("\nH3 PROMOTION (promote-if value>0; paired gap over lessons):")
    g = out["H3_promotion"]["gap_Q_minus_ATE"]
    for labk in ("helpful", "poison"):
        if labk in g:
            d = g[labk]
            print(f"    {labk:10s} Q={d['rate_a']:.2f} ATE={d['rate_b']:.2f} gap={d['gap']:+.2f} "
                  f"CI[{d['ci_lo']:+.2f},{d['ci_hi']:+.2f}] p={d['p']:.3f}")
    print("\nH4 POWER / MDE (solve-rate parity):")
    mde = out["H4_power"].get("mde", {})
    if mde:
        print(f"    observed paired SD={mde['observed_paired_sd']:.3f}, n_pairs={mde['n_pairs']} "
              f"-> MDE={100*mde['mde_at_n']:.1f}pp at {int(100*mde['power'])}% power; "
              f"n for 5pp={mde['n_for_5pp']}, 10pp={mde['n_for_10pp']}")
    print("\nABLATIONS (H2 sign-flip must survive):")
    mc = out["ablations"]["mc_seed"]
    print(f"    propensity MC seeds x n_mc: poison ATE <0 in {mc['poison_neg_fraction']:.0%} of cells, "
          f"true >0 in {mc['true_pos_fraction']:.0%}")
    for k in ("clip", "ess_frac", "propensity_source", "estimator"):
        print(f"    {k}:")
        for kk, vv in out["ablations"][k].items():
            ph = vv.get("poison"); hh = vv.get("helpful")
            print(f"      {kk:16s} poison={ph if ph is None else round(ph,3)}  helpful={hh if hh is None else round(hh,3)}")
    ts = out["theta_sweep"]
    print(f"\nTHETA SWEEP: outcome-Q separating theta={ts['outcome_Q_separating_theta']}  "
          f"causal-ATE separating theta={ts['causal_ATE_separating_theta']}")
    print("\nESS / POSITIVITY:")
    for labk, d in out["ess_positivity"].items():
        if labk.startswith("_"):
            continue
        print(f"    {labk:10s} mean ESS treat={d['mean_ess_treat']}, control={d['mean_ess_control']}, "
              f"rejected={d['rejected']}/{d['total']} (rate={d['rejection_rate']})")
    dr = out["ess_positivity"].get("_differential_rejection")
    if dr:
        print(f"    DIFFERENTIAL rejection helpful={dr['helpful']:.2f} vs poison={dr['poison']:.2f} "
              f"gap={dr['gap']:.2f} flag={dr['flag']}  (>0.1 => label-mean may be selection-biased)")


def _emit_md(out, path):
    m = out["meta"]; L = []
    L.append(f"# Lab analysis - `{m['dir']}`\n")
    L.append(f"headline condition **{m['headline_cond']}**, {m['n_runs_headline']} runs, "
             f"lessons/label = {m['n_lessons_per_label']}.\n")
    if m["anti_conservative_warning"]:
        L.append("> ⚠ **<20 lessons/label** → label-level bootstrap CIs are anti-conservative "
                 "(pilot scale). Per-lesson CIs are the trustworthy current evidence; the "
                 "label-level generalization claim awaits the scaled corpus.\n")
    L.append("## H2 - observational ATE sign-flip\n")
    L.append("| estimator | label | mean ATE | 95% CI | p | excl 0 |")
    L.append("|---|---|---|---|---|---|")
    for est in ("snipw", "aipw"):
        ll = out["H2_signflip"][est]["label_level"]
        for labk in ("helpful", "poison", "distractor"):
            d = ll.get(labk)
            if d:
                L.append(f"| {est.upper()} | {labk} | {d['mean']:+.3f} | "
                         f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] | {d['p']:.3f} | {d['excl0']} |")
    for est in ("snipw", "aipw"):
        v = out["H2_signflip"][f"verdict_{est}"]
        if v:
            L.append(f"\n**{est.upper()} verdict:** sign-flip = `{v['signflip']}` "
                     f"(poison CI<0: {v['poison_neg']}, helpful CI>0: {v['helpful_pos']}, "
                     f"non-overlapping: {v['non_overlapping']}).")
    L.append("\n## H3 - promotion gap (promote-if value>0)\n")
    g = out["H3_promotion"]["gap_Q_minus_ATE"]
    L.append("| label | outcome-Q rate | causal-ATE rate | gap | 95% CI | p |")
    L.append("|---|---|---|---|---|---|")
    for labk in ("helpful", "poison"):
        d = g.get(labk)
        if d:
            L.append(f"| {labk} | {d['rate_a']:.2f} | {d['rate_b']:.2f} | {d['gap']:+.2f} | "
                     f"[{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}] | {d['p']:.3f} |")
    mde = out["H4_power"].get("mde", {})
    if mde:
        L.append("\n## H4 - power / MDE\n")
        L.append(f"observed paired SD = {mde['observed_paired_sd']:.3f}, n_pairs = {mde['n_pairs']} "
                 f"→ **MDE = {100*mde['mde_at_n']:.1f} pp** at {int(100*mde['power'])}% power. "
                 f"n for 5pp = {mde['n_for_5pp']}, for 10pp = {mde['n_for_10pp']}.\n")
    mc = out["ablations"]["mc_seed"]
    L.append("## Robustness (H2 sign-flip must survive)\n")
    L.append(f"- propensity MC seeds×n_mc: poison ATE<0 in **{mc['poison_neg_fraction']:.0%}**, "
             f"true>0 in **{mc['true_pos_fraction']:.0%}** of {len(mc['grid'])} cells.")
    for k in ("clip", "ess_frac", "propensity_source", "estimator"):
        items = ", ".join(f"{kk}: poison={vv.get('poison')}, helpful={vv.get('helpful')}"
                          for kk, vv in out["ablations"][k].items())
        L.append(f"- {k}: {items}")
    ts = out["theta_sweep"]
    L.append(f"- θ-sweep: outcome-Q separating θ = `{ts['outcome_Q_separating_theta']}`, "
             f"causal-ATE separating θ = `{ts['causal_ATE_separating_theta']}`.")
    path.write_text("\n".join(L) + "\n")


def _emit_tex(out, path):
    if not path.parent.exists():
        return
    L = ["% Auto-generated by experiments/lab_analysis.py - do not edit by hand.", ""]

    def mac(n, v):
        L.append(f"\\providecommand{{\\{n}}}{{}}\\renewcommand{{\\{n}}}{{{v}}}")

    for est in ("snipw", "aipw"):
        ll = out["H2_signflip"][est]["label_level"]
        pre = "Snipw" if est == "snipw" else "Aipw"
        for labk, suff in (("helpful", "True"), ("poison", "Poison")):
            d = ll.get(labk)
            if d:
                mac(f"Lab{pre}{suff}", f"{d['mean']:+.3f}")
                mac(f"Lab{pre}{suff}CI", f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]")
    g = out["H3_promotion"]["gap_Q_minus_ATE"].get("poison")
    if g:
        mac("LabPromoGapPoison", f"{100*g['gap']:+.0f}\\,pp")
        mac("LabPromoGapPoisonCI", f"[{100*g['ci_lo']:+.0f}, {100*g['ci_hi']:+.0f}]")
    mde = out["H4_power"].get("mde", {})
    if mde:
        mac("LabMDE", f"{100*mde['mde_at_n']:.1f}\\,pp")
        mac("LabNforFivePP", str(mde["n_for_5pp"]))
    mc = out["ablations"]["mc_seed"]
    if mc["poison_neg_fraction"] is not None:
        mac("LabPoisonNegFrac", f"{100*mc['poison_neg_fraction']:.0f}\\%")
    mac("LabNlessonsPoison", str(out["meta"]["n_lessons_per_label"].get("poison", "--")))
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/campaign_v2")
    ap.add_argument("--cond", default="outcome", help="headline condition logs to re-score")
    args = ap.parse_args()
    run(args.dir, headline_cond=args.cond)


if __name__ == "__main__":
    main()
