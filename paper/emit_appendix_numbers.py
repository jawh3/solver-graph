#!/usr/bin/env python3
"""paper_final/emit_appendix_numbers.py — emit numbers_latent.tex for Appendix C.

Every macro is recomputed here from the committed artifacts under extra_work/results/;
none is transcribed by hand. Three deliberate differences from the source project's own analyzer
(extra_work/analyze_g2.py):

  1. The paired bootstrap resamples FAMILIES (clusters), not challenges. The 52 probe
     challenges are 13 families x 4 instances; instances within a family share a
     transform chain and are not independent.
  2. Confidence intervals are reported for the commands channel as well as rounds.
     Clustering widens five of the six marginal intervals; the detour commands interval
     is marginally narrower. It is not uniformly conservative and is not claimed to be.
  3. Censoring diagnostics and the conditional (both-solved / outcome-concordant)
     estimands are emitted, because the marginal round contrast is confounded with solve
     status under this design's round cap and must not be read as a route cost.

Run:  /path/to/python paper_final/emit_appendix_numbers.py
"""
import json, glob, math, os, re
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
EW = os.environ.get("APPENDIX_C_ARTIFACTS") or next(
    (p for p in (os.path.join(ROOT, "extra_work", "results"),
                 os.path.join(_HERE, "..", "appendix_c", "results"),
                 os.path.join(_HERE, "..", "appendix_c_artifacts"),
                 os.path.join(_HERE, "appendix_c_artifacts"))
     if os.path.isdir(p)), os.path.join(ROOT, "extra_work", "results"))
SEED = 20260821
B = 20000


def load_rows():
    rows = []
    for f in sorted(glob.glob(os.path.join(EW, "g2_step_probe", "rows_shard*.jsonl"))):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    if len(rows) != 260:
        raise SystemExit(f"FATAL: expected 260 probe rows under {EW}/g2_step_probe, "
                         f"got {len(rows)}. Emitting from a partial shard set would "
                         f"silently change every macro below.")
    return rows


def clustered_boot(pairs, fams, rng, b=B):
    """pairs: array of per-challenge paired deltas; fams: parallel family labels."""
    by_fam = defaultdict(list)
    for d, f in zip(pairs, fams):
        by_fam[f].append(d)
    keys = sorted(by_fam)
    arrs = [np.asarray(by_fam[k], dtype=float) for k in keys]
    n = len(keys)
    out = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        out[i] = np.concatenate([arrs[j] for j in idx]).mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def mcnemar_exact(b01, b10):
    n = b01 + b10
    if n == 0:
        return 1.0
    return float(binomtest(b01, n, 0.5).pvalue)


def main():
    rng = np.random.default_rng(SEED)
    rows = load_rows()
    # index: (challenge, condition) -> row
    idx = {(r["challenge"], r["condition"]): r for r in rows}
    conds = sorted({r["condition"] for r in rows})
    chals = sorted({r["challenge"] for r in rows})
    fam_of = {r["challenge"]: r["family"] for r in rows}
    fams = sorted(set(fam_of.values()))

    M = {}
    M["GtwoNRows"] = len(rows)
    M["GtwoNChal"] = len(chals)
    M["GtwoNFam"] = len(fams)
    M["GtwoModel"] = rows[0]["model"].replace("_", r"\_")
    M["GtwoMaxRounds"] = rows[0]["max_rounds"]

    # seed pairing integrity: every condition of a challenge must share a seed
    mismatch = 0
    for c in chals:
        seeds = {idx[(c, k)]["seed"] for k in conds if (c, k) in idx}
        if len(seeds) > 1:
            mismatch += 1
    M["GtwoSeedMismatch"] = mismatch

    # --- marginals per condition ---
    label = {"none": "None", "helpful": "Helpful", "latent_detour": "Detour",
             "latent_search": "Search", "sibling_poison": "Sibling"}
    for c in conds:
        rs = [idx[(ch, c)] for ch in chals if (ch, c) in idx]
        tag = label.get(c, c.title().replace("_", ""))
        M[f"Gtwo{tag}N"] = len(rs)
        M[f"Gtwo{tag}Solve"] = f"{np.mean([r['solved'] for r in rs]):.3f}"
        M[f"Gtwo{tag}Rounds"] = f"{np.mean([r['rounds_used'] for r in rs]):.2f}"
        M[f"Gtwo{tag}Cmds"] = f"{np.mean([r['total_commands'] for r in rs]):.2f}"

    # --- paired contrasts vs helpful ---
    REF = "helpful"
    for c in ["latent_detour", "latent_search", "sibling_poison", "none"]:
        tag = label.get(c, c)
        common = [ch for ch in chals if (ch, c) in idx and (ch, REF) in idx]
        f = [fam_of[ch] for ch in common]
        for chan, key in (("Rounds", "rounds_used"), ("Cmds", "total_commands")):
            d = np.array([idx[(ch, c)][key] - idx[(ch, REF)][key] for ch in common], float)
            lo, hi = clustered_boot(d, f, rng)
            M[f"Gtwo{tag}D{chan}"] = f"{d.mean():+.3f}"
            M[f"Gtwo{tag}D{chan}Lo"] = f"{lo:+.3f}"
            M[f"Gtwo{tag}D{chan}Hi"] = f"{hi:+.3f}"
        ds = np.array([idx[(ch, c)]["solved"] - idx[(ch, REF)]["solved"] for ch in common], float)
        lo, hi = clustered_boot(ds, f, rng)
        M[f"Gtwo{tag}DSolve"] = f"{ds.mean():+.3f}"
        M[f"Gtwo{tag}DSolveLo"] = f"{lo:+.3f}"
        M[f"Gtwo{tag}DSolveHi"] = f"{hi:+.3f}"
        b01 = int(sum(1 for ch in common if idx[(ch, REF)]["solved"] and not idx[(ch, c)]["solved"]))
        b10 = int(sum(1 for ch in common if idx[(ch, c)]["solved"] and not idx[(ch, REF)]["solved"]))
        p = mcnemar_exact(b01, b10)
        M[f"Gtwo{tag}Disc"] = f"{b01}/{b10}"
        M[f"Gtwo{tag}P"] = ("<0.001" if p < 1e-3 else f"{p:.3f}")
        M[f"Gtwo{tag}NPairs"] = len(common)

    # --- CENSORING DIAGNOSTICS (the reason the marginal round contrast is not usable) ---
    for c in conds:
        rs = [idx[(ch, c)] for ch in chals if (ch, c) in idx]
        tag = label.get(c, c)
        cap = sum(1 for r in rs if r["rounds_used"] >= r["max_rounds"])
        M[f"Gtwo{tag}Cap"] = f"{cap / len(rs):.3f}"
        M[f"Gtwo{tag}CapN"] = cap
        sv = [r["rounds_used"] for r in rs if r["solved"]]
        un = [r["rounds_used"] for r in rs if not r["solved"]]
        if sv:
            M[f"Gtwo{tag}RoundsSolved"] = f"{np.mean(sv):.2f}"
        if un:
            M[f"Gtwo{tag}RoundsUnsolved"] = f"{np.mean(un):.2f}"

    # --- conditional estimands + the confound they correct for ---
    for c in ["latent_detour", "latent_search"]:
        tag = label[c]
        common = [ch for ch in chals if (ch, c) in idx and (ch, REF) in idx]
        dr = np.array([idx[(ch, c)]["rounds_used"] - idx[(ch, REF)]["rounds_used"] for ch in common], float)
        ds = np.array([idx[(ch, c)]["solved"] - idx[(ch, REF)]["solved"] for ch in common], float)
        M[f"Gtwo{tag}CorrRS"] = f"{np.corrcoef(dr, ds)[0, 1]:+.2f}"
        bothcap = [ch for ch in common
                   if idx[(ch, c)]["rounds_used"] >= idx[(ch, c)]["max_rounds"]
                   and idx[(ch, REF)]["rounds_used"] >= idx[(ch, REF)]["max_rounds"]]
        M[f"Gtwo{tag}BothCapN"] = len(bothcap)
        M[f"Gtwo{tag}BothCapPct"] = f"{len(bothcap) / len(common):.3f}"
        for sub, nm in ((
                [ch for ch in common if idx[(ch, c)]["solved"] and idx[(ch, REF)]["solved"]], "Solved"),
                ([ch for ch in common if idx[(ch, c)]["solved"] == idx[(ch, REF)]["solved"]], "Conc")):
            M[f"Gtwo{tag}{nm}N"] = len(sub)
            M[f"Gtwo{tag}{nm}Fam"] = len({fam_of[ch] for ch in sub})
            for chan, key in (("Rounds", "rounds_used"), ("Cmds", "total_commands")):
                d = np.array([idx[(ch, c)][key] - idx[(ch, REF)][key] for ch in sub], float)
                lo, hi = clustered_boot(d, [fam_of[ch] for ch in sub], rng)
                M[f"Gtwo{tag}{nm}D{chan}"] = f"{d.mean():+.3f}"
                M[f"Gtwo{tag}{nm}D{chan}Lo"] = f"{lo:+.3f}"
                M[f"Gtwo{tag}{nm}D{chan}Hi"] = f"{hi:+.3f}"

    hd = [r for r in rows if r["condition"] in ("helpful", "latent_detour")]
    M["GtwoCapHDN"] = sum(1 for r in hd if r["rounds_used"] >= r["max_rounds"])
    M["GtwoCapHDTot"] = len(hd)
    M["GtwoCapAllN"] = sum(1 for r in rows if r["rounds_used"] >= r["max_rounds"])
    M["GtwoCapAllTot"] = len(rows)

    # --- censoring sensitivity: what a TRUE +k rounds/run cost would look like here ---
    hr = np.array([idx[(ch, REF)]["rounds_used"] for ch in chals], float)
    capv = float(rows[0]["max_rounds"])
    for k, nm in ((1, "One"), (2, "Two")):
        M[f"GtwoSens{nm}"] = f"{np.mean(np.minimum(hr + k, capv) - hr):+.3f}"

    # --- families that re-measured at the floor in the probe itself ---
    floored = [f for f in fams
               if all(idx[(ch, "helpful")]["solved"] == 0 and idx[(ch, "latent_detour")]["solved"] == 0
                      for ch in chals if fam_of[ch] == f)]
    M["GtwoFlooredFam"] = len(floored)
    M["GtwoFlooredChal"] = sum(1 for ch in chals if fam_of[ch] in floored)

    # --- G2 mechanism (executed-command probe, save-commands subset) ---
    mech = json.load(open(os.path.join(EW, "g2_mechanism.json")))
    M["GtwoMechN"] = mech["n_rows"]
    for c, tag in (("helpful", "Helpful"), ("latent_detour", "Detour")):
        d = mech["by_condition"][c]
        M[f"GtwoMech{tag}N"] = d["n_runs"]
        M[f"GtwoMech{tag}Pad"] = f"{d['pad_executed_rate']:.3f}"
        M[f"GtwoMech{tag}Cmds"] = f"{d['mean_commands']:.2f}"
        M[f"GtwoMech{tag}Ops"] = f"{d['mean_ops_per_command']:.2f}"
        M[f"GtwoMech{tag}MaxOps"] = d["max_ops_per_command"]

    # --- mechanism, recomputed from the per-run rows (not the aggregate) -----------
    # The aggregate g2_mechanism.json reports a MARGINAL command count, which is
    # confounded with solve status exactly as the round contrast is (unsolved runs
    # dominate it). We therefore emit the paired, family-clustered ops-per-command
    # contrast and its both-solved restriction, plus the pad-execution rate among
    # solved runs only -- the quantities that survive conditioning on outcome.
    PACK_RE = re.compile(r"\||;|&&")
    PAD_RE = re.compile(r"\brot13\b|\btr\s+['\"]?A-Za-z|codecs\.encode\([^)]*rot|\brot_13\b")
    mrows = []
    for f in sorted(glob.glob(os.path.join(EW, "g2_mechanism", "rows_shard*.jsonl"))):
        mrows += [json.loads(l) for l in open(f) if l.strip()]
    if not mrows:
        raise SystemExit(
            "FATAL: g2_mechanism/rows_shard*.jsonl not found under "
            f"{EW}. Nineteen mechanism macros cannot be emitted and the appendix will "
            "not compile. Ship those rows alongside the other artifacts.")
    if True:
        midx = {(r["challenge"], r["condition"]): r for r in mrows}
        mch = sorted({r["challenge"] for r in mrows})
        mfam = {r["challenge"]: r["family"] for r in mrows}

        def opc(r):
            c = r.get("commands") or []
            return float(np.mean([len(PACK_RE.findall(x)) + 1 for x in c])) if c else None

        pr = [c for c in mch
              if opc(midx.get((c, "helpful"), {})) is not None
              and opc(midx.get((c, "latent_detour"), {})) is not None]
        d = np.array([opc(midx[(c, "latent_detour")]) - opc(midx[(c, "helpful")]) for c in pr], float)
        lo, hi = clustered_boot(d, [mfam[c] for c in pr], rng)
        M["GtwoMechOpsN"] = len(pr)
        M["GtwoMechOpsFam"] = len({mfam[c] for c in pr})
        M["GtwoMechDOps"] = f"{d.mean():+.3f}"
        M["GtwoMechDOpsLo"] = f"{lo:+.3f}"
        M["GtwoMechDOpsHi"] = f"{hi:+.3f}"
        M["GtwoMechOpsPos"] = int((d > 0).sum())
        # arm means ON THE PAIRED SET, so the printed delta and the printed arm means are
        # the same estimand. (The aggregate g2_mechanism.json means are pooled OVER
        # COMMANDS across all runs -- a different, command-count-weighted quantity that
        # must not be printed beside a per-run paired delta.)
        M["GtwoMechHelpfulOpsPaired"] = f"{np.mean([opc(midx[(c, 'helpful')]) for c in pr]):.2f}"
        M["GtwoMechDetourOpsPaired"] = f"{np.mean([opc(midx[(c, 'latent_detour')]) for c in pr]):.2f}"
        M["GtwoMechOpsDropped"] = int(M["GtwoMechDetourN"]) - len(pr) if "GtwoMechDetourN" in M else len(mch) - len(pr)
        bs = [c for c in pr if midx[(c, "latent_detour")]["solved"] and midx[(c, "helpful")]["solved"]]
        db = np.array([opc(midx[(c, "latent_detour")]) - opc(midx[(c, "helpful")]) for c in bs], float)
        lo, hi = clustered_boot(db, [mfam[c] for c in bs], rng)
        M["GtwoMechOpsSolvedN"] = len(bs)
        M["GtwoMechOpsSolvedFam"] = len({mfam[c] for c in bs})
        M["GtwoMechSolvedDOps"] = f"{db.mean():+.3f}"
        M["GtwoMechSolvedDOpsLo"] = f"{lo:+.3f}"
        M["GtwoMechSolvedDOpsHi"] = f"{hi:+.3f}"
        rest = [c for c in pr if c not in set(bs)]
        M["GtwoMechRestN"] = len(rest)
        M["GtwoMechRestDOps"] = f"{np.mean([opc(midx[(c, 'latent_detour')]) - opc(midx[(c, 'helpful')]) for c in rest]):+.3f}"
        # The density channel's own confound check: unlike the round channel, Delta(ops)
        # is essentially orthogonal to solve status, so it is NOT a restatement of the
        # outcome. This is the appendix's warrant for treating it differently.
        dsm = np.array([midx[(c, "latent_detour")]["solved"] - midx[(c, "helpful")]["solved"]
                        for c in pr], float)
        drm = np.array([midx[(c, "latent_detour")]["rounds_used"] - midx[(c, "helpful")]["rounds_used"]
                        for c in pr], float)
        M["GtwoMechCorrOpsSolved"] = f"{np.corrcoef(d, dsm)[0, 1]:+.2f}"
        M["GtwoMechCorrRoundsSolved"] = f"{np.corrcoef(drm, dsm)[0, 1]:+.2f}"
        # pad execution among SOLVED runs only (uncensored subset)
        for c, tag in (("helpful", "Helpful"), ("latent_detour", "Detour")):
            sv = [r for r in mrows if r["condition"] == c and r["solved"]]
            M[f"GtwoMech{tag}PadSolved"] = f"{np.mean([bool(any(PAD_RE.search(x) for x in (r['commands'] or []))) for r in sv]):.3f}"
            M[f"GtwoMech{tag}SolvedN"] = len(sv)
        # the marginal command COUNT, which does NOT survive -- emitted so the paper can
        # state plainly that it does not separate the arms.
        dc = np.array([midx[(c, "latent_detour")]["total_commands"] - midx[(c, "helpful")]["total_commands"]
                       for c in pr], float)
        lo, hi = clustered_boot(dc, [mfam[c] for c in pr], rng)
        M["GtwoMechDCmds"] = f"{dc.mean():+.3f}"
        M["GtwoMechDCmdsMed"] = f"{np.median(dc):+.1f}"
        M["GtwoMechDCmdsLo"] = f"{lo:+.3f}"
        M["GtwoMechDCmdsHi"] = f"{hi:+.3f}"

    # --- G1 structural acceptance (depth-1 crypto) ---
    g1 = json.load(open(os.path.join(EW, "g1_verifier_acceptance_crypto_depth1.json")))
    sc = g1["structural_detour_claims"]
    M["GoneFamilies"] = g1["n_families_evaluated"]
    M["GonePerFamily"] = g1["params"]["n_per_family"]
    M["GoneN"] = sc["identity_checked"]
    M["GoneIdentityHolds"] = sc["identity_holds"]
    M["GoneRouteLonger"] = sc["route_strictly_longer"]
    M["GoneExtraSteps"] = f"{sc['extra_steps_mean']:.1f}"
    M["GoneExtraStepsMin"] = sc["extra_steps_min"]
    M["GonePad"] = g1["params"]["pad"].replace("_", r"\_")
    lab = {"helpful": "Helpful", "latent_detour": "Detour", "latent_search": "Search",
           "sibling_poison": "Sibling", "malignant_destroy": "Destroy",
           "malignant_exhaust": "Exhaust", "malignant_wrongflag": "Wrongflag"}
    for k, tag in lab.items():
        d = g1["by_label"][k]
        M[f"Gone{tag}Accept"] = f"{d['accept_rate_strict']:.3f}"
        M[f"Gone{tag}Wellformed"] = f"{d['accept_rate_wellformed']:.3f}"
        M[f"Gone{tag}Flag"] = f"{d['verifier_FLAG_rate_strict']:.3f}"

    # --- SQL well-formedness cross-check (Prop. 4 hardening) ---
    try:
        gsql = json.load(open(os.path.join(EW, "g1_verifier_acceptance_sql.json")))
        d = gsql["by_label"]["sibling_poison"]
        M["GoneSqlSiblingWellformed"] = f"{d['accept_rate_wellformed']:.3f}"
        M["GoneSqlSiblingAccept"] = f"{d['accept_rate_strict']:.3f}"
        M["GoneSqlN"] = d["n"]
    except Exception:
        pass

    # --- in-band family selection provenance ---
    try:
        sel = json.load(open(os.path.join(EW, "inband_selection_crypto.json")))
        M["GtwoSelNIn"] = len(sel["in_band_families"])
        M["GtwoSelNCand"] = sel.get("n_candidates_considered", sel.get("n_families_probed", "?"))
    except Exception:
        pass

    POOLED_WARN = {"GtwoMechHelpfulOps", "GtwoMechDetourOps",
                   "GtwoMechHelpfulMaxOps", "GtwoMechDetourMaxOps",
                   "GtwoMechHelpfulCmds", "GtwoMechDetourCmds"}
    out = [r"% AUTO-GENERATED by paper_final/emit_appendix_numbers.py -- DO NOT EDIT BY HAND.",
           r"% Every macro is recomputed from artifacts under extra_work/results/.",
           f"% Bootstrap: {B} family-clustered resamples, seed {SEED}.", ""]
    unfilled = []
    for k, v in M.items():
        if k.startswith("_"):
            continue
        if v is None or v == "":
            unfilled.append(k); continue
        if k in POOLED_WARN:
            out.append(r"% NOT FOR PRINT: pooled over commands/runs, unpaired and "
                       r"command-count-weighted. Do not print beside a paired delta.")
        out.append(rf"\newcommand{{\{k}}}{{{v}}}")
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "numbers_latent.tex")
    open(dest, "w").write("\n".join(out) + "\n")
    print(f"wrote {dest}: {len(out)-4} macros")
    if unfilled:
        print("UNFILLED:", unfilled)
    else:
        print("no unfilled macros")


if __name__ == "__main__":
    main()
