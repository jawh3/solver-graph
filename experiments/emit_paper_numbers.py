"""T0-3 (RIGOR_LEDGER): single source of truth for every number in the paper.

Reads ALL committed analysis artifacts (5 ladder rungs + H5 + verifier + mechanism
artifacts from T0-5) and emits paper/numbers_identity_a.tex. No number in the paper
body may be hand-typed: prose cites macros only.

Macro naming (LaTeX forbids digits in command names — rung codes are words):
  rungs : ThreeB SevenB FourteenB ThirtyTwoB LlamaB
  ATEs  : \Ate{Help|Poison}{Solve|Rew}{<rung>}{,CI,P,N}     (on-target)
          \Mate...                                          (marginal)
  CRT   : \Crt{Help|Poison}{O|R}{Frac|Med}{<rung>}          (o-/R-keyed, on-target)
  cite  : \Cite{Help|Poison}{Follow|Solve}{<rung>}, \CiteBase{<rung>}
  mech  : \CoCite, \DeltaHat{,CI}, \AOne{,CI}, \StepDetour{,CI}, \Spear{,P}  + rung
  H5    : \HFive{Empty|Clean|Forced|OutMem|CausalMem|Shuffle|Delta|DeltaCI|McNemar|N|Verdict}{FourteenB|ThirtyTwoB}
  promo : \Promo{Poison|Help}{Out|Causal}{<rung>} ("37/40" style, theta=0)
  verif : \VerifierPoisonFlag, \VerifierHelpFalseFlag, \VerifierN

Usage: python -m experiments.emit_paper_numbers [--out paper/numbers_identity_a.tex]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNGS = {
    "ThreeB": "campaign_3b_M3",
    "SevenB": "campaign_7b35_M3_rep",
    "FourteenB": "campaign_14b_M3_powered",
    "ThirtyTwoB": "campaign_32b_M3_tp2",
    "LlamaB": "campaign_llama8b_M3",
}
H5 = {"FourteenB": "h5_14b", "ThirtyTwoB": "h5_32b"}


def _j(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def f3(x):
    return f"{x:+.3f}" if x is not None else "--"


def r3(x):
    return f"{x:.3f}" if x is not None else "--"


def fp(p):
    if p is None:
        return "--"
    return r"$<\!10^{-4}$" if p < 1e-4 else f"{p:.3f}" if p >= 0.001 else f"{p:.4f}"


def ci(lo, hi, fmt=f3):
    return f"[{fmt(lo)},\\,{fmt(hi)}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper" / "numbers_identity_a.tex"))
    args = ap.parse_args()

    M: list[tuple[str, str]] = []

    def emit(name, val):
        M.append((name, str(val)))

    for code, camp in RUNGS.items():
        d = ROOT / "results" / "lab" / camp / "merged"
        ot = _j(d / "ontarget_analysis.json")
        if ot:
            for chan, key in (("Solve", "outcome"), ("Rew", "R_final")):
                for lab, mac in (("helpful", "Help"), ("poison", "Poison")):
                    for scope, pre in (("on_target", "Ate"), ("marginal", "Mate")):
                        c = (ot.get("by_label", {}).get(key, {}).get(scope, {}) or {}).get(lab)
                        if not c:
                            continue
                        b = f"{pre}{mac}{chan}{code}"
                        emit(b, f3(c.get("mean")))
                        emit(b + "CI", ci(c.get("ci_lo") if "ci_lo" in c else (c.get("ci") or [None, None])[0],
                                          c.get("ci_hi") if "ci_hi" in c else (c.get("ci") or [None, None])[1]))
                        emit(b + "P", fp(c.get("p")))
                        emit(b + "N", c.get("n", "--"))
            dp = ot.get("descriptive_poison_in_context") or {}
            if dp:
                emit(f"BaseOffTarget{code}", r3(dp.get("off_target", {}).get("solve")
                                                if isinstance(dp.get("off_target"), dict) else dp.get("off_target")))
        for keyname, kc in (("crt_test_outcome.json", "O"), ("crt_test.json", "R")):
            cr = _j(d / keyname)
            if cr:
                for lab, mac in (("helpful", "Help"), ("poison", "Poison")):
                    c = (cr.get("on_target") or {}).get(lab)
                    if c:
                        emit(f"Crt{mac}{kc}Frac{code}", r3(c.get("frac_sig_p05")))
                        emit(f"Crt{mac}{kc}Med{code}", fp(c.get("median_p")))
                        emit(f"Crt{mac}{kc}N{code}", c.get("n", "--"))
        cite = _j(d / "cite_rate.json")
        if cite:
            by = cite.get("by", cite)
            for lab, mac in (("helpful", "Help"), ("poison", "Poison")):
                on = by.get(f"{lab}:on-target")
                off = by.get(f"{lab}:off-target")
                if on:
                    emit(f"Cite{mac}Follow{code}", r3(on.get("cite_rate")))
                    emit(f"Cite{mac}Solve{code}", r3(on.get("solve_rate_when_cited")))
                    emit(f"Cite{mac}NCited{code}", on.get("n_cited", "--"))
                if off and lab == "poison":
                    emit(f"CiteBase{code}", r3(off.get("solve_rate_when_cited")))
        co = _j(d / "cocitation.json")
        if co and co.get("cited"):
            c = co["cited"]
            emit(f"CoCite{code}", r3(c.get("cocite")))
            emit(f"CoCiteN{code}", c.get("n_pc", "--"))
            emit(f"DeltaHat{code}", f3(c.get("displacement")))
            lo, hi = c.get("displacement_ci95") or [None, None]
            emit(f"DeltaHatCI{code}", ci(lo, hi))
        a1 = _j(d / "a1_contrast.json")
        if a1 and a1.get("cited"):
            c = a1["cited"]
            emit(f"AOne{code}", f3(c.get("contrast_weighted")))
            lo, hi = c.get("ci95") or [None, None]
            emit(f"AOneCI{code}", ci(lo, hi))
            itt = a1.get("sampled_ITT") or {}
            emit(f"AOneITT{code}", f3(itt.get("contrast_weighted")))
        sd = _j(d / "step_detour.json")
        if sd:
            emit(f"StepDetour{code}", f3(sd.get("detour_steps_weighted")))
            lo, hi = sd.get("ci95") or [None, None]
            emit(f"StepDetourCI{code}", ci(lo, hi))
        br = _j(d / "base_rate_sweep.json")
        if br and br.get("poison", {}).get("spearman_rho") is not None:
            emit(f"SpearPoison{code}", f3(br["poison"]["spearman_rho"]))
            emit(f"SpearPoisonP{code}", fp(br["poison"]["p"]))
            emit(f"SpearPoisonN{code}", br["poison"]["n"])
        nv = _j(d / "node_values.json")
        if nv:
            for lab, mac in (("poison", "Poison"), ("helpful", "Help")):
                rows = [v for v in nv.values() if v.get("label") == lab]
                n_out = sum(1 for v in rows if (v.get("outcome_Q") or 0) > 0)
                d_out = sum(1 for v in rows if v.get("outcome_Q") is not None)
                n_cau = sum(1 for v in rows if (v.get("causal_ATE") or 0) > 0)
                d_cau = sum(1 for v in rows if v.get("causal_ATE") is not None)
                emit(f"Promo{mac}Out{code}", f"{n_out}/{d_out}")
                emit(f"Promo{mac}Causal{code}", f"{n_cau}/{d_cau}")

    for code, h5dir in H5.items():
        s = _j(ROOT / "results" / "lab" / h5dir / "h5_summary.json")
        if not s:
            continue
        arms = s.get("arms_solve", {})
        for arm, mac in (("empty", "Empty"), ("clean", "Clean"), ("poison_forced", "Forced"),
                         ("outcome_mem", "OutMem"), ("causal_mem", "CausalMem"), ("shuffle_mem", "Shuffle")):
            if arms.get(arm) is not None:
                emit(f"HFive{mac}{code}", r3(arms[arm]))
        t = s.get("transfer_delta_causal_minus_outcome")
        if t:
            emit(f"HFiveDelta{code}", f"{100*t['delta']:+.1f}")
            emit(f"HFiveDeltaCI{code}", f"[{100*t['ci_lo']:+.1f},\\,{100*t['ci_hi']:+.1f}]")
            emit(f"HFiveMcNemar{code}", fp(t.get("mcnemar_p")))
            emit(f"HFiveN{code}", t.get("n", "--"))
            emit(f"HFiveVerdict{code}", t.get("verdict_PREREG", "--").replace("_", r"\_"))

    cb = _j(ROOT / "results" / "lab" / "h5_cluster_boot_summary.json")
    if cb:
        for tag, code in (("14b", "FourteenB"), ("32b", "ThirtyTwoB")):
            t = cb.get(tag, {}).get("family_cluster_boot")
            v = cb.get(tag, {})
            if t:
                emit(f"HFiveDeltaClus{code}", f"{t['delta_pp']:+.1f}")
                emit(f"HFiveDeltaClusCI{code}", f"[{t['ci95_pp'][0]:+.1f},\\,{t['ci95_pp'][1]:+.1f}]")
                emit(f"HFiveClusP{code}", fp(t.get("p")))
                emit(f"HFiveVerdictClus{code}", v.get("verdict_cluster", "--").replace("_", r"\_"))
                emit(f"HFiveNFamClus{code}", v.get("n_families", "--"))

    ev = _j(ROOT / "results" / "lab" / "execution_verifier.json")
    if ev:
        emit("VerifierPoisonFlag", r3(ev.get("verifier_poison_FLAG_rate")))
        emit("VerifierHelpFalseFlag", r3(ev.get("verifier_helpful_FALSE_FLAG_rate")))
        emit("VerifierN", ev.get("n_families", "--"))

    # T1-1 MALIGNANT positive control — execution verifier (deployable, deterministic; exists now)
    _MAL = (("destroy", "Destroy"), ("exhaust", "Exhaust"), ("wrongflag", "Wrongflag"))
    evm = _j(ROOT / "results" / "lab" / "execution_verifier_14b_malig.json")
    if evm and evm.get("malignant"):
        ba = evm["malignant"]["by_arm"]
        for arm, mac in _MAL:
            d = ba.get(arm)
            if d:
                emit(f"VerifierMal{mac}", r3(d.get("verifier_FLAG_rate")))
        emit("VerifierMalN", ba.get("destroy", {}).get("n", "--"))

    # T1-1 MALIGNANT positive control — 14B campaign (guarded: only after run_malignant.sh completes).
    # Per arm: on-target SNIPW ATE+CI (the estimator MUST flag these where it reads poison null),
    # solve|cited vs within-campaign base + follow-rate, and o-keyed exact CRT frac-sig.
    mdir = ROOT / "results" / "lab" / "campaign_14b_malig" / "merged"
    ot = _j(mdir / "ontarget_analysis.json")
    if ot:
        on = ot.get("by_label", {}).get("outcome", {}).get("on_target", {})
        for arm, mac in _MAL:
            c = on.get(f"malignant_{arm}")
            if c:
                emit(f"MalAte{mac}", f3(c.get("mean")))
                emit(f"MalAteCI{mac}", ci(c.get("ci_lo"), c.get("ci_hi")))
                emit(f"MalAteP{mac}", fp(c.get("p")))
                emit(f"MalAteN{mac}", c.get("n", "--"))
        cp = on.get("poison")          # poison NULL within the SAME diluted campaign = the contrast
        if cp:
            emit("MalPoisonAte", f3(cp.get("mean")))
            emit("MalPoisonAteCI", ci(cp.get("ci_lo"), cp.get("ci_hi")))
            emit("MalPoisonAteP", fp(cp.get("p")))
    mres = _j(mdir / "malignant_result.json")
    if mres:
        emit("MalBase", r3(mres.get("base")))
        for arm, mac in _MAL:
            a = (mres.get("arms") or {}).get(arm)
            if a:
                emit(f"MalSolveCited{mac}", r3(a.get("solve_when_cited")))
                emit(f"MalFollow{mac}", r3(a.get("follow_rate")))
                emit(f"MalNCited{mac}", a.get("n_cited", "--"))
    crtm = _j(mdir / "crt_test_outcome.json")
    if crtm:
        for arm, mac in _MAL:
            c = (crtm.get("on_target") or {}).get(f"malignant_{arm}")
            if c:
                emit(f"MalCrt{mac}Frac", r3(c.get("frac_sig_p05")))
                emit(f"MalCrt{mac}Med", fp(c.get("median_p")))

    # T1-1 COMBINED pooled SNIPW (both campaigns, 74/71 families) — the headline positive-control
    # numbers: every cell's CI excludes 0, both arms below .05 two-sided. Est = lesson-level (equal,
    # apples-to-apples with the poison null) + run-level (n_treat-weighted). Keys: outcome, R_final.
    for keytag, fname in (("", "malignant_pooled_combined.json"), ("Rf", "malignant_pooled_combined_rfinal.json")):
        cp = _j(mdir / fname)
        if not cp:
            continue
        for arm, mac in (("destroy", "Destroy"), ("exhaust", "Exhaust")):
            bl = (cp.get("by_label") or {}).get(f"malignant_{arm}")
            if not bl:
                continue
            for est, etag in (("equal_weight_lesson_level", ""), ("n_treat_weighted_run_level", "Run")):
                b = bl.get(est)
                if not b:
                    continue
                pre = f"MalPool{keytag}{mac}{etag}"
                emit(f"{pre}Ate", f3(b.get("ate")))
                emit(f"{pre}CI", ci(b.get("ci_lo"), b.get("ci_hi")))
                emit(f"{pre}Pone", fp(b.get("p_one_sided")))
                emit(f"{pre}Ptwo", fp(b.get("p_two_sided")))
                emit(f"{pre}Nfam", b.get("n_families", "--"))

    lines = ["% Auto-generated by experiments/emit_paper_numbers.py — DO NOT EDIT.",
             "% Source artifacts: results/lab/campaign_*/merged/*.json, results/lab/h5_*/h5_summary.json,",
             "% results/lab/execution_verifier.json. Regenerate: python -m experiments.emit_paper_numbers", ""]
    for name, val in M:
        lines.append(rf"\providecommand{{\{name}}}{{}}\renewcommand{{\{name}}}{{{val}}}")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"[emit_paper_numbers] wrote {args.out}: {len(M)} macros")


if __name__ == "__main__":
    main()
