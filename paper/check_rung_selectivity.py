#!/usr/bin/env python3
"""paper_final/check_rung_selectivity.py — mechanical guard against partial-rung reporting.

The dominant defect class in this manuscript's review history was a per-rung statistic
reported only at the rungs that agreed with the claim, with the disagreeing rungs sitting
in emitted-but-never-printed macros. That is invisible to proofreading and to LaTeX.

This script groups every emitted macro into FAMILIES by stripping the rung suffix
(ThreeB / SevenB / FourteenB / ThirtyTwoB / LlamaB) and reports any family that is
PARTIALLY printed in the body: some rungs cited, others silently omitted. A partially
printed family is not automatically wrong -- but every one must be a deliberate,
defensible choice, so the script prints the omitted values so the author can check the
omission does not run in the paper's favour.

Exit 1 if any family is partially printed and not listed in ALLOW below.
"""
import re, sys, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
NUM = ["numbers.tex", "numbers_solution.tex", "numbers_solution_defaults.tex",
       "numbers_identity_a.tex", "numbers_latent.tex"]
BODY = ["main.tex", "method_causal.tex", "results_body.tex", "results_solution.tex",
        "theorem_identifiability.tex", "appendix_latent.tex"]
RUNGS = ["ThreeB", "SevenB", "FourteenB", "ThirtyTwoB", "LlamaB"]

# Families whose partial printing is deliberate and defended in the text.
# Each entry MUST carry a reason.
ALLOW = {
    "AOne":            "all five printed in the A1 empirical check",
    "AOneCI":          "all five printed in the A1 empirical check",
    "StepDetour":      "all five printed in Hypothesis 1",
    "StepDetourCI":    "all five printed in Hypothesis 1",
    "SpearPoison":     "all five printed in the base-rate check",
    "SpearPoisonP":    "all five printed in the base-rate check",
    "SpearPoisonN":    "all five printed in the base-rate check",
    "PromoHelpOut":    "14B only, cited explicitly as a single-rung illustration of the "
                       "outcome gate promoting everything; the causal gate it is contrasted "
                       "with IS reported at all five rungs",
    "PromoPoisonOut":  "14B only, same single-rung illustration (see PromoHelpOut)",
}


def main():
    defined = {}
    for fn in NUM:
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            for name, val in re.findall(r"\\(?:new|renew)command\{\\(\w+)\}\{(.*?)\}\s*$",
                                        open(p).read(), re.M):
                defined[name] = val

    body = ""
    for fn in BODY:
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            body += re.sub(r"(?m)^\s*%.*$", "", open(p).read())
    used = set(re.findall(r"\\([A-Za-z]+)", body))

    fams = defaultdict(dict)
    for name in defined:
        for r in RUNGS:
            if name.endswith(r):
                fams[name[: -len(r)]][r] = name
                break

    problems = []
    for fam, members in sorted(fams.items()):
        if len(members) < 2:
            continue
        shown = {r: m for r, m in members.items() if m in used}
        hidden = {r: m for r, m in members.items() if m not in used}
        if shown and hidden:
            if fam in ALLOW:
                continue
            problems.append((fam, shown, hidden))

    if not problems:
        print(f"rung-selectivity: OK — {len(fams)} per-rung families, none partially printed")
        return 0

    print(f"rung-selectivity: {len(problems)} PARTIALLY-PRINTED families\n")
    for fam, shown, hidden in problems:
        print(f"  {fam}*")
        print(f"    printed : {', '.join(f'{r}={defined[m]}' for r, m in sorted(shown.items()))}")
        print(f"    OMITTED : {', '.join(f'{r}={defined[m]}' for r, m in sorted(hidden.items()))}")
    print("\nEach omission must be deliberate and must not run in the paper's favour.")
    print("Print the missing rungs, or add the family to ALLOW with a reason.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
