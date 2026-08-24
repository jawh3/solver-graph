#!/usr/bin/env python3
"""paper_final/check_estimand_vocab.py — guard against estimand mislabelling.

This manuscript has two per-lesson causal estimands over the SAME logged treatment
(T = 1[v in sampled M-set], ope/causal_value.py):

  on-target        : run set restricted to the lesson's own family (family is PRE-treatment,
                     so this is a conditional ATE, not a selection effect)
  marginal-inclusion: pooled over all families

NEITHER conditions on the lesson being CITED. The only citation-conditioned statistic in
the paper is A1's own contrast (poison-cited vs no-recipe).

Calling the on-target contrast "conditional on citation" converts a pre-treatment
conditioning into a post-treatment collider and silently concedes the paper's own rival
reading. That error survived five review rounds and two attempted fixes because it is a
WORD, not a number, and no numeric guard can see it. This one can.

Exit 1 if citation vocabulary appears in the same sentence as on-target/marginal
vocabulary.
"""
import re, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
BODY = ["main.tex", "method_causal.tex", "results_body.tex", "results_solution.tex",
        "theorem_identifiability.tex", "appendix_latent.tex",
        # the guard-rail documents ship to downstream authors and must obey the same rule
        "PROTOCOL.md", "an internal handoff document (not included in this release)"]

ESTIMAND = re.compile(r"on-target|marginal-inclusion", re.I)
# citation vocabulary that must NOT attach to those estimands
CITATION = re.compile(
    r"conditional[ -]on[ -]cit"
    r"|conditions?\s+on\s+[^.;]{0,60}?\bcite(?:d|s)?\b"      # plain 'cited', any wording
    r"|conditions?\s+on\s+[^.;]{0,60}?\bexecut(?:ed|ing)\b"  # 'actually being executed'
    r"|conditional on citation|solve\$?\\?mid\$?cited"
    r"|conditions on the agent having", re.I)
# sanctioned exceptions: sentences that are ABOUT the distinction
EXEMPT = re.compile(r"neither conditions on citation|not on citation|"
                    r"citation-conditioned statistic|A\\ref\{asm:floor\}'s own contrast", re.I)


def sentences(text):
    text = re.sub(r"(?m)^\s*%.*$", "", text)
    text = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.;])\s+", text)


def main():
    bad = []
    for fn in BODY:
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        for i, sent in enumerate(sentences(open(p).read())):
            if ESTIMAND.search(sent) and CITATION.search(sent) and not EXEMPT.search(sent):
                bad.append((fn, sent.strip()[:240]))
    if not bad:
        print("estimand-vocab: OK — no citation vocabulary attached to an "
              "on-target/marginal estimand")
        return 0
    print(f"estimand-vocab: {len(bad)} MISLABELLED sentence(s)\n")
    for fn, sent in bad:
        print(f"  {fn}: {sent}\n")
    print("The on-target contrast is FAMILY-restricted (pre-treatment), not")
    print("citation-conditioned. Rephrase, or add to EXEMPT if the sentence is about")
    print("the distinction itself.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
