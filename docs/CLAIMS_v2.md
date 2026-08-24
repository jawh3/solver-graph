> **SUPERSEDED — read `paper/PROTOCOL.md` §2 and `paper/main.pdf` first.**
>
> This is the authors' 2026-06-25 working claim ledger, kept for provenance.
> Three corrections you must apply if you read on:
>
> 1. **The `-0.0745` figure below is the ON-TARGET contrast, not the marginal.**
>    The marginal-inclusion ATE at M=1 is **-0.021, p=.25**. Presenting -0.0745
>    as the marginal claims a significant deployed causal harm the paper
>    explicitly disclaims.
> 2. **`wrongflag` is UNTESTED, not "MISSED".** The agent followed that recipe
>    on 11 of 453 sampled runs. The null reflects near-zero exposure, not a
>    demonstrated estimator blind spot.
> 3. **The ladder is FIVE rungs** (Qwen2.5 3B-32B plus Llama-3.1-8B), not the
>    "6-rung 0.5B-32B" described here. The paper is 23 pages, not 16.

---

# CLAIMS v2 — the honest claim set for the paper (post all reckonings/sweeps + 2 readiness iterations, 2026-06-25)

Every number is committed-artifact-backed and reproducible by a committed script. Reviewer-converged
(no fatal flaws); this version folds in iteration-2 consensus fixes.

## C1 (HEADLINE) — Per-item harm of a benign-false ("poison") lesson is NOT identifiable from outcomes.
Reproducible: `experiments/within_poison_validation.py` -> `active_loo_within_label.json` (source
`active_loo.json` committed). JOINT evidence (no single precise literal):
- Active-LOO gold-standard MEAN poison marginal = **+0.045, CI [−0.006,+0.098] (n=28; emitter-backed, bootstrap seed 0)** — no aggregate
  per-item poison signal (a fail-to-reject; ILL-POSED/UNINFORMATIVE, **never "zero"**).
- The cheap on-target SNIPW does NOT rank the per-item poison effect: within-poison Spearman ≈ 0
  (**SciPy +0.072 (primary) / argsort +0.028**, both ~0, emitter-backed wide bootstrap CI [−0.35,+0.49] (seed 0) — convention-robust,
  not a precise estimate). Sign-agreement among the RANKABLE nodes = **8/18 (P=0.76 = AT CHANCE / uninformative,
  NOT anti-correlated)**; the often-quoted "8/28" counts the 10 LOO ties-at-zero (a 1/12-quantized gold
  standard) as disagreements. vs within-helpful Spearman 0.64, rankable 30/30 — the estimator IS validated
  on LARGE effects only.
- WHY ~0 is honest (not just estimator failure): the feasible-n gold standard itself cannot rank the small
  poison effects — per-node LOO SE ≲ 0.20 (a worst-case p=0.5 BOUND) vs between-node SD 0.14. NEITHER the
  cheap estimator NOR the feasible-n active gold standard ranks the per-item poison effect.
- SCOPE the word "artifact": the observational on-target SNIPW reads poison mildly negative (−0.057 p=.0007
  @14B; −0.064 p=.005 @32B; nominally p<.01 at the two powered rungs) — that label-level signal is real-but-CONFOUNDED; we claim only
  that it is NOT the per-item causal estimand (non-identifiable), NOT that "no harm exists". CONSERVATIVE
  note: the active-LOO contexts were poison-FIRST (primacy) -> biases poison toward looking MORE harmful, so
  within-LOO≈0 is conservative.

## C2 — A benign-false lesson's effect is NOT a per-item constant; the in-context harm is RESOURCE-mediated (DEMOTED, confounded).
- poison ALONE is mildly recoverable-POSITIVE, not null (+0.052, CI excludes 0, p=.02; floor-bounded, base 0.04). Co-present (forced) Δ_interf = −0.31.
- The DiD interaction (Δ_interf − Δ_alone) = −0.363 [−0.479,−0.244] (all-41 CI; in-band-15 WIDER −0.367
  [−0.542,−0.192]) is **SUGGESTIVE, NOT load-bearing**: its Δ_interf arm comes from the pre-fix POISON-FIRST
  rollouts (constructed_loo_crypto.json, 2026-06-24, one day before the primacy fix) and was never re-run
  order-randomized -> it INHERITS the primacy confound. We do not headline it (and have not spent GPU to
  rescue it; order_seed exists in context_assembly if ever needed).
- The in-context harm is RESOURCE/co-presence-mediated, NOT method-conflict-specific. vs a pure budget-burner
  (`exhaust`): poison_in_context − exhaust_in_context is subset-dependent — **in-band n=15 +0.042
  [−0.242,+0.317] p=.78** (underpowered; CANNOT-reject-equality, NOT a TOST/equivalence — "cannot reject"
  ≠ "equal") AND **all-41 +0.204 [+0.040,+0.357] p=.0164** (poison hurts LESS than the maximal burner). Report
  BOTH; lead with neither as a clean magnitude. No fair distractor exists (Δ_specific empty by construction —
  poison shares the correct OUTER transport), so method-specificity cannot be tested, only excluded via the burn.
- Mechanism (OBSERVATIONAL, selection-confounded): executing the distinctive wrong-inner method co-occurs with
  failure + MORE commands (14B −0.27, +1.5 cmds; 32B −0.44, +3.4 cmds), weakening to NS when helpful co-present.
- DISCLOSED: ALL constructed-context numbers here are POISON-FIRST (primacy confound, now fixed); not over-claimed.
- NET: C2 is best read as "there is no clean, order-robust per-item poison value" — further support for C1, not
  a standalone method-conflict result.

## C3 (POSITIVE CONTROL) — The same estimator detects robustly-destructive harm (destroy/exhaust ONLY).
- destroy/exhaust DETECTED via the malignant-ALONE channel (~98% of malignant citations lack a co-present
  helpful, so detection is order-robust IN DIRECTION, not 'order-invariant'): pooled SNIPW destroy −0.016
  p2=.010, exhaust −0.017 p2=.0165 (both excl 0). **wrongflag MISSED** (SNIPW +0.004 p=.69) at EVERY mention
  — the arm most like a committed-wrong-answer / the degenerate SQL/struct failure mode, caught only by the
  deterministic execution-verifier (1.000 — benchmark-CONSTRUCTION-guaranteed deployable contrast, NOT a
  measured generalization).
- The constructed direct-harm magnitudes (−0.42..−0.82) are FORCED-EXPOSURE + malignant-FIRST (same primacy
  confound) -> DIRECTION/sign evidence only, NOT clean effect sizes.
- DISCLOSE the malignant over-sampling confound: in the malignant campaign the same-campaign poison reads
  spuriously POSITIVE (run-level +0.033 p1=.0475) because malignant lessons are over-sampled ~2.5x and depress
  the base; the poison null is the MAIN-campaign result (not the malignant campaign).

## C4 (GENERALITY, observational consistency check) — the helpful≫poison asymmetry replicates across capability.
- ON-TARGET observational SNIPW: helpful large (+0.20..+0.57 at deployed M=3; +0.747 at M=1, p<.001) vs poison small at DEPLOYED M
  (|ATE|~0.05; nominally p<.01-negative at powered rungs, null marginal) across the 6-rung Qwen ladder (0.5B–32B). "poison
  null" is precise ONLY for the MARGINAL estimand AT DEPLOYED M — at FULL exposure (M=1) the marginal poison
  harm is −0.0745 p<.001 (a real sub-floor harm; ~constant −0.05..−0.07 across M, significance power/dilution-
  gated). Frame C4 as a consistency check of the OBSERVATIONAL channel (which C1 shows is uninformative for the
  per-item estimand), not as 'poison harmless'.
- Cross-ARCHITECTURE generality = Llama-3.1-8B ONLY (single non-Qwen ladder member); the rest of the non-Qwen
  probe set (Gemma-27B n=1, Mistral-7B n=3, InternLM-7B n=2, Mistral-24B/Gemma smoke n=0) was harness-FLOORED.
  State this; do not read 'across architecture' as more than one architecture.

## C5 (HONEST LIMITS — prominent)
- The strongest RIVAL reading, named: a real-but-underpowered MARGINAL poison harm (M=1 −0.0745 p<.001;
  significance power/dilution-gated) — C1's per-item/marginal distinction holds, but the deployed-M "null" is
  partly a dilution+power phenomenon, and within crypto at full exposure poison is a measurable sub-floor harm.
- Primacy confound (disclosed + fixed; affects C2 Δ_interf/DiD, C3 constructed magnitudes, and C1 LOO
  [conservatively]); malignant over-sampling confound; base-floor on poison-alone; NO clean 2nd domain (SQL
  answer-key ceiling + struct self-verifying-flag brute-force oracle, both degenerate) -> scoped to ONE
  hand-built crypto-decode microbenchmark; observational follow-poison selection-confounded; theorem premises
  partly falsified -> demote Theorem to a scoped Proposition; CRT retracted (anti-conservative, one sentence);
  narrative reframed several times before pre-registration discipline.

## Planned paper structure
Problem -> estimator validated on helpful + positive control (destroy/exhaust detected; wrongflag missed) ->
the NEGATIVE (per-item poison ill-posed: LOO-mean-incl-0 + within-Spearman≈0 at-chance + noise floor) -> the
in-context effect is resource-mediated and order-confounded (no clean per-item value; further support for C1)
-> generality (observational ladder, deployed-M; Llama-only cross-arch) -> honest limits (incl. the M=1 sub-
floor rival). Theorem->Proposition; CRT/ladder/H5/struct+SQL degeneracy/over-sampling -> appendix/disclosure.

## Pre-write prose — DONE (2026-06-26)
- The full honest rewrite to the C1–C5 framing is committed (abstract, intro, §results2 title,
  method_causal, results_solution, results_body pilot, theorem→Propositions, discussion, conclusion).
  Two adversarial-critique iterations folded in: theorem reconciled to the reckonings (CRT demoted,
  A1 sub-floor, A3 step-detour-as-hypothesis, Theorem→Proposition), M=1 on-target/marginal estimand
  relabel, second-domain degeneracy disclosed, provenance numbers re-emitted from artifacts. Builds clean
  (16pp). Loop continues until the critique converges with no surviving falsehood.
