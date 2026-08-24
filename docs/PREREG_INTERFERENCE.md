# Pre-registration — Confirmatory replication of the crypto "interference" effect in a fresh second domain

> **STATUS: DRAFT for freezing.** Written 2026-06-25, BEFORE the new second-domain corpus is built,
> BEFORE any solver re-rollout under that domain, and BEFORE any contrast on it is computed. Once
> committed to git, the hypotheses, the single primary endpoint, the sample sizes, the analysis
> estimator, the family-inclusion rule, the equivalence margin, and the stopping rule below are
> FROZEN. The run may add detail or fix a mis-specified estimator but must NOT relax a hypothesis,
> move a threshold, change n, or add families after seeing the contrast. Every deviation is logged
> dated + reasoned in §10.
>
> **Why this exists.** The project's thesis was reframed four times, each time absorbing a failed
> result, and the one generalization prereg that bore on it (T2-7: "SQL replicates the poison null")
> FAILED and was re-narrated rather than reported as a failed prediction. The single real, well-
> measured context-dependence signal — the crypto "interference" effect (helpful+poison < helpful) —
> was discovered post-hoc and has never been pre-registered. The first attempted second domain (SQL)
> was DEGENERATE by construction (its "helpful" lesson was the literal answer key → 100% ceiling →
> the interference contrast was a sentinel `p=2.0`, undetectable by design). This document converts
> the fit-to-data interference finding into a confirmatory, falsifiable prediction in a fresh,
> non-degenerate domain.

---

## 0. Established prior findings this confirms-or-falsifies (the anchor, all VERIFIED)

Source: the project's correction log (not included in this release), "RESOLVED 2026-06-24" + "RECKONING 2026-06-25". All from the
constructed-context gold-standard LOO (`experiments/constructed_loo.py`) on the real 14B solver.

- **Crypto interference (the effect to replicate):** {helpful+poison} − {helpful}
  = **−0.311, 95% CI [−0.427, −0.195], n=41 families, p<.0001.** Forcing the wrong recipe alongside
  the correct one degrades solve.
- **helpful-alone lift = +0.805** (none 0.040 → helpful 0.845). There IS headroom; helpful is NOT at
  ceiling. This is the property the SQL domain lacked.
- **poison-ALONE (crypto) = +0.052** (0.091 vs 0.040): recoverable, not worse than no lesson.
- **Within-poison LOO-vs-cheap-SNIPW Spearman = 0.028** (sign-agree 18/28 ≈ chance). The cheap
  observational on-target SNIPW does NOT validly rank the per-item poison effect. (Within-helpful
  Spearman = 0.64 — the cheap estimator IS validated on helpful.)
- **Malignant positive control:** destroy/exhaust detected; **wrongflag NULL (combined SNIPW
  +0.0044, p=.69) — 1 of 3 arms MISSED.**
- **SQL second domain was DEGENERATE:** helpful = answer key → solve 1.00 on all 16 families →
  interference contrast = sentinel `p_two_sided=2.0`. Not a finding; a design failure.

The NEW domain (designed by a separate agent) teaches a **METHOD, not an answer**, with helpful-alone
solve targeted into a measurable, non-ceiling band (~0.5–0.85). This is the precondition that makes
the interference contrast even measurable.

---

## 1. Hypotheses — exact directional predictions, quantitative thresholds, falsifiers

### H1 (PRIMARY) — Interference replicates in the fresh domain.
On fresh solvable instances of the new domain, the constructed-context contrast
**Δ_interf = solve({helpful, poison}) − solve({helpful})** is negative.

- **CONFIRM** iff the family-cluster bootstrap 95% CI for Δ_interf has **upper bound < 0** (two-sided
  α=.05) **AND** the point estimate **≤ −0.15** (the pre-registered minimum meaningful effect, MME).
- **FAIL-TO-CONFIRM (the interesting falsification)** iff the CI **includes 0**, OR the point estimate
  is between 0 and −0.15 with a CI excluding −0.15 (a real-but-trivial effect). Either means
  **context-dependent interference does NOT generalize beyond crypto** — context-dependence is a
  crypto-specific artifact, and the honest claim shrinks to "demonstrated in one domain."
- **REVERSAL (hard falsification)** iff the CI lies **entirely above 0** (helpful+poison *better* than
  helpful): the interference story is wrong in sign; we report it as a failed prediction.

> **RISK FLAG (this can genuinely fail, by design).** The crypto mechanism is "a self-evidently-wrong
> decode confuses the agent into abandoning the right recipe." In a method-teaching domain the agent
> may *disambiguate* the two methods (as SQL's agent did once helpful was present, before ceiling
> masked it: poison_in_context there was +0.00). If the new domain's two methods are easily told
> apart, H1 fails. We pre-commit to reporting that as a negative result that BOUNDS interference to
> domains with confusable sibling methods — a real finding, not a failure to fix.

### H2 — poison-ALONE null (equivalence, NOT "p>.05").
**Δ_alone = solve({poison}) − solve({none})** is practically equivalent to zero (poison alone is
recoverable). Tested by **TOST** (§5), not by failing to reject.

- **CONFIRM-NULL** iff the 90% CI for Δ_alone lies entirely within the equivalence margin **±0.10**
  (equivalently, both one-sided tests reject at α=.05).
- **FAIL** iff either TOST bound is not rejected (CI escapes ±0.10) → poison-alone is NOT recoverable
  in this domain (it commits a wrong answer, SQL/wrongflag-style); report as a domain difference.
- Predicted point estimate: small positive, ~+0.00 to +0.10 (crypto was +0.05).

### H3 (positive control) — helpful-lift is large and non-ceiling.
**Δ_help = solve({helpful}) − solve({none})** > 0, and the helpful-alone solve rate falls in the
pre-declared **non-ceiling band [0.50, 0.85]**.

- **CONFIRM** iff Δ_help CI lower bound **> +0.20** AND mean solve({helpful}) ∈ [0.50, 0.85].
- **VALIDITY ABORT (not a result — a degeneracy kill)** iff mean solve({helpful}) **> 0.90** (ceiling,
  the SQL failure) OR **< 0.20** (floor, no headroom). If this fires, the domain is non-diagnostic and
  the interference contrast is **uninterpretable**; STOP, do NOT report H1, return the domain to design.
  This gate is checked on the FAMILY-INCLUSION probe (§4) BEFORE the confirmatory rollout, so it cannot
  be a post-hoc exclusion of the primary.

### H4 (robustness control, RUN ONLY IF the malignant track is ported) — malignant interference is large.
If the new domain ports the destroy/exhaust malignant arms, **Δ_malig = solve({helpful, malignant}) −
solve({helpful})** is strongly negative (≤ −0.40), confirming the contrast machinery can detect a
known-large harm in THIS domain (guards against "the contrast is just insensitive here").

- **CONFIRM** iff Δ_malig CI upper bound < 0 AND point ≤ −0.40, for destroy and exhaust.
- This is a machinery check, NOT evidence for H1; it is explicitly secondary.

### Crisp overall falsification of the thesis claim
The paper's generalization claim is: *"a benign-false lesson's harm is context-dependent — recoverable
alone, but interfering when co-present with the correct lesson — and this is not a crypto artifact."*
**This claim is FALSIFIED if H1 fails-to-confirm or reverses in the fresh domain.** A FAIL means we
report the effect as demonstrated in crypto only, with the new domain as a negative boundary case.

---

## 2. Primary vs secondary/exploratory endpoints (declared to kill forking paths)

- **THE ONE PRIMARY ENDPOINT: H1 / Δ_interf** = solve({helpful, poison}) − solve({helpful}), binary
  solve, constructed-context LOO, family-cluster bootstrap, on the fresh domain's in-band families.
  ONE estimand, ONE contrast, ONE estimator, ONE test.
- **Secondary (confirmatory-support, multiplicity-corrected together, §4):** H2 (poison-alone TOST),
  H3 (helpful-lift positive control).
- **Tertiary / robustness (reported, NOT in the confirmatory family):** H4 malignant machinery check;
  the within-domain LOO-vs-cheap-SNIPW Spearman (expected ≈ chance for poison, replicating 0.028);
  any reward-channel (R_final) version of Δ_interf; any per-family forest/heterogeneity (I²) plot.
- **Exploratory (labeled as such, never headlined):** anything subgroup, any post-hoc slice, any
  alternative context set (e.g. {helpful, poison, distractor}), any cross-domain pooled estimate.
  Cross-domain pooling is FORBIDDEN as a primary (the prereg reports crypto and the new domain
  separately, exactly as the model-family no-pooling rule in the original PREREG).

---

## 3. Sample size / power (fixed in advance; reasoned)

**Unit of analysis = the family** (the cluster; bootstrap resamples families). The crypto interference
was −0.311 over n=41 families. We power for the SMALLER MME of **−0.15** at **α=.05 two-sided**,
**90% power**.

**Between-family SD.** The crypto interference CI was [−0.427, −0.195], half-width ≈ 0.116 over n=41.
A 95% percentile-bootstrap half-width ≈ 1.96·SE ⇒ SE ≈ 0.059 ⇒ between-family **SD ≈ 0.059·√41 ≈
0.38**. This is the constructed-context family-level SD of the {helpful}→{helpful+poison} difference;
it is large because some families interfere hard and some not at all (heterogeneous). We use **σ_fam =
0.40** as the planning SD (rounded up; conservative).

**Families needed (one-sample family-level t/bootstrap on the per-family Δ):**
n ≈ (z_.975 + z_.90)² · σ² / δ² = (1.96 + 1.2816)² · 0.40² / 0.15²
   = 10.51 · 0.16 / 0.0225 = 10.51 · 7.111 ≈ **74.7 → n_families = 75.**

- A cross-check at the *observed* crypto effect (−0.31) needs only n ≈ (3.24·0.16)/0.0961 ≈ 5
  families — i.e. if the new domain interferes as hard as crypto, even a small run detects it. The
  n=75 budget buys power against a HALVED effect, which is the prudent confirmatory target.
- **n_families = 75** (declared). If the new-domain corpus cannot yield 75 in-band families, the
  fallback floor is **n_families = 50** (power ≈ 0.78 at δ=−0.15, σ=0.40; power ≈ 0.99 at δ=−0.25),
  and the achieved MDE is reported per §4/§10 — but n is fixed at the chosen value BEFORE the contrast,
  never grown after.

**Instances per family & the per-family Δ estimate.** Each family's Δ_interf is itself a mean over
fresh solvable instances × contexts × seeds. To keep the within-family component of variance well
below the between-family σ_fam=0.40 (so the family is genuinely the limiting unit), per family:
- **n_instances = 8** fresh solvable instances (distinct random secrets/inputs),
- **n_seeds = 3** solver seeds per (instance, context),
- ⇒ **24 rollouts per context per family**; the binary-solve SE within a family at p≈0.5 is
  √(0.25/24) ≈ 0.10 per context, ≈0.14 for the paired difference — comfortably below σ_fam, so
  between-family heterogeneity dominates as required (matches the crypto variance decomposition:
  label-level SE ≈ between/√n_fam, within ≈ 0). Contexts rolled per family: {none}, {helpful},
  {poison}, {helpful, poison} (+{helpful, malignant} only if H4 is run).
- **Total primary rollouts** ≈ 75 families × 4 contexts × 8 inst × 3 seeds = **7,200** (≈ +1,800 if
  the malignant H4 fifth context is included). This is ~1 campaign-night at 14B, consistent with the
  crypto constructed-LOO footprint.

**Seeds for the propensity/MC machinery** are NOT relevant here — constructed-context LOO uses a
FORCED context (no retrieval sampling, no propensity weights), so there is no π-reconstruction noise.
The only randomness is solver temperature (the 3 solver seeds) and the bootstrap resampling seed.

---

## 4. Analysis plan — frozen in advance

**Estimator (primary).** For each in-band family f, compute the constructed-context delta
  Δ_f = mean_solve_{inst,seed}({helpful, poison}) − mean_solve_{inst,seed}({helpful}),
both arms rolled on the SAME fresh instances (paired within instance×seed where the harness allows;
otherwise within family). The label-level estimate is the **unweighted mean over families**
  Δ_interf = mean_f Δ_f.

**CI method.** **Family-cluster percentile bootstrap**, 10,000 resamples, resampling FAMILIES with
replacement (each resampled family carries all its instance×seed rollouts), recompute Δ_interf per
resample, take the 2.5/97.5 percentiles for the two-sided 95% CI. Report ALSO the pre-registered
**one-sided** test (H_a: Δ_interf < 0): one-sided p = fraction of bootstrap replicates ≥ 0. The
one-sided test is the directional confirmatory test; the two-sided CI is reported for honesty and is
the CONFIRM gate in §1 (we use the two-sided CI excluding 0 as the bar, which is stricter than the
one-sided p — deliberately conservative).

**Ceiling/floor family handling (THE selection rule, applied BEFORE the contrast).**
A family is RETAINED iff, on the LABEL-BLIND inclusion probe (run before the confirmatory rollout and
before any Δ is computed), its **helpful-alone solve ∈ [0.20, 0.90]**.
- Families with helpful-alone solve **> 0.90** (ceiling — the SQL failure mode) are DROPPED: with no
  headroom, {helpful+poison} cannot show degradation and the contrast is undefined-by-construction.
- Families with helpful-alone solve **< 0.20** (floor) are DROPPED: helpful itself barely works, so
  the interference contrast has no base to subtract from.
- This rule conditions ONLY on the helpful-alone arm, which is a PRE-treatment (does-not-involve-
  poison) quantity → it is NOT selection-on-the-primary-outcome (the primary outcome is the
  helpful+poison vs helpful *difference*). It mirrors the original PREREG's label-blind [0.3,0.7]
  inclusion filter, widened to [0.2,0.9] to match the broader target band.
- The probe is a SEPARATE rollout set (its own fresh instances) so that retained families' confirmatory
  Δ is not computed on the same data used to select them (no selection-on-noise feedback).
- Report: n_probed, n_retained, and the helpful-alone solve distribution, so the exclusion is auditable.

**Multiplicity correction across the declared confirmatory endpoints.** The confirmatory family =
{H1 (primary), H2 (poison-alone TOST), H3 (helpful-lift)}. H1 is tested at its full α=.05 as the
single primary (no correction borne by the primary). H2 and H3 are corrected together by
**Holm–Bonferroni across the 2 secondary tests** (or BH if more secondaries are added). H4 and all
tertiary/exploratory endpoints are OUTSIDE the correction family and are labeled exploratory. There is
exactly ONE domain and ONE primary contrast, so there is no rung/label/estimator/key/scope grid to
inflate (the explicit lesson from RIGOR_LEDGER T1-5 and the n=15 episode).

**Reward channel.** Binary SOLVE is the primary estimand (invariant to step/budget rescaling). The
R_final effort-weighted version of Δ_interf is reported as tertiary robustness only, standardized
within the new domain's own solution-length stratum, never pooled with crypto.

**Frozen code.** The contrast is emitted by the existing `experiments/constructed_loo.py` path (or a
domain-parameterized variant), and the final number flows into the paper via the macro emitter
(`experiments/emit_paper_numbers.py`, RIGOR_LEDGER T0-3) — **no hand-typed figures**.

---

## 5. Equivalence vs difference (the poison-alone null is a TOST, not p>.05)

H2 (poison-alone recoverable) is a claim of ABSENCE of harm and MUST be an equivalence test.

- **Procedure:** Two One-Sided Tests (TOST) on Δ_alone = solve({poison}) − solve({none}), family-
  cluster bootstrap, **equivalence margin Δ_eq = ±0.10** on the binary-solve scale.
- **Margin justification:** 0.10 is (i) below the −0.15 MME of the primary interference effect (so
  "alone is equivalent-to-zero" and "in-context is meaningfully negative" are cleanly separable
  claims), and (ii) of the order of the crypto poison-alone point estimate (+0.05), so the test is
  neither trivially passable nor impossibly strict. It is also the original PREREG's order-of-magnitude
  for a "practically null" per-item effect.
- **Decision:** CONFIRM-NULL iff the 90% CI for Δ_alone ⊂ (−0.10, +0.10) (both one-sided bootstrap
  tests reject at α=.05). Otherwise the null FAILS and we report poison-alone as non-recoverable in
  this domain (a real domain difference, e.g. a committed-wrong-answer mechanism like SQL).
- We will NEVER write "p>.05 ⇒ poison alone is harmless." The asymmetry between H1 (a difference test)
  and H2 (an equivalence test) is deliberate and frozen.

---

## 6. What to honestly report from EXISTING data, regardless of the new run

These go in the paper as established facts whether or not H1 confirms; they keep the paper self-
consistent and pre-empt the reviewer findings already on record (RIGOR_LEDGER RECKONING 2026-06-25):

1. **Within-poison LOO-vs-cheap Spearman = 0.028** (sign-agree 18/28 ≈ chance), vs within-helpful 0.64.
   Report WITHIN-label, not the aggregate Spearman 0.78 (which is between-cluster separation /
   Simpson's paradox). The headline must be: *on the exact contrast the paper is about, the cheap
   observational SNIPW does NOT rank the per-item poison effect.* (Artifact: `active_loo_within_label.json`.)
2. **The poison "true zero" is over-claimed and will NOT be stated as zero.** It rests on one n=28 LOO
   with CI [−0.006, +0.098] = fail-to-reject. Against it stand observational SNIPW −0.057 (p=.001, 14B)
   / −0.064 (p=.005, 32B) and M=1 −0.0745 (p<.001), with constant magnitude and power-gated
   significance across the M-sweep (the textbook signature of a real-but-underpowered effect). Honest
   wording: *"the per-item poison estimand is ill-posed/uninformative (within-Spearman ≈ 0) and
   context-dependent; observational reads mildly negative; we do NOT claim a true zero."*
3. **The malignant wrongflag arm is NULL** (combined SNIPW +0.0044, p=.69). State plainly: the
   positive control detects destroy/exhaust but MISSES wrongflag — the malignant arm most analogous to
   a committed-wrong-answer. "Estimator detects malignant harm" is true for 2 of 3 arms only.
4. **The SQL second domain was DEGENERATE** (helpful = answer key → 1.00 ceiling on all 16 families →
   interference sentinel p=2.0). Report it as a design failure that MOTIVATED this prereg, not as
   "SQL shows no interference." The crypto −0.31 is the one real, well-measured context-dependence
   signal in the prior data.
5. **Narrative-drift disclosure.** The thesis was reframed four times; context-dependence and the
   constructed-LOO estimands were never pre-registered before this document. State that this prereg is
   the first frozen test of the generalization claim, and report its result — confirm OR fail — as the
   confirmatory outcome.

---

## 7. Stopping rule (prevents "add families until significant" — the prior sin)

- **n_families is FIXED at 75 (fallback floor 50) BEFORE the confirmatory rollout**, chosen in §3 from
  the power calc. Families are drawn and frozen by the label-blind inclusion probe (§4); the retained
  set is committed (its family IDs written to git) BEFORE any Δ_interf is computed.
- **No optional stopping, no peeking-then-extending.** The contrast is computed ONCE on the full frozen
  retained set. We do NOT compute Δ_interf, observe a near-miss, and then probe more families.
- If the frozen n yields a CI that straddles the CONFIRM/FAIL boundary, the result is reported as
  **INCONCLUSIVE at the pre-set n** (with the achieved CI half-width and MDE), exactly as the original
  PREREG's H5 three-way {harm / null / inconclusive} rule. Inconclusive is a publishable outcome; it is
  NOT a license to add data.
- **Sequencing gate (cannot game the primary):** the only quantity that can ABORT before the
  confirmatory run is the H3 validity check (helpful-alone ∈ [0.20, 0.90]) on the inclusion probe — a
  pre-treatment, primary-blind quantity. If it aborts, the domain returns to design (it is non-
  diagnostic, like SQL); we do NOT proceed and then exclude families to rescue significance.
- The bootstrap seed and all solver seeds are fixed and logged; the analysis is deterministic and
  byte-reproducible from the frozen logs.

---

## 8. One-paragraph summary (the freeze)

We will roll the real 14B solver on **75 in-band families** (fallback floor 50, fixed in advance) of a
fresh method-teaching second domain whose helpful-alone solve sits in **[0.20, 0.90]** (validity-gated,
primary-blind), under forced contexts {none, helpful, poison, helpful+poison}, **8 fresh instances × 3
seeds** each. The **single primary endpoint** is **Δ_interf = solve({helpful,poison}) −
solve({helpful})**, binary solve, family-cluster percentile bootstrap (10k), two-sided 95% CI plus the
pre-registered one-sided test. We **CONFIRM** the generalization of context-dependent interference iff
the two-sided CI upper bound < 0 AND the point ≤ **−0.15**; we report **FAIL** (context-dependence is
crypto-specific) iff the CI includes 0 or the effect is trivial, and **REVERSAL** iff the CI is wholly
positive. Secondary: poison-alone **TOST** (margin **±0.10**) and the helpful-lift positive control
(Holm-corrected together). We report the within-poison Spearman ≈ 0 (estimator cannot rank poison),
the wrongflag malignant miss, and the SQL degeneracy regardless of outcome. n is fixed; no optional
stopping. **H1 can genuinely fail** — if the new domain's sibling methods are easily disambiguated, the
interference vanishes — and we pre-commit to reporting that as a real negative boundary result.

---

## 9. Risk register (places this prereg can honestly fail — that is the point)

| Risk | Why it could happen | What we report if it fires |
|---|---|---|
| **H1 null** (CI includes 0) | Method-teaching agent disambiguates the two methods (SQL-in-context was +0.00 before ceiling masked it) | Context-dependent interference is crypto-specific; bound it to confusable-sibling domains. |
| **H1 reversal** (CI > 0) | A wrong-but-related method acts as a useful hint | Failed prediction, reported as such; interference story wrong in sign. |
| **H3 validity abort** | New domain ceilings (>0.90) or floors (<0.20) like SQL | Domain non-diagnostic; back to design; H1 NOT reported (no rescue exclusions). |
| **H2 null fails** (Δ_alone escapes ±0.10) | New-domain poison commits a plausible wrong answer (SQL/wrongflag mechanism) | Poison-alone non-recoverable here → a domain difference, not a contradiction. |
| **Inconclusive at n=75** | σ_fam under-estimated / effect between −0.15 and the CI noise | Report inconclusive + achieved MDE; do NOT add families. |

---

## 10. Deviation log (append dated entries; nothing relaxed)

*(empty at freeze — append here when the run refines any specification, with reason and timestamp.)*
