# Pre-registration — causal credit vs false-lesson propagation (lab-scale)

> Written 2026-06-01 on the laptop, BLIND to all lab-scale results, to bind the
> analysis before the data exists. The lab session may REFINE (add detail, fix a
> mis-specified estimator) but must NOT relax a hypothesis or stopping rule to fit an
> observed result, and must log any deviation here with a timestamp + reason.

## Hypotheses (each with its falsifier)
- **H1 — outcome credit banks poison.** The outcome-credit bandit assigns executed-but-
  recovered-from poison lessons a learned value > 0 (promotable). *Falsified if* the
  per-lesson mean Q for poison has a 95% CI that includes or sits below 0.
- **H2 — the free experiment flips it (primary result).** Re-scoring the SAME outcome-
  condition logs by IPW gives poison a NEGATIVE marginal value while true lessons stay
  positive: ATE(poison) < 0 < ATE(true), CIs non-overlapping. *Falsified if* the poison
  ATE CI includes 0, or overlaps the true-lesson CI.
- **H3 — propagation gap.** Under a promote-if-(value>0) gate, poison promotion rate is
  far lower under the causal value than under outcome Q, with true-lesson promotion
  preserved. Report as proportions with CIs over hundreds of lessons. *Falsified if* the
  two promotion rates' CIs overlap.
- **H4 — solve-rate parity is structural (predicted, not hidden).** In single-step
  retrieval the credit signals do NOT differ on immediate held-out solve rate. We
  PRE-COMMIT to reporting parity as a predicted result (see A.3 proposition), backed by
  a power/MDE analysis showing the study COULD have detected a gap of the pre-set size.
  *Note:* a significant solve-rate gap here would falsify the parity proposition and is
  reported as such — not suppressed.
- **H5 — propagation causes downstream harm (transfer experiment).** On a transfer split
  where the true lesson is ABSENT from retrieval, a memory built under outcome-credit
  promotion decisions has lower solve rate than one built under causal-credit promotion
  decisions. *Falsified if* Δsolve CI includes 0 — **a null here is a publishable result
  (it bounds where propagation matters), NOT a failure to fix.**
- **H6 — capability scaling.** From 4B→27B, recovery-masking rate rises (more poison
  recovered-from for free → outcome poison-Q rises toward true) while IPW separation is
  unchanged. *Falsified if* IPW separation degrades with capability, or recovery-masking
  does not rise.

## Primary vs secondary outcomes
- Primary: H2 (sign-flip) and H5 (transfer Δsolve). Everything else is secondary/support.
- Pre-specified estimators: SNIPW with the hardening already in `ope/causal_value.py`
  (clip [0.05,0.95], ESS<0.3n rejection, min 5 obs/arm, partial pooling toward the
  global per-node ATE with pseudo-count τ0); **per-lesson doubly-robust (AIPW)
  reported alongside** as the robustness anchor (NEW code — `ope.py`'s DR is
  whole-policy; a per-item DR/AIPW estimator is added to `causal_value.py`).
  Marginal inclusion propensity π_i(v) is Monte-Carlo estimated from the logged
  per-candidate scores + temperature (n_mc ≥ 400; the headline and every ablation
  re-uses the SAME ≥10 propensity MC seeds so MC noise is differenced out).
- **Two-level bootstrap (made exact here — clarifies the original one-line spec,
  does not relax it):**
  (i) *Per-lesson ATE CI* — for the H2 claim "ATE(poison)<0<ATE(true), per-lesson CI
  excludes 0", resample WITH REPLACEMENT over the runs contributing to that single
  lesson's treated/control arms (each lesson is observed in ~all runs, so this arm is
  well-powered even at small lesson counts), 10k resamples, percentile CI.
  (ii) *Label-level mean-ATE CI* — for the headline "poison vs true separation" and the
  promotion-rate gap, resample WITH REPLACEMENT over LESSONS (the unit of generalization),
  10k resamples; this is the CI that the n=4 pilot cannot make tight and that corpus
  scaling (≥150 lessons/label) exists to shrink. BOTH are reported; the label-level CI
  is the primary generalization claim.
- Solve-rate contrasts: paired bootstrap + exact McNemar on common (challenge,seed) items.

## Scale targets (the "irrefutable" bar — fix BEFORE running)
- ≥20–40 controlled families; **≥150 lessons per label** (helpful/poison/distractor).
- ≥1000 held-out challenges, ≥5 eval seeds (fresh random flags → zero contamination).
- Models: Qwen3.5-27B primary **+ ≥1 non-Qwen** (rule out family-specific spurious-reward
  artifacts). Report each model separately; no pooling across model families.

## Robustness ablations (each must leave the H2 sign-flip intact)
Exact grids (frozen here):
- Propensity Monte-Carlo: seeds {0..9} (≥10), n_mc ∈ {400, 1000}; report the poison/true
  label-level ATE sign across all seeds (sign-stability fraction must be 1.0).
- IPW clip range ∈ {[0.02,0.98], [0.05,0.95] (default), [0.1,0.9]}.
- ESS rejection threshold ess_frac ∈ {0.2, 0.3 (default), 0.5}.
- Propensity source ∈ {confounded-Q (current), Q=0/prior-only (free — `prior` already in
  the logged candidate pool), ε-random known-π slice (B.6, if logged)}.
- Estimator ∈ {SNIPW (default), per-lesson AIPW/DR}.
- Promotion-threshold θ sweep over [−0.2, +0.2] in steps of 0.02: show OUTCOME Q has NO θ
  that separates poison from true (poison-Q and true-Q both promote together at every θ),
  while the causal value HAS such a θ (a θ* admitting true but not poison).
Each ablation reports the label-level poison & true ATE with its bootstrap CI; the H2
sign-flip (poison CI upper bound < 0 < true CI lower bound, at the headline scale) must
survive every cell or the deviation is logged.

## Stopping rules (decide now; do not move the goalposts)
- **Data collection:** stop a campaign arm when per-label ATE CI half-width ≤ 0.03 OR the
  pre-set compute budget is hit — whichever first; report which.
- **Transfer experiment (H5):** pre-set the minimum meaningful Δsolve = 5 pp. Stop when the
  Δsolve 95% CI half-width ≤ 2.5 pp. Declare: harm if CI excludes 0 and lower bound ≥ 5pp;
  null if CI within ±2.5pp of 0; inconclusive otherwise (and say so). No optional stopping
  to chase significance.
- **Review loop:** stop when no reviewer raises a *blocking* critique (one that, if true,
  would change a conclusion); remaining critiques are logged as limitations, not erased.

## Anti-p-hacking commitments
Analysis code frozen before the headline run; any post-hoc analysis is labeled
exploratory. Negative/null results (esp. H5) are reported with equal prominence. All
numbers emit from `analyze_campaign.py`/`numbers_solution.tex` — no hand-entered figures.

## Deviation log
(Append dated entries here when the lab refines any of the above, with the reason.)

### 2026-06-01 — Phase 0 lab lock (refinements only; nothing relaxed)
- **Reproduction confirmed** (`python -m experiments.analyze_campaign --dir results/campaign_v2`,
  branch `lab-scale`): outcome poison-Q **+0.183**, true **+0.283** (both >0, H1 direction);
  observational ATE on the SAME outcome logs poison **−0.102**, true **+0.155**, distractor
  −0.054 (H2 sign-flip direction); promote-if-Q>0 admits poison **100%** under outcome.
  pytest `tests/` = 117 passed, 6 xfailed.
- **Promotion-rate headline number is 100%→0%, not 100%→25%.** The committed macro
  `\PromotePoisonObs` = promotion of poison under the causal value computed on the
  *outcome-condition logs* (`causal_ate_outcome`, the apples-to-apples cell that holds
  everything fixed but the credit signal) = **0%** (0/4). The "25%" / "1/4" in
  internal session notes (not included in this release) §2/§6 is the *different* cell `causal_ate_observational` (poison ATE
  −0.050 → 1/4). REFINEMENT: the headline promotion contrast is fixed as outcome-Q-on-
  outcome-logs (100%) vs causal-ATE-on-outcome-logs (0%); the observational-logs cell is
  reported as a secondary replication. At n=4 the 0% vs 25% gap is a single lesson — which
  is precisely why H3 is pre-committed to be reported as proportions with CIs over hundreds
  of lessons, not at n=4.
- **REPRODUCIBILITY BUG found + to-fix in Phase 1a:** `marginal_inclusion_propensities`
  is seeded with `hash(run_id) % 2**31`, and CPython's string `hash()` is randomized per
  process (PYTHONHASHSEED). Verified: 3 fresh `analyze_campaign` runs gave poison ATE
  −0.10157 / −0.09921 / −0.09545 and promotion 0% then 25%/25%/25% — i.e. the committed
  headline macro `\PromotePoisonObs{25\%}` does NOT deterministically reproduce (one
  near-zero poison lesson crosses 0 depending on the random hash seed; the sign-flip
  itself is stable, only the n=4 promotion rate is brittle). FIX: replace `hash()` with a
  stable hash (sha1) so propensities are deterministic, and report the ≥10-seed
  sign-stability ablation. This bug is itself the strongest motivation for H3-with-CIs at
  ≥150 lessons/label (a single near-zero lesson must not move a headline percentage).
- **Per-lesson DR is NEW code** (see estimators above); `ope.py::dr_estimate` is whole-policy.

### 2026-06-01 — design-panel refinements (5-lens expert panel; additions, nothing relaxed)
Pre-registered for the scaled run after a 5-lens design panel (causal-inference/OPE, CTF-realism,
statistics-power, agent-memory-novelty, adversarial-reviewer). Each is a refinement with a reason.
- **Difficulty = a GRADED step ladder (2→3→4 decode steps), reported PER BAND** with its own
  per-band label-level CI (a pre-registered sweep, not a chosen point — kills "you picked the
  difficulty"). Reason: 2-step alone reads as a toy and compresses helpful-ATE headroom; harder is
  more realistic AND gives poison real teeth, but must be calibrated in-band.
- **Per-family round budget = solution_length + slack**, slack FROZEN across bands. Headline
  **slack = 2** (the validated pilot point: 2-step + 4 rounds); robustness **sweep slack ∈ {1,2,3}**,
  H2 sign-flip must survive all three (a result holding only at slack=1 is knife-edge and flagged).
  Reason: keeps poison "teeth" uniform across difficulty so cross-family ATEs are comparable.
- **Headline estimand = binary SOLVE** for the graded run (invariant to the budget/step rescaling
  that the efficiency term e=−z(steps) introduces). Full reward R=o(1+λe) reported as secondary,
  with the RewardStandardizer **frozen across conditions** and **standardized within a
  solution-length stratum** (not the single crypto×difficulty cell). Do NOT pool reward-scale ATEs
  across difficulty bands. (Pilot, fixed budget: reward ATE is the valid headline; binary-solve ≈ 0
  there is the latent-harm signature, reported as such.)
- **Depth-matched poison**: every family's poison recipe has the SAME step count as its true recipe
  (right outer, wrong inner of equal length). Reason: a shorter poison could earn a negative ATE for
  the wrong reason (fewer steps), confounding "wrong lesson" with "shorter path." (Done in
  `benchmark/hard_ctf.py` grid; the hand-written core families have asymmetric poison — a documented
  PILOT-only caveat, not retro-fixed since campaign_v2 logs are frozen.)
- **Label-BLIND family inclusion filter**: retain a family iff its probe helpful-present solve ∈
  [0.3,0.7] (drop <0.25 floor / >0.85 ceiling), decided BEFORE computing any ATE or seeing
  helpful/poison labels. Reason: family retention must not be selection-on-the-outcome.
- **Differential-positivity audit (reported)**: per-label rate of `_hardened_ate` None-rejections;
  if helpful vs poison rejection rates differ by >0.1 the label-level mean is flagged as possibly
  selection-biased. Reason: floor families reject helpful more than poison → biased label mean.
- **Power re-allocation (statistics lens)**: between-lesson variance dominates, so cap at ~75–100
  IN-BAND lessons/label (≥75 already gives ATE-label CI half-width ≤0.03) and redirect freed compute
  to the **H5 transfer experiment: ≥600 paired (challenge,seed) items** (the only genuinely
  underpowered claim). Run a discordant-pair probe first to estimate the transfer disagreement rate.
- **Pre-registered TEETH positive control**: each retained family must show poison-only retrieval
  LOWERS solve vs true-only at its budget (else the mechanism is vacuous). Checked in the probe.
- **Staged run, gated**: calibration probe (~1–2k solves) must pass {in-band solve, teeth, poison
  ATE<0 at probe scale, non-differential rejection, byte-identical determinism re-check, non-Qwen
  tool-protocol check} BEFORE the multi-day run. No optional stopping.
- **External real-CTF track** (Cybench/NYU-CTF) for solve-rate EXTERNAL VALIDITY ONLY — explicitly
  NO per-lesson causal-label claim there; use the verifiable flag as the oracle to validate
  observational-IPW against active-LOO. Augment, do not pivot.
- **Prerequisite**: vectorize `marginal_inclusion_propensities` before the scaled ablations
  (pilot 6.7 min at n=4/label; ~150× is infeasible as-is).
- **Models available on this box** (HF cache): `Qwen3.5-27B-FP8` (H6 primary), `Qwen3.5-9B`,
  `Qwen2.5-7B-Instruct`, larger AWQ/FP8 MoEs; Ollama `mixtral:8x7b` (non-Qwen, satisfies the
  "≥1 non-Qwen" requirement). The Qwen3-4B-2507 that produced campaign_v2 is NOT cached, so
  any NEW controlled-corpus run uses a re-selected model (logged per run via `solver_model`);
  campaign_v2 stays the committed 4B anchor. Model selection may be researched/expanded
  (user-authorised 2026-06-01); each model family is reported separately (no pooling), and
  the H6 capability-scaling prediction is tested on whatever capability ladder is chosen
  (e.g. {7B/9B} → 27B), with the 4B anchor as the low rung.

## Addendum 2026-06-03 — refined primary estimand for H2 (logged deviation, NOT a relaxation)
**What changed & why.** The original H2 used the MARGINAL per-lesson ATE (effect averaged over all
runs in which the lesson was retrieved). On the 7B 35-family logs the marginal poison ATE is null
(+0.008 solve p=.46 / +0.021 reward p=.36). The 2026-06-03 benchmark audit (7 dims, 10 agents;
docs/RELATED_WORK.md, results/.../ontarget_analysis.*) confirmed this is a **dilution property of the
marginal estimand**, not inert/mislabeled poison: a lesson can only act on challenges from ITS OWN
family, and stochastic retrieval makes it on-target only ~9% of draws, so the marginal averages the
effect over ~91% irrelevant draws. Descriptively, poison-in-context solve is 0.074 on-target (n=636)
vs 0.115 off-target (n=6520).

**Refined primary estimand (pre-registered here BEFORE the replication/ladder data exists).** The
ON-TARGET (family-conditioned) per-lesson ATE: restrict each lesson's treated/control runs to
challenges of the lesson's own family, then apply the SAME SNIPW/AIPW estimator. Conditioning is on
challenge family, a **PRE-treatment** variable → effect-modification CATE, **no collider bias**
(distinct from the forbidden post-treatment conditioning on solve). H2 falsifier is unchanged
(poison on-target ATE CI must exclude 0 and not overlap the true-lesson CI). On the existing 7B logs
this gives poison −0.028 solve (p=.031) / **−0.069 reward (p=.001, CI[−.123,−.023])** and helpful
+0.164 / +0.346 — both CIs exclude 0. Effort-weighted reward gives the STRONGER poison signal (D1).

**Integrity guardrails.** (1) The MARGINAL H2 is STILL reported as the original pre-commitment — the
null is disclosed, not suppressed; on-target is presented as a refinement that EXPLAINS it. (2) On-
target was discovered POST-HOC on the 35-fam logs, so it is treated as a HYPOTHESIS there and is
pre-registered here as the primary ONLY for the FRESH replication run (distinct world-seeds,
bug-fixed benchmark) and the capability ladder, whose data did not exist at this writing. (3) It is a
single theory-motivated conditioning (the lesson's own family), not a swept subgroup — no garden of
forking paths. A headline claim requires on-target to REPLICATE on the fresh run with CI excluding 0.
- Estimator is unchanged (`ope/causal_value.observational_lesson_values`); on-target slice computed in
  `experiments/ontarget_analysis.py` and emitted by `experiments/lab_pipeline.py` for every run.

## Addendum 2026-06-11 — boundary-mapping experiments (pre-registered BEFORE any run; RIGOR_LEDGER T1-1 / T2-2 / T2-3)

### T1-1 — Malignant-poison positive control (the estimator must be shown able to detect harm)
**Motivation.** Every poison studied so far is benign-wrong (outcome-equivalent to no recipe, by the
measured A1 floor). The harm-detector (on-target SNIPW + o-keyed exact CRT) has therefore never been
shown to detect ANY harm — "poison undetectable" is not yet distinguishable from "estimator blind."
**Design.** Add a `_MALIGNANT` lesson track mirroring the poison machinery, three mechanism-resolved
sub-arms: **M-destroy** (recipe includes an artifact-corrupting step, e.g. in-place truncation of
data.bin), **M-exhaust** (recipe prescribes an unbounded brute-force loop that burns the
solution_length+2 step budget), **M-wrongflag** (recipe ends by submitting a deterministic wrong
flag, consuming the submit). Rung: 14B-AWQ (cached, 1 GPU). Probe gate FIRST (n=4/family, ~1–2k
solves): each malignant recipe must demonstrably drop solve|cited below base before scaling — if a
sub-arm has no teeth at probe, it is redesigned BEFORE the campaign, not after. Full run: 12–16
in-band families × 8 instances × 3 seeds × 8 epochs (≈1 campaign-night), same estimator config,
same analysis stack (ontarget_analysis + o-keyed CRT + cite_rate + a1_contrast).
**Pre-registered predictions (one-sided).**
(a) on-target SNIPW solve-ATE < 0 with 95% CI excluding 0, in EACH sub-arm;
(b) solve|cited < b_f (sub-floor — violating A1 by construction), family-stratified contrast CI < 0;
(c) o-keyed exact CRT frac_sig_p05 ≥ 0.5 for malignant lessons (vs the benign 0.14–0.25);
(d) the execution verifier flags M-destroy and M-wrongflag 100% (M-exhaust is out of verifier scope
    if its recipe re-executes without side effects — report whichever holds).
**KILL CONDITION (pre-committed).** If the estimator fails to flag M-destroy or M-exhaust (CI
includes 0 AND CRT frac_sig < 0.25), the paper's identifiability claim is UNSUPPORTED as scoped and
must be weakened to "untested estimator sensitivity" — reported, not suppressed.

### T2-2 — Name-leak A/B at 3B (cautionary-tale attribution)
**Motivation.** The demonstrated name-leak kill is the 7B 35-family pair (−0.0693 p=.00067 →
−0.0039 p=.938 post-fix). The 4B pilot (campaign_v2) used the same leaked names but was never
re-run post-fix; its attribution is retrospective. **Design.** Two cells at 3B (cached), identical
corpus/world-seed/config, single toggle: canonical_name = leaky (`{family}_recipe`/`{family}_wrong`)
vs neutral (content-hash). **Predictions.** Leaky cell: spurious on-target poison SNIPW negative
with CI excluding 0. Neutral cell: null (CI includes 0). Δ between cells = the measured artifact
magnitude. **Falsifier.** If the leaky cell is ALSO null at 3B, the artifact is capability-dependent
and the cautionary tale must say so (the 7B pair remains the demonstrated instance).

### T2-3 — Retrievable distractor arm (ordinary-noise control)
**Motivation.** Current distractors (steghide/binwalk/etc., category "misc") are never on-target →
0 distractor nodes in headline node_values. The benign-poison null needs an ordinary-noise
comparator. **Design.** ≥30 category-matched (crypto) real-but-irrelevant technique lessons
(e.g., RSA factoring, Vigenère) that enter the candidate pool and get sampled but can never match
a trace signature; ride along in the T1-1 campaign as a third label. **Predictions.** Distractor
on-target ATE ≈ 0 with CI narrower than poison's (no displacement teeth); promotion rate under
outcome credit below poison's. **Falsifier.** A significant negative distractor ATE would indicate
context-clutter harm — reported as a real (and theorem-relevant) finding, not suppressed.

**Integrity guardrails (all three).** These entries are committed to git BEFORE any of the runs
launch; analysis uses the frozen estimator config and the same multiplicity plan as the ladder
(per-rung primary = on-target solve SNIPW; everything else exploratory). Nulls are publishable
results, not failures.
