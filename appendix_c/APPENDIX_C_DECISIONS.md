# APPENDIX_C_DECISIONS.md — parameter choices for the Appendix C latent-poison probe

Provenance ledger for the probe reported in Appendix C of *What a Verifiable Reward Can
and Cannot Credit in Self-Extending Agent Memory*. Each entry records what changed, from
what to what, why, and **when it was decided relative to the contrast it could affect**.

These four decisions (D-01--D-04) govern the Appendix C probe. Every one was fixed
**before** the comparison it could influence was observed. No analysis reported in
Appendix C was chosen after seeing a result it could affect.

This file is extracted from a larger working ledger kept for a separate course project;
entries beyond D-04 concern experiments that are **not** part of this paper and are not
reproduced here.

---

## D-01 — Solver model: Qwen3.5-27B-FP8 → Qwen2.5-7B-Instruct
**When:** 2026-07-26, before any experiment ran.
**Why:** hardware, not preference. FP8 inference requires Hopper/Ada (sm_89+); both cards
are RTX A5000 (Ampere, sm_86). The FP8 checkpoint cannot be served natively here.
Qwen2.5-7B-Instruct is on disk, was used by the prior work's lab campaigns, and fits one
card — which also lets both GPUs run independent workers.
**Affects:** all real-model results. Scoping consequence stated in the report: claims are
about this solver, not about the capability ladder.
**Alternative retained:** Qwen2.5-32B-Instruct-AWQ (TP=2, verified working) as a
capability check if the schedule allows.

## D-02 — Solver `max_output_tokens`: 384 → 768
**When:** 2026-07-26, after the smoke run, before any reported contrast.
**Why:** harness defect, not result-shaping. At 384 the 7B solver frequently truncated
mid-JSON, producing repeated *"Could not parse LLM response as JSON, nudging…"* rounds
that consumed the round budget without issuing a command. That noise is unrelated to the
lesson under test and inflates step counts for **every** condition, which would corrupt
the step-contrast measurement depends on.
**Decided blind to any contrast:** the change was made from parse-failure counts in the
smoke log, before any latent-vs-helpful comparison existed.
**Affects:** all real-model runs from calibration onward; the 384-token smoke run is not
reported as a result.

## D-03 — Round budget: generous for the step probe
**When:** 2026-07-26, at design time.
**Why:** the step probe measures the *step cost* of a longer route, so the budget must be
generous or the cost is censored at the cap and reads as zero. A flat 8 rounds was chosen
against a corpus whose correct methods run 1--4 rounds. (A separate campaign, not reported
in this paper, needed the opposite regime — a tight budget — and the two were deliberately
kept apart.) **Note in hindsight:** 8 rounds proved *not* generous enough. Appendix C
reports that 55 of the 104 helpful and detour runs still exhaust it (196/260 across all
five conditions), that the marginal round contrast is
therefore confounded with solve status, and that no step-cost bound can be claimed from
this design. The reasoning in this entry was sound; the budget was not.
**Affects:** the two regimes are reported separately; step deltas are not
transplanted across budgets.

## D-04 — In-band family selection by calibration probe
**When:** 2026-07-26, before the campaigns.
**Why:** the prior work documents that several models were "harness-FLOORED" on this
corpus. A family where the solver never solves (floor) or always solves (ceiling) carries
no signal about a lesson either way. Families are selected by a `none` vs `helpful` probe
**that never sees a latent-poison condition**, so selection cannot bias the latent arms.
**Selection rule, fixed in advance:** keep families with `helpful` solve rate strictly
above `none`, and `none` below ceiling.
**Affects:** which families enter the step probe; the selection artifact and rule are
reported. Appendix C additionally reports that four of the selected families re-measured
at the floor, and that the selection bias runs in favor of the null it reports.
