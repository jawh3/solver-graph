# PROTOCOL — briefing for producing an extended abstract or poster from this work

**Read this file first.** It is written for an AI assistant (or a person) who has this
bundle and nothing else, and who has been asked to produce a 1-page extended abstract, a
conference poster, a talk, or a summary. It states what the paper claims, what it
deliberately does *not* claim, how every experiment was run, and where each number comes
from. The manuscript itself (`main.pdf`, `main.tex`) is the authority; this file is the
map and the guard-rails.

---

## 0. The single most important instruction

**This is a negative-result paper, and its value is the negative result.** The natural
instinct when posterizing it is to lead with the positive finding — "we built a
causal-credit estimator that recovers lesson value for free from logged retrieval
randomness!" That framing is *wrong for this paper* and the authors have repeatedly
rejected it.

The headline is: **the per-item harm of a benign-false lesson is not identifiable from a
verifiable binary outcome.** The free causal-credit construction is the *instrument* that
establishes the boundary, not the contribution. A poster that leads with the estimator
misrepresents the work.

Second guard-rail: **"non-identifiable" means "not per-item rankable," not "harmless."**
A small sub-floor harm *does* persist when poison is actually executed. Any headline of
the form "false lessons don't hurt" is a misreading and the paper explicitly disowns it
(`main.tex`, Discussion).

---

## 1. Bibliographic front matter

| Field | Value |
|---|---|
| **Title** | What a Verifiable Reward Can and Cannot Credit in Self-Extending Agent Memory: A Non-Identifiability Result for False-Lesson Harm |
| **Authors** | James Hugglestone (first author); Samuel Jacob Chacko (second author); Xiuwen Liu (advisor) |
| **Affiliation** | Florida State University |
| **Status** | Preprint. arXiv category: cs.LG primary, cross-list cs.AI and cs.CR |
| **Length** | 22 pages, two-column, 10pt |
| **Code / artifacts** | `https://github.com/jawh3/solver-graph` — **verify this is public before printing it on a poster or QR code** |
| **License** | arXiv perpetual non-exclusive licence 1.0 (deliberately *not* CC-BY). Repo: MIT for code/data; paper text all-rights-reserved |
| **Plain abstract** | `abstract_plain.txt` (1,918 chars; arXiv's metadata limit is 1,920 — there is ~2 characters of headroom, so do not add words to the abstract) |
| **AI-use disclosure** | The manuscript carries an explicit statement that Claude (Anthropic) was used for code and manuscript drafting. If you produce derived material, carry that disclosure forward. |

---

## 2. The claim ledger — what may and may not be said

This is the authors' own honest claim set. **Do not state a claim more strongly than it
appears here.**

### C1 — HEADLINE (negative)
The per-item harm of a *benign-false* ("poison") lesson — one that names a wrong method
which the agent executes and then recovers from — is **not identifiable from binary
outcomes**.

- Assumption-light active leave-one-out gold standard, mean poison marginal
  **+0.045, 95% CI [−0.006, +0.098], n=28**. This is a *fail-to-reject* — the estimand is
  ill-posed/uninformative. **Never describe it as "zero."**
- The cheap observational SNIPW estimator does not *rank* the per-item effect:
  within-poison Spearman **≈ 0** (+0.072 primary; bootstrap CI [−0.35, +0.49]).
  Sign-agreement among rankable nodes **8/18, P=0.76 — at chance**, not anti-correlated.
- Contrast that with the same estimator on *helpful* lessons: within-helpful Spearman
  **0.64**, rankable **30/30**. The estimator is validated — on large effects only.
- The honesty note that makes C1 a real result rather than an estimator failure: the
  *gold standard itself* cannot rank the small poison effects at feasible n (per-node LOO
  SE ≲ 0.20 against a between-node SD of 0.14). Neither the cheap estimator nor the
  expensive one ranks it.

### C2 — DEMOTED, do not headline
The in-context harm is resource/co-presence mediated rather than method-conflict
specific. The supporting DiD interaction inherits a **primacy confound** (its rollouts
were poison-first, from before the ordering fix, and were never re-run
order-randomized). Read C2 as further support for C1, not a standalone result. If a
poster has no room, **cut C2 entirely.**

### C3 — POSITIVE CONTROL (this is what delimits the boundary)
The same estimator *does* detect robustly-destructive lessons:
- `destroy` −0.016 (p=.010) and `exhaust` −0.017 (p=.0165) — both detected, both
  floor-breaching.
- `wrongflag` (a confidently-wrong answer) returns a null from outcomes (+0.004, p=.69) —
  but **do not call this a demonstrated blind spot.** The agent followed that recipe on
  only 11 of 453 sampled runs (follow rate **0.024**, vs 0.978 for `destroy` and 0.805 for
  `exhaust`), so the arm carries almost no treated exposure and is *untested* rather than
  shown invisible. The execution-verifier flags it at 1.000 regardless.
- Disclose the malignant campaign's **~2.5× over-sampling confound**: within that
  campaign the same-campaign benign poison reads spuriously *positive* (+0.033). The
  benign-poison null is the **main**-campaign result, not the malignant campaign's.

### C4 — GENERALITY (consistency check only)
The helpful ≫ poison asymmetry replicates across the capability ladder.

**Do not confuse the two poison estimands — this is the single easiest error to make
here, and the paper disclaims the stronger reading.** The **marginal-inclusion** ATE (the
deployed causal quantity: the effect of *including* a poison lesson in the retrieved set)
is null at **every** retrieval breadth — M=1 −0.021 p=.25; M=3 +0.004 p=.79; M=6 −0.002
p=.86. The honest rival is the **on-target** contrast, which restricts the run set to
challenges of the poison lesson's own family (family is *pre-treatment* — this is a
conditional ATE, not a selection or citation effect): −0.0745 p<.001 (CI [−0.125, −0.033], n=9) at M=1, −0.057 p<.001 at
M=3, and non-significant at M=6. **Never present the on-target number as the marginal
one** — doing so claims a significant deployed causal harm the paper explicitly disclaims.

Cross-*architecture* generality is **one** architecture (Llama-3.1-8B); every other
non-Qwen probe was harness-floored at n ≤ 3. Never write "across architectures" plural.

### C5 — LIMITS THAT MUST APPEAR
Any honest poster needs at least these:
- Scoped to **one** hand-built crypto-decode microbenchmark. A second (SQL) author-built
  domain **breaks the deployed-M null** (on-target −0.25, p=.003); a third
  (struct-extraction) degenerates (self-verifying via the `picoCTF{` prefix,
  brute-forceable). This is the most important external-validity caveat in the paper.
- The strongest rival reading, named by the authors themselves: the deployed-M null is
  partly a dilution-and-power phenomenon, and at full exposure poison *is* a measurable
  sub-floor harm.
- The formal result is a scoped **Proposition**, not a Theorem — it was deliberately
  demoted when its premises were found partly falsified.
- The tasks are a minimal instrument (single fixed-transform decodes, 1–4 shell rounds),
  not Cybench/NYU-CTF-scale exploitation. Value is *known ground-truth lesson causality*,
  not realism.

---

## 3. The system under study

An agentic CTF solver whose memory is a **self-extending typed-node knowledge graph** of
lessons, retrieved by a **contextual bandit** over graph edges, feeding an LLM solver with
sandboxed shell tools. Four false-lesson modes are each matched to one oracle-free lever:

| Failure mode | What it is | Lever aimed at it | Stage |
|---|---|---|---|
| **Birth defect** | A lesson trusted after a single success, before its reliability is known | Gated write-back (provisional → trusted only after repeated cited successes) | write-back |
| **Latent poison** | False premise, still reaches the flag, but by a longer route | Step-count reward `R = o(1 + λe)`, `e = −z(steps)` | reward |
| **Recovery masking** | Lesson sits in context during a success it did not cause | Action-grounded credit: credit only `cited ∩ sampled`, where "cited" means the lesson's tool/technique appears in the agent's **executed shell commands** — never its self-narration | credit |
| **Benign-false (the negative result)** | Wrong method, executed, recovered from | *(none — this is what the paper shows outcomes cannot reach)* | — |

**Retrieval pipeline** (this is the load-bearing detail for the causal claims):
recall top-N by embedding similarity → rerank → MMR diversity → **sequential softmax
sample of M without replacement, at temperature τ, logging the per-draw propensity and
the sequential set log-probability**.

That last clause is the whole methodological trick: **retrieval is a free randomized
experiment.** Each lesson's inclusion is a stochastic draw with a *logged* propensity, and
the reward is re-checkable without an oracle. Inverse-propensity weighting over the
system's own existing logs therefore yields a causal estimate at **zero extra inference
cost**. If a poster has one "method" panel, this is it.

**Bandit value blend:** `score = (n·Q + κ·sim)/(n + κ)` — similarity is retained as a
decaying prior so a cold lesson is sampled by its similarity rather than an arbitrary
uniform weight. Bandit context is coarse (three category nodes: crypto/forensics/misc),
which is itself a disclosed limitation.

---

## 4. The benchmark instrument

Public CTF corpora are unusable here for two reasons: **contamination** (the solver's
pretraining has seen the flags and write-ups) and **no ground-truth lesson causality**
(you cannot know which lesson caused a solve). So the corpus is built.

- Every task embeds a **freshly random flag** `picoCTF{…}` → zero contamination by
  construction.
- Every family has a **known correct method** plus a planted **poison** lesson naming a
  plausible-but-wrong method → ground-truth causal labels.
- Task descriptions are **method-neutral**, so the retrieved *lesson*, not the prompt,
  carries the method. This is what makes the lesson the treatment.

**Corpus calibration is the key design step,** and it is worth a poster sentence. A
controlled corpus only measures lesson dynamics inside the band where a lesson *matters*:
too easy and the model solves regardless (poison cannot hurt); too hard and it fails even
when correctly instructed. The recalibrated corpus is a **two-layer decode** — an *outer*
transport encoding (Base64/hex, so `file` reveals only "ASCII text") wrapping an *inner*
transform (gzip, raw zlib, single-byte XOR, or reverse) that is invisible in the raw
bytes. The model can execute each step but cannot guess the inner transform, so the
recipe must come from memory. The round budget is deliberately tight (four) so the model
cannot brute-force every decoding.

**External validity was attempted and blocked, honestly.** A live `play.picoctf.org`
probe with valid credentials and a real browser hit commercial bot-mitigation. The
authors made a single legitimate attempt and **did not deploy evasion** (stealth drivers,
CAPTCHA solving). Reported as Appendix A. Live public CTF is contaminated, label-free,
*and* operationally gated against automation — which is the argument for the built
instrument.

---

## 5. Experimental structure

### Two tiers — do not conflate them
- **The 4B pilot** *illustrates* the boundary. One 8 GB laptop GPU, Qwen3-4B-Instruct-2507
  in 4-bit, ~10 families. Its magnitudes are **illustrative only** — it ran on an earlier
  benchmark revision whose poison-contract/oracle-labeling/grid-duplication bugs were
  fixed in commit `63db02d` and was not re-run. Figures from it are captioned as such.
- **The lab-scale ladder** *carries the headline*. Five capability rungs at M=3:
  **Qwen2.5-Instruct 3B / 7B / 14B-AWQ / 32B-AWQ**, plus one non-Qwen rung,
  **Llama-3.1-8B-Instruct**. Each rung logs **2.2k–15.6k runs** over **11–43 families**
  (~15.6k/15.5k at the powered 14B/32B rungs). A malignant positive-control campaign at
  14B adds 4,264 runs (later expanded by 5,096 more).

Note the generation gap and state it if you mention both: the pilot is Qwen3-generation,
the ladder is Qwen2.5, and the *intended* solver (Qwen3.5-27B-FP8) was never run.

### Baselines / ablations
| Label | What it is |
|---|---|
| **B0** | No graph at all — the lift denominator |
| **B2** | Static similarity retrieval + rerank + diversity, no bandit |
| **B4** | Full method: bandit + cited-only credit + step reward |
| *all-M credit* ablation | Credit every in-context lesson, not just cited ones — isolates **recovery masking** |
| *outcome-only* ablation (λ=0) | Isolates **latent poison** |
| **B5 / B6** | Gated vs. ungated write-back |

Each learning system streams the corpus for two epochs; B4 is then **frozen** and
evaluated paired against B0/B2 on common random seeds.

### Three credit signals, trading cost against precision
1. **Active leave-one-out re-rollout** — the assumption-light gold standard. Re-solves
   each challenge with and without the lesson, holding the other sampled lessons fixed
   (so it is interference-robust). Expensive; stays a diagnostic.
2. **Observational IPW/SNIPW** — reads the existing logs at **zero extra inference**.
   Hardened with self-normalized IPW, effective-sample-size rejection, propensity
   clipping to [0.05, 0.95], ≥5 observations per arm, and partial pooling of sparse cells.
3. **Online bandit** — recomputes its values from its own logged exploration each epoch.

### Statistics — non-negotiable and worth stating on a poster
Paired **within-challenge** comparisons on common seeds; **bootstrap** confidence
intervals; **exact McNemar** for paired binary outcomes; ground-truth difficulty controls
(family point value, not an LLM grader — the grader and solver would both be Qwen);
**efficiency measured in steps** (`rounds_used` / `total_commands`), never wall-clock,
because wall-clock on a shared GPU is irreproducible.

---

## 6. Appendix C — the latent-poison probe (read the scope line before using it)

Appendix C is a **single-author addition by J.H., developed as a graduate course project**
and merged at appendix strength. It closes a gap the paper had confessed four times: the
step-count reward was *specified* as the lever against latent poison, but no corpus
instantiated latent poison, so the lever went unexercised.

**Construction — the "involution pad."** Where the correct recipe is a transform chain
`T`, the poison recipe prescribes `π ∘ π ∘ T` for a self-inverse `π` (rot13; likewise
xor_k, rev). Since `π∘π` is the identity, the recovered flag is **bit-identical** to the
correct method's while the prescribed route is **strictly longer**. Over 52 depth-1 crypto
families × 8 instances (n=416): output identity holds 416/416, route strictly longer
416/416, by exactly 2 extra steps.

**Result 1 — the execution-verifier is blind to it, by construction.** It accepts the
detour arm at **1.000**, exactly the rate of a genuinely helpful lesson, while flagging
every arm it is supposed to catch at 1.000. This is near-tautological and the appendix
says so: the verifier's criterion is well-formedness plus input-sensitivity, and a recipe
that recovers the true flag satisfies both however long its route. That is precisely the
bound on what the verifier can reach.

**Result 2 — the step contrast is censored and NOT usable. No bound may be stated.**
The marginal contrast is Δrounds = −0.250 [−1.21, +0.65], but that number is **not a
measurement of route cost** and the paper explicitly disowns it. The flat 8-round budget
censors 30/52 helpful and 25/52 detour runs; 20 pairs (38.5%) are capped in *both* arms
and contribute Δ≡0 by construction; solved runs finish in ~1.4–1.6 rounds while unsolved
sit at the cap. The result is corr(Δrounds, Δsolved) = **−0.84** — the round variable is
close to a restatement of the outcome. A censoring sensitivity check shows the scale of
the problem: a *true* cost of one round per run would read as only **+0.42** rounds here.
**Do not state a step-cost bound. Do not say the interval "excludes" anything.**

**There is no validated positive control for the round channel.** The `search` arm was
meant to be one and looks like one marginally (+2.08 rounds), but it is censored at
0.904, its round excess tracks its solve failure (corr −0.95), and on outcome-concordant
pairs it falls to +0.270 [+0.053, +0.550], against the detour arm's own **concordant**
−0.150 [−0.475, +0.136]. The two *do* separate on that estimand, but 30 of search's 37
concordant pairs are capped in **both** arms and contribute Δ≡0 by construction, so the
concordant estimand is diluted, not uncensored. It validates *outcome* sensitivity, not
route-cost sensitivity. **Never compare a concordant contrast against the marginal
interval** — that was a defect in an earlier draft.

**Result 3 — the mechanism, which is the substantive finding.** The agent *does* run the
longer route: the pad appears in executed shell commands on **0.962** of detour traces vs
0.000 of helpful, and among *solved* runs alone — where no censoring can act — the
separation is complete (**11/11** detour vs **0/12** helpful). The cost shows up as
**denser** commands, not more of them: paired shell operations per issued command rise
**+2.78 [+1.78, +3.83]** (21 of 22 pairs positive), and this **survives conditioning on
outcome** (+2.29 [+2.00, +2.75] on both-solved pairs — but that is only **7 pairs in 4
families**, so quote the n whenever you quote the interval). The issued-command **count** does
*not* separate the arms (its interval spans zero, median −1.0, and on the main probe's
both-solved estimand it runs the other way) — **never quote the raw command count as a
cost.** The point is about the reward's *units*: one component of the execution trace
registers the longer route, and `e` counts rounds.

**The reading is a CONJECTURE, not a result.** The natural suggestion is that the relevant
property is route *compressibility* rather than route *length*. The appendix labels this a
conjecture: the mechanism data is consistent with it, and the step contrast neither
confirms nor refutes it. Label it a conjecture anywhere you use it.

**Scope — state it wherever you use this.** One model (Qwen2.5-7B-Instruct), one crypto
domain, 13 families, n=52 paired. It uses the same model as the ladder's 7B rung but a
different design — forced single-lesson context, flat 8-round budget, selected depth-1
families — so it is **not a rung of that ladder**. It is a marginal and conditional step
contrast, **not a per-item ranking**: the LOO gold standard was never run against this
construction, so it says nothing about per-item identifiability. Do not let it near C1.

**Poster/abstract guard-rails for this appendix:** do not produce a figure of the marginal
detour/search round contrasts — the paper disowns that contrast. If you want a figure
here, plot the mechanism: pad-execution rate and ops-per-command, both conditioned on
solved runs. And in most cases the right call for a 1-page abstract or a poster is to
**omit Appendix C entirely** — it is appendix-strength material that enters no headline
claim.

**What was deliberately excluded** from the paper and must not be reintroduced: the
companion project's adversarial write-back "accumulation" campaign. Its poison fraction
*declines* monotonically over epochs, its crypto arm is 85–95% censored at a round cap
below the correct method's mean run length, and its "the gate does not discriminate"
finding is contradicted by its own ungated rows, where no gate logic runs at all and the
differential is *larger*. No adversary or threat-model framing belongs in this paper.

---

## 7. Figure inventory

All seven are **vector PDF** (no embedded rasters) and scale cleanly to A0.
**Caveat:** they embed **Type 3 fonts**, which arXiv warns about — harmless for a poster,
worth regenerating with `matplotlib.rcParams['pdf.fonttype'] = 42` before arXiv submission.
Captions live in the `.tex`, not inside the figures, so **any poster use needs a
freshly-written standalone caption.**

| File | Shows | Poster value |
|---|---|---|
| `fig_pipeline.pdf` | The solve-and-update loop; each lever attached to the stage it defends | **Best method hero.** But it is 416×147 pt (≈2.8:1) with ~6.3 pt type — awkward on a portrait board. Consider redrawing at poster scale |
| `fig_separation.pdf` | Same logs, two credit signals. Left: outcome-bandit Q by ground-truth label — executed poison earns a positive value, numerically beside true lessons. Right: observational IPW ATE — poison mildly negative, true strongly positive | **Best single result figure.** It *is* the paper's thesis in one image |
| `fig_loo.pdf` | Leave-one-out marginal contribution by ground-truth lesson type | Strong support for C1 |
| `fig_q_by_config.pdf` | Final learned Q by lesson type across configurations; cited-only credit zeroes unused distractors, all-M credit rewards them, but executed poison stays indistinguishable from true | Good for the "which lever works" story |
| `fig_solve_rates.pdf` | B0 vs B2 vs B4 solve rate and paired lift | Context; note B4 does **not** beat B2 |
| `fig_q_trajectory.pdf` | Learned value over training | Optional |
| `fig_ope_sweep.pdf` | Off-policy value of counterfactual sampling temperatures | Sanity check; probably cut for a poster |

There is **no figure for Appendix C**, and the obvious one is forbidden: do **not** plot
the paired detour/search round contrasts — §6 explains that the paper disowns that
contrast as censoring-confounded. If you want an Appendix C panel, plot the mechanism
instead: pad-execution rate (0.962 marginal; 11/11 vs 0/12 among solved runs) and
ops-per-command. **Label whichever ops figure you plot** — the marginal paired arm means
are 2.66 → 5.44 over 22 pairs; conditioned on both-solved they are 1.71 → 4.00 on only
7 pairs in 4 families. Do not print the marginal under a "conditioned on solved" label.
Always show the n.

**Tables:** the paper has only three `tabular` environments (`tables_pilot.tex` ×2,
`results_solution.tex` ×1) plus the two new ones in `appendix_latent.tex`. The lab-ladder
per-rung results are **prose-only** in `theorem_identifiability.tex`, so a poster that
wants a ladder table must build it from the macros in `numbers_identity_a.tex`.

---

## 8. Where every number lives

Nothing in this paper is hand-typed. Each `\macro` in the prose is defined in one of:

| File | Contents |
|---|---|
| `numbers.tex` | 4B pilot |
| `numbers_solution.tex` + `numbers_solution_defaults.tex` | The causal-credit study |
| `numbers_identity_a.tex` | **The lab-scale ladder** — 375 macros, the load-bearing numbers |
| `numbers_latent.tex` | Appendix C, re-derived by `emit_appendix_numbers.py` |
| `tables_pilot.tex` | The pilot tables |

To resolve any macro you see in the text, grep these files. The two emitter scripts in
this bundle (`emit_appendix_numbers.py`, `make_abstract_plain.py`) are runnable; the
paper's other emitters live in the repo under `experiments/`.

---

## 9. Suggested spine for a 1-page extended abstract

1. **Hook** — self-extending agent memory compounds competence *and* error; prior work
   shows consolidating even flawless solutions can *lower* success rate.
2. **The turn** — CTF is a *verifiable-reward* domain. A flag is checkable with no oracle.
   So: can the verifiable signal alone detect and suppress a false lesson?
3. **Why it must be per-item** — every hygiene lever a deployed memory has (gating,
   pruning, demotion) acts on *individual* lessons, so it needs a *per-item* signal.
4. **The free experiment** — stochastic retrieval with logged propensities + a
   re-checkable reward ⇒ IPW over existing logs, zero extra inference.
5. **What it reaches** — helpful lessons, all the way (Spearman 0.64, 30/30 rankable).
6. **What it does not** — the per-item harm of a benign-false lesson. Gold standard
   +0.045 [−0.006, +0.098]; within-poison Spearman ≈ 0, sign agreement at chance.
7. **Why** — under a measured base-rate floor, executing a benign-wrong recipe is, for a
   *binary* outcome, nearly equivalent to executing none.
8. **The boundary's other edge** — destructive lessons *are* caught; a confidently-wrong
   answer is **untested**, not shown invisible (the agent followed that recipe on only 11
   of 453 sampled runs, follow rate 0.024 vs 0.978/0.805 for destroy/exhaust, and under
   *forced* exposure a constructed leave-one-out does detect it at −0.43); a deterministic
   execution-verifier recovers recipe *validity*, a different and recoverable target.
9. **Limits** — one microbenchmark domain; a second breaks it; not "harmless."
10. **Artifact** — a released contamination-free, ground-truth-labeled instrument.

For a poster, the same spine works with panels 4–6 as the visual centre
(`fig_pipeline.pdf` → `fig_separation.pdf` → `fig_loo.pdf`) and a large, unhedged
takeaway line. A good one:

> **A verifiable reward tells you *that* a run succeeded — not *which* lesson made it succeed.**

---
