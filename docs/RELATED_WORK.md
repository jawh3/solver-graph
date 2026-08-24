# RELATED_WORK.md — verified literature landscape (sweep 2026-06-03)

> Built by an 8-angle web sweep + per-hit verification (14/14 confirmed real). Drives the
> paper's positioning and the post-pivot action plan. **One scoop-risk is UNVERIFIED and must
> be confirmed before any "first" claim: MemoryR2 (2605.21768).**

## Headline verdict
Our **core method is genuinely novel**: *per-individual-memory-item off-policy causal credit
(IPW/SNIPW/AIPW ATE-on-reward) over **logged stochastic-retrieval propensities**, used to
separate genuinely-useful lessons from self-banked poison and to **suppress false-lesson
PROPAGATION at the write-back step**, in a **verifiable-reward CTF** domain with action-grounded
cited-only credit.* **None** of the 14 confirmed papers do per-item propensity-based causal credit.

The three pivot directions are **mostly already published** and must be **demoted to supporting
analyses inside the core frame**, never standalone discoveries:
- **D1 (effort credit): motivation already public.** Mazhar et al. *Trace-Level Contamination*
  (2604.27586) already shows the binary-success-is-leaky decoupling + token/step cost, and
  **explicitly defers per-item causal attribution to future work** — that deferral is our gap.
- **D2 (capability-dependence): partly scooped AND partly contradicted (adversarial regime).**
  Skill-Inject runs a capability ladder and concludes robustness is **non-monotonic in capability
  / "will not be solved through model scaling."** Reframe D2 to the **benign-false-lesson
  DETECTABILITY** regime on a controlled same-family ladder (distinct from adversarial).
- **D3 (adversarial skills): largely DONE.** Skill-Inject + MemMorph + ZJU survey + MINJA +
  "Your Agent Is Mine" collectively own "agents trust and follow malicious skills." **Drop the
  D3 benchmark build;** at most a small causal-credit extension, or defer to a separate paper.

## Confirmed papers (arXiv IDs verified)
| Paper | ID / venue | Why it matters to us |
|---|---|---|
| **Memory-R2** | 2605.21768 *(CONFIRMED 2026-06-03)* | "Fair Credit Assignment for Long-Horizon Memory-Augmented LLM Agents" (Yan…Tresp, Ma). **LoGo-GRPO with LOCAL RE-ROLLOUTS — needs extra inference; NO off-policy IPW/propensities, NO per-item causal ATE, NO poison/write-back gating.** Same core insight (trajectory-level reward is biased credit for memory ops) but different, costlier mechanism → we are the **cheap observational off-policy alternative + the poison-suppression angle they lack.** Cite as concurrent work. NOT a scoop. |
| Mazhar *Trace-Level Contamination* | 2604.27586 | **D1 anchor — VERIFIED 2026-06-03.** "Behavioral detours with recovery, **40.3% of runs**" (diverged trace, still-correct outcome); decoupling: **76.2% of low-cost runs are WRONG, only 16.3% of high-cost runs are RIGHT**; 15.3% silent-semantic-corruption; token overhead 1.5×/2.4×. Backends GPT-5-mini/LLaMA-3.1-70B/Qwen3-235B; 614 paired runs / 32 GAIA tasks. **Does NOT use an oracle** — divergence is vs the PAIRED CLEAN run (edit distance + first-divergence t*). Defers per-item attribution → our gap. |
| Skill-Inject | 2602.20156 | **D2/D3 must-engage.** 202 malicious-skill tasks, ASR 41–79%, action-outcome detection; capability ladder → **non-monotonic**, contradicts naive D2. |
| MemMorph | 2605.26154 | D3: memory-poison→malicious tool selection; 10-model ladder incl. our Qwen2.5-7B but **no capability-trend analysis**. |
| Experience-Following | 2505.16067 | Motivation: experience-following / error-propagation / misaligned replay. Deterministic top-k, no causal credit. |
| Evo-Memory | 2511.20857 | Contemporaneous self-evolving-memory benchmark; deterministic retrieval, steps reported descriptively only. |
| RSCB-MC | 2604.27283 | Contextual bandit over memory — but **explicitly "not causal,"** no write-back, no propensities. |
| SkillOS | 2605.06614 | Self-evolving skill curation; composite-reward **but uniform-advantage RL, deterministic BM25**; reports efficiency wins (so D1 metric not novel). |
| Skill Shadowing | 2605.24050 | "More skills → worse" at the selection stage; observational only, no causal/propensity machinery. |
| ZJU Agent-Skills survey | 2602.12430 | D3 landscape: 26.1% of 42,447 real skills vulnerable to file/password exfil; governance framing. |
| AgentPoison | 2407.12784 (NeurIPS'24) | Canonical RAG/memory backdoor attack; attacker-trigger, no detection/credit. |
| PoisonedRAG | 2402.07867 (USENIX'25) | Canonical corpus poisoning; text-surface defenses only — we go beyond with causal attribution. |
| BadRAG | 2406.00083 | Retrieval backdoor; attack-side only. |
| MINJA | 2503.03704 (NeurIPS'25) | Self-stored malicious memory → attacker action; **embedding sanitization FAILS (entangled)** → motivates causal credit over similarity filtering. |
| "Your Agent Is Mine" | 2604.08407 (CCS'26) | Malicious API routers; harm outside the task-success loop (adjacent D3 framing). |
| Faulty Memory | 2605.12978 | The negative result our thesis inverts (self-consolidation degrades) — framing foil. |

## Precise novelty we still hold (lead with these)
1. **Per-item causal credit from "free randomized experiment"** — IPW on logged stochastic-retrieval
   propensities → per-lesson ATE; **observational variant needs no extra inference** (vs MemoryR2 re-rollout).
2. **False-lesson PROPAGATION suppression at the crediting/write-back step** — causal-ATE gate vs naive
   outcome credit, on the *same* logs that fool outcome credit (use our actual logged promotion-gap numbers).
3. **D1 (scoped):** effort-weighted reward *as the dependent variable inside per-item causal credit* —
   detection null on binary solve becomes significant on effort-weighted reward (fills Mazhar's deferred gap).
4. **D2 (scoped):** capability-dependent **detectability** of **benign** false lessons on a controlled
   same-family ladder (harm + causal signal shrink as the model recovers "for free"). Engage Skill-Inject's
   adversarial non-monotonic null as an explicit contrast (different regime).

## Hard "don't waste time / don't overclaim" rules
- **Don't** claim first-to-show agents follow malicious skills (Skill-Inject et al. own it). **Drop the D3 benchmark.**
- **Don't** claim adversarial harm monotonically worsens with capability (Skill-Inject contradicts).
- **Don't** claim invention of IPW / causal credit / memory credit broadly. **Confirm MemoryR2 first.**
- **Do** state SUTVA/interference + propensity-circularity limits up front (sign-flip already survives Q=0/prior-only propensities).
