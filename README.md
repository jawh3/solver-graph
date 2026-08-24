# solver-graph

Research artifact for **"What a Verifiable Reward Can and Cannot Credit in
Self-Extending Agent Memory: A Non-Identifiability Result for False-Lesson Harm."**

> James Hugglestone, Samuel Jacob Chacko, Xiuwen Liu (advisor) — Florida State University

An LLM agent with a self-extending memory writes lessons from its own runs and retrieves
them later. Some of those lessons are causally wrong. In Capture-The-Flag the reward is
**verifiable** — a flag is a high-entropy string the environment accepts or rejects, with
no oracle, no human label and no LLM judge — so it is tempting to use that signal to keep
the memory clean.

**The result is negative, and that is the contribution.** The per-item harm of a
*benign-false* lesson — one naming a wrong method the agent executes and then recovers
from — is **not per-item rankable** from the verifiable outcome. Neither a cheap
observational estimate nor an assumption-light leave-one-out gold standard can rank it.
The same machinery ranks *helpful* lessons perfectly well, which is what makes the
negative result meaningful rather than an estimator failure.

"Not identifiable" means **not per-item rankable** — it does *not* mean harmless.

## Layout

| path | what |
|---|---|
| `benchmark/` | the contamination-free micro-CTF generator: fresh random flag per task, known-correct method plus a planted poison, method-neutral descriptions |
| `retrieval/` | recall → rerank → MMR → propensity-logged sampling. This is what makes retrieval a *free randomized experiment* |
| `ope/` | IPS / SNIPS / doubly-robust estimators over the system's own logs |
| `bandit/`, `reward/`, `writeback/`, `pipeline/` | the memory system: contextual bandit, step-count reward, gated write-back, orchestration |
| `evaluation/` | baselines B0–B6, paired bootstrap, McNemar, dataset partitions |
| `experiments/` | campaign drivers and the emitters that turn artifacts into the paper's numbers |
| `results/pilot/` | 4B pilot artifacts and figures |
| `results/campaign_v2/` | the 4B causal-credit study |
| `results/lab/` | the capability-ladder campaigns (Qwen2.5 3B–32B + Llama-3.1-8B) — the load-bearing artifacts |
| `appendix_c/` | artifacts and scripts for the paper's Appendix C (constructed longer-route poison) |
| `paper/` | manuscript sources, figures, and the compiled PDF |
| `docs/` | pre-registrations, the working claim ledger, related work, architecture diagram |

## Reproducing the numbers

Nothing in the paper is hand-transcribed; the numbers are emitted from artifacts.

```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd)

pytest tests/                                        # mock mode, no GPU required
python -m experiments.analyze                        # pilot  -> paper/numbers.tex
python -m experiments.analyze_campaign --dir results/campaign_v2
python -m experiments.emit_paper_numbers             # ladder -> paper/numbers_identity_a.tex
python3 paper/emit_appendix_numbers.py               # Appendix C -> paper/numbers_latent.tex
cd paper && pdflatex main && pdflatex main
```

`paper/` also carries three guards that fail the build rather than let an error ship:
`check_rung_selectivity.py` (no per-rung statistic may be reported at only the rungs that
agree), `check_estimand_vocab.py` (the on-target contrast is family-restricted and must
never be described as citation-conditioned), and `check_coverage.py` (every
pre-registered item must be reported, superseded, or explicitly excluded).

## Scope

Results are scoped to **one** hand-built crypto-decode microbenchmark and a five-rung
capability ladder (Qwen2.5 3B/7B/14B/32B plus Llama-3.1-8B — "across architecture" here
means *one* architecture). A second, author-built SQL domain **breaks** the deployed-M
null; a third degenerates. `docs/PREREG.md` records the pre-registration; `paper/` states
the limitations in full.

## Provenance

This is a curated release. The complete development history lives in a private
repository; it is excluded here because it contains internal working notes, unrelated
material, and machine-specific configuration. Nothing required to reproduce a reported
number was withheld — see `release-manifest.md` for the full contents of this release.

## Citing

If you use the benchmark generator or the artifacts, please cite the paper. A BibTeX
entry will be added once the preprint has an arXiv identifier.

## License

MIT for code and data (`LICENSE`). The manuscript text and figures under `paper/` are
all rights reserved.

## AI assistance

Claude (Anthropic) was used substantially for code implementation and manuscript
drafting; the paper contains a full disclosure. All experimental design decisions,
result verification and final claims are the authors'.
