# solver-graph

Code and data for the CARS 2026 poster **"Learning Which Lessons to Show a CTF Agent:
Learned Choice Wins on Synthetic Tasks, Not Yet on Real CTFs."**

> James Hugglestone, Samuel Jacob Chacko, Xiuwen Liu (advisor), Florida State University

[Poster (PDF)](poster/Hugglestone_CARS26_poster.pdf) ·
[Abstract (PDF)](poster/Hugglestone_CARS26_abstract.pdf) ·
[Full results and how to reproduce them](RESULTS.md)

An LLM agent can keep a memory of short **lessons**: notes naming a technique to try, with a
short recipe. Which lessons should it see for a new challenge? In capture-the-flag (CTF), an
exact-match flag check gives a **verifiable reward** with no human or LLM judge. The learner
here is updated from only two things: whether the flag was right, and which shown lessons turn
up in the agent's commands. We ask whether that is enough to pick better lessons than a fixed
relevance ranking.

## Results

Differences are in percentage points of solve rate, paired by task; brackets are 95%
confidence intervals.

| test bed | comparison | difference [95% CI] |
|---|---|---|
| synthetic, Qwen2.5-14B | learned - static | **+13.2** [+6.1,&nbsp;+20.9] |
| synthetic, Qwen2.5-32B | learned - static | **+25.2** [+16.5,&nbsp;+33.6] |
| NYU CTF Bench, 176 tasks × 3 seeds | static - no memory | +0.6 [-2.1,&nbsp;+3.4] |
| NYU CTF Bench, 58 file-only tasks | static - no memory | +1.7 [-4.0,&nbsp;+7.5] |

- **no memory**: no lessons.
- **static**: the top 3 lessons by a fixed text-relevance ranking (a cross-encoder); it never
  changes.
- **learned**: 3 lessons chosen by values learned from the flag.
- **file-only**: tasks that give the agent only files, no live server.

Learned beats static at both model sizes on the synthetic benchmark (22 decoding-puzzle
families, fresh random flags). There the learner is scored on new items from the same families
it trained on, and in about a quarter of its training runs it explored using each task's family
label to choose which lessons to try (not which one is right); a learner trained without that
label was not tested (RESULTS.md §1, §5). On NYU CTF Bench (Qwen3.8-27B), static lessons gave no
detectable gain; the real-CTF results cover static lessons only. Controls, per-seed results
and the limits are in [RESULTS.md](RESULTS.md).

## What is here

- `benchmark/`: the synthetic task generator: a fresh random flag per task, a known-correct
  method, neutral names.
- `retrieval/`, `bandit/`, `reward/`, `pipeline/`: how lessons are chosen and credited
  (cross-encoder ranking, the contextual bandit, the reward, one run end to end).
- `solver/`, `batch_test/`: the agent loop, its sandboxed executor, and the model adapters
  the synthetic runs use.
- `realbench/`: NYU CTF Bench and Cybench adapters: task lists, a container per solve with
  no internet, the flag check, lesson writing.
- `experiments/`: the drivers and analyses of the two campaigns the poster reports, plus a
  few earlier analysis modules that the campaign runner and the tests import. The reported
  numbers come from `phase2_ladder.py`, `phase2_pool_draws.py` and `nyu_rerun_analysis.py`.
- `results/lab/phase2_14b/`, `results/lab/phase2_32b/`: synthetic-benchmark results: each
  condition's held-out metrics (about 33 MB) and the computed ladders.
- `results/realbench_nyu_v2/`: every NYU run record with every command and the start of its
  output (about 48 MB),
  the analysis output and the task health check (the task list is in `docs/phase3/`).
- `results/realbench_nyu_mem/`: the 41 lessons used on NYU CTF Bench, and what each lesson
  writer saw.
- `poster/`: the poster and the one-page abstract.
- `tests/`: runs in mock mode with no GPU; tests that need Docker or a benchmark skip
  without them.
- Also here: `config.py` (the single source of the swept and ablated parameters), `local_llm/`
  (a small model server for laptop tests; not used for the reported runs), `writeback/`
  (adds new lessons to memory; not used in these results), `ope/`, `knowledge_graph/`,
  `instrumentation/`, `evaluation/`, `OPEN_DECISIONS.md` (the design decision log) and
  `CTF_SOLVER_ARCHITECTURE_v2.md`, the May 2026 design spec that code comments cite as
  "spec §N". Its research framing predates the poster.

## Reproducing the numbers

Every number in the poster and in `RESULTS.md` is computed from files in this repository,
with no GPU (Python 3.10 or newer). Two exceptions: which NYU lesson writers saw a solution is
a judgment made by reading each writer's sources, recorded with those source lists in
`results/realbench_nyu_mem/qwen_qwen3.8-27b-fp8/lesson_sources.tsv` (rebuilding the lists needs a
checkout of NYU CTF Bench; `experiments/nyu_lesson_sources.py`); and the count of static runs that
ran after the no-memory condition finished comes from the runner's log.

```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
pytest tests/

# synthetic benchmark, 14B and 32B
python experiments/phase2_ladder.py --root results/lab/phase2_14b --out /tmp/ladder_14b.md
python experiments/phase2_ladder.py --root results/lab/phase2_32b --out /tmp/ladder_32b.md
# causal credit over 3 draws
python experiments/phase2_pool_draws.py --root results/lab/phase2_32b
# NYU CTF Bench
python experiments/nyu_rerun_analysis.py
```

Tests that need Docker or a benchmark checkout skip when it is absent. Running the agent
also needs `requirements-agent.txt`, a vLLM server and, for NYU CTF Bench, Docker 28 or newer
and a checkout of the benchmark; see `RESULTS.md` §6.

## Benchmark data

NYU CTF Bench: Shao et al., "NYU CTF Bench: a scalable open-source benchmark dataset for
evaluating LLMs in offensive security," NeurIPS Datasets and Benchmarks, 2024
(<https://github.com/NYU-LLM-CTF/NYU_CTF_Bench>, GPL-2.0). The benchmark is not included.
`results/realbench_nyu_v2/` contains flags and agent transcripts for its test tasks; please do
not use these files to train models that may be evaluated on NYU CTF Bench.

## Earlier release

The August 2026 release of this repository (the manuscript "What a Verifiable Reward Can and
Cannot Credit in Self-Extending Agent Memory" and its artifacts) is superseded and should not
be cited or relied on. It is not part of the current tree. It remains in this repository's
history at commit [`43ecf5d`](https://github.com/jawh3/solver-graph/tree/43ecf5d).

## Citing

Please cite the poster: J. Hugglestone, S. J. Chacko, X. Liu, "Learning Which Lessons to
Show a CTF Agent: Learned Choice Wins on Synthetic Tasks, Not Yet on Real CTFs," poster,
IEEE CARS 2026. Contact: jah21e@fsu.edu.

## Provenance

This is a curated release. The complete development history lives in a private
repository; it is excluded here because it contains internal working notes, unrelated
material, and machine-specific configuration. Nothing required to reproduce a reported
number was withheld. See `release-manifest.md` for the full contents of this release.

## License

MIT for code and data (`LICENSE`). The poster and abstract PDFs under `poster/` are all
rights reserved. Run records and lessons quote material from NYU CTF Bench, which keeps its
original licenses (GPL-2.0, copy in `LICENSES/GPL-2.0.txt`).

## AI assistance

Claude (Anthropic) was used substantially throughout: writing code and analysis scripts,
running and monitoring experiments, drafting protocols, reviewing code and analyses, and
drafting this documentation and the poster and abstract text. The authors directed the work,
approved the experimental designs and analyses, and are responsible for all claims.
