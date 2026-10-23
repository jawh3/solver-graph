# Results

The numbers behind the CARS 2026 poster
([PDF](poster/Hugglestone_CARS26_poster.pdf)), with the setup, the controls and the commands
that recompute each one from the files in this repository. Differences are in percentage
points of solve rate; brackets are 95% confidence intervals.

## 1. Setup

**Agent.** A frozen LLM solves each task with shell commands: on the synthetic benchmark in a
fresh working directory per solve holding a copy of the task files, on NYU CTF Bench in its own
container. We compare three conditions; in the two with memory, three lessons are put in the
agent's prompt.

- **no memory**: no lessons.
- **static**: the top 3 lessons by text relevance from a pretrained cross-encoder. On NYU CTF
  Bench it is the top 1 plus two more chosen by maximal marginal relevance (MMR: relevance
  balanced against diversity). The choice never changes.
- **learned**: 3 lessons sampled (temperature 0.1) by a learned value Q per (context,
  lesson). The context is the alphabet of the challenge file's encoding, read from the file
  itself (for example hex digits vs base64 characters): 4 contexts for the 22 families. After
  each run, shown lessons whose command words appear in the agent's commands get credit,
  Q ← mean(R - b), with R = 1 + 0.5 · speed if solved and 0 otherwise, and b a running mean
  of R per context. Speed is the negative standardized step count, not wall-clock time.

**Synthetic benchmark.** 22 decoding-puzzle families (2,376 items: 1,782 for training, 594
held out), a fresh random flag per task, neutral task names. The families were chosen from 124
candidates by a probe with the 14B solver that kept only families with room for a lesson to
help (solved at most half the time without lessons, at least 40% of the time with the right
lesson, and at least 30 points more often); 56 passed and 22 were picked, up to six per
alphabet (`results/lab/phase2_14b/probe/`). The 32B rung used the same families. The
benchmark is therefore built so that the right lesson matters, and its gains are a best case. Lesson pool: 48 (a right and a
half-right recipe per family, plus 4 tool lessons). Solvers Qwen2.5-14B-Instruct-AWQ and
Qwen2.5-32B-Instruct-AWQ. Each learner trains on its own runs, then is scored on the held-out
items: 22 families × 27 items × 2 seeds = 1,188 held-out runs per condition, paired item by
item. In about 25% of training runs the learner explores: the task's two family lessons are
each shown with probability ½ and the remaining slots are filled at random from the other
lessons. Exploration uses each training task's family to pick its candidates, but not which
of the two is right.

**NYU CTF Bench** (Shao et al., "NYU CTF Bench: a scalable open-source benchmark dataset for
evaluating LLMs in offensive security," NeurIPS Datasets and Benchmarks, 2024;
<https://github.com/NYU-LLM-CTF/NYU_CTF_Bench>). 176 of the 200 test tasks are run. The
other 24 are not: 2 are unsafe on a shared host, 11 are broken or cannot be run as specified,
2 are trivial welcome tasks, and 9 have a flag format the solver cannot submit (see
`results/realbench_nyu_v2/TASK_HEALTH.md`). 41 lessons, written from 41 of the 45 usable
development tasks (the writer failed on the other 4). Each was written by the model from that task's reference
material: its description, README, write-ups, hints and reference solver scripts, never its
handouts or a solve attempt. 23 of the 41 writers saw a solution (12 a solver script; 11 a README or
write-up describing the solution, judged by reading them) and 18 did not (`lesson_sources.tsv`). No test task is a
lesson source. Solver Qwen3.8-27B-FP8 (chosen by an earlier comparison of solver models on solve rate without memory;
served with vLLM 0.19.1); 10 rounds, each one model turn of up to 5 commands
(300 s per command); a Kali container per solve (Cybench's agent image), internet blocked; seeds 1000 to 1002. Both
conditions ran at the same time against one model server, except that 55 static runs (46 not
yet finished, per the runner's log, and 9 that had errored, listed in `static.jsonl.no_retry`) ran after the no-memory condition finished its main pass. A run that ended in an error got one retry.
4 of the 528 pairs still had an error on one side, all because the conversation outgrew the
model's 32k-token context window, and are dropped, leaving 524 pairs.

**Statistics.** Synthetic: family-clustered bootstrap (4,000 resamples) and a
sign-permutation test over families. NYU: task-clustered bootstrap (10,000 resamples,
seed 0) and an exact two-sided sign test on discordant pairs (pairs where only one condition
solved). A gain is called detectable only if the CI excludes 0 and p < 0.05. Most NYU pairs are ties,
so the bootstrap bounds move a little with its random draws: with 200,000 resamples the CIs are
[-2.1,&nbsp;+3.2] (all tasks) and [-4.0,&nbsp;+8.0] (file-only).

## 2. Synthetic benchmark: learned beat static

Held-out solve rate (%):

| model | no memory | static | learned | oracle set |
|---|---|---|---|---|
| Qwen2.5-14B | 1.5 | 5.4 | **18.6** | 27.4 |
| Qwen2.5-32B | 2.7 | 11.3 | **36.4** | 56.8 |

"Oracle set" forces the correct lesson in. It is a reference point, not a method.

| contrast | 14B | 32B |
|---|---|---|
| learned - static | **+13.2** [+6.1,&nbsp;+20.9]; 16 families ahead, 3 behind; p = 0.0014 (sign-permutation) | **+25.2** [+16.5,&nbsp;+33.6]; 19 ahead, 3 behind; p < 0.001 |
| learned - static, two more independent training draws (same static run) | +11.2, +13.7 | +24.2, +23.6 |
| same learner at temperature 0.2 - static | +3.2 [-0.4,&nbsp;+6.7] | +12.2 [+5.0,&nbsp;+18.4] |
| same exploration, no context (one Q for all) - static | -1.9 [-5.5,&nbsp;+1.1] | -0.7 [-7.7,&nbsp;+5.6] |
| causal credit - learned | -0.9 [-5.2,&nbsp;+3.2] | +9.1 [+2.4,&nbsp;+15.8]; over 3 independent draws **+10.6** [+5.4,&nbsp;+15.9] |

The no-context control explores exactly as the learner does but keeps one Q for all
challenges. It shows no detectable gain over static, and the learner beats it directly by
+15.1 [+8.7,&nbsp;+22.2] (14B; 17 families ahead, 2 behind) and +25.8 [+19.9,&nbsp;+31.1] (32B; 21
ahead, 1 behind), so exploration alone does not produce the gain; the per-context values are
needed. A learner trained without family-guided exploration was not
tested. At temperature 0.2 the gain shrinks, and at 14B it is no longer detectable. Causal
credit re-solves a task without each used lesson and credits each lesson by what the
re-solve loses without it. It uses about 3.5× (14B) and 5.5 to 5.6× (32B) the tokens of the learned
condition per run, counting training and held-out runs (`tokens_per_run` in
`results/lab/phase2_*/compute_report.json`).

The ladder files also list conditions not discussed here: other context keys (including the
task's family label), other temperatures, and stress tests with poisoned lesson pools.

## 3. NYU CTF Bench: static, no detectable gain

| subset | static - no memory | no memory | static | pairs (tasks) | discordant: static only / no memory only | sign test p |
|---|---|---|---|---|---|---|
| all tasks | **+0.6** [-2.1,&nbsp;+3.4] | 14.9% (78) | 15.5% (81) | 524 (176) | 25 / 22 | 0.77 |
| file-only | +1.7 [-4.0,&nbsp;+7.5] | 23.6% | 25.3% | 174 (58) | 15 / 12 | 0.70 |
| live service | 0.0 [-2.6,&nbsp;+2.6] | 10.6% | 10.6% | 350 (118) | 10 / 10 | 1.00 |
| seed 1000 | -1.1 [-5.7,&nbsp;+3.4] | 15.4% | 14.3% | 175 | 7 / 9 | 0.80 |
| seed 1001 | -0.6 [-5.2,&nbsp;+4.0] | 16.7% | 16.1% | 174 | 7 / 8 | 1.00 |
| seed 1002 | +3.4 [-1.1,&nbsp;+8.0] | 12.6% | 16.0% | 175 | 11 / 5 | 0.21 |

File-only tasks give the agent only files; live-service tasks also run a server the agent
must attack. A gain above about 3.4 points is outside the 95% CI. Leaving out the 3 tasks
whose handout contains the flag gives +0.6 [-2.0,&nbsp;+3.3] (515 pairs).

A shown lesson appears in the agent's commands (a distinctive word of the lesson, matched
literally) in 58 of the 524 static runs; matched against the paired no-memory runs, the same
lessons appear in 46 by chance. The pool has 41 lessons; 39 were ever shown.

## 4. Why the gap? (hypotheses, not tested)

- **Few wins to learn from.** 45 development tasks at the ~15% no-memory solve rate would give
  a learner only about 7 solved runs per pass; the synthetic learners had 510 to 1,245 solved
  training runs in two passes.
- **Coarse credit.** "Used" is a keyword match, not proof that a lesson helped.
- **Self-written lessons.** The NYU lessons were written by the model from other tasks'
  reference material, 18 of 41 with no solution to learn from.

## 5. Limits

- The lesson pool is fixed; a self-extending memory (one that writes new lessons as it runs)
  is not tested.
- Real-CTF results cover static lessons only.
- Synthetic training explored using each task's family label; label-blind exploration is not
  tested.
- NYU CTF Bench: one model, 3 seeds, 10 rounds of at most 5 commands, internet blocked. About
  a quarter of model turns were malformed or cut-off replies with no command (23% without
  memory, 26% with static lessons), and 85 to 87% of runs used all 10 rounds. Public tasks may be in
  the models' pretraining data.
- The synthetic benchmark is one author-built domain (decoding puzzles).

## 6. Reproducing

With `pip install -r requirements.txt` (Python 3.10 or newer):

```bash
export PYTHONPATH=$(pwd)

python experiments/phase2_ladder.py --root results/lab/phase2_14b --out /tmp/ladder_14b.md
python experiments/phase2_ladder.py --root results/lab/phase2_32b --out /tmp/ladder_32b.md
python experiments/phase2_pool_draws.py --root results/lab/phase2_32b
python experiments/nyu_rerun_analysis.py
```

The two ladder runs reproduce `results/lab/phase2_14b/LADDER.md` and
`results/lab/phase2_32b/LADDER.md` (the `.json` alongside to floating-point precision). In the ladder, `B0` is no memory,
`B2det` static, `deployed_alphabet_tau01` learned (`_rep1`, `_rep2`: the independent draws;
`_tau02`: temperature 0.2), `deployed_global_tau01` the no-context control,
`deployed_causal_alphabet` causal credit and `oracle_static` the oracle set.
`experiments/nyu_lesson_sources.py` rebuilds `lesson_sources.tsv`; it needs a checkout of NYU
CTF Bench.

| file | contents |
|---|---|
| `results/lab/phase2_*/arms/<arm>/shard_00/progress.json` | per-condition held-out metrics (item, seed, family, solved) and training progress |
| `results/lab/phase2_*/families.json` | the 22 families and the family-selection seed (item names are in the progress files) |
| `results/lab/phase2_*/compute_report.json` | runs, tokens and solver-hours per condition |
| `results/realbench_nyu_v2/qwen_qwen3.8-27b-fp8/off.jsonl`, `static.jsonl` | every NYU run record (no memory, static): outcome, rounds, every command, and the first 1,000 characters of each command's output |
| `results/realbench_nyu_v2/qwen_qwen3.8-27b-fp8/analysis_final.json` | the analysis output for the numbers in §3 |
| `results/realbench_nyu_mem/qwen_qwen3.8-27b-fp8/lessons.jsonl` | the 41 lessons in the static condition's pool |
| `results/realbench_nyu_mem/qwen_qwen3.8-27b-fp8/lesson_sources.tsv` | what each lesson writer saw, and whether it included a solution |
| `docs/phase3/NYU_USABLE_test.txt`, `NYU_USABLE_development.txt` | the 176 test tasks run and the 45 usable development tasks |
| `results/realbench_nyu_v2/TASK_HEALTH.md` | the health check behind the task list |

**Running the agent again** needs `requirements-agent.txt` and a vLLM server.

- Synthetic benchmark: `experiments/phase2_launch.sh` (`STAGE=arms` launches every condition,
  including the causal-credit arms and the replication draws; the 14B rung ran without the extra
  causal arms, `CAUSAL_EXTRA=0`; one GPU for 14B, two for 32B). Set `ROOT` to a fresh directory and
  `FAMS_FILE=results/lab/phase2_14b/families.txt`; the script's header lists the other settings.
- NYU CTF Bench: `experiments/nyu_rerun.sh` with `ROOT` set to a fresh directory, two GPUs, the
  `cybench/cybench` image, Docker 28 or newer and `NYUCTF_ROOT` pointing at a checkout of the
  benchmark; see `realbench/README.md`.
