# Release manifest - curated public tree

Built 2026-10-23 by `release/build_release.py` from the private working repo.

**236 files including this manifest, 83M.** Only paths on the include list below are copied.

## Included

| path | why |
|---|---|
| `.gitignore` | ignore rules [REPLACED] |
| `CTF_SOLVER_ARCHITECTURE_v2.md` | the system design spec |
| `LICENSE` | MIT for code/data; poster PDFs all rights reserved; NYU CTF Bench excerpts GPL-2.0 |
| `LICENSES/GPL-2.0.txt` | the GPL-2.0 text, for the NYU CTF Bench excerpts in the run records |
| `OPEN_DECISIONS.md` | design decisions with reasoning and reversibility |
| `README.md` | repo entry point, written for a public audience |
| `RESULTS.md` | setup, results, controls and reproduction commands |
| `bandit` | contextual bandit + hygiene (C6, C9) |
| `batch_test` | the real-model solve adapters the synthetic-benchmark campaigns use [SANITIZED] |
| `benchmark` | the synthetic micro-CTF generator |
| `config.py` | single-source swept/ablated parameters |
| `docs/phase3/NYU_USABLE_development.txt` | the 45 NYU CTF Bench development tasks (lesson sources) |
| `docs/phase3/NYU_USABLE_test.txt` | the 176 NYU CTF Bench test tasks run (read by the analysis) |
| `evaluation` | baselines, paired bootstrap, datasets (C4) |
| `experiments` | the drivers and analyses of the two campaigns the poster reports (EXPERIMENTS_KEEP) [SANITIZED] |
| `instrumentation` | the run-record schema and JSONL logger |
| `knowledge_graph` | typed-node lesson graph (C1) |
| `local_llm` | OpenAI-compatible server shim |
| `ope` | IPS/SNIPS/DR estimators |
| `pipeline` | orchestrator, system specs, causal credit (C8) |
| `poster/Hugglestone_CARS26_abstract.pdf` | the one-page CARS 2026 abstract |
| `poster/Hugglestone_CARS26_poster.pdf` | the CARS 2026 poster (40 x 30 in) |
| `realbench` | real-CTF adapters (NYU CTF Bench, Cybench): provisioning, flag check, lessons [SANITIZED] |
| `requirements-agent.txt` | additional dependencies for running the agent (vLLM, embedders, ...) |
| `requirements.txt` | dependencies for the analysis and the tests |
| `results/lab/phase2_14b/` | synthetic benchmark (Phase 2): families, compute report, and each arm's held-out metrics (arms/*/shard_00/progress.json) |
| `results/lab/phase2_14b/LADDER.md` | the held-out ladder (.md, .json), recomputed at build time from the shipped data; numbers identical to the recorded one |
| `results/lab/phase2_14b/probe/probe.json` | the family probe: per-family solve rates of the 124 candidates |
| `results/lab/phase2_14b/probe/selection.json` | the families that passed the probe's criterion |
| `results/lab/phase2_32b/` | synthetic benchmark (Phase 2): families, compute report, and each arm's held-out metrics (arms/*/shard_00/progress.json) |
| `results/lab/phase2_32b/LADDER.md` | the held-out ladder (.md, .json), recomputed at build time from the shipped data; numbers identical to the recorded one |
| `results/realbench_nyu_mem/qwen_qwen3.8-27b-fp8/lesson_sources.tsv` | what each lesson writer saw (experiments/nyu_lesson_sources.py) |
| `results/realbench_nyu_mem/qwen_qwen3.8-27b-fp8/lessons.jsonl` | the 41 lessons in the static condition's pool |
| `results/realbench_nyu_v2/TASK_HEALTH.md` | which test tasks are run, and why the others are not |
| `results/realbench_nyu_v2/qwen_qwen3.8-27b-fp8/analysis_final.json` | the NYU analysis output, recomputed at build time from the shipped records (identical numbers) |
| `results/realbench_nyu_v2/qwen_qwen3.8-27b-fp8/off.jsonl` | NYU CTF Bench: every no-memory run record |
| `results/realbench_nyu_v2/qwen_qwen3.8-27b-fp8/off.summary.json` | per-arm run summary |
| `results/realbench_nyu_v2/qwen_qwen3.8-27b-fp8/static.jsonl` | NYU CTF Bench: every static-lessons run record |
| `results/realbench_nyu_v2/qwen_qwen3.8-27b-fp8/static.jsonl.no_retry` | the 9 static pairs whose one retry happened at resume (runner input) |
| `results/realbench_nyu_v2/qwen_qwen3.8-27b-fp8/static.summary.json` | per-arm run summary |
| `retrieval` | recall/rerank/MMR/propensity-logged sampling (C2) |
| `reward` | R = o(1+lambda*e) and grading (C5) |
| `solver` | the solver and sandboxed tool executor |
| `tests` | test suite; runs in mock mode with no GPU |
| `writeback` | gated promotion + provenance (C7) |

## Excluded

The public tree omits internal working notes, session logs, legacy scratch directories, unrelated material, machine-specific configuration, the scripts of experiments this release does not report (for example the comparison of solver models that chose the NYU solver), and the superseded August 2026 release (still in this repository's history at 43ecf5d). Nothing required to reproduce a reported number was withheld.

## Sanitized files

These were copied with user- and machine-specific names and absolute paths replaced by environment variables or neutral placeholders.

- `CTF_SOLVER_ARCHITECTURE_v2.md`
- `OPEN_DECISIONS.md`
- `bandit/__init__.py`
- `bandit/bandit.py`
- `bandit/hygiene.py`
- `bandit/linucb.py`
- `batch_test/README.md`
- `batch_test/__init__.py`
- `batch_test/real_adapters.py`
- `batch_test/run_batch.py`
- `batch_test/smoke_real.py`
- `benchmark/__init__.py`
- `benchmark/hard_ctf.py`
- `benchmark/micro_ctf.py`
- `benchmark/sql_domain.py`
- `benchmark/struct_domain.py`
- `config.py`
- `evaluation/baselines.py`
- `evaluation/datasets.py`
- `evaluation/evaluate.py`
- `evaluation/harness.py`
- `evaluation/run_eval.py`
- `evaluation/stats.py`
- `evaluation/synthetic.py`
- `experiments/a1_contrast.py`
- `experiments/compute_report.py`
- `experiments/constructed_loo.py`
- `experiments/execution_verifier.py`
- `experiments/lab_analysis.py`
- `experiments/lab_probe.py`
- `experiments/learning_curve.py`
- `experiments/loo_report.py`
- `experiments/nyu_lesson_sources.py`
- `experiments/nyu_rerun.sh`
- `experiments/nyu_rerun_analysis.py`
- `experiments/nyu_rerun_finish.sh`
- `experiments/ontarget_analysis.py`
- `experiments/phase2_32b_after_causal.sh`
- `experiments/phase2_32b_causal_extra_after.sh`
- `experiments/phase2_channels.py`
- `experiments/phase2_estimands.py`
- `experiments/phase2_gpu1_after.sh`
- `experiments/phase2_gpu1_chain.sh`
- `experiments/phase2_health.sh`
- `experiments/phase2_keeper.sh`
- `experiments/phase2_ladder.py`
- `experiments/phase2_launch.sh`
- `experiments/phase2_pool_draws.py`
- `experiments/phase2_start_32b.sh`
- `experiments/phase2_within_poison.py`
- `experiments/run_campaign.py`
- `experiments/smoke_checks.py`
- `instrumentation/__init__.py`
- `instrumentation/jsonl_logger.py`
- `instrumentation/records.py`
- `knowledge_graph/DEV_LOG.md`
- `knowledge_graph/README.md`
- `knowledge_graph/SWEEPS.md`
- `local_llm/__init__.py`
- `local_llm/server.py`
- `ope/__init__.py`
- `ope/causal_value.py`
- `ope/ope.py`
- `ope/sweep.py`
- `pipeline/__init__.py`
- `pipeline/causal_credit.py`
- `pipeline/orchestrator.py`
- `pipeline/system_spec.py`
- `realbench/README.md`
- `realbench/__init__.py`
- `realbench/bakeoff_summary.py`
- `realbench/cybench.py`
- `realbench/distill.py`
- `realbench/learn.py`
- `realbench/lessons.py`
- `realbench/memory_analysis.py`
- `realbench/nyu_screen_analysis.py`
- `realbench/nyuctf.py`
- `realbench/oracle_check.py`
- `realbench/run_cybench.py`
- `realbench/select_models.py`
- `realbench/solve_adapter.py`
- `results/realbench_nyu_v2/qwen_qwen3.8-27b-fp8/off.jsonl`
- `results/realbench_nyu_v2/qwen_qwen3.8-27b-fp8/static.jsonl`
- `retrieval/DEV_LOG.md`
- `retrieval/README.md`
- `retrieval/SWEEPS.md`
- `retrieval/context_resolution.py`
- `retrieval/diversity.py`
- `retrieval/embedder.py`
- `retrieval/memory_client.py`
- `retrieval/reranker.py`
- `retrieval/retriever.py`
- `retrieval/sampling.py`
- `reward/__init__.py`
- `reward/grader.py`
- `reward/reward.py`
- `solver/README.md`
- `solver/citation.py`
- `solver/container_executor.py`
- `solver/context_assembly.py`
- `solver/logger.py`
- `solver/main.py`
- `solver/mock_solver.py`
- `solver/scraper.py`
- `solver/solver.py`
- `solver/submitter.py`
- `solver/tool_executor.py`
- `tests/test_benchmark_audit_fixes.py`
- `tests/test_causal_value_ci.py`
- `tests/test_container_executor.py`
- `tests/test_corpus_scaling.py`
- `tests/test_execution_verifier.py`
- `tests/test_fix_catalog.py`
- `tests/test_malignant_teeth.py`
- `tests/test_observational_causal.py`
- `tests/test_observational_nonadditive.py`
- `tests/test_online_observational.py`
- `tests/test_orchestrator.py`
- `tests/test_realbench_cybench.py`
- `tests/test_realbench_learn.py`
- `tests/test_struct_domain.py`
- `tests/test_tool_executor.py`
- `tests/test_tool_filesize_guard.py`
- `writeback/__init__.py`
- `writeback/forge_competitor.py`
- `writeback/writeback.py`

## Known gaps

- Code comments and docstrings in `experiments/`, `realbench/` and `tests/` cite internal working notes under `docs/` that are not part of this release.
- The raw per-run training records of the synthetic benchmark (5.8 GB) are not shipped; the held-out metrics every reported number uses are.

