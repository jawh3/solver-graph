# evaluation

> Early component notes (May 2026). They describe the first design and may be out of date; see the root
> README and RESULTS.md for the current state.


Metrics, analysis, and experiment harness.

## Purpose

Provides tools for measuring and analyzing solver performance:

- Solve rate (overall, per-category, per-difficulty)
- Time-to-flag distributions
- RAG lift measurement (with vs without knowledge graph context)
- Per-(category × difficulty) standardized reward computation
- False lesson propagation analysis

## Key Components

- Result loading and analysis from JSONL files
- Per-category, per-model, per-worker statistics
- Standardized reporting format

## Provenance

- Analysis functions extracted from `baseline-solver/evaluate.py`
- Architecture per `system_description.md` §Update Loop (reward computation)
