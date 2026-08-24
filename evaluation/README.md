# evaluation

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

- Analysis functions extracted from `the superseded baseline solver (not in this release)`
- Architecture per a superseded design note (not included in this release) §Update Loop (reward computation)
