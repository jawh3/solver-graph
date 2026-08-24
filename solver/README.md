# solver

Multi-turn tool-augmented LLM solver for CTF challenges.

## Purpose

The core solve loop:

1. Scrape challenge metadata from PicoGym (or accept pre-scraped data)
2. Receive context from the retrieval system (M sampled knowledge graph nodes)
3. Multi-turn LLM interaction with tool use (shell commands, file analysis)
4. Flag extraction and verification
5. Flag submission to PicoCTF via Playwright

This module is the **reproducibility anchor** — its behavior defines the baseline against which knowledge-graph-augmented performance is measured.

## Key Files

- `solver.py` — Multi-turn tool-augmented LLM solver
- `tool_executor.py` — Sandboxed shell command runner
- `scraper.py` — PicoGym challenge scraper (Playwright)
- `submitter.py` — Flag submission via Playwright
- `main.py` — CLI orchestrator
- `logger.py` — Structured console logging

## Provenance

- Copied from `baseline-solver/` (functional code unchanged)
- Original repo: `baseline-solver` on branch `refactor/reorganization`
