"""
benchmark/ - the controlled, REAL-LLM micro-CTF testbed (laptop pilot).

Why this exists and why it is the *primary* instrument for the false-lesson paper:
live picoCTF gives (a) pretraining contamination (Qwen has seen the flags) and
(b) NO ground-truth causal labels (we can't know which lesson actually caused a
solve). This corpus gives the opposite: every flag is freshly RANDOM
(`picoCTF{<random>}`) so there is zero contamination, and every challenge family
has a KNOWN true solution method plus a planted FALSE/POISON lesson - so
false-lesson *exposure* can be measured exactly against ground truth, oracle-free.

A real small LLM (`local_llm/server.py`) actually solves these with the real shell
tools via the real `solver.solve_challenge`. The flag is the only reward signal.
"""

from .micro_ctf import (
    generate_corpus, build_seed_graph, check_flag, FAMILIES, family_label,
)

__all__ = ["generate_corpus", "build_seed_graph", "check_flag", "FAMILIES", "family_label"]
