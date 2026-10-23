"""
batch_test/ - coordinated REAL-GPU run harness for the GPU server.

Everything in the rest of the repo is mock-testable on a laptop. This package is
the *only* place that touches the real GPU: it swaps the mock embedder/reranker/
solver for the real ones and runs the SAME eval harness (`evaluation/`) so the
metrics, logs (§12), and OPE (§13) are identical to the mock smoke test - only the
backends change.

Run order on the GPU machine (see README.md):
  1. `python -m batch_test.smoke_real`   - preflight: vLLM up? embed? one real solve?
  2. `python -m batch_test.run_batch ...` - the coordinated eval ladder on real data.

Because vLLM is a scarce, coordinated event (§2), the runner maximizes information
per run: it logs propensities for OPE so the parameter sweep happens OFFLINE
afterward (`python -m ope.sweep`) with NO additional GPU.
"""
