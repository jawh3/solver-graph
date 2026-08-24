#!/bin/bash
# Run the baseline PicoCTF solver using the ctf_solver conda environment.
# All arguments are forwarded to main.py.
#
# Examples:
#   ./run.sh                     # Solve all unsolved challenges
#   ./run.sh --limit 5           # Attempt at most 5
#   ./run.sh --category Crypto   # Only Crypto challenges
#   ./run.sh --dry-run --limit 1 # Quick test

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR"
${PYBIN:-python3} -m solver.main "$@"
