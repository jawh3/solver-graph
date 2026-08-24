#!/usr/bin/env bash
# paper_final/make_bundle.sh — build the self-contained handoff bundle.
#
# Produces solver-graph-paper-bundle.zip: everything a downstream reader (human or AI)
# needs to understand the paper's claims, the full experimental protocol, and the
# provenance of every number -- with no access to this repository.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
OUT="$HERE/solver-graph-paper-bundle"
ZIP="$HERE/solver-graph-paper-bundle.zip"

rm -rf "$OUT" "$ZIP"
mkdir -p "$OUT"/{paper,figures,protocol,numbers,appendix_c_artifacts,arxiv}

# --- 1. the manuscript -------------------------------------------------------
cp "$HERE/main.pdf"              "$OUT/paper/"
cp "$HERE"/*.tex                 "$OUT/paper/"
cp "$HERE/main.bbl"              "$OUT/paper/"
cp "$HERE/references.bib"        "$OUT/paper/"
cp "$HERE/abstract_plain.txt"    "$OUT/paper/"
cp "$HERE"/fig_*.pdf             "$OUT/paper/"   # main.tex sets \graphicspath{{./}}

# --- 2. figures (vector, A0-scalable) ---------------------------------------
for f in fig_pipeline fig_separation fig_loo fig_q_by_config fig_solve_rates \
         fig_q_trajectory fig_ope_sweep; do
  cp "$HERE/$f.pdf" "$OUT/figures/" 2>/dev/null || true
done

# --- 3. the briefing + guard-rails ------------------------------------------
cp "$HERE/PROTOCOL.md"            "$OUT/protocol/"
# CLAIMS_v2.md is a 2026-06-25 working ledger, superseded by the manuscript. It carries
# two dangerous errors, so it ships with a correction banner rather than bare.
{ cat <<'BANNER'
> **SUPERSEDED — read `PROTOCOL.md` §2 and `main.pdf` first.**
>
> This is the authors' 2026-06-25 working claim ledger, kept for context. Three
> corrections you must apply if you read on:
>
> 1. **C4/C5's `-0.0745` is the ON-TARGET contrast, not the marginal.** The
>    marginal-inclusion ATE at M=1 is **-0.021, p=.25**. Presenting -0.0745 as the
>    marginal claims a significant deployed causal harm the paper explicitly disclaims.
> 2. **`wrongflag` is UNTESTED, not "MISSED".** The agent followed that recipe on 11 of
>    453 sampled runs (follow rate 0.024). The null comes from non-exposure, not from a
>    demonstrated estimator blind spot; under forced exposure a constructed leave-one-out
>    detects it at -0.43.
> 3. **The ladder is FIVE rungs** (Qwen2.5 3B-32B plus Llama-3.1-8B), not the "6-rung
>    0.5B-32B" this file describes. The paper is 22+ pages, not 16.

---

BANNER
  cat "$ROOT/docs/CLAIMS_v2.md"; } > "$OUT/protocol/CLAIMS_v2.md"
cp "$HERE/ARXIV_SUBMISSION.md"    "$OUT/protocol/"
cp "$ROOT/README.md"              "$OUT/protocol/repo_README.md"
cp "$ROOT/docs/architecture.html" "$OUT/protocol/" 2>/dev/null || true
cp "$ROOT/LICENSE"                "$OUT/protocol/"
cp "$ROOT/extra_work/APPENDIX_C_DECISIONS.md" "$OUT/appendix_c_artifacts/"

# --- 4. every number, and the scripts that emit them ------------------------
cp "$HERE"/numbers*.tex "$HERE/tables_pilot.tex" "$OUT/numbers/" 2>/dev/null || true
cp "$HERE/emit_appendix_numbers.py" "$HERE/make_abstract_plain.py" "$OUT/numbers/"

# --- 5. Appendix C source artifacts (so its numbers are re-derivable) -------
for f in g1_verifier_acceptance_crypto_depth1 g1_verifier_acceptance_sql \
         g2_step_contrast g2_mechanism inband_selection_crypto; do
  cp "$ROOT/extra_work/results/$f.json" "$OUT/appendix_c_artifacts/" 2>/dev/null || true
  cp "$ROOT/extra_work/results/$f.provenance.json" "$OUT/appendix_c_artifacts/" 2>/dev/null || true
done
mkdir -p "$OUT/appendix_c_artifacts/g2_step_probe" "$OUT/appendix_c_artifacts/g2_mechanism"
cp "$ROOT/extra_work/results/g2_step_probe/rows_shard"*.jsonl \
   "$OUT/appendix_c_artifacts/g2_step_probe/" 2>/dev/null || true
# the mechanism per-run rows: the aggregate g2_mechanism.json is pooled over commands and
# cannot reproduce any paired or both-solved quantity, so the rows must ship.
cp "$ROOT/extra_work/results/g2_mechanism/rows_shard"*.jsonl \
   "$OUT/appendix_c_artifacts/g2_mechanism/" 2>/dev/null || true

# --- 6. the arXiv upload package --------------------------------------------
# REGENERATE, never copy: the package is built from the same sources and silently goes
# stale the moment any .tex is edited after the last make_arxiv.py run. A stale package
# compiles clean and looks correct, so nothing signals the defect.
echo "regenerating the arXiv package from current sources..."
( cd "$HERE" && "${PYTHON:-python3}" make_arxiv.py ) || { echo "  make_arxiv.py FAILED"; exit 1; }
cp "$HERE/arxiv_submission.tar.gz" "$OUT/arxiv/"
# the packaged appendix must match the working copy, or the upload carries stale claims
if ! diff -q <(tar xzOf "$HERE/arxiv_submission.tar.gz" appendix_latent.tex) \
             <(sed -E 's/^[[:space:]]*%.*$//' "$HERE/appendix_latent.tex" | grep -v '^$') \
             >/dev/null 2>&1; then
  a=$(tar xzOf "$HERE/arxiv_submission.tar.gz" numbers_latent.tex | grep -c newcommand)
  b=$(grep -c newcommand "$HERE/numbers_latent.tex")
  [ "$a" = "$b" ] || { echo "  FATAL: packaged numbers_latent.tex has $a macros, working copy has $b"; exit 1; }
fi
echo "  arXiv package regenerated and macro-count matched"

cp "$HERE/an internal handoff document (not included in this release)" "$OUT/README.md"

# --- self-test the commands BUNDLE_README documents, before shipping ---
# Mode-(iii) guard: no per-rung statistic may be reported at only some of its rungs.
echo "self-test: rung-selectivity..."
( cd "$HERE" && "${PYTHON:-python3}" check_rung_selectivity.py ) \
  || { echo "  FATAL: partially-printed per-rung family (see above)"; exit 1; }

echo "self-test: estimand vocabulary..."
( cd "$HERE" && "${PYTHON:-python3}" check_estimand_vocab.py ) \
  || { echo "  FATAL: citation vocabulary attached to an on-target/marginal estimand"; exit 1; }

echo "self-test: pre-registration coverage..."
( cd "$HERE" && "${PYTHON:-python3}" check_coverage.py ) \
  || { echo "  FATAL: pre-registered item with no disposition"; exit 1; }

echo "self-test: re-deriving Appendix C numbers from the bundled artifacts..."
# Run in a THROWAWAY dir: the emitter writes beside itself, so running it in $OUT
# would overwrite the shipped numbers/numbers_latent.tex with whatever the (possibly
# incomplete) artifacts produce -- and then ship the disagreement silently.
S="$(mktemp -d)"; cp "$OUT/numbers/"*.py "$S/"
( cd "$S" && APPENDIX_C_ARTIFACTS="$OUT/appendix_c_artifacts" \
    "${PYTHON:-python3}" emit_appendix_numbers.py ) > "$S/emit.log" 2>&1 \
  || { cat "$S/emit.log"; echo "  FAILED"; rm -rf "$S"; exit 1; }
grep -q "^UNFILLED:" "$S/emit.log" \
  && { cat "$S/emit.log"; echo "  FATAL: unfilled macros"; rm -rf "$S"; exit 1; }
cmp -s "$S/numbers_latent.tex" "$OUT/numbers/numbers_latent.tex" \
  || { echo "  FATAL: re-derived numbers_latent.tex differs from the shipped one"; rm -rf "$S"; exit 1; }
rm -rf "$S"; echo "  emit_appendix_numbers.py OK"
( cd "$OUT" && "${PYTHON:-python3}" numbers/make_abstract_plain.py >/dev/null ) \
  && echo "  make_abstract_plain.py OK" || { echo "  FAILED"; exit 1; }
if command -v pdflatex >/dev/null; then
  # Build in a THROWAWAY dir: building in $OUT/paper would overwrite the good main.pdf
  # with a single-pass build whose refs are all "??" -- and the page count would be
  # unchanged, so nothing would signal it. Two passes, and fail on unresolved refs.
  T="$(mktemp -d)"; cp "$OUT/paper/"* "$T/"
  ( cd "$T" && pdflatex -interaction=nonstopmode main >/dev/null 2>&1 \
             && pdflatex -interaction=nonstopmode main >/dev/null 2>&1
    grep -qa "not found" main.log && { echo "  pdflatex: missing assets"; exit 1; }
    grep -qaE "Reference .* undefined|Citation .* undefined|There were undefined references" main.log \
      && { echo "  FATAL: bundle sources do not resolve refs/citations"; exit 1; }
    true ) \
    || { rm -rf "$T"; exit 1; }
  rm -rf "$T"; echo "  pdflatex OK"
fi
# the shipped PDF must be the working PDF, and must have no unresolved references
cmp -s "$HERE/main.pdf" "$OUT/paper/main.pdf" \
  || { echo "FATAL: bundled main.pdf is not the working main.pdf"; exit 1; }
if command -v pdftotext >/dev/null; then
  # grep -c (not -q): -q exits at the first match, pdftotext takes SIGPIPE, and under
  # `set -o pipefail` the pipeline returns 141 -- so a -q guard is SILENT exactly when
  # the PDF is bad. -c drains stdin; || true absorbs grep's exit 1 on zero matches.
  n=$(pdftotext "$OUT/paper/main.pdf" - | grep -c '??' || true)
  [ "$n" = 0 ] || { echo "FATAL: bundled PDF has $n unresolved references"; exit 1; }
fi

( cd "$HERE" && zip -qr "$ZIP" "$(basename "$OUT")" )
echo "wrote $ZIP ($(du -h "$ZIP" | cut -f1)), $(find "$OUT" -type f | wc -l) files"
