#!/usr/bin/env python3
"""paper/make_arxiv.py — build a self-contained arXiv submission package.

Run from paper/:  python3 make_arxiv.py
Produces arxiv_dist/ (flattened, comment-stripped, test-compilable in isolation)
and arxiv_submission.tar.gz. arXiv cannot see files outside the upload, so the
../results/... graphicspath and tables input are rewritten to local copies; the
.bbl is included because arXiv does not run bibtex the way this repo does.
Full-line % comments are stripped (internal working notes must not ship in the
public source); trailing/inline % (spacing control) is preserved.
"""
import re, shutil, subprocess, sys, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE / "arxiv_dist"
PILOT_FIGS = HERE.parent / "results" / "pilot" / "figures"
if not PILOT_FIGS.exists():          # figures were staged locally in paper_final/
    PILOT_FIGS = HERE

TEX_FILES = ["main.tex", "method_causal.tex", "results_body.tex",
             "results_solution.tex", "theorem_identifiability.tex",
             "appendix_latent.tex",
             "numbers.tex", "numbers_solution_defaults.tex",
             "numbers_solution.tex", "numbers_identity_a.tex",
             "numbers_latent.tex"]
PILOT_FIG_NAMES = ["fig_pipeline.pdf", "fig_solve_rates.pdf", "fig_q_by_config.pdf",
                   "fig_q_trajectory.pdf", "fig_loo.pdf", "fig_ope_sweep.pdf"]
LOCAL_FIGS = ["fig_separation.pdf"]

FULL_LINE_COMMENT = re.compile(r"^\s*%")

def strip_comments(text: str) -> str:
    kept = [l for l in text.splitlines() if not FULL_LINE_COMMENT.match(l)]
    out, blank = [], False
    for l in kept:                      # collapse the blank runs stripping leaves behind
        if l.strip() == "":
            if blank: continue
            blank = True
        else:
            blank = False
        out.append(l)
    return "\n".join(out) + "\n"

def main():
    if DIST.exists(): shutil.rmtree(DIST)
    DIST.mkdir()

    for name in TEX_FILES:
        text = (HERE / name).read_text()
        if name == "main.tex":
            text = text.replace(r"\graphicspath{{../results/pilot/figures/}}",
                                r"\graphicspath{{./}}")   # no-op if already local
        if name == "results_body.tex":
            text = text.replace(r"\input{../results/pilot/tables.tex}",
                                r"\input{tables_pilot.tex}")
        (DIST / name).write_text(strip_comments(text))

    tables_src = HERE / "tables_pilot.tex"
    if not tables_src.exists():
        tables_src = HERE.parent / "results" / "pilot" / "tables.tex"
    (DIST / "tables_pilot.tex").write_text(strip_comments(tables_src.read_text()))
    for f in PILOT_FIG_NAMES: shutil.copy(PILOT_FIGS / f, DIST / f)
    for f in LOCAL_FIGS: shutil.copy(HERE / f, DIST / f)
    shutil.copy(HERE / "references.bib", DIST / "references.bib")
    shutil.copy(HERE / "main.bbl", DIST / "main.bbl")

    # abstract_plain.txt: ARXIV_SUBMISSION.md tells the submitter to paste this into the
    # arXiv metadata box, but nothing used to generate it. Generate it here so it exists
    # and so the 1,920-char limit is checked before the submission form rejects it.
    subprocess.run([sys.executable, str(HERE / "make_abstract_plain.py")], check=True)
    shutil.copy(HERE / "abstract_plain.txt", DIST / "abstract_plain.txt")

    # test-compile in isolation (twice for refs); the .bbl supplies the bibliography
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main"],
                           cwd=DIST, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    if "Output written on main.pdf" not in r.stdout:
        print(r.stdout[-3000:]); sys.exit("dist compile FAILED")
    pages = re.search(r"Output written on main.pdf \((\d+) pages", r.stdout).group(1)
    log = (DIST / "main.log").read_text(errors="ignore")
    problems = [l for l in log.splitlines() if "undefined" in l.lower() or "Overfull" in l]
    print(f"dist compiles: {pages} pages; {len(problems)} undefined/overfull lines")
    for l in problems[:10]: print("  ", l)

    # package only the sources arXiv needs (no build artifacts)
    tar_path = HERE / "arxiv_submission.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in sorted(DIST.iterdir()):
            if f.suffix in {".tex", ".bbl", ".pdf"} and f.name != "main.pdf":
                tar.add(f, arcname=f.name)
            elif f.name in ("references.bib", "abstract_plain.txt"):
                tar.add(f, arcname=f.name)
    print(f"wrote {tar_path.name} ({tar_path.stat().st_size//1024} KB)")

if __name__ == "__main__":
    main()
