#!/usr/bin/env python3
"""paper_final/make_abstract_plain.py — emit abstract_plain.txt for the arXiv metadata box.

ARXIV_SUBMISSION.md instructs the submitter to paste arxiv_dist/abstract_plain.txt, but
no such file was ever generated (make_arxiv.py does not emit one). This does. It de-TeXes
the abstract from main.tex, resolving the macros the abstract uses, and prints the
character count against arXiv's 1,920-char limit so the submitter finds out here rather
than in the submission form.
"""
import re, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
# main.tex and the numbers*.tex live beside this script in the repo, but in the handoff
# bundle they are in ../paper and ../numbers. Resolve both layouts.
SRC = next((d for d in (HERE, os.path.join(HERE, "..", "paper"))
            if os.path.exists(os.path.join(d, "main.tex"))), HERE)
NUM = next((d for d in (HERE, os.path.join(HERE, "..", "numbers"))
            if os.path.exists(os.path.join(d, "numbers.tex"))), HERE)
LIMIT = 1920


def macros():
    m = {}
    for fn in ("numbers.tex", "numbers_solution_defaults.tex", "numbers_solution.tex",
               "numbers_identity_a.tex", "numbers_latent.tex"):
        p = os.path.join(NUM, fn)
        if not os.path.exists(p):
            continue
        for name, val in re.findall(r"\\(?:new|renew|provide)command\{\\(\w+)\}\{([^}]*)\}",
                                    open(p).read()):
            m[name] = val
    return m


def detex(s, m):
    s = re.sub(r"^\s*%.*$", "", s, flags=re.M)
    for _ in range(3):                                   # macros can nest
        s = re.sub(r"\\(\w+)\{\}", lambda x: m.get(x.group(1), x.group(0)), s)
        s = re.sub(r"\\(\w+)(?![a-zA-Z])", lambda x: m.get(x.group(1), x.group(0)), s)
    reps = [(r"\\,\\%", "%"), (r"\\emph\{([^{}]*)\}", r"\1"), (r"\\textbf\{([^{}]*)\}", r"\1"),
            (r"\\text\{([^{}]*)\}", r"\1"), (r"\\noindent", ""), (r"\\,", " "),
            (r"\\%", "%"), (r"\\\$", "$"), (r"\\ ", " "), (r"---", " - "), (r"--", "-"),
            (r"``", '"'), (r"''", '"'), (r"\\citep\{[^}]*\}", ""), (r"\\cite\{[^}]*\}", ""),
            (r"\$([^$]*)\$", r"\1"), (r"\\[a-zA-Z]+", ""), (r"[{}]", "")]
    for a, b in reps:
        s = re.sub(a, b, s)
    s = re.sub(r"\s*\n\s*", " ", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def main():
    src = open(os.path.join(SRC, "main.tex")).read()
    body = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1)
    txt = detex(body, macros())
    dest = os.path.join(HERE, "abstract_plain.txt")
    open(dest, "w").write(txt + "\n")
    n = len(txt)
    print(f"wrote {dest}: {n} chars (arXiv limit {LIMIT}) -> "
          f"{'OK' if n <= LIMIT else f'OVER BY {n - LIMIT}'}")
    if n > LIMIT:
        sys.exit(1)


if __name__ == "__main__":
    main()
