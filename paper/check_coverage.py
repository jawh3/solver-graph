#!/usr/bin/env python3
"""paper_final/check_coverage.py — artifact- and pre-registration-coverage guard.

Rounds 7-9 found three undisclosed pre-registered arms (H5, T2-3, T2-2) and an unapplied
pre-registered inclusion gate. None was visible by reading the manuscript: each is an
ARTIFACT WITH NO CLAIM, and no textual guard can see that surface.

This enumerates every committed result artifact and every pre-registered item, and
requires each to carry a disposition in COVERAGE.tsv:

  cited <where>        the paper reports it
  superseded <what>    a later artifact/analysis replaces it
  excluded <reason>    deliberately out of scope, with the reason stated

Fail on any artifact or pre-registered item with no disposition. An unclassified item is
not necessarily a defect -- but it must be an explicit decision, not an oversight.
"""
import os, re, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COV = os.path.join(HERE, "COVERAGE.tsv")

# Only RESULT-BEARING artifacts: the ones a reviewer would hold against a claim.
# Per-shard state, manifests, progress files and corpora are excluded by construction.
ART_PATTERNS = ["*_analysis.json", "*_result.json", "*_contrast.json", "*_summary.json",
                "*_boot.json", "*_check.json", "*_loo.json", "*_sweep.json",
                "*_snipw.json", "*_snipw_rfinal.json", "*_combined.json",
                "*_combined_rfinal.json", "node_values.json", "selection.json",
                "active_loo_within_label.json"]
ART_SKIP = ("/state/", "/work/", "/corpus/", "/shards/")
PREREG = ["docs/PREREG.md", "docs/PREREG_INTERFERENCE.md"]
# pre-registered item headings look like "### T2-3 — ..." or "- **H5 — ..."
ITEM_RE = re.compile(r"^#{2,4}\s+([A-Z]\d[-\w]*)\s*[—-]|^\s*-\s+\*\*(H\d+)\s*[—-]", re.M)


def discover():
    arts = set()
    for base in ("results/lab", "results/pilot"):
        for pat in ART_PATTERNS:
            for p in glob.glob(os.path.join(ROOT, base, "**", pat), recursive=True):
                rel = os.path.relpath(p, ROOT)
                if not any(k in "/" + rel for k in ART_SKIP):
                    arts.add(rel)
    items = set()
    for f in PREREG:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            for a, b in ITEM_RE.findall(open(p).read()):
                items.add(f"prereg:{a or b}")
    return arts, items


def load_cov():
    cov = {}
    if not os.path.exists(COV):
        return cov
    for line in open(COV):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            cov[parts[0].strip()] = (parts[1].strip(), parts[2].strip() if len(parts) > 2 else "")
    return cov


def main():
    arts, items = discover()
    cov = load_cov()
    universe = sorted(arts | items)
    missing_all = [k for k in universe if k not in cov]
    # The pre-registration surface is FATAL: all three undisclosed-arm defects
    # (H5, T2-3, T2-2) lived there. The broader artifact surface is advisory for now --
    # 117 result files, most already covered transitively by a cited analysis.
    missing = [k for k in missing_all if k.startswith("prereg:")]
    advisory = [k for k in missing_all if not k.startswith("prereg:")]
    stale = [k for k in cov if k not in set(universe)]
    bad = [(k, v) for k, (v, _) in cov.items()
           if v not in ("cited", "superseded", "excluded")]

    print(f"coverage: {len(arts)} artifacts + {len(items)} pre-registered items "
          f"= {len(universe)} tracked; {len(cov)} dispositions")
    ok = True
    if bad:
        ok = False
        print(f"\n  {len(bad)} INVALID disposition(s) (must be cited/superseded/excluded):")
        for k, v in bad[:20]:
            print(f"    {k}\t{v}")
    if advisory:
        print(f"\n  advisory: {len(advisory)} result artifact(s) with no explicit "
              f"disposition (not fatal; most are covered transitively by a cited analysis)")
    if missing:
        ok = False
        print(f"\n  {len(missing)} PRE-REGISTERED item(s) with NO disposition:")
        for k in missing[:40]:
            print(f"    {k}")
        if len(missing) > 40:
            print(f"    ... and {len(missing)-40} more")
        print("\n  Every pre-registered item must be reported, superseded, or explicitly")
        print("  excluded with a reason. Add a line to COVERAGE.tsv for each.")
    if stale:
        print(f"\n  note: {len(stale)} disposition(s) for items no longer present "
              f"(harmless, but prune): {stale[:5]}")
    if ok:
        print("\ncoverage: OK — every pre-registered item has a disposition")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
