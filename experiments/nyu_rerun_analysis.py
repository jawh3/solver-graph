#!/usr/bin/env python3
"""experiments/nyu_rerun_analysis.py - the NYU CTF Bench analysis: static lessons vs no memory.

Unit: a (task, seed) pair present and error-free in both arms. Primary analysis = seeds BOTH arms completed (every task
has a record in both); a partly finished seed is reported separately and never pooled.

  H1  RD(static - OFF), all tasks: task-cluster bootstrap 95% CI (10,000 resamples, seed 0) + exact two-sided sign test on
      discordant (task, seed) pairs. Claimed only if the CI excludes 0 AND p < 0.05. The
      task-level sign test (which respects clustering) is reported beside it.
  H2  the same on the 58 file-only tasks (same rule); the service-task RD is reported alongside.
  Descriptive: solve rates (Wilson), per category, without flag-in-handout tasks, sandbox counts, rounds on solved tasks.

    PYTHONPATH=<repo> python experiments/nyu_rerun_analysis.py [--dir results/realbench_nyu_v2/qwen_qwen3.8-27b-fp8]
        [--out docs/phase3/NYU_RERUN_RESULTS_<date>.md] [--json <path>] [--provisional]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from realbench.memory_analysis import cluster_bootstrap, sign_test, wilson  # noqa: E402

N_BOOT = 10_000
LESSONS = REPO / "results/realbench_nyu_mem/qwen_qwen3.8-27b-fp8/lessons.jsonl"


def citation_vs_chance(off: dict, st: dict, keys: list, lessons_path: Path = LESSONS) -> dict:
    """Descriptive: share of pairs in which a shown lesson is cited in the static trace, vs the chance
    rate - the same lessons matched against the paired no-memory trace (same matcher as the run: distinctive, literal,
    minus generic_keywords)."""
    if not lessons_path.exists():
        return {}
    from realbench.lessons import LessonPool, generic_keywords
    from solver.citation import match_citations
    by_id = {n.id: n for n in LessonPool.load(lessons_path).nodes}
    on_c = ch_c = n = 0
    for key in keys:
        mem = (st[key].get("extra") or {}).get("memory") or {}
        nodes = [by_id[i] for i in mem.get("lessons") or [] if i in by_id]
        if not nodes:
            continue
        n += 1
        on_c += bool(mem.get("cited"))
        generic = sorted(set(generic_keywords(nodes)))
        ms = match_citations(nodes, off[key].get("trace") or [], mode="distinctive", literal=True, generic_tokens=generic)
        ch_c += any(m.cited for m in ms)
    return {"pairs": n, "cited_static": on_c, "cited_chance_off": ch_c}


def load(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["task_id"], int((r.get("extra") or {}).get("trial", 0)))] = r
    return out


def sandbox_ok(r: dict) -> bool:
    """True unless a command was blocked or ran outside the container. A record with no sandbox counts never started a
    solve (provisioning failed before any command): it is an errored record, counted under no_solve, not a violation."""
    s = (r.get("extra") or {}).get("sandbox") or {}
    return not s or (not s.get("rejected") and s.get("dispatched") == s.get("container_runs"))


def no_solve(r: dict) -> bool:
    return not ((r.get("extra") or {}).get("sandbox")) and bool(r.get("error"))


def mcnemar_exact(rows: list) -> tuple:
    """Exact two-sided sign test on discordant (task, seed) pairs: b = static solved & OFF not, c = the reverse."""
    b = sum(1 for r in rows if r["on"] == 1 and r["off"] == 0)
    c = sum(1 for r in rows if r["on"] == 0 and r["off"] == 1)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    return b, c, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def estimand(rows: list) -> dict:
    rd, lo, hi = cluster_bootstrap(rows, n_boot=N_BOOT, seed=0)
    b, c, p = mcnemar_exact(rows)
    tp, tn, tpv = sign_test(rows)
    claim = (lo > 0 or hi < 0) and p < 0.05
    return {"n_pairs": len(rows), "n_tasks": len({r["task_id"] for r in rows}), "rd": rd, "ci": [lo, hi],
            "discordant": {"static_only": b, "off_only": c, "p": p}, "task_sign": {"pos": tp, "neg": tn, "p": tpv},
            "off_rate": sum(r["off"] for r in rows) / max(1, len(rows)), "static_rate": sum(r["on"] for r in rows) / max(1, len(rows)),
            "claim": bool(claim)}


def analyse(d: Path) -> dict:
    off, st = load(d / "off.jsonl"), load(d / "static.jsonl")
    tasks = sorted({t for t, _k in off} | {t for t, _k in st})
    trials = sorted({k for _t, k in off} | {k for _t, k in st})
    task_ids = None
    usable = REPO / "docs/phase3/NYU_USABLE_test.txt"
    if usable.exists():
        # the usable list keys by display name; map to task_id through the records
        disp = {l.split("\t")[0] for l in usable.read_text().splitlines() if l.strip()}
        by_disp = {r["display_name"]: r["task_id"] for r in list(off.values()) + list(st.values())}
        task_ids = {by_disp[x] for x in disp if x in by_disp}
        n_expected = len(disp)
    else:
        n_expected = len(tasks)
    complete, partial = [], []
    for k in trials:
        n_off = sum(1 for (t, kk) in off if kk == k); n_st = sum(1 for (t, kk) in st if kk == k)
        (complete if n_off >= n_expected and n_st >= n_expected else partial).append(k)
    meta_rows = {}
    leaky = {r["task_id"] for r in list(off.values()) + list(st.values()) if r.get("flag_leaks")}

    def rows_for(ks):
        rows, dropped = [], 0
        for (t, k), ro in off.items():
            if k not in ks or (t, k) not in st:
                continue
            rs = st[(t, k)]
            if ro.get("error") or rs.get("error"):
                dropped += 1; continue
            rows.append({"task_id": t, "trial": k, "off": int(bool(ro.get("solved"))), "on": int(bool(rs.get("solved"))),
                         "needs_service": bool(ro.get("needs_service")), "category": ro.get("category") or "?",
                         "leaky": t in leaky, "rounds_off": ro.get("rounds_used"), "rounds_on": rs.get("rounds_used")})
        return rows, dropped

    rows, dropped = rows_for(complete)
    prow, pdropped = rows_for(partial)
    out = {"dir": str(d), "n_tasks_expected": n_expected, "trials_complete": complete, "trials_partial": partial,
           "records": {"off": len(off), "static": len(st)}, "dropped_errored_pairs": dropped,
           "sandbox_bad": {"off": sum(1 for r in off.values() if not sandbox_ok(r)), "static": sum(1 for r in st.values() if not sandbox_ok(r))},
           "no_solve": {"off": sum(1 for r in off.values() if no_solve(r)), "static": sum(1 for r in st.values() if no_solve(r))},
           "egress": sorted({str((r.get("extra") or {}).get("egress")) for r in list(off.values()) + list(st.values())}),
           "errors": {"off": sum(1 for r in off.values() if r.get("error")), "static": sum(1 for r in st.values() if r.get("error"))}}
    if rows:
        out["H1_all"] = estimand(rows)
        out["H2_file_only"] = estimand([r for r in rows if not r["needs_service"]])
        out["service"] = estimand([r for r in rows if r["needs_service"]])
        out["without_flag_in_handout"] = estimand([r for r in rows if not r["leaky"]])
        out["per_seed"] = {k: estimand([r for r in rows if r["trial"] == k]) for k in complete}
        cats = defaultdict(list)
        for r in rows:
            cats[r["category"]].append(r)
        out["per_category"] = {c: {"n": len(v), "off": sum(x["off"] for x in v), "static": sum(x["on"] for x in v)} for c, v in sorted(cats.items())}
        n = len(rows)
        out["rates"] = {"off": wilson(sum(r["off"] for r in rows), n), "static": wilson(sum(r["on"] for r in rows), n), "n": n}
        so = [r["rounds_off"] for r in rows if r["off"] and r["rounds_off"]]; ss = [r["rounds_on"] for r in rows if r["on"] and r["rounds_on"]]
        out["rounds_on_solved"] = {"off": sum(so) / max(1, len(so)), "static": sum(ss) / max(1, len(ss))}
        out["citation"] = citation_vs_chance(off, st, [(r["task_id"], r["trial"]) for r in rows])
    if prow:
        out["partial_seed"] = estimand(prow) | {"dropped_errored_pairs": pdropped}
    return out


def pct(x):
    return f"{x * 100:+.1f}" if x == x else "nan"


def markdown(a: dict, provisional: bool) -> str:
    L = [f"# NYU CTF Bench: no memory (OFF) vs static lessons (ON-static){' (PROVISIONAL)' if provisional else ''}", "",
         "Qwen3.8-27B-FP8; each solve in its own container (fails closed), internet blocked, host unreachable; both arms "
         "against one model server.", "",
         f"Seeds complete in both arms: {[1000 + k for k in a['trials_complete']] or 'none'}; partial: {[1000 + k for k in a['trials_partial']] or 'none'}. "
         f"Records: OFF {a['records']['off']}, static {a['records']['static']}. Errored pairs dropped: {a['dropped_errored_pairs']}. "
         f"Records with a command blocked or run outside the container: {a['sandbox_bad']}. Records where no solve started "
         f"(provisioning failed; errored, dropped unless the retry succeeds): {a.get('no_solve')}. Egress: {a['egress']}.", ""]
    if "H1_all" not in a:
        L.append("No complete seed yet."); return "\n".join(L) + "\n"
    r = a["rates"]
    L += [f"Solve rate over {r['n']} pairs: OFF {r['off'][0]:.3f} [{r['off'][1]:.3f}, {r['off'][2]:.3f}], "
          f"static {r['static'][0]:.3f} [{r['static'][1]:.3f}, {r['static'][2]:.3f}] (Wilson 95%).", "",
          "| estimand | pairs (tasks) | OFF | static | RD, points [95% task-cluster CI] | discordant static/OFF, exact p | task sign +/-, p | rule met |",
          "|---|---|---|---|---|---|---|---|"]
    for name, key in [("**H1** all tasks", "H1_all"), ("**H2** file-only", "H2_file_only"), ("service (reported)", "service"),
                      ("all, without flag-in-handout", "without_flag_in_handout")]:
        e = a[key]
        L.append(f"| {name} | {e['n_pairs']} ({e['n_tasks']}) | {e['off_rate']:.3f} | {e['static_rate']:.3f} | {pct(e['rd'])} [{pct(e['ci'][0])}, {pct(e['ci'][1])}] | "
                 f"{e['discordant']['static_only']}/{e['discordant']['off_only']}, p = {e['discordant']['p']:.3f} | "
                 f"{e['task_sign']['pos']}/{e['task_sign']['neg']}, p = {e['task_sign']['p']:.3f} | {'yes' if e['claim'] else 'no'} |")
    L += ["", "Per seed (H1):", ""]
    for k, e in a["per_seed"].items():
        L.append(f"- seed {1000 + int(k)}: RD {pct(e['rd'])} [{pct(e['ci'][0])}, {pct(e['ci'][1])}], OFF {e['off_rate']:.3f}, static {e['static_rate']:.3f}")
    if "partial_seed" in a:
        e = a["partial_seed"]
        L += ["", f"Partial seed (not pooled): {e['n_pairs']} pairs, RD {pct(e['rd'])} [{pct(e['ci'][0])}, {pct(e['ci'][1])}]."]
    L += ["", "Per category (pairs, OFF solved, static solved): " + ", ".join(f"{c} {v['n']}/{v['off']}/{v['static']}" for c, v in a["per_category"].items()),
          "", f"Mean rounds on solved pairs: OFF {a['rounds_on_solved']['off']:.1f}, static {a['rounds_on_solved']['static']:.1f}.", ""]
    c = a.get("citation") or {}
    if c:
        L += [f"Lesson use (descriptive): a shown lesson was cited in {c['cited_static']} of {c['pairs']} static traces; the same lessons "
              f"'match' {c['cited_chance_off']} of the paired no-memory traces by chance.", ""]
    L.append("Decision rule: a lift is claimed only if the CI excludes 0 **and** the exact sign test on discordant pairs gives p < 0.05; "
             "otherwise the result is a bounded null with the interval shown.")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "results/realbench_nyu_v2/qwen_qwen3.8-27b-fp8"))
    ap.add_argument("--out", default=""); ap.add_argument("--json", default=""); ap.add_argument("--provisional", action="store_true")
    args = ap.parse_args(argv)
    a = analyse(Path(args.dir))
    md = markdown(a, args.provisional)
    if args.out:
        Path(args.out).write_text(md)
    if args.json:
        Path(args.json).write_text(json.dumps(a, indent=1, default=float))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
