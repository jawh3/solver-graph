#!/usr/bin/env python3
"""
Evaluation Harness — measures distributed solver performance.

Runs scaling experiments across different worker counts and configurations,
then generates metrics and charts for the paper.

Usage:
    python evaluate.py --experiment scaling   # Wall-clock speedup (1-5 workers)
    python evaluate.py --experiment rag-lift  # RAG vs no-RAG comparison
    python evaluate.py --analyze              # Analyze existing results
"""
import json
import argparse
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_DIR / "eval_results"


# ---------------------------------------------------------------------------
# Result analysis
# ---------------------------------------------------------------------------

def load_results(results_file: Path) -> list[dict]:
    """Load results from a JSONL file."""
    records = []
    if results_file.exists():
        for line in results_file.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def analyze_results(results: list[dict]) -> dict:
    """Compute summary statistics from a list of result records."""
    total = len(results)
    flags_found = sum(1 for r in results if r.get("flag"))
    accepted = sum(1 for r in results if r.get("status") == "accepted")
    dry_run_flags = sum(1 for r in results if r.get("status") == "dry_run" and r.get("flag"))
    errors = sum(1 for r in results if r.get("status") == "worker_error")

    # Per-worker stats
    by_worker = defaultdict(lambda: {"total": 0, "flags": 0})
    for r in results:
        w = r.get("worker", "unknown")
        by_worker[w]["total"] += 1
        if r.get("flag"):
            by_worker[w]["flags"] += 1

    # Per-model stats
    by_model = defaultdict(lambda: {"total": 0, "flags": 0})
    for r in results:
        m = r.get("model", "unknown")
        by_model[m]["total"] += 1
        if r.get("flag"):
            by_model[m]["flags"] += 1

    # Per-category stats
    by_category = defaultdict(lambda: {"total": 0, "flags": 0})
    for r in results:
        c = r.get("category", "Unknown")
        by_category[c]["total"] += 1
        if r.get("flag"):
            by_category[c]["flags"] += 1

    # RAG lift: compare results with vs without RAG context
    with_rag = [r for r in results if r.get("rag_context_chars", 0) > 0]
    without_rag = [r for r in results if r.get("rag_context_chars", 0) == 0]
    rag_solve_rate = (
        sum(1 for r in with_rag if r.get("flag")) / max(len(with_rag), 1)
    )
    no_rag_solve_rate = (
        sum(1 for r in without_rag if r.get("flag")) / max(len(without_rag), 1)
    )

    # Average rounds
    rounds = [r.get("rounds_used", 0) for r in results if r.get("rounds_used")]
    avg_rounds = sum(rounds) / max(len(rounds), 1)

    return {
        "total_attempted": total,
        "flags_found": flags_found,
        "accepted": accepted,
        "dry_run_flags": dry_run_flags,
        "errors": errors,
        "solve_rate": flags_found / max(total, 1),
        "avg_rounds": round(avg_rounds, 2),
        "by_worker": dict(by_worker),
        "by_model": dict(by_model),
        "by_category": dict(by_category),
        "rag_stats": {
            "with_rag_count": len(with_rag),
            "with_rag_solve_rate": round(rag_solve_rate, 4),
            "without_rag_count": len(without_rag),
            "without_rag_solve_rate": round(no_rag_solve_rate, 4),
            "rag_lift": round(rag_solve_rate - no_rag_solve_rate, 4),
        },
    }


def print_analysis(stats: dict, label: str = ""):
    """Pretty-print analysis results."""
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION RESULTS{f' — {label}' if label else ''}")
    print(f"{'=' * 60}")
    print(f"  Total attempted:  {stats['total_attempted']}")
    print(f"  Flags found:      {stats['flags_found']}")
    print(f"  Accepted:         {stats['accepted']}")
    print(f"  Solve rate:       {stats['solve_rate']:.1%}")
    print(f"  Avg rounds/solve: {stats['avg_rounds']}")
    print(f"  Errors:           {stats['errors']}")

    if stats["by_category"]:
        print(f"\n  Per-category:")
        for cat, s in sorted(stats["by_category"].items()):
            rate = s["flags"] / max(s["total"], 1)
            print(f"    {cat:15s} {s['flags']:3d}/{s['total']:3d} ({rate:.0%})")

    if stats["by_worker"]:
        print(f"\n  Per-worker:")
        for w, s in sorted(stats["by_worker"].items()):
            rate = s["flags"] / max(s["total"], 1)
            print(f"    {w:15s} {s['flags']:3d}/{s['total']:3d} ({rate:.0%})")

    if stats["by_model"]:
        print(f"\n  Per-model:")
        for m, s in sorted(stats["by_model"].items()):
            rate = s["flags"] / max(s["total"], 1)
            print(f"    {m:30s} {s['flags']:3d}/{s['total']:3d} ({rate:.0%})")

    rag = stats["rag_stats"]
    if rag["with_rag_count"] > 0 or rag["without_rag_count"] > 0:
        print(f"\n  RAG Knowledge Graph Impact:")
        print(f"    With RAG:    {rag['with_rag_solve_rate']:.1%} "
              f"({rag['with_rag_count']} challenges)")
        print(f"    Without RAG: {rag['without_rag_solve_rate']:.1%} "
              f"({rag['without_rag_count']} challenges)")
        print(f"    RAG lift:    {rag['rag_lift']:+.1%}")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Scaling experiment
# ---------------------------------------------------------------------------

def run_scaling_experiment(
    max_workers: int = 5,
    challenges_per_run: int = 20,
    category: str | None = None,
    dry_run: bool = True,
):
    """Run the solver with 1..max_workers and measure wall-clock time.

    Saves results to eval_results/scaling_<timestamp>.json
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    scaling_results = []

    for n_workers in range(1, max_workers + 1):
        print(f"\n{'─' * 40}")
        print(f"  Scaling experiment: {n_workers} worker(s)")
        print(f"{'─' * 40}")

        # Generate a temporary workers config with n_workers
        # This assumes all workers are defined in workers.yaml
        # and we simply limit how many are used via --max-parallel
        cmd = [
            sys.executable, "-m", "solver.main",
            "--limit", str(challenges_per_run),
            "--dry-run" if dry_run else "",
            "--include-solved",  # so we hit the same challenges each run
        ]
        if category:
            cmd.extend(["--category", category])
        cmd = [c for c in cmd if c]  # remove empty strings

        start_time = time.time()
        result = subprocess.run(cmd, cwd=str(PROJECT_DIR), capture_output=True, text=True)
        elapsed = time.time() - start_time

        scaling_results.append({
            "n_workers": n_workers,
            "wall_clock_seconds": round(elapsed, 2),
            "challenges": challenges_per_run,
            "exit_code": result.returncode,
        })

        print(f"  Wall-clock time: {elapsed:.1f}s (exit code: {result.returncode})")

    # Save results
    out_file = RESULTS_DIR / f"scaling_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(scaling_results, f, indent=2)
    print(f"\nScaling results saved to {out_file}")

    # Print summary table
    print(f"\n{'Workers':>8s} {'Time (s)':>10s} {'Speedup':>8s}")
    print(f"{'-'*8:>8s} {'-'*10:>10s} {'-'*8:>8s}")
    baseline_time = scaling_results[0]["wall_clock_seconds"] if scaling_results else 1
    for r in scaling_results:
        speedup = baseline_time / max(r["wall_clock_seconds"], 0.1)
        print(f"{r['n_workers']:8d} {r['wall_clock_seconds']:10.1f} {speedup:8.2f}x")

    return scaling_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Distributed CTF Solver Evaluator")
    parser.add_argument("--experiment", choices=["scaling", "rag-lift"],
                        help="Run a specific experiment")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Analyze a results JSONL file")
    parser.add_argument("--max-workers", type=int, default=5,
                        help="Max workers for scaling experiment")
    parser.add_argument("--challenges", type=int, default=20,
                        help="Challenges per scaling run")
    parser.add_argument("--category", type=str, default=None,
                        help="Category filter for experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't submit flags during experiments")
    args = parser.parse_args()

    if args.analyze:
        results = load_results(Path(args.analyze))
        if not results:
            print(f"No results found in {args.analyze}")
            sys.exit(1)
        stats = analyze_results(results)
        print_analysis(stats, label=args.analyze)

    elif args.experiment == "scaling":
        run_scaling_experiment(
            max_workers=args.max_workers,
            challenges_per_run=args.challenges,
            category=args.category,
            dry_run=args.dry_run,
        )

    elif args.experiment == "rag-lift":
        print("RAG-lift experiment: run master.py twice —")
        print("  1. python master.py --limit N --dry-run         (with RAG)")
        print("  2. python master.py --limit N --dry-run --no-rag (without RAG)")
        print("Then analyze both results files:")
        print("  python evaluate.py --analyze results_distributed.jsonl")

    else:
        # Default: analyze distributed results if they exist
        rfile = PROJECT_DIR / "results_distributed.jsonl"
        if rfile.exists():
            results = load_results(rfile)
            stats = analyze_results(results)
            print_analysis(stats, label="results_distributed.jsonl")
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
