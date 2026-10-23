#!/usr/bin/env python3
"""
Baseline PicoCTF Solver - Main Orchestrator.

Logs into PicoCTF, scrapes challenges from the Gym, feeds each to a local LLM,
and submits the proposed flag.  Results are logged to results.jsonl.

Usage:
    python main.py                     # Solve all unsolved challenges
    python main.py --limit 5           # Solve at most 5 challenges
    python main.py --category Crypto   # Only Crypto challenges
    python main.py --dry-run           # Scrape + LLM but do NOT submit flags
    python main.py --no-tools           # Disable tool use (single-shot LLM only, for ablation)
    python main.py --dry-run --limit 1 # Quick test with one challenge
"""
import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright
from dotenv import load_dotenv

from solver.scraper import login, iter_challenge_pages, get_challenge_details
from solver.solver import solve_challenge, check_llm
from solver.submitter import submit_flag
from solver.logger import ChallengeLogger, print_banner, print_final_summary
from retrieval.memory_client import record_attempt_sync
try:
    from knowledge_graph.kg import get_knowledge_graph_context
except ModuleNotFoundError:  # legacy module, superseded by ContextRetriever
    def get_knowledge_graph_context(*a, **k):
        raise RuntimeError(
            "the legacy knowledge_graph.kg helper was retired; use "
            "retrieval.ContextRetriever with solve_challenge(injected_rag_context=...)")

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
RESULTS_FILE = PROJECT_DIR / "results.jsonl"
COMPARE_RESULTS_FILE = PROJECT_DIR / "results_compare.jsonl"
COOKIES_FILE = PROJECT_DIR / "cookies.json"
LOGS_DIR = PROJECT_DIR / "logs"


def load_previous_results() -> set[str]:
    """Load challenge names that have already been attempted from results.jsonl."""
    attempted: set[str] = set()
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                attempted.add(record["name"])
            except (json.JSONDecodeError, KeyError):
                continue
    return attempted


def save_result(record: dict):
    """Append a single result record to results.jsonl."""
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def save_compare_result(record: dict):
    """Append a compare-mode result record to results_compare.jsonl."""
    with open(COMPARE_RESULTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def write_challenge_log(challenge_name: str, trace: list[dict], trace_json: bool = False):
    """Write a per-challenge log file with full trace details."""
    from solver.scraper import sanitize_name
    LOGS_DIR.mkdir(exist_ok=True)

    if trace_json:
        # Write structured JSON trace for detailed debugging
        log_path = LOGS_DIR / f"{sanitize_name(challenge_name)}_trace.json"
        with open(log_path, "w") as f:
            json.dump({
                "challenge": challenge_name,
                "timestamp": datetime.utcnow().isoformat(),
                "rounds": trace,
            }, f, indent=2, default=str)

    # Always write the human-readable text log
    log_path = LOGS_DIR / f"{sanitize_name(challenge_name)}.log"
    with open(log_path, "w") as f:
        f.write(f"=== {challenge_name} ===\n\n")
        for r in trace:
            f.write(f"--- Round {r['round']} ---\n")
            if r.get("thinking"):
                f.write(f"[Thinking]\n{r['thinking']}\n\n")
            if r.get("action"):
                f.write(f"[Action] {r['action']}\n")
            if r.get("commands"):
                for cmd in r["commands"]:
                    f.write(f"  $ {cmd}\n")
            if r.get("results_summary"):
                f.write(f"[Results] {r['results_summary']}\n")
            if r.get("flag"):
                f.write(f"[Flag] {r['flag']}\n")
            f.write("\n")


def _should_attempt(ch, skip_solved, category, previously_attempted):
    """Return True if this challenge should be attempted."""
    if skip_solved and ch.get("already_solved"):
        return False
    if ch["name"] in previously_attempted:
        return False
    if category and ch.get("category", "").lower() != category.lower():
        return False
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run(
    limit: int | None = None,
    category: str | None = None,
    dry_run: bool = False,
    skip_solved: bool = True,
    use_tools: bool = True,
    use_memory: bool = False,
    compare_mode: bool = False,
    trace_enabled: bool = False,
    fresh: bool = False,
    headed: bool = False,
):
    # ---- Pre-flight checks ----
    print_banner(use_tools)

    if not check_llm():
        print("\n⚠  LLM server is not reachable. Start it with:")
        print("   bash infrastructure/llm.sh")
        sys.exit(1)

    # Optionally clear previous results for a fresh run
    if fresh and RESULTS_FILE.exists():
        backup = RESULTS_FILE.with_suffix(".jsonl.bak")
        RESULTS_FILE.rename(backup)
        print(f"ℹ  Moved old results to {backup.name} (fresh run)")

    previously_attempted = load_previous_results()
    if previously_attempted:
        print(f"ℹ  {len(previously_attempted)} challenges previously attempted (in results.jsonl)")

    # ---- Launch browser & log in ----
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=not headed,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        )

        # ---- Load cookies instead of manual login ----
        if not COOKIES_FILE.exists():
            print("\n✗  cookies.json not found!")
            print("   Run save_login.py on your local machine first, then")
            print("   transfer cookies.json to this directory.")
            sys.exit(1)

        with open(COOKIES_FILE, "r") as f:
            cookies = json.load(f)

        SAME_SITE_MAP = {
            "strict": "Strict",
            "lax": "Lax",
            "none": "None",
            "no_restriction": "None",
            "unspecified": "Lax",
        }

        cleaned_cookies = []
        for cookie in cookies:
            raw = (cookie.get("sameSite") or "lax").lower()
            cleaned_cookies.append({
                "name": cookie["name"],
                "value": cookie["value"],
                "domain": cookie["domain"],
                "path": cookie.get("path", "/"),
                "secure": cookie.get("secure", False),
                "httpOnly": cookie.get("httpOnly", False),
                "sameSite": SAME_SITE_MAP.get(raw, "Lax"),
                **({"expires": cookie["expirationDate"]} if "expirationDate" in cookie else {}),
            })

        await context.add_cookies(cleaned_cookies)
        print(f"✓  Loaded {len(cookies)} cookies from {COOKIES_FILE}")

        page = await context.new_page()

        # Verify session is still valid by navigating to practice page
        print("🔐  Verifying session...")
        await page.goto("https://play.picoctf.org/practice")
        await page.wait_for_load_state("networkidle")

        if "login" in page.url.lower():
            print("\n✗  Session expired or cookies are invalid!")
            print("   Re-run save_login.py on your local machine and")
            print("   transfer a fresh cookies.json to this directory.")
            sys.exit(1)

        print("✓  Session valid - proceeding to solve challenges\n")

        # ---- Process challenges page by page ----
        stats = {"attempted": 0, "flags_found": 0, "submitted": 0, "accepted": 0}
        challenge_num = 0
        done = False

        async for page_num, page_challenges in iter_challenge_pages(page):
            if done:
                break

            # Filter challenges on this page
            targets = [ch for ch in page_challenges
                       if _should_attempt(ch, skip_solved, category, previously_attempted)]

            for challenge in targets:
                if limit and challenge_num >= limit:
                    done = True
                    break

                challenge_num += 1
                # --- Create logger for this challenge ---
                log = ChallengeLogger(
                    name=challenge["name"],
                    number=challenge_num,
                    category=challenge.get("category", ""),
                    points=challenge.get("points", "?"),
                    difficulty=challenge.get("difficulty", ""),
                )

                # 1. Get full details (expand modal, download files)
                log.phase("SCRAPE")
                details = await get_challenge_details(page, challenge, ARTIFACTS_DIR)

                if details.get("error"):
                    log.error(f"Scraper error: {details['error']}")
                    log.challenge_summary("scrape_error")
                    save_result({
                        "name": challenge["name"],
                        "category": challenge.get("category"),
                        "status": "scrape_error",
                        "error": details["error"],
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                    continue

                n_files = len(details.get("local_files", []))
                n_hints = len(details.get("hints", []))
                log.info(f"{n_files} file(s), {n_hints} hint(s)")

                stats["attempted"] += 1

                if compare_mode:
                    # ---- COMPARE MODE: run baseline then memory ----
                    log.phase("BASELINE", "solving without memory")
                    baseline = solve_challenge(details, use_tools=use_tools,
                                              use_memory=False, log=log)

                    log.phase("MEMORY", "solving with memory")
                    mem = solve_challenge(details, use_tools=use_tools,
                                         use_memory=True, log=log)

                    def _outcome(r: dict) -> str:
                        if r["flag"]:
                            return "flag_found"
                        if r.get("gave_up_reason"):
                            return "gave_up"
                        return "no_flag"

                    compare_record = {
                        "name": challenge["name"],
                        "category": challenge.get("category"),
                        "points": challenge.get("points"),
                        "difficulty": challenge.get("difficulty"),
                        "mode": "compare",
                        "baseline": {
                            "flag": baseline["flag"],
                            "rounds_used": baseline["rounds_used"],
                            "total_commands": baseline["total_commands"],
                            "outcome": _outcome(baseline),
                        },
                        "memory": {
                            "flag": mem["flag"],
                            "rounds_used": mem["rounds_used"],
                            "total_commands": mem["total_commands"],
                            "outcome": _outcome(mem),
                        },
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    # Determine overall outcome label
                    b_found = bool(baseline["flag"])
                    m_found = bool(mem["flag"])
                    if b_found and m_found:
                        compare_record["result"] = "both_found"
                    elif m_found and not b_found:
                        compare_record["result"] = "memory_wins"
                    elif b_found and not m_found:
                        compare_record["result"] = "baseline_wins"
                    else:
                        compare_record["result"] = "none_found"

                    save_compare_result(compare_record)
                    write_challenge_log(challenge["name"] + "_baseline", baseline["trace"])
                    write_challenge_log(challenge["name"] + "_memory", mem["trace"])

                    result_label = compare_record["result"]
                    log.info(f"Compare result: {result_label} "
                             f"(baseline={'✓' if b_found else '✗'}, "
                             f"memory={'✓' if m_found else '✗'})")

                    # Record memory attempt back to memory (respects dry-run)
                    if not dry_run:
                        record_attempt_sync(
                            challenge_id=challenge["name"],
                            success=m_found,
                            key_insight=mem["flag"] if m_found else None,
                            failure_reason=mem.get("gave_up_reason") if not m_found else None,
                        )

                    if m_found:
                        stats["flags_found"] += 1

                    continue  # skip normal submit flow in compare mode

                # ---- NORMAL MODE ----
                # 2. Ask the LLM to solve it
                solve_result = solve_challenge(details, use_tools=use_tools,
                                               use_memory=use_memory, log=log)
                flag = solve_result["flag"]
                trace = solve_result["trace"]
                rounds_used = solve_result["rounds_used"]
                total_commands = solve_result["total_commands"]

                result_record = {
                    "name": challenge["name"],
                    "category": challenge.get("category"),
                    "points": challenge.get("points"),
                    "difficulty": challenge.get("difficulty"),
                    "flag": flag,
                    "rounds_used": rounds_used,
                    "total_commands": total_commands,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                if not flag:
                    result_record["status"] = "no_flag"
                    if solve_result.get("gave_up_reason"):
                        result_record["gave_up_reason"] = solve_result["gave_up_reason"]
                    save_result(result_record)
                    write_challenge_log(challenge["name"], trace, trace_json=trace_enabled)
                    log.challenge_summary("no_flag", rounds_used=rounds_used,
                                          commands_run=total_commands)
                    continue

                stats["flags_found"] += 1

                # 3. Submit the flag (unless dry-run)
                if dry_run:
                    result_record["status"] = "dry_run"
                    log.submit_dry_run(flag)
                else:
                    log.phase("SUBMIT")
                    submission = await submit_flag(page, challenge["name"], flag)
                    result_record["submission"] = submission
                    stats["submitted"] += 1

                    if submission.get("accepted"):
                        result_record["status"] = "accepted"
                        stats["accepted"] += 1
                        log.submit_result(True)
                    else:
                        result_record["status"] = "rejected"
                        log.submit_result(False, message=submission.get("message", ""))

                save_result(result_record)
                write_challenge_log(challenge["name"], trace, trace_json=trace_enabled)
                log.challenge_summary(
                    result_record["status"],
                    rounds_used=rounds_used,
                    commands_run=total_commands,
                    flag=flag,
                )

                # Record attempt back to memory (lifelong learning)
                # Skip during dry-run or if memory is not enabled.
                if use_memory and not dry_run:
                    record_attempt_sync(
                        challenge_id=challenge["name"],
                        success=result_record["status"] == "accepted",
                        key_insight=flag if flag else None,
                        failure_reason=solve_result.get("gave_up_reason") if not flag else None,
                        time_spent_seconds=None,
                    )

        # ---- Summary ----
        await browser.close()

        print_final_summary(stats, str(RESULTS_FILE))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline PicoCTF Solver")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Maximum number of challenges to attempt")
    parser.add_argument("--category", "-c", type=str, default=None,
                        help="Filter by category (e.g. Crypto, Forensics, Web)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Scrape + solve with LLM but do NOT submit flags")
    parser.add_argument("--include-solved", action="store_true",
                        help="Also attempt challenges already solved on PicoCTF")
    parser.add_argument("--no-tools", action="store_true",
                        help="Disable tool-augmented solving (single-shot LLM only, for ablation)")
    parser.add_argument("--memory", action="store_true",
                        help="Enable memory MCP server - retrieve past knowledge and record results")
    parser.add_argument("--compare", action="store_true",
                        help="Run each challenge twice: baseline (no memory) then with memory. "
                             "Results saved to results_compare.jsonl")
    parser.add_argument("--trace", action="store_true",
                        help="Write detailed JSON trace files per challenge to logs/")
    parser.add_argument("--fresh", action="store_true",
                        help="Clear results.jsonl before running (re-attempt all challenges)")
    parser.add_argument("--headed", action="store_true",
                        help="Run browser in visible (non-headless) mode for debugging")
    args = parser.parse_args()

    asyncio.run(run(
        limit=args.limit,
        category=args.category,
        dry_run=args.dry_run,
        skip_solved=not args.include_solved,
        use_tools=not args.no_tools,
        use_memory=args.memory,
        compare_mode=args.compare,
        trace_enabled=args.trace,
        fresh=args.fresh,
        headed=args.headed,
    ))



if __name__ == "__main__":
    main()