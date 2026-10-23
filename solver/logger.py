#!/usr/bin/env python3
"""
ChallengeLogger - structured, readable console output for the CTF solver.

Provides colored, phase-based logging so you can follow the solver's
reasoning instead of staring at opaque emoji lines.
"""
import textwrap


# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

class _C:
    """ANSI escape sequences."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    BG_RED  = "\033[41m"
    BG_GREEN = "\033[42m"


def _col(text: str, *codes: str) -> str:
    return "".join(codes) + text + _C.RESET


# ---------------------------------------------------------------------------
# ChallengeLogger
# ---------------------------------------------------------------------------

class ChallengeLogger:
    """Structured per-challenge logger.

    Usage:
        log = ChallengeLogger("Crack the Gate 1", 3,
                              category="Web", points=100, difficulty="Easy")
        log.phase("SCRAPE")
        log.info("Downloaded 2 files")
        log.phase("SOLVE")
        log.thinking("This looks like a brute force challenge...")
        log.command("curl -s http://...", stdout="<html>...", stderr="", rc=0)
        log.llm_action("run_commands", detail="curl, grep")
        log.flag_found("picoCTF{brut4_f0rc4_3c6b118b}")
        log.summary("accepted", rounds=3, commands_run=5)
    """

    INDENT = "  "  # base indent for all output under the challenge header

    def __init__(self, name: str, number: int, *,
                 category: str = "", points: str | int = "?",
                 difficulty: str = ""):
        self.name = name
        self.number = number
        self.category = category
        self.points = points
        self.difficulty = difficulty
        self._current_phase = ""
        self._print_header()

    # ---- Header / Phase ----

    def _print_header(self):
        bar = "━" * 64
        print(f"\n{_col(bar, _C.CYAN)}")
        print(f"  {_col(f'[{self.number}]', _C.BOLD, _C.WHITE)}  "
              f"{_col(self.name, _C.BOLD, _C.CYAN)}")
        meta = f"  {self.category}  •  {self.points} pts  •  {self.difficulty}"
        print(f"{_col(meta, _C.DIM)}")
        print(f"{_col(bar, _C.CYAN)}")

    def phase(self, name: str, detail: str = ""):
        """Print a phase separator (SCRAPE / SOLVE / SUBMIT)."""
        self._current_phase = name
        label = _col(f"── {name} ", _C.BOLD, _C.MAGENTA)
        tail = _col("─" * max(0, 50 - len(name)), _C.DIM)
        extra = f"  {_col(detail, _C.DIM)}" if detail else ""
        print(f"\n{self.INDENT}{label}{tail}{extra}")

    def round_start(self, round_num: int, max_rounds: int):
        """Print round header within the SOLVE phase."""
        print(f"\n{self.INDENT}  {_col(f'Round {round_num}/{max_rounds}', _C.BOLD, _C.BLUE)}")

    # ---- LLM Thinking ----

    def thinking(self, text: str, max_lines: int = 12):
        """Display the LLM's chain-of-thought in a dim, indented block."""
        if not text or not text.strip():
            return
        prefix = f"{self.INDENT}  {_col('│', _C.DIM)} "
        print(f"{self.INDENT}  {_col('┌─ LLM Thinking ─────────────────────', _C.DIM)}")
        lines = text.strip().splitlines()
        shown = lines[:max_lines]
        for line in shown:
            # Wrap long lines
            wrapped = textwrap.wrap(line, width=90) or [""]
            for wl in wrapped:
                print(f"{prefix}{_col(wl, _C.DIM)}")
        if len(lines) > max_lines:
            print(f"{prefix}{_col(f'... ({len(lines) - max_lines} more lines in log file)', _C.DIM, _C.ITALIC)}")
        print(f"{self.INDENT}  {_col('└───────────────────────────────────', _C.DIM)}")

    # ---- LLM Actions ----

    def llm_action(self, action: str, *, detail: str = ""):
        """Log what the LLM decided to do."""
        icons = {
            "run_commands": "🔧",
            "submit_flag": "🚩",
            "give_up": "⏭",
        }
        icon = icons.get(action, "❓")
        extra = f"  {_col(detail, _C.DIM)}" if detail else ""
        print(f"{self.INDENT}    {icon}  Action: {_col(action, _C.BOLD)}{extra}")

    def llm_gave_up(self, reason: str):
        print(f"{self.INDENT}    ⏭  {_col('Gave up:', _C.YELLOW)} {reason}")

    def llm_invalid_flag(self, flag: str):
        print(f"{self.INDENT}    ⚠  {_col('Rejected invalid/placeholder flag:', _C.YELLOW)} {flag}")

    def llm_parse_error(self, detail: str = ""):
        msg = "Could not parse LLM response as JSON"
        if detail:
            msg += f" - {detail}"
        print(f"{self.INDENT}    ⚠  {_col(msg, _C.YELLOW)}")

    def llm_error(self, error: str):
        print(f"{self.INDENT}    ✗  {_col(f'LLM error: {error}', _C.RED)}")

    # ---- Command Execution ----

    def command(self, cmd: str, *, stdout: str = "", stderr: str = "",
                rc: int = 0, error: str | None = None,
                max_lines: int = 8):
        """Print a command and a smart-truncated version of its output."""
        print(f"{self.INDENT}    $ {_col(cmd, _C.CYAN)}")

        if error:
            print(f"{self.INDENT}      {_col(f'ERROR: {error}', _C.RED)}")
            return

        if rc != 0:
            print(f"{self.INDENT}      {_col(f'exit {rc}', _C.YELLOW)}")

        # Print stdout with smart truncation
        if stdout.strip():
            self._print_output(stdout, "stdout", max_lines)

        # Print stderr (always show if non-empty - often contains clues)
        if stderr.strip():
            self._print_output(stderr, "stderr", max_lines=4, color=_C.YELLOW)

    def _print_output(self, text: str, label: str, max_lines: int,
                      color: str = ""):
        lines = text.rstrip().splitlines()
        indent = f"{self.INDENT}      "

        if len(lines) <= max_lines:
            for line in lines:
                highlighted = self._highlight_flags(line)
                if color:
                    print(f"{indent}{_col(highlighted, color)}")
                else:
                    print(f"{indent}{highlighted}")
        else:
            # Show first N-2 and last 2 lines
            head_count = max(max_lines - 2, 1)
            for line in lines[:head_count]:
                highlighted = self._highlight_flags(line)
                if color:
                    print(f"{indent}{_col(highlighted, color)}")
                else:
                    print(f"{indent}{highlighted}")
            omitted = len(lines) - head_count - 2
            print(f"{indent}{_col(f'  ... ({omitted} lines omitted) ...', _C.DIM)}")
            for line in lines[-2:]:
                highlighted = self._highlight_flags(line)
                if color:
                    print(f"{indent}{_col(highlighted, color)}")
                else:
                    print(f"{indent}{highlighted}")

    @staticmethod
    def _highlight_flags(line: str) -> str:
        """Highlight picoCTF{...} in output with bright green."""
        import re
        return re.sub(
            r'(picoCTF\{[^\s}]+\})',
            _col(r'\1', _C.BOLD, _C.GREEN),
            line
        )

    # ---- Flags ----

    def flag_found(self, flag: str, source: str = ""):
        src = f" ({source})" if source else ""
        print(f"\n{self.INDENT}  🚩  {_col(f'FLAG FOUND{src}:', _C.BOLD, _C.GREEN)} {_col(flag, _C.BOLD, _C.GREEN)}")

    def no_flag(self):
        print(f"\n{self.INDENT}  ⏭  {_col('No flag found', _C.YELLOW)}")

    # ---- Submit ----

    def submit_result(self, accepted: bool, message: str = ""):
        if accepted:
            print(f"{self.INDENT}  ✅  {_col('Flag ACCEPTED!', _C.BOLD, _C.GREEN)}")
        else:
            detail = f" - {message}" if message else ""
            print(f"{self.INDENT}  ❌  {_col(f'Flag rejected{detail}', _C.RED)}")

    def submit_dry_run(self, flag: str):
        print(f"{self.INDENT}  🏁  {_col('[DRY RUN]', _C.YELLOW)} Would submit: {flag}")

    # ---- General ----

    def info(self, msg: str):
        print(f"{self.INDENT}  ℹ  {msg}")

    def warn(self, msg: str):
        print(f"{self.INDENT}  ⚠  {_col(msg, _C.YELLOW)}")

    def error(self, msg: str):
        print(f"{self.INDENT}  ✗  {_col(msg, _C.RED)}")

    def success(self, msg: str):
        print(f"{self.INDENT}  ✓  {_col(msg, _C.GREEN)}")

    # ---- Per-challenge summary ----

    def challenge_summary(self, status: str, *, rounds_used: int = 0,
                          commands_run: int = 0, flag: str | None = None):
        """Compact end-of-challenge recap."""
        bar = "─" * 50
        print(f"\n{self.INDENT}{_col(bar, _C.DIM)}")
        status_colors = {
            "accepted": (_C.GREEN, "✅ ACCEPTED"),
            "rejected": (_C.RED, "❌ REJECTED"),
            "no_flag": (_C.YELLOW, "⏭  NO FLAG"),
            "dry_run": (_C.CYAN, "🏁 DRY RUN"),
            "scrape_error": (_C.RED, "✗  SCRAPE ERROR"),
            "error": (_C.RED, "✗  ERROR"),
        }
        color, label = status_colors.get(status, (_C.WHITE, f"?  {status}"))
        print(f"{self.INDENT}  {_col(label, _C.BOLD, color)}"
              f"  │  Rounds: {rounds_used}  │  Commands: {commands_run}"
              + (f"  │  Flag: {flag}" if flag else ""))
        print(f"{self.INDENT}{_col(bar, _C.DIM)}")


# ---------------------------------------------------------------------------
# Global helpers (for non-per-challenge output)
# ---------------------------------------------------------------------------

def print_banner(use_tools: bool = True):
    """Print the startup banner."""
    bar = "=" * 64
    print(f"\n{_col(bar, _C.CYAN, _C.BOLD)}")
    print(f"  {_col('Baseline PicoCTF Solver', _C.BOLD, _C.WHITE)}")
    if not use_tools:
        print(f"  {_col('Mode: NO TOOLS (single-shot LLM only)', _C.YELLOW)}")
    print(f"{_col(bar, _C.CYAN, _C.BOLD)}")


def print_final_summary(stats: dict, results_file: str):
    """Print the end-of-run summary."""
    bar = "=" * 64
    print(f"\n{_col(bar, _C.CYAN, _C.BOLD)}")
    print(f"  {_col('RESULTS SUMMARY', _C.BOLD, _C.WHITE)}")
    print(f"{_col(bar, _C.CYAN, _C.BOLD)}")

    def _stat(label, val, color=_C.WHITE):
        print(f"  {label:<16} {_col(str(val), _C.BOLD, color)}")

    _stat("Attempted:",   stats["attempted"])
    _stat("Flags found:", stats["flags_found"], _C.CYAN)
    _stat("Submitted:",   stats["submitted"])
    _stat("Accepted:",    stats["accepted"],    _C.GREEN)

    rejected = stats["submitted"] - stats["accepted"]
    if rejected > 0:
        _stat("Rejected:",  rejected, _C.RED)

    print(f"\n  Results saved to: {results_file}")
    print(f"{_col(bar, _C.CYAN, _C.BOLD)}")
