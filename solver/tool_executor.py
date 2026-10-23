#!/usr/bin/env python3
"""
Tool Executor - sandboxed shell command runner for CTF challenge analysis.

Executes commands requested by the LLM against downloaded challenge artifacts,
with an allowlist of safe tools and resource limits.
"""
import os
import re
import signal
import subprocess
import shlex
import base64 as b64module
import resource
import tempfile
import threading
import time
from pathlib import Path

# Hard cap on the size of ANY file a tool command may create (RLIMIT_FSIZE). CTF decode
# outputs are tiny (a flag is ~30 bytes); a legit gunzip/zlib output is well under this.
# This prevents a decompression bomb / runaway write from filling the disk - one such
# command wrote a 681 GB file overnight (2026-06-02) and filled the shared drive.
MAX_FILE_BYTES = 64 * 1024 * 1024  # 64 MB


def _limit_child_resources():
    """preexec_fn: cap file size (+ a generous CPU/AS guard) for the sandboxed command.
    Exceeding RLIMIT_FSIZE delivers SIGXFSZ (terminates the writer)."""
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Safety configuration
# ---------------------------------------------------------------------------

# No command allowlist - all commands are permitted.
# Only truly destructive patterns are blocked below.

# Patterns that are always blocked (even inside allowed commands)
BLOCKED_PATTERNS = [
    r"\brm\s+-rf\s+/(\s|$|\*)",  # rm -rf / (root) or /* only - trailing \b failed at EOS (fixed 2026-05-31)
    r"\brm\b.*--no-preserve-root",
    r"\bmkfs\b",
    r"\bdd\b\s+.*of=/dev",
    r"\bshutdown\b",
    r"\breboot\b",
    r":\(\)\s*\{",          # fork bomb
    r"\b>\s*/dev/sd",
    r"\bsystemctl\b",
    r"(?:\s|^)(~|/home|/root)(?:/|\b)",  # Stop wandering in ALL home directories
]

COMMAND_TIMEOUT = 60       # seconds per command (60s for SSH/network)
MAX_OUTPUT_BYTES = 50_000  # 50 KB max captured output

# Bound the parent's memory when a tool floods its output. Captured output goes to a
# temp file (not a pipe), so this only caps what we read back, preserving head+tail.
READ_CAP_BYTES = 16 * 1024 * 1024
KILL_GRACE_S = 3.0         # SIGTERM -> grace -> SIGKILL, per command teardown


def _kill_process_group(proc, pgid: int | None, grace: float = KILL_GRACE_S) -> None:
    """Tear down the command's whole PROCESS GROUP, not just the direct child.

    `shell=True` means the direct child is `/bin/sh`; the real tool, every member of a
    pipeline, and anything backgrounded with `&` are GRANDCHILDREN. Killing only the
    child (what `subprocess.run(timeout=...)` does) leaves those reparented to init,
    where they accumulate indefinitely - this machine collected 35 such orphans, the
    oldest running 55 days, before this was fixed (2026-07-26).

    `pgid` must be captured at SPAWN time (it equals `proc.pid`, because
    `start_new_session=True` makes the child a new group leader). Looking it up after
    the child is reaped raises ESRCH, which would silently skip the sweep in exactly
    the case that leaks: a shell that exits 0 having backgrounded a grandchild.
    The new session also guarantees `killpg` can never reach our own processes.

    Costs nothing on the common path: if the group is already empty, the first
    `killpg` raises ESRCH and we return without waiting out the grace period.
    """
    if pgid is None or pgid == os.getpgid(0):
        # Never signal our own group; fall back to killing just the child.
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=grace)
        except Exception:
            pass
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return                             # group already empty - clean exit, no survivors
    except Exception:
        return
    try:
        proc.wait(timeout=grace)           # give the group a chance to die politely
    except Exception:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)    # sweep survivors (grandchildren, backgrounded jobs)
    except Exception:
        pass
    try:
        proc.wait(timeout=grace)
    except Exception:
        pass


def _read_capped(fh) -> str:
    """Read back a captured-output temp file under a memory bound, keeping head+tail
    (the command-aware truncation below needs both ends)."""
    try:
        fh.flush()
    except Exception:
        pass
    fh.seek(0, os.SEEK_END)
    size = fh.tell()
    fh.seek(0)
    if size <= READ_CAP_BYTES:
        data = fh.read()
    else:
        half = READ_CAP_BYTES // 2
        head = fh.read(half)
        fh.seek(-half, os.SEEK_END)
        data = head + b"\n... [output capped] ...\n" + fh.read(half)
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_pipeline(cmd: str) -> list[str]:
    """Split a shell command into pipeline/chain segments, respecting quotes.

    Only splits on |, ;, && when they appear outside of single quotes,
    double quotes, or backticks.  This prevents false splits inside
    arguments like: python3 -c "from X import Y; from Z import W"
    or: grep 'foo\\|bar'
    """
    segments = []
    current = []
    i = 0
    in_single = False
    in_double = False
    in_backtick = False

    while i < len(cmd):
        ch = cmd[i]

        # Track quote state (no nesting - shell rules)
        if ch == "'" and not in_double and not in_backtick:
            in_single = not in_single
            current.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single and not in_backtick:
            in_double = not in_double
            current.append(ch)
            i += 1
            continue
        if ch == '`' and not in_single and not in_double:
            in_backtick = not in_backtick
            current.append(ch)
            i += 1
            continue

        # Handle backslash-escaped characters (skip next char)
        if ch == '\\' and not in_single and i + 1 < len(cmd):
            current.append(ch)
            current.append(cmd[i + 1])
            i += 2
            continue

        # Only split when outside all quotes
        if not in_single and not in_double and not in_backtick:
            # && operator
            if ch == '&' and i + 1 < len(cmd) and cmd[i + 1] == '&':
                segments.append(''.join(current))
                current = []
                i += 2
                continue
            # | (pipe) or ; (sequence)
            if ch in ('|', ';'):
                segments.append(''.join(current))
                current = []
                i += 1
                continue

        current.append(ch)
        i += 1

    # Last segment
    segments.append(''.join(current))
    return segments


def _extract_base_command(segment: str) -> str | None:
    """Extract the base command name from a single pipeline segment.

    Uses shlex to properly handle quoting, falling back to naive split.
    """
    segment = segment.strip()
    if not segment:
        return None

    # Try shlex-aware tokenization
    try:
        tokens = shlex.split(segment)
    except ValueError:
        # Unbalanced quotes - fall back to naive split
        tokens = segment.split()

    for token in tokens:
        # Skip env var assignments (FOO=bar) and redirections
        if "=" in token and not token.startswith("-"):
            continue
        if token.startswith("<") or token.startswith(">"):
            continue
        return os.path.basename(token)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_command(cmd: str) -> tuple[bool, str]:
    """Check if a command is safe to run.

    Returns (is_safe, reason).
    """
    cmd_stripped = cmd.strip()
    if not cmd_stripped:
        return False, "empty command"

    # Check blocked patterns
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, cmd_stripped):
            return False, f"blocked pattern: {pattern}"

    return True, "ok"


def get_available_tools() -> list[str]:
    """Return a list of common CTF tools found in the PATH (of this thread's container backend, if any)."""
    backend = get_backend()
    if backend is not None and hasattr(backend, "available_tools"):
        try:
            return list(backend.available_tools())
        except Exception:
            return []
    if _SANDBOX_REQUIRED:
        return []                                 # never describe the host's PATH to a solver that must be sandboxed
    common_tools = [
        "exiftool", "binwalk", "zsteg", "steghide", "strings", "grep",
        "find", "nasm", "gcc", "g++", "nc", "ssh", "python3", "identify",
        "convert", "foremost", "tesseract", "jq", "curl", "wget", "nmap",
        "smbclient", "sqlmap", "nikto", "gobuster", "dirb"
    ]
    available = []
    for tool in common_tools:
        if subprocess.run(f"command -v {tool}", shell=True, capture_output=True).returncode == 0:
            available.append(tool)
    return available


def get_tool_hint(cmd_name: str) -> str | None:
    """Return a hint if a common tool is missing."""
    hints = {
        "exiftool": "exiftool is missing. Try 'strings' or 'identify -verbose' for metadata.",
        "binwalk": "binwalk is missing. Try 'strings' or 'grep -oba' to find common signatures.",
        "zsteg": "zsteg is missing. Try manually inspecting LSB or using 'identify' to check image properties.",
        "steghide": "steghide is missing. This is a specialized tool; check if the pass is actually needed.",
        "strings": "strings is missing? Try 'cat -v' or 'hexdump -C'.",
        "nmap": "nmap is missing. Try 'nc -zv' for simple port scanning.",
        "cupp.py": "cupp.py is missing. You may need to create a custom wordlist with a simple script.",
    }
    return hints.get(cmd_name)


# ---------------------------------------------------------------------------
# Execution backend (Phase 3, 2026-09-14): a registered backend (solver.container_executor.ContainerSession)
# receives every command whose working directory lies inside its mounted tree; with none registered the host
# subprocess path below runs unchanged (the reproducibility anchor).
#
# 2026-10-19 (audit N1): the backend used to be ONE module global. The realbench runners solve 3 tasks at once in
# threads, so each solve's `use_backend` overwrote the others' and a command whose working directory did not match
# the current backend fell through to a HOST subprocess (~30% of commands in the Phase-3 campaigns). Now:
#   - the backend is per thread (`threading.local`), so concurrent solves cannot see each other's session;
#   - inside `use_backend` a command outside the session's tree is BLOCKED, never run on the host;
#   - `require_sandbox()` (set by the realbench runners) blocks any command in a thread with no backend at all;
#   - `use_backend.stats` counts what was dispatched / blocked, so a solve can check every command ran in its container.
# The Phase-2 host path (no backend, sandbox not required) is unchanged.
# ---------------------------------------------------------------------------
_LOCAL = threading.local()
_SANDBOX_REQUIRED = False
_VIOLATIONS_LOCK = threading.Lock()
_SANDBOX_VIOLATIONS = 0


def set_backend(backend) -> None:
    _LOCAL.backend = backend


def get_backend():
    return getattr(_LOCAL, "backend", None)


def require_sandbox(on: bool = True) -> None:
    """Process-wide: when on, a command in a thread without a backend is blocked instead of run on the host."""
    global _SANDBOX_REQUIRED
    _SANDBOX_REQUIRED = bool(on)


def sandbox_violations() -> int:
    """Commands blocked because the sandbox was required and the thread had no backend (should stay 0)."""
    return _SANDBOX_VIOLATIONS


def _note_violation() -> None:
    global _SANDBOX_VIOLATIONS
    with _VIOLATIONS_LOCK:
        _SANDBOX_VIOLATIONS += 1


class use_backend:
    """Context manager: `with use_backend(session): ...` routes this thread's commands through `session`, then restores.
    `stats` = {"dispatched": commands sent to the session, "rejected": commands blocked because their working directory
    was outside the session's tree} for the commands run inside the block."""

    def __init__(self, backend):
        self.backend, self._prev, self._prev_stats = backend, None, None
        self.stats = {"dispatched": 0, "rejected": 0}

    def __enter__(self):
        self._prev, self._prev_stats = get_backend(), getattr(_LOCAL, "stats", None)
        self.stats = {"dispatched": 0, "rejected": 0}
        set_backend(self.backend)
        _LOCAL.stats = self.stats
        return self.backend

    def __exit__(self, *exc):
        set_backend(self._prev)
        _LOCAL.stats = self._prev_stats


def _finalize_result(result: dict, cmd: str, stdout: str, stderr: str, returncode: int) -> dict:
    """Command-aware truncation, hard caps and missing-tool hints (shared by the host and container paths)."""
    base_cmd = _extract_base_command(cmd)
    if base_cmd in ("ls", "find", "dir"):
        lines = stdout.splitlines()
        if len(lines) > 100:
            stdout = "\n".join(lines[:50]) + f"\n\n... [{len(lines) - 100} lines truncated for brevity] ...\n\n" + "\n".join(lines[-50:])
    elif base_cmd in ("xxd", "hexdump"):
        lines = stdout.splitlines()
        if len(lines) > 60:
            stdout = "\n".join(lines[:30]) + f"\n\n... [{len(lines) - 60} lines of hex dump truncated] ...\n\n" + "\n".join(lines[-30:])
    if len(stdout) > MAX_OUTPUT_BYTES:
        stdout = stdout[:MAX_OUTPUT_BYTES] + f"\n... [truncated at {MAX_OUTPUT_BYTES} bytes]"
    if len(stderr) > MAX_OUTPUT_BYTES:
        stderr = stderr[:MAX_OUTPUT_BYTES] + f"\n... [truncated at {MAX_OUTPUT_BYTES} bytes]"
    result["stdout"] = stdout
    result["stderr"] = stderr
    result["returncode"] = returncode
    is_not_found = (returncode == 127 or
                    "command not found" in stderr.lower() or
                    f"{base_cmd}: not found" in stderr.lower() or
                    f"{base_cmd}: command not found" in stderr.lower())
    if is_not_found:
        hint = get_tool_hint(base_cmd)
        if hint:
            result["stderr"] = (result["stderr"] + f"\n[HINT] {hint}").strip()
    return result


def run_command(cmd: str, working_dir: str | Path) -> dict:
    """Execute a single shell command in the given working directory.

    Returns:
        {"command": str, "stdout": str, "stderr": str, "returncode": int, "error": str|None}
    """
    result = {
        "command": cmd,
        "stdout": "",
        "stderr": "",
        "returncode": -1,
        "error": None,
    }

    # Validate
    is_safe, reason = validate_command(cmd)
    if not is_safe:
        result["error"] = f"BLOCKED: {reason}"
        return result

    working_dir = str(working_dir)

    backend = get_backend()
    if backend is not None:
        stats = getattr(_LOCAL, "stats", None)
        if not backend.matches(working_dir):
            # fail closed: inside a sandboxed solve nothing runs on the host (audit N1)
            if stats is not None:
                stats["rejected"] += 1
            result["error"] = "BLOCKED: working directory is outside the sandbox container (no host fallback)"
            return result
        if stats is not None:
            stats["dispatched"] += 1
        raw = backend.run(cmd, working_dir)
        if raw.get("error"):
            result["error"] = raw["error"]
        return _finalize_result(result, cmd, raw.get("stdout", ""), raw.get("stderr", ""), int(raw.get("returncode", -1)))
    if _SANDBOX_REQUIRED:
        _note_violation()
        result["error"] = "BLOCKED: sandbox required but no container backend is registered for this solve"
        return result

    try:
        # Scrub sensitive environment variables from the subprocess
        safe_env = {k: v for k, v in os.environ.items() if "TOKEN" not in k and "PASS" not in k and "KEY" not in k}
        safe_env["LC_ALL"] = "C"
        
        # Capture to temp FILES, not pipes. `subprocess.run(capture_output=True)` calls
        # communicate() after a timeout kill, which blocks until every writer closes the
        # pipe - an orphaned grandchild holding the inherited fd hangs the parent forever,
        # so the nominal 60s timeout was not actually enforced. Files cannot hang.
        with tempfile.TemporaryFile() as fout, tempfile.TemporaryFile() as ferr:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=working_dir,
                stdout=fout,
                stderr=ferr,
                stdin=subprocess.DEVNULL,            # a tool that reads stdin exits instead of hanging
                env=safe_env,
                preexec_fn=_limit_child_resources,   # cap file size: no decompression-bomb disk fill
                start_new_session=True,              # own process group -> group teardown below
            )
            pgid = proc.pid                          # == the new group's id (start_new_session)
            timed_out = False
            try:
                proc.wait(timeout=COMMAND_TIMEOUT)   # waits on the PID, never reads a pipe
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                _kill_process_group(proc, pgid)      # always sweep, including after a clean exit
            stdout = _read_capped(fout)
            stderr = _read_capped(ferr)

        if timed_out:
            result["error"] = f"TIMEOUT after {COMMAND_TIMEOUT}s"

        returncode = proc.returncode if proc.returncode is not None else -1
        _finalize_result(result, cmd, stdout, stderr, returncode)

    except subprocess.TimeoutExpired:
        result["error"] = f"TIMEOUT after {COMMAND_TIMEOUT}s"
    except Exception as e:
        result["error"] = str(e)

    return result


def run_commands(commands: list[str], working_dir: str | Path,
                 log=None, wall_cap_s: float | None = None) -> list[dict]:
    """Execute a list of commands sequentially, returning all results.

    If *log* is a ChallengeLogger, use it for rich output display.

    *wall_cap_s* is an EPISODE-level wall-clock budget spanning this whole command
    list: once it is exhausted the remaining commands are skipped rather than run, so
    one episode cannot consume unbounded time even when every individual command stays
    under COMMAND_TIMEOUT. Default None reads `CTF_TOOL_EPISODE_CAP_S` from the
    environment; unset means no cap, which preserves the legacy behaviour the
    reproducibility baseline was measured under.
    """
    if wall_cap_s is None:
        try:
            wall_cap_s = float(os.environ.get("CTF_TOOL_EPISODE_CAP_S", "") or 0) or None
        except ValueError:
            wall_cap_s = None
    t0 = time.monotonic()
    results = []
    for cmd in commands:
        elapsed = time.monotonic() - t0
        if wall_cap_s is not None and elapsed >= wall_cap_s:
            res = {"command": cmd, "stdout": "", "stderr": "", "returncode": -1,
                   "error": f"SKIPPED: episode wall-clock cap {wall_cap_s:g}s exhausted "
                            f"({elapsed:.1f}s elapsed)"}
        else:
            res = run_command(cmd, working_dir)

        if log:
            log.command(
                cmd,
                stdout=res.get("stdout", ""),
                stderr=res.get("stderr", ""),
                rc=res.get("returncode", -1),
                error=res.get("error"),
            )
        else:
            print(f"      🔧  Running: {cmd}")
            if res["error"]:
                print(f"         ✗ {res['error']}")
            elif res["returncode"] != 0:
                print(f"         ⚠ Exit code {res['returncode']}")
            else:
                out_preview = res["stdout"][:100].replace("\n", " ")
                print(f"         ✓ {out_preview}{'...' if len(res['stdout']) > 100 else ''}")

        results.append(res)
    return results


MAX_OUTPUT_CHARS = 2000  # per-command output cap to prevent context overflow

def format_results_for_llm(results: list[dict]) -> str:
    """Format command results into a string for the LLM."""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"### Command {i}: `{r['command']}`")
        if r["error"]:
            parts.append(f"**Error:** {r['error']}")
        else:
            parts.append(f"**Exit code:** {r['returncode']}")
            stdout = (r["stdout"] or "").strip()
            stderr = (r["stderr"] or "").strip()
            if stdout:
                if len(stdout) > MAX_OUTPUT_CHARS:
                    # Keep head + tail so the LLM sees both start and end
                    half = MAX_OUTPUT_CHARS // 2
                    stdout = stdout[:half] + f"\n\n... [{len(stdout) - MAX_OUTPUT_CHARS} chars truncated] ...\n\n" + stdout[-half:]
                parts.append(f"**stdout:**\n```\n{stdout}\n```")
            if stderr:
                if len(stderr) > MAX_OUTPUT_CHARS:
                    stderr = stderr[:MAX_OUTPUT_CHARS] + f"\n... [{len(stderr) - MAX_OUTPUT_CHARS} chars truncated]"
                parts.append(f"**stderr:**\n```\n{stderr}\n```")
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Auto-decode: find encoded flags in command output
# ---------------------------------------------------------------------------

# Matches base64-like strings (20+ chars, valid base64 alphabet, optional padding)
_BASE64_RE = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')

# Matches hex strings (40+ hex chars)
_HEX_RE = re.compile(r'(?:[0-9a-fA-F]{2}){20,}')

_FLAG_RE = re.compile(r'picoCTF\{[^\s}]{3,100}\}')


def extract_encoded_flags(text: str) -> list[str]:
    """Scan text for base64/hex/URL-encoded strings, decode them, and return any flags found.

    This bypasses the LLM entirely - flags come straight from decoded bytes.
    """
    from urllib.parse import unquote
    flags = []

    # 1. Direct flag search
    for m in _FLAG_RE.finditer(text):
        flags.append(m.group(0))

    # 2. URL-decode (catches picoCTF%7B...%7D)
    if '%7B' in text or '%7b' in text:
        decoded_url = unquote(text)
        for fm in _FLAG_RE.finditer(decoded_url):
            if fm.group(0) not in flags:
                print(f"      🔓  Auto-decoded URL → {fm.group(0)}")
                flags.append(fm.group(0))

    # 3. Base64 decode
    for m in _BASE64_RE.finditer(text):
        candidate = m.group(0)
        try:
            decoded = b64module.b64decode(candidate + "==").decode("utf-8", errors="replace")
            for fm in _FLAG_RE.finditer(decoded):
                print(f"      🔓  Auto-decoded base64 → {fm.group(0)}")
                flags.append(fm.group(0))
        except Exception:
            pass

    # 4. Hex decode
    for m in _HEX_RE.finditer(text):
        candidate = m.group(0)
        try:
            decoded = bytes.fromhex(candidate).decode("utf-8", errors="replace")
            for fm in _FLAG_RE.finditer(decoded):
                print(f"      🔓  Auto-decoded hex → {fm.group(0)}")
                flags.append(fm.group(0))
        except Exception:
            pass

    return flags
