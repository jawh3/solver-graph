#!/usr/bin/env python3
"""
LLM Solver — multi-turn tool-augmented CTF solving.

The LLM can request shell commands to analyze challenge files (strings, exiftool,
base64, python3, etc.).  Commands are executed in a sandbox and results fed back
until a flag is found or rounds are exhausted.
"""
import os
import re
import json
from pathlib import Path
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

from solver.tool_executor import run_commands, format_results_for_llm, extract_encoded_flags, get_available_tools
from solver.logger import ChallengeLogger
from retrieval.memory_client import get_tools, call_tool_sync, format_tools_for_prompt

# Load memory tools once at startup (gracefully empty if server is down)
_MEMORY_TOOLS: list[dict] = []
_MEMORY_TOOLS_PROMPT: str = ""

def _init_memory_tools():
    """Populate _MEMORY_TOOLS and _MEMORY_TOOLS_PROMPT (called lazily on first solve)."""
    global _MEMORY_TOOLS, _MEMORY_TOOLS_PROMPT
    if _MEMORY_TOOLS_PROMPT:  # already initialised
        return
    _MEMORY_TOOLS = get_tools()
    _MEMORY_TOOLS_PROMPT = format_tools_for_prompt(_MEMORY_TOOLS)
    if _MEMORY_TOOLS:
        print(f"✓ Memory server: {len(_MEMORY_TOOLS)} retrieval tool(s) available")
    else:
        print("ℹ  Memory server not reachable — solving without memory context")

load_dotenv()

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-32B-AWQ")

client = OpenAI(
    base_url=VLLM_URL,
    api_key="not-needed",
)

MAX_TOOL_ROUNDS = 8       # max tool-use iterations (fail fast, attempt more)
MAX_COMMANDS_PER_ROUND = 5  # max commands per LLM request
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# System prompt template — {memory_tools_section} is filled in at solve time.
_SYSTEM_PROMPT_TEMPLATE = """You are an expert CTF solver with access to a Linux shell.

You will be given a PicoCTF challenge. You can either:
1. Solve it directly if you know the answer, OR
2. Request shell commands to analyze the provided files, OR
3. Query your memory of past CTF challenges for helpful techniques and insights.

**Response format — you MUST respond with valid JSON only (no markdown, no extra text):**

To run commands:
{{"action": "run_commands", "commands": ["strings file.bin | grep picoCTF", "pdftotext file.pdf -"]}}

To query memory:
{{"action": "query_memory", "tool": "memory_search", "args": {{"query": "RSA small exponent", "category": "crypto", "k": 3}}}}

To submit a flag you have computed/found:
{{"action": "submit_flag", "flag": "picoCTF{{the_actual_flag}}"}}

If you truly cannot solve it:
{{"action": "give_up", "reason": "explanation"}}{memory_tools_section}

**Available shell tools:**
You have access to ALL standard Linux commands. This includes but is not limited to:
strings, file, xxd, hexdump, base64, openssl, python3, grep, cat, head, tail,
awk, sed, sort, uniq, tr, cut, rev, wc, echo, dd, od, ls, find, stat, diff, tac,
pdftotext, pdfinfo, exiftool, binwalk, foremost, steghide, tesseract,
unzip, tar, gzip, curl, wget, nc, ncat, nmap, socat,
r2, rabin2, objdump, readelf, nm, gdb, ltrace, strace,
john, hashcat, ssh, sshpass, git, scp, sftp, sudo, expect, mkdir, cp, mv, chmod

**Strategy — use memory tools wisely:**
- Start with `memory_search` or `memory_search_failures` to check what worked (or failed) for similar challenges.
- Use `memory_by_technique` or `memory_by_vulnerability` once you have a hypothesis.
- Then pivot to shell commands to apply what you learned.

**Available Python libraries (use via `python3 -c "..."`):**
pytesseract, PIL/Pillow, pyzbar, pycryptodome (Crypto), gmpy2, z3 (z3-solver),
angr, capstone, ropper, struct, binascii, hashlib, itertools, codecs

**Critical rules:**
- ALWAYS use shell commands to decode data. NEVER decode base64/hex/rot13 in your head — you WILL get it wrong.
  Example: if you see base64 like "cGljb0NURns...", run: echo "cGljb0NURns..." | base64 -d
- ONLY submit a flag you have ACTUALLY SEEN in command output. Never guess or reconstruct.
- Real picoCTF flags contain unique identifiers (hex hashes, random strings). If it looks generic, it's wrong.
- Files are in the current working directory — reference them by filename.
- You can run up to 6 commands per round, with up to 7 rounds total.
- **CRITICAL: `cd` does NOT persist between commands!** Each command runs in its own subprocess. Use `cd DIR && command` to run a command in a subdirectory. Example: `cd challenge && cat flag.txt` NOT `cd challenge` then `cat flag.txt` as separate commands.
- **CRITICAL: Do NOT explore your local filesystem (ls ~/, find ~/ etc.) — that is the SOLVER machine, NOT the challenge. Challenge files are ONLY in the working directory or on the remote server.**
- **CRITICAL: NEVER run `git push` unless the remote points to the challenge server (picoctf.net). Check with `cd REPO && git remote -v` first.**
- If you see ROT13/encoded text in HTML comments, DECODE IT FIRST before using the information.

**Tips — File Analysis:**
- For PDFs: `pdftotext file.pdf -`
- For metadata: `exiftool file`
- For binaries: `strings file | grep picoCTF`
- For encoding: `echo "encoded" | base64 -d` — ALWAYS use commands to decode
- For ROT13: `echo "text" | tr 'A-Za-z' 'N-ZA-Mn-za-m'`
- For hidden data: `binwalk file`, `xxd file | head`
- For multi-part flags: use `grep -n FLAGPART file` to preserve order. Do NOT sort.
- For corrupted files: use `python3 -c "data=open('file','rb').read(); ..."` to fix headers/bytes

**Tips — Images & Stego:**
- For OCR (text in images): `python3 -c "import pytesseract; from PIL import Image; print(pytesseract.image_to_string(Image.open('img.png')))"`
- For QR codes: `python3 -c "from pyzbar.pyzbar import decode; from PIL import Image; print(decode(Image.open('img.png')))"`
- For steghide: first decode the passphrase with a command, then: `steghide extract -sf img.jpg -p 'PASSPHRASE'`
- For zsteg: `zsteg img.png`

**Tips — SSH/Git Challenges (VERY IMPORTANT):**
- For SSH with a password: `sshpass -p 'PASSWORD' ssh -o StrictHostKeyChecking=no user@host -p PORT 'COMMAND'`
- Run commands on the remote server by appending them: `sshpass -p 'PASSWORD' ssh -o StrictHostKeyChecking=no user@host -p PORT 'ls -la; cat flag.txt'`
- Chain multiple remote commands: `sshpass -p 'PASSWORD' ssh ... 'cmd1 && cmd2 && cmd3'`
- For sudo on SSH (if terminal needed): `sshpass -p 'PASSWORD' ssh ... 'echo PASSWORD | sudo -S cat /root/flag.txt'`
- For NOPASSWD sudo on editors (e.g. emacs): `sshpass -p 'PASSWORD' ssh ... 'sudo emacs -q --batch --eval "(princ (with-temp-buffer (insert-file-contents \\"/root/flag.txt\\") (buffer-string)))"'`
- For sudo -l: `sshpass -p 'PASSWORD' ssh ... 'sudo -l'`
- For git clone with password: `GIT_SSH_COMMAND="sshpass -p PASSWORD ssh -o StrictHostKeyChecking=no" git clone ssh://git@HOST:PORT/path/repo.git`
- **CRITICAL:** If a Git repository requires `sshpass` for cloning, you **MUST** ALSO use `GIT_SSH_COMMAND` for ALL other network operations (`push`, `pull`, `fetch`). If you forget it, the command WILL hang forever. Example: `cd repo && GIT_SSH_COMMAND="sshpass -p PASSWORD ssh -o StrictHostKeyChecking=no" git push origin master`
- After cloning: `cd repo && cat README.md && git log --oneline && git show HEAD`
- **DO NOT try to SSH without sshpass** — interactive password entry will hang.
- **DO NOT explore ~/**, that is the solver machine. Everything you need is on the REMOTE server.

**Tips — Web Exploitation (VERY IMPORTANT for Web challenges):**
- ALWAYS use `curl` to interact with web services. You have full HTTP access.
- For source code: `curl -s URL`
- For headers: `curl -sI URL`
- For custom headers: `curl -s -H 'X-Custom: value' URL`
- For POST data: `curl -s -X POST -d 'key=value' URL`
- For cookies: `curl -s -b 'name=value' URL`
- **Cookie jar for multi-step flows**: `curl -s -c /tmp/cookies.txt -L URL` then `curl -s -b /tmp/cookies.txt URL2`
- For login + access pattern: `curl -s -c /tmp/cookies.txt -L -X POST -d 'username=X&password=Y' URL/login && curl -s -b /tmp/cookies.txt URL/flag`
- **CRITICAL**: Always use `curl -L` to follow redirects when POSTing to forms
- For SSTI (Jinja2): first test with `curl -s -L -X POST -d 'content={{7*7}}' URL` — if you see 49 in output, it's vulnerable
  Then: `curl -s -L -X POST -d 'content={{config}}' URL`
  Then: `curl -s -L -X POST -d 'content={{request.application.__globals__.__builtins__.__import__("os").popen("cat flag.txt").read()}}' URL`
  Or try: `curl -s -L -X POST -d 'content={{lipsum.__globals__["os"].popen("cat flag*").read()}}' URL`
- For SQL injection: `curl -s 'URL?param=1%27+OR+1%3D1--'`
- For path traversal: `curl -s 'URL/../../../../etc/passwd'`
- **CRITICAL**: If HTML comments contain ROT13 text, ALWAYS decode with `echo 'TEXT' | tr 'A-Za-z' 'N-ZA-Mn-za-m'` BEFORE using the information
- If you find dev headers/bypass hints, USE THEM with curl immediately

**Tips — Binary Exploitation & Networking:**
- Read the source code FIRST if provided — `cat source.c`
- For reversing: `r2 -q -c 'aaa; pdf @ main' binary` or `objdump -d binary | head -200`
- For symbols: `nm binary` or `objdump -t binary | grep -i flag`
- **Networking/Netcat:** `nc HOST PORT` WILL HANG and time out if you just use it interactively. 
  - To read the initial banner only: `nc -w 3 HOST PORT`
  - To send a simple string and get a response: `echo 'INPUT' | nc -w 3 HOST PORT`
- For PIE/ASLR with a leaked address: write a Python script that connects, reads the leak, computes the target, and sends it:
  `python3 -c "import socket; s=socket.create_connection(('HOST',PORT)); data=s.recv(4096).decode(); print(data); addr=int(data.split('0x')[1].split()[0],16); win=addr-MAIN_OFFSET+WIN_OFFSET; s.sendall((hex(win)+'\n').encode()); print(s.recv(4096).decode()); print(s.recv(4096).decode())"`
- Do the ENTIRE exploit in ONE python command — connect, read, compute, send, read flag

**Tips — Missing Tools:**
- If a Linux command-line tool you need is missing (like `smbclient`, `exiftool`, `steghide`), you can install it yourself:
  `sudo apt-get update && sudo apt-get install -y smbclient`

**Tips — Crypto:**
- For crypto: `python3 -c "from Crypto.Cipher import AES; ..."` or `python3 -c "import gmpy2; ..."`
- For constraint solving: `python3 -c "from z3 import *; ..."`
    - To obtain the flag, you may need to use tools or write python scripts.
    - If you see a flag format like picoCTF{{...}}, ALWAYS output it clearly!
    - Available tools on this system: {available_tools}
"""

CHALLENGE_PROMPT = """**Challenge:**
- Name: {name}
- Category: {category}
- Points: {points}
- Difficulty: {difficulty}

**Description:**
{description}

**Hints:**
{hints}

**Code Blocks / Connection Info:**
{code_blocks}
{server_urls}
**Files in working directory:**
{file_listing}

Analyze this challenge. Respond with JSON only."""

TOOL_RESULTS_PROMPT = """Here are the results of your commands:

{results}

Based on these results, either:
- Request more commands if you need more analysis
- Submit the flag if you found it
- Give up if you cannot solve it

Respond with JSON only."""

FALLBACK_PROMPT = """You are an expert CTF solver. Solve this PicoCTF challenge step by step.

**Challenge:**
- Name: {name}
- Category: {category}
- Points: {points}
- Difficulty: {difficulty}

**Description:**
{description}

**Hints:**
{hints}

**Code Blocks / Connection Info:**
{code_blocks}

**File Contents:**
{file_contents}

**CRITICAL INSTRUCTIONS:**
1. Only output a flag if you have ACTUALLY COMPUTED or DECODED it from the data above.
2. Real picoCTF flags contain unique random identifiers like hex hashes (e.g. picoCTF{{s0me_w0rds_a1b2c3d4}}).
3. NEVER guess a flag based on the challenge theme. Flags like picoCTF{{hidden_in_comment}} or picoCTF{{fixed_the_header}} are WRONG — they are generic descriptions, not real flags.
4. If the challenge requires interacting with a server, running commands, or performing steps you cannot do here, say CANNOT_SOLVE.
5. If you cannot determine the EXACT flag with certainty, say CANNOT_SOLVE.

Provide the flag in format picoCTF{{...}} ONLY if you have computed it. Otherwise say CANNOT_SOLVE."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_thinking(content: str) -> tuple[str, str]:
    """Extract <think>...</think> content and return (thinking_text, cleaned_content)."""
    thinking = ""
    match = re.search(r'<think>(.*?)</think>', content, flags=re.DOTALL)
    if match:
        thinking = match.group(1).strip()
    cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return thinking, cleaned

# Flag must: no whitespace inside braces, 3-100 chars long
FLAG_PATTERN = re.compile(r'picoCTF\{[^\s}]{3,100}\}')

PLACEHOLDER_FLAGS = {
    "picoCTF{flag_here}", "picoCTF{FLAG}", "picoCTF{flag}",
    "picoCTF{your_flag_here}", "picoCTF{the_flag}", "picoCTF{example_flag}",
    "picoCTF{placeholder}", "picoCTF{PLACEHOLDER}", "picoCTF{insert_flag}",
    "picoCTF{actual_flag}", "picoCTF{real_flag}", "picoCTF{...}",
    "picoCTF{???}", "picoCTF{REDACTED}", "picoCTF{redacted}",
    "picoCTF{todo}", "picoCTF{TODO}", "picoCTF{hidden_flag_here}",
    "picoCTF{flag_content_here}", "picoCTF{the_actual_flag}",
    "picoCTF{flag_value}", "picoCTF{flag_goes_here}",
    "picoCTF{replace_me}", "picoCTF{sample_flag}",
    # Code template patterns (Python f-strings, format strings)
    "picoCTF{{{result}", "picoCTF{{result}}", "picoCTF{result}",
}

# Common English words that appear in hallucinated flags but never in real ones
_GENERIC_WORDS = {
    "flag", "hidden", "found", "secret", "answer", "solution", "correct",
    "fixed", "header", "comment", "content", "here", "value", "result",
    "decoded", "cracked", "solved", "extracted", "discovered",
    "the", "this", "that", "your", "file", "data",
    "rot13", "notes", "flame", "crazy", "simple", "easy",
}

def _is_likely_placeholder(flag: str) -> bool:
    """Heuristic: reject flags that are just common English words or dummy patterns.

    Real picoCTF flags typically contain unique hex suffixes, leet-speak, or
    random-looking character sequences — not plain English words.
    """
    if flag in PLACEHOLDER_FLAGS:
        return True
    # Extract content inside braces
    inner = flag.split('{', 1)[-1].rstrip('}')

    # Reject flags that contain HTTP header names (hallucinated from challenge hints)
    if ':' in inner and any(w in inner.lower() for w in ['access', 'header', 'auth', 'dev', 'admin']):
        return True

    # Split on underscores
    parts = inner.lower().replace('-', '_').split('_')
    # If ALL parts are generic/common words → likely hallucinated
    if len(parts) >= 1 and all(p in _GENERIC_WORDS for p in parts if p):
        return True

    # Check for dummy pattern: all parts are simple leet-substitutions of generic words
    # e.g. "s0me_w0rds_a1b2c3d4" — the last part looks like dummy hex
    if len(parts) >= 2:
        last = parts[-1]
        # If the last part is exactly 8 hex chars with a running pattern (a1b2c3d4, abc123, etc.)
        if re.match(r'^[a-f0-9]{6,8}$', last):
            # Check if it's a sequential/patterned dummy like a1b2c3d4, abc12345
            if last in ('a1b2c3d4', 'abc123', '12345678', 'abcd1234', 'deadbeef'):
                return True

    return False


def _format_hints(hints: list) -> str:
    if not hints:
        return "No hints available."
    return "\n".join(f"- {h}" for h in hints)


def _format_code_blocks(code_blocks: list) -> str:
    if not code_blocks:
        return "None."
    return "\n".join(f"```\n{block}\n```" for block in code_blocks)


def _get_file_listing(local_files: list) -> str:
    """List downloaded files with their sizes and types."""
    if not local_files:
        return "No files."

    parts = []
    for f in local_files:
        path = f.get("local_path")
        if not path or not Path(path).exists():
            parts.append(f"- {f.get('text', 'file')}: not downloaded")
            continue
        p = Path(path)
        size = p.stat().st_size
        parts.append(f"- {p.name} ({size} bytes)")
    return "\n".join(parts) if parts else "No files."


def _get_server_urls(challenge: dict) -> str:
    """Extract live server URLs from challenge data for web/network challenges.

    These are URLs the LLM should use with curl/nc instead of localhost.
    """
    import re as _re
    urls = set()

    # Check local_files for HTTP URLs pointing to challenge servers (not CDN downloads)
    for f in challenge.get("local_files", []):
        url = f.get("url", "")
        # Challenge server URLs have a high port number and are NOT file downloads
        if _re.match(r'https?://[^/]+:\d{4,}', url):
            # Strip path — we want the base URL
            from urllib.parse import urlparse
            parsed = urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            urls.add(base)

    # Check files list from scraper (before download) for server URLs
    for f in challenge.get("files", []):
        url = f.get("url", "")
        if _re.match(r'https?://[^/]+:\d{4,}', url):
            from urllib.parse import urlparse
            parsed = urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            urls.add(base)

    # Check code_blocks for nc/connect info
    for block in challenge.get("code_blocks", []):
        # Look for patterns like "nc hostname port" or URLs
        for m in _re.finditer(r'https?://\S+:\d{4,}\S*', block):
            urls.add(m.group(0).rstrip('/'))
        for m in _re.finditer(r'nc\s+(\S+)\s+(\d+)', block):
            urls.add(f"nc {m.group(1)} {m.group(2)}")

    # Check description for URLs
    desc = challenge.get("description", "")
    for m in _re.finditer(r'https?://\S+:\d{4,}\S*', desc):
        urls.add(m.group(0).rstrip('/'))

    if not urls:
        return ""
    return "\n**\U0001f310 Live Server URLs (use these with curl/nc, NOT localhost):**\n" + "\n".join(f"- {u}" for u in sorted(urls)) + "\n"

def _read_file_contents(local_files: list, max_bytes: int = 8000) -> str:
    """Read downloaded artifact files and include their contents (for fallback)."""
    if not local_files:
        return "No files attached."

    parts = []
    for f in local_files:
        path = f.get("local_path")
        if not path or not Path(path).exists():
            continue
        p = Path(path)
        size = p.stat().st_size
        if size > max_bytes:
            parts.append(f"- {p.name}: binary/large file ({size} bytes, skipped)")
            continue
        try:
            content = p.read_text(errors="replace")
            parts.append(f"--- {p.name} ---\n{content}\n--- end {p.name} ---")
        except Exception:
            parts.append(f"- {p.name}: binary file ({size} bytes)")
    return "\n".join(parts) if parts else "No readable files."


def extract_flag(text: str) -> Optional[str]:
    """Extract a picoCTF{...} flag from text, rejecting placeholders."""
    matches = FLAG_PATTERN.findall(text)
    for flag in matches:
        if not _is_likely_placeholder(flag):
            return flag
    if matches:
        print(f"      ⚠  Rejected placeholder flag(s): {matches}")
    return None


def _normalize_action(obj):
    """Map common *valid-JSON* action variants onto the canonical {'action': ...} schema.

    STRICT NO-OP when 'action' is already present: Qwen/Llama emit the canonical form and are
    byte-identical under this path (so their frozen campaign results are unaffected). This only
    triggers for models that emit reasonable-but-variant schemas — e.g. Gemma's
    {'run_commands': [{'command': '...', 'shell': 'bash'}]} or a bare {'commands': [...]} without
    an 'action' key. Without this, such models lose ~1-in-4 rounds to schema-mismatch nudges and
    floor on multi-round tasks — a harness artifact, not a capability limit. Lenient parser is a
    SUPERSET of the strict one (it accepts more valid actions; it never reinterprets a strict one)."""
    if not isinstance(obj, dict) or obj.get("action"):
        return obj

    def _cmds(v):
        out = []
        if isinstance(v, str):
            out = [v]
        elif isinstance(v, list):
            for c in v:
                if isinstance(c, str):
                    out.append(c)
                elif isinstance(c, dict):
                    cc = c.get("command") or c.get("cmd") or c.get("commands")
                    if isinstance(cc, str):
                        out.append(cc)
                    elif isinstance(cc, list):
                        out.extend(x for x in cc if isinstance(x, str))
        return out

    for k in ("run_commands", "commands", "command"):
        if k in obj:
            cmds = _cmds(obj[k])
            if cmds:
                return {"action": "run_commands", "commands": cmds}
    for k in ("submit_flag", "flag"):
        if isinstance(obj.get(k), str):
            return {"action": "submit_flag", "flag": obj[k]}
    if isinstance(obj.get("query_memory"), dict):
        return {"action": "query_memory", **obj["query_memory"]}
    for k in ("give_up", "giveup"):
        if k in obj:
            r = obj.get(k)
            return {"action": "give_up", "reason": r if isinstance(r, str) else "model gave up"}
    return obj


def _parse_llm_json(content: str) -> Optional[dict]:
    """Try to parse JSON from LLM output, handling markdown fences and thinking tags.

    Routes every successful parse through _normalize_action so valid-but-variant action schemas
    (common in non-Qwen models) are accepted; canonical {'action': ...} output is unchanged."""
    # Strip <think>...</think> tags if present
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = content.strip()

    candidates = [content]
    # markdown code fence
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    if m:
        candidates.append(m.group(1).strip())
    # first flat { ... } block (no nested braces)
    m = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    # greedy first-{ to last-} span (handles nested objects, e.g. Gemma's command dicts)
    m = re.search(r'\{.*\}', content, re.DOTALL)
    if m:
        candidates.append(m.group(0))

    for cand in candidates:
        try:
            return _normalize_action(json.loads(cand))
        except (json.JSONDecodeError, TypeError):
            continue

    return None


def _get_working_dir(challenge: dict) -> str:
    """Get the working directory for command execution (where files are)."""
    local_files = challenge.get("local_files", [])
    for f in local_files:
        path = f.get("local_path")
        if path and Path(path).exists():
            return str(Path(path).parent)
    # Fallback to artifacts dir
    return str(Path(__file__).parent / "artifacts")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_llm() -> bool:
    """Quick health-check for the vLLM server."""
    try:
        import requests
        base = VLLM_URL.replace("/v1", "")
        r = requests.get(f"{base}/health", timeout=5)
        if r.status_code == 200:
            print(f"✓ vLLM server healthy at {base}")
            return True
        models = client.models.list()
        print(f"✓ vLLM server responding — models: {[m.id for m in models.data]}")
        return True
    except Exception as e:
        print(f"✗ vLLM server not reachable: {e}")
        return False


def solve_challenge(
    challenge: dict,
    use_tools: bool = True,
    use_memory: bool = False,
    log=None,
    injected_rag_context: str = "",
    max_rounds: int = 15,
    max_output_tokens: int = 0,
    context_window: int = 32768,
    working_dir: str = None,
    llm_seed: int = None,
) -> dict:
    """Solve a challenge using multi-turn tool-augmented LLM interaction.

    If use_tools is False, uses single-shot LLM only (for ablation).
    If use_memory is True, memory MCP tools are available to the LLM.

    Returns a dict:
        {"flag": str|None, "trace": list[dict], "rounds_used": int,
         "gave_up_reason": str|None, "total_commands": int}
    """
    name = challenge.get("name", "Unknown")
    category = challenge.get("category", "Unknown")
    points = challenge.get("points", "?")
    difficulty = challenge.get("difficulty", "?")
    description = challenge.get("description", "No description.")
    hints = _format_hints(challenge.get("hints", []))
    code_blocks = _format_code_blocks(challenge.get("code_blocks", []))
    file_listing = _get_file_listing(challenge.get("local_files", []))
    working_dir = working_dir or _get_working_dir(challenge)
    server_urls = _get_server_urls(challenge)

    trace = []           # per-round records
    total_commands = 0
    total_tokens = 0     # cumulative LLM tokens (effort signal, D1; additive — does not affect solving)
    gave_up_reason = None
    recent_commands: list[str] = []  # loop detection: track last N commands
    flags_seen_in_output: set[str] = set()  # anti-hallucination: track flags from command output
    rejected_flag_attempts: dict[str, int] = {}  # track how many times each flag was rejected
    effective_max_tokens = max_output_tokens if max_output_tokens > 0 else DEFAULT_MAX_OUTPUT_TOKENS

    # Ablation mode: skip tools entirely — no flag submission (high hallucination risk)
    if not use_tools:
        if log:
            log.warn("No-tools mode: skipping (high hallucination risk without tool verification)")
        return {"flag": None, "trace": [], "rounds_used": 0,
                "gave_up_reason": "no_tools_mode_disabled", "total_commands": 0, "tokens": 0}

    # Initialise memory context
    memory_context = ""
    if use_memory:
        _init_memory_tools()

    if log:
        log.phase("SOLVE", f"up to {MAX_TOOL_ROUNDS} rounds, {MAX_COMMANDS_PER_ROUND} cmds/round")

    # Build system prompt — memory tools section only included when use_memory is on
    memory_tools_section = ("\n" + _MEMORY_TOOLS_PROMPT) if (use_memory and _MEMORY_TOOLS_PROMPT) else ""
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        memory_tools_section=memory_tools_section,
        available_tools=", ".join(get_available_tools())
    )

    # --- Pre-fetch memory context so the LLM sees it on its first turn ---
    memory_context = ""
    if use_memory and _MEMORY_TOOLS:
        memory_query = f"{name} {category} {difficulty}"
        if log:
            log.info(f"Pre-fetching memory: {memory_query!r}")

        # 1. Semantic search for similar past challenges
        similar = call_tool_sync("memory_search", {
            "query": memory_query,
            "category": category.lower() if category and category != "Unknown" else "",
            "k": 3,
        })
        # 2. Check for known failures to avoid repeating mistakes
        failures = call_tool_sync("memory_search_failures", {
            "query": memory_query,
            "k": 2,
        })

        parts = []
        if similar and similar != "(no results)":
            parts.append(f"**Similar past challenges:**\n{similar}")
        if failures and failures != "(no results)":
            parts.append(f"**Known failures to avoid:**\n{failures}")

        if parts:
            memory_context = (
                "\n\n---\n**Memory context from past CTF challenges "
                "(use as background — do NOT copy flags from this):**\n\n"
                + "\n\n".join(parts)
            )
            if log:
                log.info("Memory context injected into prompt")
        else:
            if log:
                log.info("No relevant memory found")

    # Combine local memory_context with injected remote RAG context
    if injected_rag_context:
        memory_context += "\n\n---\n**Distributed RAG Context:**\n" + injected_rag_context

    # Build initial messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": CHALLENGE_PROMPT.format(
            name=name, category=category, points=points,
            difficulty=difficulty, description=description,
            hints=hints, code_blocks=code_blocks,
            server_urls=server_urls,
            file_listing=file_listing
        ) + memory_context},
    ]

    rounds_used = 0
    for round_num in range(1, max_rounds + 1):
        rounds_used = round_num

        # Strategy Shift Prompt: if we are near exhaustion, push the LLM to change approach
        near_limit_note = ""
        if round_num > (max_rounds * 0.7):
            near_limit_note = f"\n\n[SYSTEM] ATTENTION: You are in Round {round_num}/{max_rounds}. If your current approach is not yielding results, try a DIFFERENT strategy now!"
        elif round_num > 1:
            near_limit_note = f"\n\n[SYSTEM] Current progress: Round {round_num}/{max_rounds}."

        # --- Context window management: token-estimation based trimming ---
        # Estimate tokens at ~4 chars/token, keep under 60% of context window
        max_input_tokens = int(context_window * 0.60)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = total_chars // 4
        while estimated_tokens > max_input_tokens and len(messages) > 4:
            # Remove oldest assistant+user pair after the initial system+user
            messages.pop(2)
            if len(messages) > 2:
                messages.pop(2)
            total_chars = sum(len(m.get("content", "")) for m in messages)
            estimated_tokens = total_chars // 4
        # Fallback: also cap at a fixed message count
        MAX_CONTEXT_MESSAGES = 14
        if len(messages) > MAX_CONTEXT_MESSAGES:
            messages = messages[:2] + messages[-(MAX_CONTEXT_MESSAGES - 2):]
        round_record = {"round": round_num, "thinking": "", "action": None,
                        "commands": [], "results_summary": "", "flag": None}

        if log:
            log.round_start(round_num, max_rounds)

        try:
            _create_kwargs = dict(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=effective_max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            # llm_seed makes generation reproducible per-call — required for the
            # causal-credit re-rolls (vary the seed to sample distinct trajectories).
            if llm_seed is not None:
                _create_kwargs["seed"] = int(llm_seed)
            response = client.chat.completions.create(**_create_kwargs)
            raw_content = response.choices[0].message.content
            _usage = getattr(response, "usage", None)
            if _usage is not None:
                total_tokens += int(getattr(_usage, "total_tokens", 0) or 0)

            # Extract thinking BEFORE stripping it
            thinking, content = _extract_thinking(raw_content)
            round_record["thinking"] = thinking

            if log and thinking:
                log.thinking(thinking)

        except Exception as e:
            if log:
                log.llm_error(str(e))
            else:
                print(f"      ✗ LLM error: {e}")
            round_record["action"] = "error"
            round_record["results_summary"] = str(e)
            trace.append(round_record)
            break

        # Try to parse structured JSON response
        parsed = _parse_llm_json(content)
        round_record["llm_response"] = content[:3000]  # store for tracing

        if parsed:
            action = parsed.get("action", "")
            round_record["action"] = action

            if action == "submit_flag":
                flag = parsed.get("flag", "")
                round_record["flag"] = flag

                # Gate 1: basic format and placeholder check
                if not flag or _is_likely_placeholder(flag) or not FLAG_PATTERN.match(flag):
                    if log:
                        log.llm_invalid_flag(flag)
                    else:
                        print(f"      ⚠  Invalid/placeholder flag from LLM: {flag}")
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content":
                        "That flag appears to be invalid or a placeholder. "
                        "Please run commands to find the ACTUAL flag in the output. "
                        "Respond with JSON only."})
                    trace.append(round_record)
                    continue

                # Gate 2: anti-hallucination — check if flag was seen in output
                # Relaxed: accept exact match, substring match, or inner-content match
                flag_verified = False
                if flag in flags_seen_in_output:
                    flag_verified = True
                else:
                    # Check if the flag's inner content appears in any seen flag
                    flag_inner = flag[len('picoCTF{'):-1] if flag.startswith('picoCTF{') else flag
                    for seen in flags_seen_in_output:
                        seen_inner = seen[len('picoCTF{'):-1] if seen.startswith('picoCTF{') else seen
                        if flag_inner and (flag_inner in seen_inner or seen_inner in flag_inner):
                            flag_verified = True
                            break
                        if flag in seen or seen in flag:
                            flag_verified = True
                            break

                # Soft verification: if the LLM keeps submitting the same flag
                # after being told to verify it, it probably computed it correctly
                # (e.g. assembled from FLAGPART entries, or computed via crypto)
                if not flag_verified:
                    rejected_flag_attempts[flag] = rejected_flag_attempts.get(flag, 0) + 1
                    if rejected_flag_attempts[flag] >= 2:
                        flag_verified = True
                        if log:
                            log.info(f"Soft-verified flag after {rejected_flag_attempts[flag]} attempts: {flag}")

                if flag_verified:
                    if log:
                        log.llm_action("submit_flag")
                        log.flag_found(flag, source="LLM submit (verified)")
                    else:
                        print(f"      🚩  Flag found (verified): {flag}")
                    trace.append(round_record)
                    return {"flag": flag, "trace": trace,
                            "rounds_used": rounds_used,
                            "gave_up_reason": None,
                            "total_commands": total_commands, "tokens": total_tokens}
                else:
                    # Flag not seen in any prior command output — likely hallucinated
                    if log:
                        log.warn(f"Flag not found in any command output (possible hallucination): {flag}")
                    else:
                        print(f"      ⚠  Flag not in command output (hallucination?): {flag}")
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content":
                        f"The flag '{flag}' was NOT found in any of your previous command outputs. "
                        "You may be hallucinating. Run a command that will print the actual flag "
                        "(e.g. `strings file | grep picoCTF` or `echo encoded | base64 -d`). "
                        "ONLY submit a flag you see printed in command output. "
                        "Respond with JSON only."})
                    trace.append(round_record)
                    continue

            elif action == "run_commands":
                commands = parsed.get("commands", [])
                if not commands:
                    if log:
                        log.warn("LLM requested run_commands but provided no commands")
                    else:
                        print("      ⚠  No commands provided")
                    trace.append(round_record)
                    break

                # Limit commands per round
                commands = commands[:MAX_COMMANDS_PER_ROUND]
                round_record["commands"] = commands
                total_commands += len(commands)

                # --- Loop detection: break if LLM keeps running the same commands ---
                cmd_key = ";".join(sorted(commands))
                recent_commands.append(cmd_key)
                if len(recent_commands) >= 3 and recent_commands[-1] == recent_commands[-2] == recent_commands[-3]:
                    if log:
                        log.warn("Loop detected — LLM repeated the same commands 3 times. Breaking.")
                    else:
                        print("      ⚠  Loop detected — same commands repeated 3x")
                    gave_up_reason = "loop_detected"
                    trace.append(round_record)
                    break

                if log:
                    log.llm_action("run_commands", detail=f"{len(commands)} command(s)")

                results = run_commands(commands, working_dir, log=log)

                # Build a compact results summary for trace
                summaries = []
                for r in results:
                    out = (r.get("stdout", "") or "").strip()
                    if not out:
                        out = (r.get("stderr", "") or "").strip()
                    if not out and r.get("error"):
                        out = r.get("error")
                    
                    if out:
                        out_first_line = out[:1000].replace('\n', '\\n')
                    else:
                        out_first_line = "(empty)"
                        
                    summaries.append(f"{r['command']}: {out_first_line}")
                round_record["results_summary"] = "; ".join(summaries)

                # Check if any command output contains a flag (including encoded)
                for r in results:
                    output_text = (r.get("stdout", "") or "") + (r.get("stderr", "") or "")
                    
                    # Track all raw flag strings seen in output
                    for m in FLAG_PATTERN.finditer(output_text):
                        flags_seen_in_output.add(m.group(0))

                    found_flags = extract_encoded_flags(output_text)
                    for flag in found_flags:
                        if flag not in PLACEHOLDER_FLAGS:
                            flags_seen_in_output.add(flag)
                            if log:
                                log.flag_found(flag, source="command output")
                            else:
                                print(f"      🚩  Flag found in command output: {flag}")
                            
                            # Auto-return: flag came directly from command output,
                            # no need to wait for the LLM to (mis)type it.
                            round_record["flag"] = flag
                            trace.append(round_record)
                            return {"flag": flag, "trace": trace,
                                    "rounds_used": rounds_used,
                                    "gave_up_reason": None,
                                    "total_commands": total_commands, "tokens": total_tokens}

                # Feed results back to LLM
                formatted = format_results_for_llm(results)
                messages.append({"role": "assistant", "content": content})
                
                # Build prompt for the LLM
                if round_num == 1:
                    prompt = "Results:\n\n" + formatted + near_limit_note
                else:
                    prompt = f"Round {round_num}/{max_rounds} results:\n\n" + formatted + near_limit_note
                
                messages.append({"role": "user", "content": prompt})
                trace.append(round_record)
                continue

            elif action == "query_memory":
                tool_name = parsed.get("tool", "")
                args = parsed.get("args", {})

                known_tools = {t["name"] for t in _MEMORY_TOOLS}
                if not tool_name or tool_name not in known_tools:
                    errmsg = f"Unknown memory tool: {tool_name!r}. Available: {sorted(known_tools)}"
                    if log:
                        log.warn(errmsg)
                    else:
                        print(f"      \u26a0  {errmsg}")
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content":
                        f"{errmsg}. Respond with JSON only."})
                    trace.append(round_record)
                    continue

                if log:
                    log.llm_action("query_memory", detail=f"{tool_name}({args})")
                else:
                    print(f"      \U0001f9e0  Memory query: {tool_name}({args})")

                memory_result = call_tool_sync(tool_name, args)
                round_record["results_summary"] = f"memory:{tool_name} -> {memory_result[:120]}"

                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content":
                    f"**Memory result from `{tool_name}`:**\n\n{memory_result}\n\n"
                    "Use this context to decide your next step. Respond with JSON only."})
                trace.append(round_record)
                continue

            elif action == "give_up":
                reason = parsed.get("reason", "unknown")
                gave_up_reason = reason
                if log:
                    log.llm_gave_up(reason)
                else:
                    print(f"      \u23ed  LLM gave up: {reason}")
                trace.append(round_record)
                break

        # JSON parse failed — try extracting flag from raw text
        round_record["action"] = "parse_error"
        round_record["llm_response"] = content[:3000]
        flag = extract_flag(content)
        if flag:
            # Use the same relaxed matching as the structured path
            flag_verified = flag in flags_seen_in_output
            if not flag_verified:
                flag_inner = flag[len('picoCTF{'):-1] if flag.startswith('picoCTF{') else flag
                for seen in flags_seen_in_output:
                    seen_inner = seen[len('picoCTF{'):-1] if seen.startswith('picoCTF{') else seen
                    if flag_inner and (flag_inner in seen_inner or seen_inner in flag_inner):
                        flag_verified = True
                        break
                    if flag in seen or seen in flag:
                        flag_verified = True
                        break
            # Soft verification for raw text path too
            if not flag_verified:
                rejected_flag_attempts[flag] = rejected_flag_attempts.get(flag, 0) + 1
                if rejected_flag_attempts[flag] >= 2:
                    flag_verified = True
                    if log:
                        log.info(f"Soft-verified raw flag after {rejected_flag_attempts[flag]} attempts: {flag}")
            if flag_verified:
                if log:
                    log.flag_found(flag, source="raw text (verified)")
                else:
                    print(f"      🚩  Flag found (raw, verified): {flag}")
                round_record["flag"] = flag
                trace.append(round_record)
                return {"flag": flag, "trace": trace,
                        "rounds_used": rounds_used,
                        "gave_up_reason": None,
                        "total_commands": total_commands, "tokens": total_tokens}
            else:
                if log:
                    log.warn(f"Flag in raw text but not in command output (skipping): {flag}")

        # Nudge the LLM to retry in JSON format
        if log:
            log.llm_parse_error("nudging to retry")
        else:
            print("      ⚠  Could not parse LLM response as JSON, nudging...")
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content":
            "Your response was not valid JSON. You MUST respond with ONLY a JSON object. "
            "Either: {\"action\": \"run_commands\", \"commands\": [\"...\"]}, "
            "{\"action\": \"submit_flag\", \"flag\": \"picoCTF{...}\"}, or "
            "{\"action\": \"give_up\", \"reason\": \"...\"}. "
            "Respond with JSON only."})
        trace.append(round_record)
        continue

    # All rounds exhausted — give up cleanly (no fallback to avoid hallucination)
    if log:
        log.warn("All rounds exhausted — no verified flag found")
    else:
        print("      ⏭  All rounds exhausted — no verified flag found")
    return {"flag": None, "trace": trace,
            "rounds_used": rounds_used,
            "gave_up_reason": gave_up_reason or "rounds_exhausted",
            "total_commands": total_commands, "tokens": total_tokens}


def _fallback_solve(challenge: dict,
                    log: ChallengeLogger | None = None) -> Optional[str]:
    """Single-shot fallback solver (no tool use)."""
    name = challenge.get("name", "Unknown")
    category = challenge.get("category", "Unknown")
    points = challenge.get("points", "?")
    difficulty = challenge.get("difficulty", "?")
    description = challenge.get("description", "No description.")
    hints = _format_hints(challenge.get("hints", []))
    code_blocks = _format_code_blocks(challenge.get("code_blocks", []))
    file_contents = _read_file_contents(challenge.get("local_files", []))

    prompt = FALLBACK_PROMPT.format(
        name=name, category=category, points=points,
        difficulty=difficulty, description=description,
        hints=hints, code_blocks=code_blocks,
        file_contents=file_contents,
    )

    if log:
        log.phase("FALLBACK", "single-shot, no tools")
    else:
        print(f"      🤖  Fallback attempt...")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        raw_content = response.choices[0].message.content

        # Extract and display thinking
        thinking, content = _extract_thinking(raw_content)
        if log and thinking:
            log.thinking(thinking)

        flag = extract_flag(content)
        if flag:
            if log:
                log.flag_found(flag, source="fallback")
            else:
                print(f"      🚩  Flag found (fallback): {flag}")
            return flag

        if log:
            log.no_flag()
        else:
            print("      ✗ No flag found in fallback")
    except Exception as e:
        if log:
            log.llm_error(f"Fallback: {e}")
        else:
            print(f"      ✗ Fallback LLM error: {e}")

    return None
