"""Regression test: a sandboxed tool command cannot write a giant file (decompression
bomb / runaway write). On 2026-06-02 a solver decode wrote a 681 GB file and filled the
shared drive; RLIMIT_FSIZE now caps any single file at MAX_FILE_BYTES."""
import os
import tempfile

from solver.tool_executor import run_command, MAX_FILE_BYTES


def test_runaway_write_is_capped():
    d = tempfile.mkdtemp()
    run_command("cat /dev/zero > bomb.bin", d)
    p = os.path.join(d, "bomb.bin")
    sz = os.path.getsize(p) if os.path.exists(p) else 0
    # killed by SIGXFSZ near the cap — must NOT be gigabytes
    assert sz <= MAX_FILE_BYTES + 4 * 1024 * 1024, f"file not capped: {sz} bytes"
