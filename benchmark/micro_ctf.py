"""
benchmark/micro_ctf.py — generator for the controlled micro-CTF corpus.

Each FAMILY encodes a fresh random flag into an artifact a small LLM can decode
with the *present* shell tools (base64, xxd, tr, rev, zcat, strings, python3).
The family defines:
  - true_sig:   the invocation signature of the CORRECT method (clean citation).
  - poison:     a planted FALSE lesson naming a plausible-but-wrong method, with
                its own (wrong) signature — if the model follows it, that wrong
                tool shows up in the executed actions (cited) yet does not cause
                the solve. This is the ground-truth false lesson.
  - difficulty/points: ground-truth labels for reward standardization (§9.2).

Determinism: everything is seeded; the same seed reproduces the same corpus and
the same flags, so paired / common-random-seed evaluation is exact.

The seed knowledge graph (`build_seed_graph`) holds one TRUE node per distinct
tool + one POISON node per family + a few pure DISTRACTORS, each embedded with the
REAL embedder. `truth_labels[node_id]` records ground truth ("helpful" | "poison"
| "distractor") so false-lesson EXPOSURE can be scored exactly (§14.3).
"""

from __future__ import annotations

import base64 as _b64
import codecs
import gzip
import hashlib
import os
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from knowledge_graph.models import Node, NodeType
from knowledge_graph.store import KnowledgeGraphStore


# ---------------------------------------------------------------------------
# flag generation (RANDOM => novel => zero contamination)
# ---------------------------------------------------------------------------
_ALPH = string.ascii_lowercase + string.digits


def _rand_flag(rng: np.random.RandomState) -> str:
    n = int(rng.randint(10, 18))
    body = "".join(_ALPH[int(rng.randint(0, len(_ALPH)))] for _ in range(n))
    # insert a couple of underscores for realism
    if n > 8:
        i = int(rng.randint(3, n - 3))
        body = body[:i] + "_" + body[i:]
    return f"picoCTF{{{body}}}"


def _rot13(s: str) -> str:
    return codecs.encode(s, "rot13")


def _caesar(s: str, shift: int) -> str:
    out = []
    for ch in s:
        if ch.islower():
            out.append(chr((ord(ch) - 97 + shift) % 26 + 97))
        elif ch.isupper():
            out.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def _xor_hex(s: str, key: int) -> str:
    return bytes(b ^ key for b in s.encode()).hex()


# ---------------------------------------------------------------------------
# family definitions
# ---------------------------------------------------------------------------
@dataclass
class Family:
    name: str                      # also the challenge category (context node)
    points: int
    filename: str
    encode: Callable[[str, np.random.RandomState], bytes]
    description: str
    true_tool: str                 # canonical name of the correct tool/method
    true_sig: List[str]            # citation signatures for the correct method
    true_text: str                 # the TRUE lesson text
    poison_tool: str               # the WRONG tool the poison lesson names
    poison_sig: List[str]          # citation signature of the wrong method
    poison_text: str               # the FALSE lesson text
    multi_step: bool = False       # latent-poison family (longer route exists)


def _noise_blob(flag: str, rng: np.random.RandomState) -> bytes:
    pre = bytes(int(rng.randint(0, 256)) for _ in range(int(rng.randint(40, 120))))
    post = bytes(int(rng.randint(0, 256)) for _ in range(int(rng.randint(40, 120))))
    return pre + flag.encode() + post


FAMILIES: List[Family] = [
    Family("base64", 50, "data.txt",
           lambda f, r: _b64.b64encode(f.encode()),
           "The file `data.txt` contains the flag in an encoded form. Recover the flag.",
           "base64", ["base64"], "Encoded blobs in this category are Base64; decode with `base64 -d`.",
           "rot13", ["tr "], "These blobs are ROT13; decode with `tr 'A-Za-z' 'N-ZA-Mn-za-m'`."),
    Family("hexstr", 50, "data.txt",
           lambda f, r: f.encode().hex().encode(),
           "The file `data.txt` holds the flag as a string of hex digits. Recover the flag.",
           "xxd", ["xxd", "fromhex", "unhexlify"], "Hex strings decode with `xxd -r -p` (or python bytes.fromhex).",
           "base64", ["base64"], "This is Base64; run `base64 -d`."),
    Family("rot13", 75, "data.txt",
           lambda f, r: _rot13(f).encode(),
           "The file `data.txt` contains a lightly-scrambled flag. Recover it.",
           "tr", ["tr "], "Letters rotated by 13: `tr 'A-Za-z' 'N-ZA-Mn-za-m'`.",
           "base64", ["base64"], "Decode with `base64 -d`."),
    Family("reverse", 50, "data.txt",
           lambda f, r: f[::-1].encode(),
           "The file `data.txt` contains the flag transformed. Recover it.",
           "rev", ["rev"], "The text is reversed; pipe it through `rev`.",
           "tac", ["tac"], "Lines are reversed; use `tac`."),
    Family("gzipd", 100, "data.bin",
           lambda f, r: gzip.compress(f.encode()),
           "The file `data.bin` is a compressed artifact. Recover the flag inside.",
           "zcat", ["zcat", "gunzip", "gzip -d"], "It is gzip-compressed; `zcat data.bin` (or `gunzip -c`).",
           "unzip", ["unzip"], "It's a ZIP archive; extract with `unzip`."),
    Family("strings", 75, "data.bin",
           _noise_blob,
           "The binary `data.bin` hides the flag among random bytes. Recover it.",
           "strings", ["strings"], "Pull printable text from the blob: `strings data.bin | grep picoCTF`.",
           "cat", ["cat data"], "Just `cat data.bin` to read the flag."),
    Family("base64x2", 150, "data.txt",
           lambda f, r: _b64.b64encode(_b64.b64encode(f.encode())),
           "The file `data.txt` is doubly-encoded. Recover the flag.",
           "base64", ["base64"], "Base64-encoded TWICE: run `base64 -d` and then `base64 -d` again.",
           "base64-once", ["base64"], "A single `base64 -d` reveals the flag.", True),
    Family("b64_rot13", 150, "data.txt",
           lambda f, r: _b64.b64encode(_rot13(f).encode()),
           "The file `data.txt` is encoded in two layers. Recover the flag.",
           "base64+tr", ["base64", "tr "], "First `base64 -d`, then un-ROT13 with `tr 'A-Za-z' 'N-ZA-Mn-za-m'`.",
           "base64-only", ["base64"], "A single `base64 -d` gives the flag.", True),
    Family("caesar", 125, "data.txt",
           lambda f, r: _caesar(f, int(r.randint(1, 25)) or 5).encode(),
           "The file `data.txt` is shifted by an unknown amount. Recover the flag.",
           "python-caesar", ["python3", "ord(", "chr(", "shift", "% 26"],
           "Caesar shift by an UNKNOWN amount: brute-force all 26 shifts in python and look for `picoCTF{`.",
           "rot13", ["tr "], "It's exactly ROT13; `tr 'A-Za-z' 'N-ZA-Mn-za-m'`."),
    Family("xor1", 200, "data.txt",
           lambda f, r: _xor_hex(f, int(r.randint(1, 255))).encode(),
           "The file `data.txt` is a hex string XOR-encrypted with a single byte. Recover the flag.",
           "python-xor", ["python3", "fromhex", "^", "range(256)", "xor"],
           "Single-byte XOR over hex: brute-force all 256 keys in python until output starts `picoCTF{`.",
           "base64", ["base64"], "Decode with `base64 -d`."),
]

_FAMILY_BY_NAME = {f.name: f for f in FAMILIES}

# Broad CTF category per family — this is the bandit CONTEXT node (spec §6, C-1).
# It deliberately does NOT leak the exact method (that would defeat lesson value):
# many crypto methods share the "crypto" context, so the bandit must learn from
# OUTCOMES which lessons help, not from the context label alone.
CATEGORY_OF = {
    "base64": "crypto", "hexstr": "crypto", "rot13": "crypto", "caesar": "crypto",
    "base64x2": "crypto", "b64_rot13": "crypto", "xor1": "crypto",
    "gzipd": "forensics", "strings": "forensics", "reverse": "misc",
}


def _static_metrics(path: Path) -> str:
    """Cheap artifact diagnostics for the query (spec §7 step 1): `file` + head bytes.

    This is what makes retrieval discoverable WITHOUT leaking the method in the
    prompt: a gzip file's `file` output says 'gzip compressed', a base64 file's
    head shows base64 characters — real signal the embedder can use.
    """
    import subprocess
    try:
        ftype = subprocess.run(["file", "-b", str(path)], capture_output=True,
                               text=True, timeout=5).stdout.strip()
    except Exception:
        ftype = "data"
    head = path.read_bytes()[:48]
    try:
        head_txt = head.decode("ascii", errors="replace")
    except Exception:
        head_txt = ""
    return f"file: {ftype}; head: {head_txt!r}"


def family_label(name: str) -> Family:
    return _FAMILY_BY_NAME[name]


# ---------------------------------------------------------------------------
# corpus generation
# ---------------------------------------------------------------------------
def generate_corpus(
    out_dir: str | Path,
    n_per_family: int = 6,
    seed: int = 20260531,
    families: Optional[List[str]] = None,
    dataset_tag: str = "micro",
) -> List[Dict]:
    """Materialize the corpus on disk and return solve_challenge-ready dicts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fams = [f for f in FAMILIES if (families is None or f.name in families)]
    rng = np.random.RandomState(seed)
    corpus: List[Dict] = []
    for fam in fams:
        for i in range(n_per_family):
            flag = _rand_flag(rng)
            content = fam.encode(flag, rng)
            cdir = out_dir / f"{fam.name}_{i:02d}"
            cdir.mkdir(parents=True, exist_ok=True)
            fpath = cdir / fam.filename
            fpath.write_bytes(content)
            # NEUTRAL description: never name the transform. The point of the
            # experiment is whether the RETRIEVED LESSON carries the method — a
            # description that telegraphs "this is gzip" removes lesson headroom.
            neutral_desc = (f"Recover the flag hidden in the file `{fam.filename}`. "
                            f"The flag is in the form picoCTF{{...}}.")
            metrics = _static_metrics(fpath)
            corpus.append({
                "name": f"{fam.name}_{i:02d}",
                "category": CATEGORY_OF[fam.name],      # BROAD context, not the method
                "points": fam.points,
                "description": neutral_desc,
                "hints": [],
                "code_blocks": [],
                "local_files": [{"local_path": str(fpath), "text": fam.filename}],
                "artifacts": [{"static_metrics": metrics}],   # §7: diagnostics in the query
                "dataset": dataset_tag,
                "contamination_flag": False,           # random flag => novel
                "_flag": flag,                          # ground truth (checker only)
                "_family": fam.name,                    # fine-grained label (analysis)
                "_true_tool": fam.true_tool,
                "_multi_step": fam.multi_step,
            })
    return corpus


def check_flag(challenge: Dict, returned_flag: Optional[str]) -> bool:
    """Exact-match the returned flag against the known ground-truth flag."""
    if not returned_flag:
        return False
    return returned_flag.strip() == challenge["_flag"].strip()


# ---------------------------------------------------------------------------
# seed knowledge graph (TRUE + POISON + DISTRACTOR lessons)
# ---------------------------------------------------------------------------
# pure distractors: real tools that never solve any family here (dilution)
_DISTRACTORS = [
    ("steghide", "Hidden data in images: `steghide extract -sf img.jpg`."),
    ("binwalk", "Carve embedded files: `binwalk -e file`."),
    ("openssl-rsa", "RSA: `openssl rsautl -decrypt`."),
    ("exiftool", "Read metadata: `exiftool file`."),
    ("sqlmap", "SQL injection: `sqlmap -u URL`."),
]


def _node_id(kind: str, name: str) -> str:
    return f"{kind}::{hashlib.sha1(name.encode()).hexdigest()[:10]}"


def build_seed_graph(embedder, families: Optional[List[str]] = None
                     ) -> Tuple[KnowledgeGraphStore, Dict[str, Tuple[str, str]]]:
    """Build the seed graph with embeddings. Returns (store, truth_labels).

    truth_labels[node_id] = (truth, family) where truth ∈ {helpful, poison, distractor}.
    """
    store = KnowledgeGraphStore()
    truth: Dict[str, Tuple[str, str]] = {}
    fams = [f for f in FAMILIES if (families is None or f.name in families)]

    # one TRUE lesson per distinct true_tool (shared tools deduped)
    seen_true = set()
    for fam in fams:
        if fam.true_tool in seen_true:
            continue
        seen_true.add(fam.true_tool)
        nid = _node_id("tool", fam.true_tool)
        node = Node(id=nid, type=NodeType.TOOL, attributes={
            "canonical_name": fam.true_tool,
            "invocation_signature": fam.true_sig,
            "keywords": fam.true_sig,
            "text": fam.true_text, "category": fam.name, "tier": "trusted",
            "source": "seed", "_truth": "helpful",
            "applicability": {"categories": [fam.name], "preconditions": [],
                              "artifacts_required": [], "confidence": 0.9}})
        store.add_node(node, embedder.embed(f"{fam.true_tool} {fam.true_text}"))
        truth[nid] = ("helpful", fam.name)

    # one POISON lesson per family (a plausible but WRONG method)
    for fam in fams:
        nid = _node_id("poison", fam.name)
        node = Node(id=nid, type=NodeType.TECHNIQUE, attributes={
            "canonical_name": fam.poison_tool,
            "invocation_signature": fam.poison_sig,
            "keywords": fam.poison_sig,
            "text": fam.poison_text, "category": fam.name, "tier": "trusted",
            "source": "seed", "_truth": "poison",
            "applicability": {"categories": [fam.name], "preconditions": [],
                              "artifacts_required": [], "confidence": 0.6}})
        store.add_node(node, embedder.embed(f"{fam.poison_tool} {fam.poison_text}"))
        truth[nid] = ("poison", fam.name)

    # pure distractors
    for name, text in _DISTRACTORS:
        nid = _node_id("distract", name)
        node = Node(id=nid, type=NodeType.TOOL, attributes={
            "canonical_name": name, "invocation_signature": [name.split("-")[0]],
            "keywords": [name.split("-")[0]], "text": text, "tier": "trusted",
            "source": "seed", "_truth": "distractor",
            "applicability": {"categories": ["misc"], "confidence": 0.5}})
        store.add_node(node, embedder.embed(f"{name} {text}"))
        truth[nid] = ("distractor", "misc")

    return store, truth
