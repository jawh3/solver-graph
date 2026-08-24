"""
solver/citation.py — action-grounded citation matcher + reward-hacking guard (spec §9.3, §11).

THE recovery-masking lever (C-4). A sampled lesson node `v` is `cited` iff its
*invocation signature* appears in the agent's **executed actions** — the shell
commands actually run — NOT in the model's self-narration. Self-reported
attribution is biased and unreliable (MemoryR2 2605.21768; WhoAndWhen
2505.00212); the entire point is to deny credit to a lesson that sat in context
during a success but was never used (recovery masking).

Matching rules by node type (§3.1):
  - Tool:      `invocation_signature` (regex or keyword list) vs executed commands.
  - Technique: technique keywords (canonical_name tokens + declared keywords)
               vs executed commands.
  - Artifact:  produced-artifact filename appears in a command.
Deterministic matcher first; an LLM-judge fallback is allowed only as
`confidence="llm"` and flagged low-confidence (we do not implement the LLM judge
here — it is a coordinated-GPU add-on).

Reward-hacking guard (§11; LLMsGamingVerifiers 2604.15149, RewardHackingBench
2605.02964): flag a *non-exploit capture* — the flag was obtained by trivially
reading a file (`cat flag.txt`) for a category that is supposed to require
exploitation. This is logged as a distinct outcome, not silently counted as a
solve; it is the residual contamination mode in a verifiable-reward domain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from knowledge_graph.models import Node, NodeType

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-]+")


def _executed_commands(trace: List[Dict]) -> List[str]:
    """Flatten all executed shell commands from a solver trace."""
    cmds: List[str] = []
    for rec in trace or []:
        for c in rec.get("commands", []) or []:
            cmds.append(str(c))
    return cmds


def _node_keywords(node: Node) -> List[str]:
    """Keywords that ground a citation for this node."""
    a = node.attributes
    kws: List[str] = []
    sig = a.get("invocation_signature")
    if sig:
        if isinstance(sig, list):
            kws.extend(str(s) for s in sig)
        else:
            kws.append(str(sig))
    name = a.get("canonical_name")
    if name:
        kws.append(str(name))
    for kw in a.get("keywords", []) or []:
        kws.append(str(kw))
    # Tool nodes: the tool name itself is the strongest signal
    if node.type is NodeType.TOOL and name:
        kws.append(str(name))
    # de-dupe, drop empties / too-generic 1-char tokens
    seen = set()
    out = []
    for k in kws:
        k = k.strip()
        if len(k) >= 2 and k.lower() not in seen:
            seen.add(k.lower())
            out.append(k)
    return out


@dataclass
class CitationMatch:
    node_id: str
    cited: bool
    match_evidence: str = ""
    confidence: str = "deterministic"
    sampled: bool = True


def _matches(keyword: str, command: str) -> Optional[str]:
    """Return the matched evidence if `keyword` grounds against `command`.

    Treats a keyword as a regex if it looks like one (contains regex metachars),
    else as a case-insensitive whole-token/substring match. Returns the matched
    text for `match_evidence`.
    """
    kw = keyword.strip()
    if not kw:
        return None
    # regex signature?
    if any(ch in kw for ch in r"\^$.|?*+()[]{}"):
        try:
            m = re.search(kw, command, flags=re.IGNORECASE)
            if m:
                return m.group(0)
        except re.error:
            pass
    # token / substring match (case-insensitive)
    if kw.lower() in command.lower():
        return kw
    return None


def match_citations(
    sampled_nodes: List[Node],
    trace: List[Dict],
) -> List[CitationMatch]:
    """Deterministically mark which sampled nodes were action-grounded-cited."""
    commands = _executed_commands(trace)
    blob = "\n".join(commands)
    results: List[CitationMatch] = []
    for node in sampled_nodes:
        evidence = ""
        for kw in _node_keywords(node):
            ev = _matches(kw, blob)
            if ev:
                evidence = f"{kw} ~ {ev}"
                break
        results.append(CitationMatch(
            node_id=node.id, cited=bool(evidence), match_evidence=evidence,
        ))
    return results


# ---------------------------------------------------------------------------
# Reward-hacking guard (§11)
# ---------------------------------------------------------------------------

# Categories that are SUPPOSED to require exploitation (a trivial file read of
# the flag indicates a non-exploit capture / mis-provisioned challenge).
_EXPLOIT_CATEGORIES = {"pwn", "binary exploitation", "web", "web exploitation",
                       "crypto", "cryptography", "reverse engineering", "rev"}

# A trivial flag read: cat/strings/grep/head/tail/less of a local flag file.
_TRIVIAL_READ_RE = re.compile(
    r"\b(cat|strings|grep|head|tail|less|more|xxd|od|nl)\b[^|;&]*\bflag",
    re.IGNORECASE,
)


def detect_non_exploit_capture(challenge: Dict, flag: Optional[str],
                               trace: List[Dict]) -> bool:
    """Heuristic: did a flag come from a trivial file read in an exploit category?

    Logged as `non_exploit_capture` on the RunRecord (§12) so the eval harness
    can report solve rate WITH and WITHOUT these, exposing checker-abuse / mis-
    provisioned challenges instead of silently inflating the headline (§14.3).
    """
    if not flag:
        return False
    category = str(challenge.get("category", "")).strip().lower()
    if category not in _EXPLOIT_CATEGORIES:
        return False
    # the flag-producing command, if any, is the last command that printed it.
    for rec in trace or []:
        for c in rec.get("commands", []) or []:
            if _TRIVIAL_READ_RE.search(str(c)):
                return True
    return False


def citation_quality(matches: List[CitationMatch], key: Dict[str, bool]) -> Dict[str, float]:
    """Precision/recall of the matcher vs a hand-labeled key (C3 acceptance)."""
    tp = fp = fn = 0
    for m in matches:
        truth = key.get(m.node_id, False)
        if m.cited and truth:
            tp += 1
        elif m.cited and not truth:
            fp += 1
        elif not m.cited and truth:
            fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}
