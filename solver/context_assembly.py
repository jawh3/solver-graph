"""
solver/context_assembly.py - render M sampled nodes into solver context (spec §7 step 7).

Static injection (the v2 default - NOT the MCP dynamic path; spec §7-§8). The
rendered block is passed to `solve_challenge(injected_rag_context=...)`. Two
research-critical rendering rules:

  1. **Applicability conditions are shown** (§3.4): a lesson's preconditions
     (e.g. "no PIE", "format string in printf") are rendered so the solver can
     self-filter - a lesson applied out of its conditions is a false-lesson
     pathway (interference).
  2. **Provisional nodes are marked low-confidence** (§9.4): a write-back node
     that has not passed the promotion gate must never appear as "trusted". The
     tier flag is surfaced in-context and its influence is weight-capped upstream
     (bandit.score). This is the birth-defect lever at the prompt layer.

The rendering is deterministic and stable (sorted by tier then id) so the same M
set yields the same context string - important for paired/seeded eval (§14.4).
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from knowledge_graph.models import Node, NodeType


def _order_key(n: Node, order_seed: Optional[int]):
    """Within-tier order key. LABEL-NEUTRAL by construction: hashes the rendered TEXT, NOT node.id.
    (node.id encodes the label via its 'true::' / 'poison::' prefix, and 'poison::' < 'true::'
    lexically, so the old id-sort rendered the WRONG recipe FIRST in 100% of {helpful,poison}
    contexts - a primacy confound that aliased the constructed 'interference' effect with position.
    See RIGOR_LEDGER RECKONING 2026-06-25.) order_seed!=None gives a deterministic per-run shuffle
    (for an explicit lesson-ORDER-as-factor experiment)."""
    tier_rank = 0 if n.attributes.get("tier", "trusted") == "trusted" else 1
    basis = n.attributes.get("text", "") or n.id
    if order_seed is not None:
        basis = f"{order_seed}:{basis}"
    return (tier_rank, hashlib.sha1(basis.encode()).hexdigest())


def _fmt_applicability(app: Dict) -> str:
    if not app:
        return ""
    bits = []
    if app.get("categories"):
        bits.append("categories=" + ",".join(app["categories"]))
    if app.get("preconditions"):
        bits.append("preconditions=" + "; ".join(app["preconditions"]))
    if app.get("artifacts_required"):
        bits.append("requires=" + ",".join(app["artifacts_required"]))
    return " | ".join(bits)


def render_node(node: Node) -> str:
    """Render one lesson node as a compact, citation-friendly line."""
    a = node.attributes
    tier = a.get("tier", "trusted")
    name = a.get("canonical_name") or a.get("title") or node.id
    text = a.get("text", "")
    parts = [f"[{node.type.value}] {name}"]
    if tier == "provisional":
        parts.append("(PROVISIONAL - unverified, treat as a weak hint)")
    if node.type is NodeType.TOOL and a.get("invocation_signature"):
        parts.append(f"invocation: {a['invocation_signature']}")
    if text:
        parts.append(f"- {text}")
    app = _fmt_applicability(a.get("applicability", {}))
    if app:
        parts.append(f"[applies when: {app}]")
    return " ".join(parts)


def assemble_context(nodes: List[Node], header: bool = True, order_seed: Optional[int] = None) -> str:
    """Render the M sampled nodes into the injected RAG context block.

    Ordering: trusted first, then provisional; within a tier, by a LABEL-NEUTRAL content hash
    (NOT node.id - see _order_key; the id-sort caused a primacy confound). Deterministic given the
    same nodes (+ order_seed), so paired/seeded eval is reproducible; pass order_seed for a per-run
    randomized lesson order (the fix for the primacy confound in future runs / an order-as-factor arm).
    """
    if not nodes:
        return ""
    ordered = sorted(nodes, key=lambda n: _order_key(n, order_seed))
    lines = [render_node(n) for n in ordered]
    body = "\n".join(f"- {ln}" for ln in lines)
    if not header:
        return body
    return (
        "Retrieved lessons from past CTF challenges (use the relevant ones; "
        "ignore lessons whose 'applies when' conditions do not match this "
        "challenge; PROVISIONAL lessons are unproven hints):\n" + body
    )
