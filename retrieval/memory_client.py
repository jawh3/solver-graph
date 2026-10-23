#!/usr/bin/env python3
"""
Memory MCP Client - connects to the custom memory server over SSE.

Provides:
  - get_tools()              List retrieval tools available on the server.
  - call_tool_sync()         Call any memory tool by name with arbitrary args.
  - record_attempt_sync()    Record a solve attempt (lifelong learning).
  - format_tools_for_prompt() Render tool descriptions for injection into the LLM prompt.
"""

import os
import json
import asyncio
import logging
import concurrent.futures
from typing import Any, Optional

log = logging.getLogger(__name__)

MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8006/sse")

# Tools the LLM is allowed to call (read-only retrieval tools only).
# We exclude write/management tools: memory_record_attempt, memory_graph_stats.
RETRIEVAL_TOOLS = {
    "memory_search",
    "memory_by_technique",
    "memory_by_vulnerability",
    "memory_by_category",
    "memory_search_failures",
}

# ---------------------------------------------------------------------------
# Internal async core
# ---------------------------------------------------------------------------

async def _get_tools() -> list[dict]:
    """Return a list of tool dicts from the MCP server."""
    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
    except ImportError:
        return []

    try:
        async with sse_client(MEMORY_SERVER_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                tools = []
                for t in result.tools:
                    tools.append({
                        "name": t.name,
                        "description": t.description or "",
                        "input_schema": t.inputSchema or {},
                    })
                return tools
    except Exception as e:
        log.warning("Memory server unreachable (list_tools): %s", e)
        return []


async def _call_tool(tool_name: str, arguments: dict) -> str:
    """Call a single tool on the MCP server and return formatted text output."""
    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
    except ImportError:
        return "(mcp package not installed)"

    try:
        async with sse_client(MEMORY_SERVER_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                texts = [item.text for item in result.content if hasattr(item, "text") and item.text]
                return "\n\n".join(texts) if texts else "(no results)"
    except Exception as e:
        log.warning("Memory tool %r failed: %s", tool_name, e)
        return f"(memory tool error: {e})"


# ---------------------------------------------------------------------------
# Sync wrappers - safe to call from non-async contexts
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine from any context (async or sync)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an existing async loop (e.g. Playwright's loop in main.py)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=20)
        else:
            return loop.run_until_complete(coro)
    except Exception as e:
        log.warning("_run_async error: %s", e)
        return None


def get_tools() -> list[dict]:
    """Return all retrieval tools available on the memory server."""
    tools = _run_async(_get_tools()) or []
    return [t for t in tools if t["name"] in RETRIEVAL_TOOLS]


def call_tool_sync(tool_name: str, arguments: dict) -> str:
    """Call a memory tool synchronously. Returns formatted text output."""
    if tool_name not in RETRIEVAL_TOOLS:
        return f"(tool {tool_name!r} not in allowed retrieval tools)"
    result = _run_async(_call_tool(tool_name, arguments))
    return result if result is not None else "(memory server unreachable)"


def record_attempt_sync(
    challenge_id: str,
    success: bool,
    techniques_used: list[str] | None = None,
    tools_used: list[str] | None = None,
    key_insight: str | None = None,
    code_used: str | None = None,
    failure_reason: str | None = None,
    time_spent_seconds: int | None = None,
) -> None:
    """Record a solve attempt result back into memory (lifelong learning).

    This is a fire-and-forget call - failures are silently logged.
    """
    args: dict[str, Any] = {
        "challenge_id": challenge_id,
        "success": success,
    }
    if techniques_used is not None:
        args["techniques_used"] = techniques_used
    if tools_used is not None:
        args["tools_used"] = tools_used
    if key_insight:
        args["key_insight"] = key_insight
    if code_used:
        args["code_used"] = code_used
    if failure_reason:
        args["failure_reason"] = failure_reason
    if time_spent_seconds is not None:
        args["time_spent_seconds"] = time_spent_seconds

    try:
        _run_async(_call_tool("memory_record_attempt", args))
        log.debug("Recorded attempt for %r (success=%s)", challenge_id, success)
    except Exception as e:
        log.warning("Failed to record attempt for %r: %s", challenge_id, e)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_tools_for_prompt(tools: list[dict]) -> str:
    """Render memory tool descriptions into a string for the LLM system prompt."""
    if not tools:
        return ""

    lines = [
        "\n**Memory Tools** - query your knowledge base of past CTF challenges:",
        "To call a memory tool, respond with:",
        '{"action": "query_memory", "tool": "TOOL_NAME", "args": {...}}',
        "",
        "Available memory tools:",
    ]
    for t in tools:
        lines.append(f"- **{t['name']}**: {t['description'].splitlines()[0]}")
        props = t.get("input_schema", {}).get("properties", {})
        if props:
            param_str = ", ".join(
                f"`{p}`" + (f" (optional)" if p not in t.get("input_schema", {}).get("required", []) else "")
                for p in props
            )
            lines.append(f"  Params: {param_str}")
    lines.append(
        "\nCall memory tools early - they give you techniques, tools, and key insights from similar past challenges."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    tools = get_tools()
    print(f"=== {len(tools)} retrieval tools available on {MEMORY_SERVER_URL} ===\n")
    for t in tools:
        print(f"  {t['name']}: {t['description'].splitlines()[0]}")

    print()
    query = " ".join(sys.argv[1:]) or "RSA weak key picoCTF crypto"
    print(f"Calling memory_search({query!r})...\n")
    result = call_tool_sync("memory_search", {"query": query, "k": 3})
    print(result[:2000])
