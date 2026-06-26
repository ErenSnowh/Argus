"""
mcp_connection.py
-------------------
Wires each ADK sub-agent to the ARGUS MCP server (mcp_server/server.py) over
stdio, using `tool_filter` to enforce the exact same least-privilege
allowlists defined in security/guardrails.TOOL_ALLOWLISTS. This means the
access control isn't just a policy on paper -- the agent's underlying
ADK tool registry physically does not contain tools outside its role, so an
LLM hallucinating a tool call outside its job description has nothing to
call.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp import StdioServerParameters
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

from security.guardrails import TOOL_ALLOWLISTS

_SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "mcp_server.server"],
    cwd=str(PROJECT_ROOT),
    env={**os.environ},  # Pass through WINDIR, PATH, etc. for scapy + Windows
)

_CONNECTION_PARAMS = StdioConnectionParams(server_params=_SERVER_PARAMS, timeout=30.0)


def toolset_for(agent_role: str) -> McpToolset | None:
    """Return an McpToolset scoped to exactly the tools `agent_role` is
    allowed to call. Raises if the role has no registered allowlist, so a
    new agent can never accidentally get full, unfiltered tool access by
    omission. Returns None for roles with an explicitly empty allowlist
    (e.g. report_agent, which only synthesizes text and gets no tool access
    at all)."""
    allowed = TOOL_ALLOWLISTS.get(agent_role)
    if allowed is None:
        raise KeyError(
            f"No tool allowlist registered for '{agent_role}'. "
            "Add one to security/guardrails.TOOL_ALLOWLISTS before wiring this agent."
        )
    if not allowed:
        return None
    return McpToolset(connection_params=_CONNECTION_PARAMS, tool_filter=sorted(allowed))
