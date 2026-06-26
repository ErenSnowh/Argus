#!/usr/bin/env python3
"""
run_live.py
------------
Executes the real ARGUS multi-agent ADK pipeline end-to-end against Gemini.

The API key is auto-loaded from config.py — no manual setup needed. Anyone
who clones this repo can run this immediately:

    python agents/run_live.py "Investigate this flow: ..."

This is the code path judges should read to evaluate ADK usage;
`run_offline_demo.py` is the deterministic fallback for seeing the pipeline's
shape without any API calls.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401  -- auto-sets GOOGLE_API_KEY in os.environ

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.orchestrator import argus_pipeline

APP_NAME = "argus_soc"

# Retry settings for transient Gemini 503 UNAVAILABLE errors
MAX_RETRIES = 3
RETRY_BASE_DELAY = 4  # seconds, doubled each retry


async def run_incident(incident_description: str) -> None:
    """Run the live ADK pipeline, printing each agent's output to stdout."""
    events = []
    async for event in run_live_incident(incident_description):
        events.append(event)
        author = event.get("author", "?")
        if event.get("text"):
            print(f"\n--- [{author}] ---")
            print(event["text"])
        if event.get("tool_call"):
            tc = event["tool_call"]
            print(f"  [{author}] tool_call -> {tc['name']}({tc['args']})")


async def run_live_incident(incident_description: str):
    """Async generator that yields structured event dicts. Used by both the
    CLI runner above and the dashboard's /api/investigate/live endpoint.
    Includes retry logic for transient 503 errors."""

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async for event in _run_pipeline(incident_description):
                yield event
            return  # success — exit retry loop
        except Exception as e:
            import traceback
            trace_str = traceback.format_exc()
            error_str = f"{type(e).__name__}: {e}\n{trace_str}"
            last_error = e
            if "503" in error_str or "UNAVAILABLE" in error_str:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                yield {
                    "author": "system",
                    "text": f"Gemini returned 503 UNAVAILABLE (attempt {attempt}/{MAX_RETRIES}). "
                            f"Retrying in {delay}s...\nError detail: {error_str}",
                }
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(delay)
                    continue
            # Non-retryable error — raise immediately
            yield {"author": "system", "text": f"Pipeline error: {error_str}"}
            return

    # All retries exhausted
    yield {
        "author": "system",
        "text": f"All {MAX_RETRIES} attempts failed. Last error: {last_error}",
    }


async def _run_pipeline(incident_description: str):
    """Core ADK pipeline execution — no retry logic, just a clean run."""
    session_service = InMemorySessionService()
    user_id = "analyst"
    session_id = str(uuid.uuid4())
    await session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )

    runner = Runner(
        app_name=APP_NAME, agent=argus_pipeline, session_service=session_service
    )

    content = types.Content(
        role="user", parts=[types.Part(text=incident_description)]
    )

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        author = getattr(event, "author", "?")
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    yield {"author": author, "text": part.text}
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    yield {
                        "author": author,
                        "tool_call": {"name": fc.name, "args": dict(fc.args)},
                    }


if __name__ == "__main__":
    description = " ".join(sys.argv[1:]) or (
        "Investigate this network flow: a single source IP made 200 short-lived "
        "TCP SYN connections to sequential destination ports on host 10.0.4.50 "
        "within 16 milliseconds. flow_duration_ms=8, total_fwd_packets=2, "
        "total_bwd_packets=1, unique_dst_ports_per_src=190, syn_flag_count=2, "
        "flow_packets_per_sec=260. PCAP available at data/sample_portscan.pcap."
    )
    asyncio.run(run_incident(description))
