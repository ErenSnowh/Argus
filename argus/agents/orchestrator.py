"""
orchestrator.py
-----------------
Composes the five ARGUS sub-agents into a multi-agent pipeline using ADK's
native workflow agents -- this is the "Agent / Multi-agent system (ADK)" key
concept artifact for the capstone rubric.

Pipeline shape:

    incoming flow / pcap / IOC
            |
            v
      [triage_agent]                (classify + ATT&CK lookup, decides escalation)
            |
            v
   [ParallelAgent] -----------------------
       |                                 |
   [enrichment_agent]              [forensics_agent]   (run concurrently --
   (IOC reputation)                 (pcap / file hash)  independent evidence
       |                                 |              streams, no shared state
        ---------------------------------
            |
            v
     [remediation_agent]           (propose-only playbook, human approval required)
            |
            v
       [report_agent]              (pure synthesis, zero tool access)

SequentialAgent and ParallelAgent are both native ADK workflow primitives;
nesting a ParallelAgent inside a SequentialAgent is the standard ADK pattern
for "fan out for independent evidence gathering, then fan back in."
"""

from __future__ import annotations

from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent

from agents.sub_agents import (
    enrichment_agent,
    forensics_agent,
    remediation_agent,
    report_agent,
    triage_agent,
)

evidence_gathering = ParallelAgent(
    name="evidence_gathering",
    description="Runs threat-intel enrichment and packet/file forensics concurrently on independent evidence streams.",
    sub_agents=[enrichment_agent, forensics_agent],
)

argus_pipeline = SequentialAgent(
    name="argus_soc_pipeline",
    description=(
        "ARGUS end-to-end SOC pipeline: triage -> parallel evidence gathering "
        "(enrichment + forensics) -> remediation proposal -> incident report."
    ),
    sub_agents=[triage_agent, evidence_gathering, remediation_agent, report_agent],
)
