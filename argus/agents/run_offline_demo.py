"""
run_offline_demo.py
----------------------
Runs the SAME five-stage ARGUS pipeline shape (triage -> enrichment +
forensics -> remediation -> report) but with each "agent" step implemented
as a direct, deterministic Python call into the MCP tool functions instead
of an LLM-driven ADK agent.

Why this exists:
  - The public demo link / dashboard must work for anyone, with zero
    credentials, zero cost, and zero latency variance.
  - It is the fastest way for a judge to see ARGUS's actual decision logic
    (detector -> ATT&CK mapping -> enrichment -> forensics -> playbook ->
    report) without provisioning a Gemini API key first.
  - It exercises the exact same MCP tool layer (mcp_server/*.py) that the
    real ADK agents in agents/run_live.py call -- so this is not a separate,
    unrelated mock; it is the production tool layer with a rule-based
    orchestrator standing in for the LLM-driven one.

For the "real" multi-agent reasoning (the agents deciding what to call, in
what order, and writing freeform analyst prose) see agents/run_live.py,
which requires GOOGLE_API_KEY and runs the actual ADK SequentialAgent /
ParallelAgent pipeline defined in agents/orchestrator.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server.ioc_intel import enrich_ioc
from mcp_server.mitre_mapping import map_to_attack
from mcp_server.pcap_forensics import analyze_pcap_summary
from mcp_server.playbooks import propose_playbook
from ml.model import FlowClassifier
from security.guardrails import AuditLogger, redact

AUDIT_LOG = AuditLogger(Path(__file__).resolve().parent.parent / "data" / "audit_log.jsonl")


def run_offline_incident(
    flow_features: dict,
    suspect_ip: str | None = None,
    pcap_path: str | None = None,
) -> dict:
    """Run the full ARGUS pipeline deterministically and return a structured
    incident dict + rendered Markdown report. Mirrors what the LLM-driven
    pipeline produces, minus the freeform narrative reasoning."""

    # ---- Stage 1: Triage -------------------------------------------------
    classifier = FlowClassifier()
    triage = classifier.predict(flow_features)
    AUDIT_LOG.log("agent_step", "triage_agent", {"verdict": triage["predicted_label"], "confidence": triage["confidence"]})

    attack = map_to_attack(triage["predicted_label"])
    escalate = triage["predicted_label"] != "BENIGN" or triage["confidence"] < 0.7

    # ---- Stage 2: Evidence gathering (enrichment + forensics) -----------
    ioc_result = None
    if escalate and suspect_ip:
        ioc_result = enrich_ioc(suspect_ip)
        AUDIT_LOG.log("agent_step", "enrichment_agent", {"indicator": suspect_ip, "verdict": ioc_result.verdict})

    forensics_result = None
    if escalate and pcap_path:
        forensics_result = analyze_pcap_summary(pcap_path)
        AUDIT_LOG.log("agent_step", "forensics_agent", {"pcap": pcap_path, "packets": forensics_result.get("packet_count")})

    # ---- Stage 3: Remediation proposal -----------------------------------
    playbook = None
    if escalate:
        playbook = propose_playbook(triage["predicted_label"], triage["confidence"])
        AUDIT_LOG.log("agent_step", "remediation_agent", {"severity": playbook.severity})

    # ---- Stage 4: Report synthesis ---------------------------------------
    report_md = _render_report(triage, attack, ioc_result, forensics_result, playbook, escalate)
    AUDIT_LOG.log("agent_step", "report_agent", {"escalated": escalate})

    return {
        "triage": triage,
        "attack_mapping": attack.__dict__ if attack else None,
        "ioc_result": ioc_result.__dict__ if ioc_result else None,
        "forensics_result": forensics_result,
        "playbook": playbook.__dict__ if playbook else None,
        "escalated": escalate,
        "report_markdown": report_md,
    }


def _render_report(triage, attack, ioc_result, forensics_result, playbook, escalate) -> str:
    lines = ["# ARGUS Incident Report", ""]
    lines += ["## Summary",
               f"Detection: **{triage['predicted_label']}** (confidence {triage['confidence']:.2%}). "
               f"{'Escalated for investigation.' if escalate else 'Below escalation threshold -- benign, no further action.'}",
               ""]

    lines += ["## Detection Details"]
    for label, prob in sorted(triage["class_probabilities"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {label}: {prob:.2%}")
    lines.append("")

    if attack:
        lines += ["## MITRE ATT&CK Mapping",
                   f"**{attack.technique_id} — {attack.technique_name}** (Tactic: {attack.tactic})",
                   attack.description, ""]

    if ioc_result:
        red_indicator, _ = redact(ioc_result.indicator)
        lines += ["## Threat Intel",
                   f"Indicator `{red_indicator}` -> **{ioc_result.verdict}** "
                   f"({ioc_result.malicious_votes} votes, categories: {', '.join(ioc_result.categories) or 'none'}, "
                   f"source: {ioc_result.source})", ""]

    if forensics_result and "error" not in forensics_result:
        lines += ["## Forensics",
                   f"PCAP `{forensics_result['file']}` (sha256 `{forensics_result['sha256'][:16]}...`): "
                   f"{forensics_result['packet_count']} packets, {forensics_result['total_bytes']} bytes, "
                   f"{forensics_result['unique_dst_ports_contacted']} unique destination ports contacted "
                   f"over {forensics_result['duration_sec']}s. Protocol mix: {forensics_result['protocol_mix']}.",
                   ""]

    if playbook:
        lines += ["## Recommended Actions (Pending Human Approval)",
                   f"Severity: **{playbook.severity}**"]
        lines += [f"- {step}" for step in playbook.steps]
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    sample_flow = {
        "flow_duration_ms": 8, "total_fwd_packets": 2, "total_bwd_packets": 1,
        "total_fwd_bytes": 70, "total_bwd_bytes": 30, "fwd_packet_len_mean": 40,
        "fwd_packet_len_std": 2, "bwd_packet_len_mean": 30, "bwd_packet_len_std": 1,
        "flow_bytes_per_sec": 9000, "flow_packets_per_sec": 260, "flow_iat_mean": 3,
        "flow_iat_std": 1, "fwd_iat_mean": 3, "bwd_iat_mean": 5, "syn_flag_count": 2,
        "ack_flag_count": 0, "rst_flag_count": 1, "psh_flag_count": 0, "fin_flag_count": 0,
        "unique_dst_ports_per_src": 190, "packets_per_flow": 3, "avg_packet_size": 40,
        "down_up_ratio": 0.5,
    }
    result = run_offline_incident(
        sample_flow,
        suspect_ip="185.220.101.45",
        pcap_path="data/sample_portscan.pcap",
    )
    print(result["report_markdown"])
