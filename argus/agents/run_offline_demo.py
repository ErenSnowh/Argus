"""
run_offline_demo.py
----------------------
Runs the SAME six-stage ARGUS pipeline shape (triage -> enrichment +
forensics -> remediation -> compliance -> report) but with each "agent"
step implemented as a direct, deterministic Python call into the MCP tool
functions instead of an LLM-driven ADK agent.

Now includes all enhanced capabilities:
  - SHAP-based ML explainability
  - Composite threat scoring
  - Alert correlation / campaign detection
  - CERT-In compliance reporting
  - Deep PCAP forensics (DNS tunneling, beaconing, payload anomalies)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server.ioc_intel import enrich_ioc
from mcp_server.mitre_mapping import map_to_attack
from mcp_server.pcap_forensics import analyze_pcap_summary
from mcp_server.playbooks import propose_playbook
from mcp_server.certin_report import generate_certin_report
from ml.model import FlowClassifier
from ml.threat_score import compute_threat_score
from ml.correlation import correlate_alerts
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

    # ---- Stage 1: Triage (with SHAP explainability) ----------------------
    classifier = FlowClassifier()
    triage = classifier.predict_with_explanation(flow_features)
    AUDIT_LOG.log("agent_step", "triage_agent", {"verdict": triage["predicted_label"], "confidence": triage["confidence"]})

    attack = map_to_attack(triage["predicted_label"])
    escalate = triage["predicted_label"] != "BENIGN" or triage["confidence"] < 0.7

    # ---- Stage 1b: Threat Score ------------------------------------------
    threat_score_result = None
    if escalate:
        ts = compute_threat_score(
            predicted_label=triage["predicted_label"],
            ml_confidence=triage["confidence"],
            source_ip=suspect_ip,
        )
        threat_score_result = {
            "score": ts.score,
            "severity": ts.severity,
            "sla_recommendation": ts.sla_recommendation,
            "breakdown": ts.breakdown,
            "explanation": ts.explanation,
            "temporal_multiplier": ts.temporal_multiplier,
        }
        AUDIT_LOG.log("agent_step", "triage_agent", {"threat_score": ts.score, "severity": ts.severity})

    # ---- Stage 1c: Alert Correlation -------------------------------------
    correlation_result = None
    if escalate and suspect_ip:
        correlation_result = correlate_alerts(
            source_ip=suspect_ip,
            target_ip="10.0.4.50",  # Default target for demo
            predicted_label=triage["predicted_label"],
            confidence=triage["confidence"],
            threat_score=threat_score_result["score"] if threat_score_result else 0.0,
        )
        AUDIT_LOG.log("agent_step", "triage_agent", {"campaign_detected": correlation_result.get("campaign_detected", False)})

    # ---- Stage 1d: World Model Forecast ----------------------------------
    forecast_result = None
    if escalate:
        try:
            from ml.world_model.predictor import forecast_infiltration as _forecast
            forecast_result = _forecast(flow_features, k_steps=5)
            AUDIT_LOG.log("agent_step", "triage_agent", {
                "forecast_risk": forecast_result.get("risk_level"),
                "forecast_max_prob": forecast_result.get("max_infiltration_prob"),
            })
        except (FileNotFoundError, ImportError) as e:
            # World model not trained yet — gracefully skip
            forecast_result = {"error": str(e), "risk_level": "UNKNOWN"}
            AUDIT_LOG.log("agent_step", "triage_agent", {"forecast_error": str(e)})

    # ---- Stage 2: Evidence gathering (enrichment + forensics) -----------
    ioc_result = None
    if escalate and suspect_ip:
        ioc_result = enrich_ioc(suspect_ip)
        AUDIT_LOG.log("agent_step", "enrichment_agent", {"indicator": suspect_ip, "verdict": ioc_result.verdict})

        # Update threat score with IOC context
        if threat_score_result:
            ts_updated = compute_threat_score(
                predicted_label=triage["predicted_label"],
                ml_confidence=triage["confidence"],
                ioc_verdict=ioc_result.verdict,
                source_ip=suspect_ip,
            )
            threat_score_result["score"] = ts_updated.score
            threat_score_result["severity"] = ts_updated.severity
            threat_score_result["explanation"] = ts_updated.explanation

    forensics_result = None
    if escalate and pcap_path:
        forensics_result = analyze_pcap_summary(pcap_path)
        AUDIT_LOG.log("agent_step", "forensics_agent", {"pcap": pcap_path, "packets": forensics_result.get("packet_count")})

    # ---- Stage 3: Remediation proposal -----------------------------------
    playbook = None
    if escalate:
        playbook = propose_playbook(triage["predicted_label"], triage["confidence"])
        AUDIT_LOG.log("agent_step", "remediation_agent", {"severity": playbook.severity})

    # ---- Stage 4: CERT-In Compliance Report ------------------------------
    certin_result = None
    if escalate:
        certin = generate_certin_report(
            predicted_label=triage["predicted_label"],
            confidence=triage["confidence"],
            threat_score=threat_score_result["score"] if threat_score_result else None,
            threat_severity=threat_score_result["severity"] if threat_score_result else None,
            mitre_technique_id=attack.technique_id if attack else None,
            mitre_technique_name=attack.technique_name if attack else None,
            mitre_tactic=attack.tactic if attack else None,
            suspect_ips=[suspect_ip] if suspect_ip else [],
            ioc_results=[ioc_result.__dict__] if ioc_result else [],
            playbook_steps=playbook.steps if playbook else [],
        )
        certin_result = {
            "report_id": certin.report_id,
            "incident_category": certin.incident_category,
            "severity_class": certin.severity_class,
            "reporting_deadline_ist": certin.reporting_deadline_ist,
            "is_within_deadline": certin.is_within_deadline,
            "report_markdown": certin.report_markdown,
        }
        AUDIT_LOG.log("agent_step", "compliance_agent", {"report_id": certin.report_id, "category": certin.incident_category})

    # ---- Stage 5: Report synthesis ---------------------------------------
    report_md = _render_report(triage, attack, ioc_result, forensics_result, playbook, escalate, threat_score_result, correlation_result, certin_result, forecast_result)
    AUDIT_LOG.log("agent_step", "report_agent", {"escalated": escalate})

    return {
        "triage": triage,
        "attack_mapping": attack.__dict__ if attack else None,
        "ioc_result": ioc_result.__dict__ if ioc_result else None,
        "forensics_result": forensics_result,
        "playbook": playbook.__dict__ if playbook else None,
        "threat_score": threat_score_result,
        "correlation": correlation_result,
        "certin_report": certin_result,
        "forecast": forecast_result,
        "escalated": escalate,
        "report_markdown": report_md,
    }


def _render_report(triage, attack, ioc_result, forensics_result, playbook, escalate, threat_score, correlation, certin_result, forecast=None) -> str:
    lines = ["# ARGUS Incident Report", ""]

    # Summary with threat score
    summary = f"Detection: **{triage['predicted_label']}** (confidence {triage['confidence']:.2%}). "
    if threat_score:
        summary += f"Threat Score: **{threat_score['score']}/100** ({threat_score['severity']}). "
    if forecast and forecast.get("risk_level") and forecast.get("risk_level") != "UNKNOWN":
        summary += f"Forecast Risk: **{forecast['risk_level']}**. "
    summary += 'Escalated for investigation.' if escalate else 'Below escalation threshold - benign, no further action.'
    lines += ["## Summary", summary, ""]

    if threat_score:
        lines += ["## Threat Score", threat_score['explanation'], ""]

    # World Model Infiltration Forecast
    if forecast and not forecast.get("error"):
        lines += ["## World Model Infiltration Forecast"]
        lines.append(f"**Risk Level**: {forecast.get('risk_level', 'N/A')}")
        lines.append(f"**Max Infiltration Probability**: {forecast.get('max_infiltration_prob', 0):.1%}")
        lines.append("")
        lines.append("### Probability Timeline")
        probs = forecast.get("probability_timeline", [])
        stages = forecast.get("predicted_stages", [])
        for i, (prob, stage) in enumerate(zip(probs, stages)):
            filled = int(prob * 20)
            bar = "#" * filled + "-" * (20 - filled)
            lines.append(f"  T+{i+1}: [{bar}] {prob:.1%}  [{stage}]")
        lines.append("")
        if forecast.get("forecast_explanation"):
            lines.append(forecast["forecast_explanation"])
            lines.append("")
    elif forecast and forecast.get("error"):
        lines += ["## World Model Infiltration Forecast",
                   f"[!] Forecast unavailable: {forecast['error']}", ""]

    # Detection details with SHAP explanation
    lines += ["## Detection Details"]
    for label, prob in sorted(triage["class_probabilities"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {label}: {prob:.2%}")
    lines.append("")

    if "explanation_text" in triage:
        lines += ["### ML Explainability (SHAP Analysis)", triage["explanation_text"], ""]

    if attack:
        lines += ["## MITRE ATT&CK Mapping",
                   f"**{attack.technique_id} - {attack.technique_name}** (Tactic: {attack.tactic})",
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
                   f"over {forensics_result['duration_sec']}s. Protocol mix: {forensics_result['protocol_mix']}.", ""]

        # Deep forensics findings
        risk_flags = forensics_result.get("forensic_risk_flags", [])
        if risk_flags:
            lines.append(f"### [!] Deep Forensic Risk Flags: {', '.join(risk_flags)}")
            if "dns_tunneling" in forensics_result:
                dt = forensics_result["dns_tunneling"]
                lines.append(f"- **DNS Tunneling**: {dt['verdict']}")
            if "beaconing" in forensics_result:
                bc = forensics_result["beaconing"]
                lines.append(f"- **C2 Beaconing**: {bc['verdict']}")
            if "payload_anomalies" in forensics_result:
                pa = forensics_result["payload_anomalies"]
                lines.append(f"- **Payload Anomalies**: {pa['verdict']}")
            lines.append("")

    # Alert correlation / campaign
    if correlation and correlation.get("campaign_detected"):
        campaign = correlation["campaign"]
        lines += ["## Alert Correlation - Campaign Detected",
                   f"**Campaign ID**: `{campaign['campaign_id']}`",
                   f"**Severity**: {campaign['campaign_severity']} (Score: {campaign['campaign_score']})",
                   f"**Alert Count**: {campaign['alert_count']}",
                   campaign["description"], ""]
        if campaign.get("kill_chain_progression"):
            lines.append(f"**Kill-Chain Progression**: {' -> '.join(campaign['kill_chain_progression'])}")
            lines.append("")

    if playbook:
        lines += ["## Recommended Actions (Pending Human Approval)",
                   f"Severity: **{playbook.severity}**"]
        lines += [f"- {step}" for step in playbook.steps]
        lines.append("")

    # CERT-In Compliance
    if certin_result:
        lines += ["## CERT-In Compliance",
                   f"**Report ID**: `{certin_result['report_id']}`",
                   f"**Category**: {certin_result['incident_category']}",
                   f"**Severity Class**: {certin_result['severity_class']}",
                   f"**Reporting Deadline**: {certin_result['reporting_deadline_ist']}",
                   f"**Within Deadline**: {'Yes' if certin_result['is_within_deadline'] else 'No'}",
                   f"Submit to: incident@cert-in.org.in", ""]

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

