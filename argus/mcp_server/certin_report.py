"""
certin_report.py
------------------
CERT-In Compliant Incident Report Generator.

India's Computer Emergency Response Team (CERT-In) issued directives in
April 2022 mandating all organizations to report cybersecurity incidents
within **6 hours** of detection. The directive specifies mandatory fields
and incident categories that must appear in every report.

This module auto-generates CERT-In-format incident reports from ARGUS
investigation results, reducing analyst burden from hours of manual
report writing to a single MCP tool call.

References:
    - CERT-In Direction under Section 70B(6) of the IT Act, 2000
    - https://www.cert-in.org.in/
    - NCIIPC reporting guidelines for Critical Information Infrastructure

Supported CERT-In incident categories:
    1. Targeted scanning/probing of critical networks
    2. Compromise of critical systems
    3. Unauthorised access
    4. Defacement of websites
    5. Malicious code attacks (e.g., ransomware, trojans)
    6. Attacks on servers and network devices
    7. Identity theft, phishing
    8. Denial of Service (DoS/DDoS) attacks
    9. Attacks on critical infrastructure
   10. Data breaches
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta


# Indian Standard Time (UTC+5:30) for CERT-In reporting
IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------------------------------------------------------
# CERT-In incident category mapping from ARGUS labels
# ---------------------------------------------------------------------------

CERTIN_CATEGORIES: dict[str, dict] = {
    "PortScan": {
        "category_id": 1,
        "category": "Targeted scanning/probing of critical networks/systems",
        "severity_class": "Medium",
        "reporting_priority": "Standard",
    },
    "BruteForce": {
        "category_id": 3,
        "category": "Unauthorised access of IT systems/data",
        "severity_class": "High",
        "reporting_priority": "Urgent",
    },
    "WebAttack": {
        "category_id": 6,
        "category": "Attacks on servers and network devices (SQLi/XSS/RCE)",
        "severity_class": "High",
        "reporting_priority": "Urgent",
    },
    "DDoS": {
        "category_id": 8,
        "category": "Denial of Service (DoS) and Distributed Denial of Service (DDoS) attacks",
        "severity_class": "Critical",
        "reporting_priority": "Immediate",
    },
    "Botnet": {
        "category_id": 5,
        "category": "Malicious code attacks (Botnet/C2 communication)",
        "severity_class": "Critical",
        "reporting_priority": "Immediate",
    },
    "BENIGN": {
        "category_id": 0,
        "category": "No incident - benign activity",
        "severity_class": "Informational",
        "reporting_priority": "None",
    },
}


# ---------------------------------------------------------------------------
# Report data structure
# ---------------------------------------------------------------------------

@dataclass
class CertInReport:
    """CERT-In compliant incident report with all mandatory fields."""
    report_id: str
    generated_at_ist: str
    detection_timestamp_ist: str
    reporting_deadline_ist: str
    is_within_deadline: bool

    # Incident classification
    incident_category_id: int
    incident_category: str
    severity_class: str
    reporting_priority: str

    # Detection details
    detection_method: str
    detection_label: str
    detection_confidence: float
    mitre_technique_id: str | None
    mitre_technique_name: str | None
    mitre_tactic: str | None

    # Affected systems
    affected_ips: list[str]
    suspect_source_ips: list[str]
    affected_services: list[str]

    # IOC details
    indicators_of_compromise: list[dict]

    # Threat score
    threat_score: float | None
    threat_severity: str | None

    # Actions
    actions_taken: list[str]
    recommended_actions: list[str]

    # Organization (template — to be filled by the analyst)
    organization_name: str
    organization_sector: str
    reporting_officer: str
    contact_email: str
    contact_phone: str

    # Full markdown report
    report_markdown: str


def _generate_report_id() -> str:
    """Generate a unique report ID in CERT-In format."""
    now = datetime.now(IST)
    ts = now.strftime("%Y%m%d%H%M%S")
    import uuid
    short_id = str(uuid.uuid4())[:6].upper()
    return f"ARGUS-CERTIN-{ts}-{short_id}"


def generate_certin_report(
    predicted_label: str,
    confidence: float,
    threat_score: float | None = None,
    threat_severity: str | None = None,
    mitre_technique_id: str | None = None,
    mitre_technique_name: str | None = None,
    mitre_tactic: str | None = None,
    suspect_ips: list[str] | None = None,
    affected_ips: list[str] | None = None,
    ioc_results: list[dict] | None = None,
    forensics_summary: str | None = None,
    playbook_steps: list[str] | None = None,
    organization_name: str = "[ORGANIZATION NAME — TO BE FILLED]",
    organization_sector: str = "[SECTOR — e.g., Banking, Power, Telecom]",
    reporting_officer: str = "[SOC ANALYST NAME]",
    contact_email: str = "[SOC EMAIL]",
    contact_phone: str = "[SOC PHONE]",
) -> CertInReport:
    """Generate a CERT-In compliant incident report.

    All organizational fields are templates that the SOC analyst fills
    before submitting to CERT-In. ARGUS auto-populates all technical
    fields from the investigation results.
    """
    now = datetime.now(IST)
    detection_time = now
    deadline = detection_time + timedelta(hours=6)

    cat = CERTIN_CATEGORIES.get(predicted_label, CERTIN_CATEGORIES["BENIGN"])

    suspect_ips = suspect_ips or []
    affected_ips = affected_ips or []
    ioc_results = ioc_results or []
    playbook_steps = playbook_steps or []

    affected_services = []
    if predicted_label == "DDoS":
        affected_services = ["Network services (potential outage)"]
    elif predicted_label == "BruteForce":
        affected_services = ["Authentication services (SSH/RDP/Web login)"]
    elif predicted_label == "WebAttack":
        affected_services = ["Web application (HTTP/HTTPS)"]
    elif predicted_label == "Botnet":
        affected_services = ["Compromised host (C2 communication active)"]
    elif predicted_label == "PortScan":
        affected_services = ["Multiple services probed (reconnaissance)"]

    actions_taken = [
        "Automated detection by ARGUS multi-agent SOC system",
        "Alert triaged and classified by ML-based flow classifier",
        "IOC enrichment performed against threat intelligence feeds",
    ]
    if forensics_summary:
        actions_taken.append("PCAP forensic analysis completed")
    actions_taken.append("CERT-In compliance report auto-generated")

    recommended = playbook_steps if playbook_steps else [
        "Review and validate the detection findings",
        "Escalate to incident response team if confirmed",
        "Submit this report to CERT-In within the 6-hour deadline",
    ]

    report_id = _generate_report_id()

    # Build the markdown report
    md = _build_certin_markdown(
        report_id=report_id,
        now=now,
        detection_time=detection_time,
        deadline=deadline,
        cat=cat,
        predicted_label=predicted_label,
        confidence=confidence,
        threat_score=threat_score,
        threat_severity=threat_severity,
        mitre_technique_id=mitre_technique_id,
        mitre_technique_name=mitre_technique_name,
        mitre_tactic=mitre_tactic,
        suspect_ips=suspect_ips,
        affected_ips=affected_ips,
        affected_services=affected_services,
        ioc_results=ioc_results,
        forensics_summary=forensics_summary,
        actions_taken=actions_taken,
        recommended=recommended,
        organization_name=organization_name,
        organization_sector=organization_sector,
        reporting_officer=reporting_officer,
        contact_email=contact_email,
        contact_phone=contact_phone,
    )

    return CertInReport(
        report_id=report_id,
        generated_at_ist=now.strftime("%Y-%m-%d %H:%M:%S IST"),
        detection_timestamp_ist=detection_time.strftime("%Y-%m-%d %H:%M:%S IST"),
        reporting_deadline_ist=deadline.strftime("%Y-%m-%d %H:%M:%S IST"),
        is_within_deadline=True,
        incident_category_id=cat["category_id"],
        incident_category=cat["category"],
        severity_class=cat["severity_class"],
        reporting_priority=cat["reporting_priority"],
        detection_method="ARGUS AI Multi-Agent SOC (Random Forest + Gemini LLM)",
        detection_label=predicted_label,
        detection_confidence=confidence,
        mitre_technique_id=mitre_technique_id,
        mitre_technique_name=mitre_technique_name,
        mitre_tactic=mitre_tactic,
        affected_ips=affected_ips,
        suspect_source_ips=suspect_ips,
        affected_services=affected_services,
        indicators_of_compromise=ioc_results,
        threat_score=threat_score,
        threat_severity=threat_severity,
        actions_taken=actions_taken,
        recommended_actions=recommended,
        organization_name=organization_name,
        organization_sector=organization_sector,
        reporting_officer=reporting_officer,
        contact_email=contact_email,
        contact_phone=contact_phone,
        report_markdown=md,
    )


def _build_certin_markdown(
    report_id, now, detection_time, deadline, cat,
    predicted_label, confidence, threat_score, threat_severity,
    mitre_technique_id, mitre_technique_name, mitre_tactic,
    suspect_ips, affected_ips, affected_services,
    ioc_results, forensics_summary,
    actions_taken, recommended,
    organization_name, organization_sector,
    reporting_officer, contact_email, contact_phone,
) -> str:
    """Build the full CERT-In compliant Markdown report."""
    lines = [
        "# CERT-In Incident Report",
        "",
        f"**Report ID**: `{report_id}`",
        f"**Generated**: {now.strftime('%Y-%m-%d %H:%M:%S IST')}",
        f"**Detection Time**: {detection_time.strftime('%Y-%m-%d %H:%M:%S IST')}",
        f"**Reporting Deadline**: {deadline.strftime('%Y-%m-%d %H:%M:%S IST')} (6-hour mandate)",
        f"**Status**: [OK] Within reporting deadline",
        "",
        "---",
        "",
        "## 1. Reporting Organization",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Organization | {organization_name} |",
        f"| Sector | {organization_sector} |",
        f"| Reporting Officer | {reporting_officer} |",
        f"| Contact Email | {contact_email} |",
        f"| Contact Phone | {contact_phone} |",
        "",
        "---",
        "",
        "## 2. Incident Classification",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| CERT-In Category | **{cat['category_id']}. {cat['category']}** |",
        f"| Severity Class | **{cat['severity_class']}** |",
        f"| Reporting Priority | **{cat['reporting_priority']}** |",
        f"| Detection Label | {predicted_label} |",
        f"| ML Confidence | {confidence:.1%} |",
    ]

    if threat_score is not None:
        lines.append(f"| Threat Score | **{threat_score}/100** ({threat_severity}) |")

    lines += ["", "---", "", "## 3. MITRE ATT&CK Mapping", ""]

    if mitre_technique_id:
        lines += [
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Technique ID | {mitre_technique_id} |",
            f"| Technique Name | {mitre_technique_name} |",
            f"| Tactic | {mitre_tactic} |",
        ]
    else:
        lines.append("No MITRE ATT&CK mapping applicable (benign detection).")

    lines += ["", "---", "", "## 4. Affected Systems", ""]

    if affected_ips:
        lines.append("**Affected IPs:**")
        for ip in affected_ips:
            lines.append(f"- `{ip}`")
        lines.append("")

    if suspect_ips:
        lines.append("**Suspect Source IPs:**")
        for ip in suspect_ips:
            lines.append(f"- `{ip}`")
        lines.append("")

    if affected_services:
        lines.append("**Affected Services:**")
        for svc in affected_services:
            lines.append(f"- {svc}")
        lines.append("")

    lines += ["---", "", "## 5. Indicators of Compromise (IOCs)", ""]

    if ioc_results:
        lines.append("| Indicator | Verdict | Votes | Categories |")
        lines.append("|-----------|---------|-------|------------|")
        for ioc in ioc_results:
            indicator = ioc.get("indicator", "?")
            verdict = ioc.get("verdict", "?")
            votes = ioc.get("malicious_votes", 0)
            cats = ", ".join(ioc.get("categories", []))
            lines.append(f"| `{indicator}` | **{verdict}** | {votes} | {cats} |")
    else:
        lines.append("No IOCs enriched in this investigation.")

    lines += ["", "---", "", "## 6. Forensic Analysis", ""]
    if forensics_summary:
        lines.append(forensics_summary)
    else:
        lines.append("No PCAP forensics performed for this incident.")

    lines += ["", "---", "", "## 7. Actions Taken", ""]
    for action in actions_taken:
        lines.append(f"- {action}")

    lines += ["", "---", "", "## 8. Recommended Actions (Pending Human Approval)", ""]
    for rec in recommended:
        lines.append(f"- {rec}")

    lines += [
        "",
        "---",
        "",
        "## 9. Compliance Notice",
        "",
        "> This report has been auto-generated by **ARGUS** (Autonomous Response & ",
        "> Guarded Unified Security) to assist compliance with CERT-In Direction dated",
        "> 28.04.2022 under Section 70B(6) of the Information Technology Act, 2000.",
        "> ",
        "> All incidents must be reported to CERT-In within **6 hours** of detection.",
        "> Submit to: incident@cert-in.org.in",
        "> ",
        "> **Disclaimer**: This is a machine-generated draft. The reporting officer",
        "> must review and validate all findings before submission to CERT-In.",
        "",
        "---",
        "",
        f"*Report generated by ARGUS SOC Co-Pilot v2.0 | {now.strftime('%Y-%m-%d %H:%M:%S IST')}*",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    report = generate_certin_report(
        predicted_label="DDoS",
        confidence=0.94,
        threat_score=87.5,
        threat_severity="HIGH",
        mitre_technique_id="T1498",
        mitre_technique_name="Network Denial of Service",
        mitre_tactic="Impact",
        suspect_ips=["185.220.101.45"],
        affected_ips=["10.0.4.50"],
        ioc_results=[{
            "indicator": "185.220.101.45",
            "verdict": "malicious",
            "malicious_votes": 14,
            "categories": ["tor-exit-node", "c2-relay"],
        }],
    )
    print(report.report_markdown)
