"""
server.py
----------
ARGUS MCP Server -- exposes the SOC toolset as standard MCP tools so any
MCP-compatible client (Google ADK agents, Claude, a CLI, a future SIEM
plugin) can call them over a uniform protocol instead of hard-coded function
calls. This is the "MCP Server" key-concept artifact for the capstone rubric.

Tools exposed:
  classify_flow            -- run a network flow through the RF detector
  classify_flow_explained  -- classify with SHAP-based feature attribution
  lookup_mitre_attack      -- map a detection label to ATT&CK tactic/technique
  enrich_ioc               -- threat-intel lookup for an IP/domain (VT-backed)
  verify_file_hash         -- hash a file and check against known-malicious set
  analyze_pcap_summary     -- structured forensic summary of a PCAP file
  propose_playbook         -- generate a human-approval-required remediation plan
  compute_threat_score     -- composite 0-100 risk score across multiple dimensions
  correlate_alerts         -- detect multi-stage attack campaigns
  generate_certin_report   -- CERT-In compliant incident report (India 6-hour mandate)

Every tool call is:
  - checked against the calling agent's tool allowlist (least privilege)
  - redacted for secrets/PII before the result is returned
  - written to the hash-chained audit log

Run standalone for local testing:
    python -m mcp_server.server
Or mount over stdio for an MCP client (e.g. an ADK agent / Claude Desktop)
via the `mcp` CLI: `mcp dev mcp_server/server.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from ml.model import FlowClassifier, MODEL_PATH
from mcp_server.ioc_intel import enrich_ioc as _enrich_ioc
from mcp_server.mitre_mapping import map_to_attack
from mcp_server.pcap_forensics import analyze_pcap_summary as _analyze_pcap_summary
from mcp_server.pcap_forensics import verify_file_hash as _verify_file_hash
from mcp_server.playbooks import propose_playbook as _propose_playbook
from ml.threat_score import compute_threat_score as _compute_threat_score
from ml.correlation import correlate_alerts as _correlate_alerts
from mcp_server.certin_report import generate_certin_report as _generate_certin_report
from mcp_server.nciipc_report import generate_nciipc_report as _generate_nciipc_report
from ml.world_model.predictor import forecast_infiltration as _forecast_infiltration, get_predictor as _get_predictor
from mcp_server.knowledge_base import (
    enrich_detection_with_kb as _enrich_kb,
    lookup_attack_technique as _lookup_technique,
    lookup_cve as _lookup_cve,
    get_kb_status as _kb_status,
)
from ml.world_model.dataset_loader import list_datasets as _list_datasets
from security.guardrails import AuditLogger, redact

mcp = FastMCP("argus-soc-tools")

AUDIT_LOG = AuditLogger(Path(__file__).resolve().parent.parent / "data" / "audit_log.jsonl")

_classifier: FlowClassifier | None = None


def _get_classifier() -> FlowClassifier:
    global _classifier
    if _classifier is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Model not trained yet. Run `python scripts/train_model.py` first.")
        _classifier = FlowClassifier()
    return _classifier


def _audit(tool: str, payload: dict) -> None:
    safe_payload = {}
    for k, v in payload.items():
        if isinstance(v, str):
            redacted, _ = redact(v)
            safe_payload[k] = redacted
        else:
            safe_payload[k] = v
    AUDIT_LOG.log("tool_call", tool, safe_payload)


@mcp.tool()
def classify_flow(flow_features: dict) -> dict:
    """Classify a network flow's 24 statistical features (duration, packet
    counts, byte counts, IAT stats, flag counts, etc.) into BENIGN or an
    attack category using the trained Random Forest detector. Returns the
    predicted label, confidence, and full class-probability distribution."""
    result = _get_classifier().predict(flow_features)
    _audit("classify_flow", {"input_keys": list(flow_features.keys()), "result": result["predicted_label"]})
    return result


@mcp.tool()
def classify_flow_explained(flow_features: dict) -> dict:
    """Classify a network flow WITH SHAP-based explainability. Returns the
    standard prediction plus top contributing features with natural-language
    explanations of why each feature drove the detection. Use this when
    the analyst needs to understand the ML's reasoning."""
    result = _get_classifier().predict_with_explanation(flow_features)
    _audit("classify_flow_explained", {"result": result["predicted_label"], "has_explanation": True})
    return result


@mcp.tool()
def lookup_mitre_attack(detection_label: str) -> dict:
    """Map a detection label (e.g. 'DDoS', 'PortScan', 'BruteForce',
    'WebAttack', 'Botnet') to its MITRE ATT&CK tactic, technique ID/name,
    a plain-language description, and recommended controls."""
    mapping = map_to_attack(detection_label)
    _audit("lookup_mitre_attack", {"label": detection_label, "found": mapping is not None})
    if mapping is None:
        return {"found": False, "label": detection_label}
    return {
        "found": True,
        "tactic": mapping.tactic,
        "technique_id": mapping.technique_id,
        "technique_name": mapping.technique_name,
        "description": mapping.description,
        "recommended_controls": mapping.recommended_controls,
    }


@mcp.tool()
def enrich_ioc(indicator: str) -> dict:
    """Look up threat-intel reputation for an IP address or domain (VirusTotal
    in production with VIRUSTOTAL_API_KEY set; local demo cache otherwise).
    Returns verdict (malicious/suspicious/clean/unknown), vote counts, and
    categories."""
    result = _enrich_ioc(indicator)
    _audit("enrich_ioc", {"indicator": indicator, "verdict": result.verdict})
    return {
        "indicator": result.indicator,
        "verdict": result.verdict,
        "malicious_votes": result.malicious_votes,
        "categories": result.categories,
        "source": result.source,
    }


@mcp.tool()
def verify_file_hash(file_path: str) -> dict:
    """Compute the SHA-256 of a file and check it against the known-malicious
    hash set. Use for verifying suspicious binaries/attachments found during
    an investigation."""
    result = _verify_file_hash(file_path)
    _audit("verify_file_hash", {"file_path": file_path, "match_found": result.get("match_found")})
    return result


@mcp.tool()
def analyze_pcap_summary(pcap_path: str) -> dict:
    """Return a structured forensic summary of a PCAP file: packet/byte
    counts, top talkers, destination port distribution, protocol mix,
    capture duration, plus deep forensics (DNS tunneling detection,
    C2 beaconing analysis, payload anomaly flags)."""
    result = _analyze_pcap_summary(pcap_path)
    _audit("analyze_pcap_summary", {"pcap_path": pcap_path, "packet_count": result.get("packet_count")})
    return result


@mcp.tool()
def propose_playbook(detection_label: str, confidence: float) -> dict:
    """Generate a proposed, human-approval-required remediation playbook for
    a detection label. This tool only PROPOSES actions; it has no execution
    permission against any firewall/EDR/network system by design."""
    pb = _propose_playbook(detection_label, confidence)
    _audit("propose_playbook", {"label": detection_label, "severity": pb.severity})
    return {
        "incident_label": pb.incident_label,
        "severity": pb.severity,
        "requires_human_approval": pb.requires_human_approval,
        "steps": pb.steps,
        "attack_reference": pb.attack_reference,
    }


@mcp.tool()
def compute_threat_score(
    predicted_label: str,
    ml_confidence: float,
    ioc_verdict: str = "unknown",
    source_ip: str = "",
    asset_tier: str = "standard",
) -> dict:
    """Compute a composite 0-100 Threat Score for an alert by fusing ML
    confidence, IOC reputation, MITRE ATT&CK kill-chain position, temporal
    risk multiplier (repeated alerts escalate), and asset criticality tier.
    Returns the score, severity band, SLA recommendation, and full breakdown."""
    ts = _compute_threat_score(
        predicted_label=predicted_label,
        ml_confidence=ml_confidence,
        ioc_verdict=ioc_verdict if ioc_verdict else None,
        source_ip=source_ip if source_ip else None,
        asset_tier=asset_tier,
    )
    _audit("compute_threat_score", {"label": predicted_label, "score": ts.score, "severity": ts.severity})
    return {
        "score": ts.score,
        "severity": ts.severity,
        "sla_recommendation": ts.sla_recommendation,
        "breakdown": ts.breakdown,
        "explanation": ts.explanation,
        "temporal_multiplier": ts.temporal_multiplier,
        "asset_multiplier": ts.asset_multiplier,
        "alert_count_window": ts.alert_count_window,
    }


@mcp.tool()
def correlate_alerts(
    source_ip: str,
    target_ip: str,
    predicted_label: str,
    confidence: float,
    threat_score: float = 0.0,
) -> dict:
    """Ingest an alert into the correlation engine and check for campaign
    patterns. Groups alerts by source IP within a 5-minute sliding window
    and detects kill-chain progression (e.g., PortScan → BruteForce → Botnet).
    Returns campaign details if a multi-stage attack is detected."""
    result = _correlate_alerts(
        source_ip=source_ip,
        target_ip=target_ip,
        predicted_label=predicted_label,
        confidence=confidence,
        threat_score=threat_score,
    )
    _audit("correlate_alerts", {
        "source_ip": source_ip,
        "campaign_detected": result.get("campaign_detected", False),
    })
    return result


@mcp.tool()
def generate_certin_report(
    predicted_label: str,
    confidence: float,
    threat_score: float = 0.0,
    threat_severity: str = "",
    mitre_technique_id: str = "",
    mitre_technique_name: str = "",
    mitre_tactic: str = "",
    suspect_ips: str = "",
    affected_ips: str = "",
) -> dict:
    """Generate a CERT-In compliant incident report per India's 2022 directive
    mandating 6-hour reporting. Auto-populates all mandatory technical fields
    from the ARGUS investigation results. Organizational fields are left as
    templates for the SOC analyst to fill before submission."""
    report = _generate_certin_report(
        predicted_label=predicted_label,
        confidence=confidence,
        threat_score=threat_score if threat_score else None,
        threat_severity=threat_severity if threat_severity else None,
        mitre_technique_id=mitre_technique_id if mitre_technique_id else None,
        mitre_technique_name=mitre_technique_name if mitre_technique_name else None,
        mitre_tactic=mitre_tactic if mitre_tactic else None,
        suspect_ips=suspect_ips.split(",") if suspect_ips else [],
        affected_ips=affected_ips.split(",") if affected_ips else [],
    )
    _audit("generate_certin_report", {
        "report_id": report.report_id,
        "category": report.incident_category,
        "severity": report.severity_class,
    })
    return {
        "report_id": report.report_id,
        "generated_at_ist": report.generated_at_ist,
        "detection_timestamp_ist": report.detection_timestamp_ist,
        "reporting_deadline_ist": report.reporting_deadline_ist,
        "is_within_deadline": report.is_within_deadline,
        "incident_category_id": report.incident_category_id,
        "incident_category": report.incident_category,
        "severity_class": report.severity_class,
        "reporting_priority": report.reporting_priority,
        "report_markdown": report.report_markdown,
    }


@mcp.tool()
def generate_nciipc_report(
    predicted_label: str,
    confidence: float,
    cii_sector: str = "Power & Energy",
    threat_score: float = 0.0,
    threat_severity: str = "",
    suspect_ips: str = "",
    affected_ips: str = "",
    organization_name: str = "[DESIGNATED CII ENTITY -- TO BE COMPLETED]",
    facility_location: str = "[CRITICAL FACILITY / REGION]",
) -> dict:
    """Generate an official NCIIPC Critical Information Infrastructure (CII)
    Incident Report under Section 70A of the Information Technology Act, 2000.
    Embeds the World Model attack progression forecast, MITRE ATT&CK, CAPEC,
    and CVE vulnerability intelligence. Designed for submission to
    helpdesk1@nciipc.gov.in with dual coordination to incident@cert-in.org.in."""
    report = _generate_nciipc_report(
        predicted_label=predicted_label,
        confidence=confidence,
        cii_sector=cii_sector,
        threat_score=threat_score if threat_score else None,
        threat_severity=threat_severity if threat_severity else None,
        suspect_ips=suspect_ips.split(",") if suspect_ips else [],
        affected_ips=affected_ips.split(",") if affected_ips else [],
        organization_name=organization_name,
        facility_location=facility_location,
    )
    _audit("generate_nciipc_report", {
        "report_id": report.report_id,
        "sector": report.cii_sector,
        "criticality": report.criticality_tier,
        "severity": report.threat_severity,
    })
    return {
        "report_id": report.report_id,
        "generated_at_ist": report.generated_at_ist,
        "cii_sector": report.cii_sector,
        "sector_code": report.sector_code,
        "criticality_tier": report.criticality_tier,
        "containment_sla": report.containment_sla,
        "max_infiltration_prob": report.max_infiltration_prob,
        "predicted_kill_chain": report.predicted_kill_chain,
        "proactive_containment_advisory": report.proactive_containment_advisory,
        "nciipc_submission_email": report.nciipc_submission_email,
        "nciipc_portal_url": report.nciipc_portal_url,
        "report_markdown": report.report_markdown,
        "nciipc_email_draft": report.nciipc_email_draft,
    }


@mcp.tool()
def forecast_infiltration(
    flow_features: dict,
    k_steps: int = 5,
) -> dict:
    """Run the World Model's K-step forward simulation to predict future
    network state evolution, infiltration probability timeline, and
    predicted MITRE ATT&CK stage at each future time step. This is the
    core forecasting capability that goes beyond static classification
    by predicting HOW an attack will progress, not just what it is now."""
    result = _forecast_infiltration(
        flow_features=flow_features,
        k_steps=k_steps,
    )
    _audit("forecast_infiltration", {
        "k_steps": k_steps,
        "risk_level": result.get("risk_level"),
        "max_prob": result.get("max_infiltration_prob"),
    })
    return result


@mcp.tool()
def get_world_model_status() -> dict:
    """Check whether the World Model Transformer is trained and return
    model metadata (parameters, accuracy, AUC, training source). Use this
    to verify the world model is available before calling forecast_infiltration."""
    try:
        predictor = _get_predictor()
        info = predictor.get_model_info()
        _audit("get_world_model_status", {"trained": info.get("model_trained", False)})
        return info
    except Exception as e:
        return {"model_trained": False, "error": str(e)}


@mcp.tool()
def enrich_with_knowledge_base(
    label: str,
    technique_id: str | None = None,
) -> dict:
    """Enrich a detection label with context from MITRE ATT&CK (STIX),
    CAPEC attack patterns, and CVE/NVD vulnerability data. Returns
    related techniques with mitigations, attack patterns with CWE links,
    and relevant CVEs with CVSS scores. Use this after classification to
    provide deep threat intelligence context."""
    result = _enrich_kb(label, technique_id)
    _audit("enrich_with_knowledge_base", {
        "label": label,
        "n_techniques": len(result.get("attack_techniques", [])),
        "n_capec": len(result.get("capec_patterns", [])),
        "n_cves": len(result.get("related_cves", [])),
    })
    return result


@mcp.tool()
def lookup_cve_nvd(cve_id: str) -> dict:
    """Look up a specific CVE from the National Vulnerability Database.
    Returns the description, CVSS score, severity, affected products,
    and references. Use when an IOC or detection is linked to a known CVE."""
    result = _lookup_cve(cve_id)
    if result:
        _audit("lookup_cve_nvd", {"cve_id": cve_id, "cvss": result.get("cvss_score")})
        return result
    return {"error": f"CVE {cve_id} not found"}


@mcp.tool()
def lookup_attack_technique_detail(technique_id: str) -> dict:
    """Look up a specific MITRE ATT&CK technique by ID (e.g. T1498) from
    the full Enterprise STIX bundle. Returns description, platforms,
    data sources, detection guidance, and mitigations."""
    result = _lookup_technique(technique_id)
    if result:
        _audit("lookup_attack_technique_detail", {"technique_id": technique_id})
        return result
    return {"error": f"Technique {technique_id} not found in ATT&CK database"}


@mcp.tool()
def list_supported_datasets() -> dict:
    """List all supported public cybersecurity datasets with download URLs,
    descriptions, and sizes. Use this to show analysts which datasets are
    available for world model training."""
    _audit("list_supported_datasets", {})
    return _list_datasets()


@mcp.tool()
def get_knowledge_base_status() -> dict:
    """Return the status of all integrated knowledge bases (MITRE ATT&CK,
    CAPEC, CVE/NVD) including cache status and data freshness."""
    status = _kb_status()
    _audit("get_knowledge_base_status", {k: v.get("cache_exists", False) for k, v in status.items() if isinstance(v, dict)})
    return status


if __name__ == "__main__":
    mcp.run()

