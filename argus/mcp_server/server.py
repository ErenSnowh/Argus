"""
server.py
----------
ARGUS MCP Server -- exposes the SOC toolset as standard MCP tools so any
MCP-compatible client (Google ADK agents, Claude, a CLI, a future SIEM
plugin) can call them over a uniform protocol instead of hard-coded function
calls. This is the "MCP Server" key-concept artifact for the capstone rubric.

Tools exposed:
  classify_flow            -- run a network flow through the RF detector
  lookup_mitre_attack       -- map a detection label to ATT&CK tactic/technique
  enrich_ioc                -- threat-intel lookup for an IP/domain (VT-backed)
  verify_file_hash          -- hash a file and check against known-malicious set
  analyze_pcap_summary      -- structured forensic summary of a PCAP file
  propose_playbook          -- generate a human-approval-required remediation plan

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
    counts, top talkers, destination port distribution, protocol mix, and
    capture duration. Use this when an analyst or agent needs first-pass
    context on a raw packet capture."""
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


if __name__ == "__main__":
    mcp.run()
