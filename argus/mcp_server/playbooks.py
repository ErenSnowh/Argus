"""
playbooks.py
-------------
Generates a *proposed* remediation playbook for a given ATT&CK-mapped
incident. By design, this module only proposes actions -- it never calls an
EDR/firewall API to execute them. ARGUS's remediation_agent is deliberately
denied an "execute_playbook" tool (see security/guardrails.py
TOOL_ALLOWLISTS): a SOC co-pilot that can autonomously fire firewall rules or
isolate hosts without a human-in-the-loop approval step is a far bigger risk
than the incidents it's meant to catch. This is a conscious security-by-design
trade-off, not a missing feature.
"""

from __future__ import annotations

from dataclasses import dataclass

from mcp_server.mitre_mapping import AttackMapping, map_to_attack


@dataclass
class Playbook:
    incident_label: str
    severity: str
    requires_human_approval: bool
    steps: list[str]
    attack_reference: str | None


_SEVERITY_BY_LABEL = {
    "DDoS": "High",
    "Botnet": "Critical",
    "BruteForce": "Medium",
    "WebAttack": "High",
    "PortScan": "Low",
}


def propose_playbook(label: str, confidence: float, context: dict | None = None) -> Playbook:
    context = context or {}
    mapping: AttackMapping | None = map_to_attack(label)
    severity = _SEVERITY_BY_LABEL.get(label, "Medium")

    if confidence < 0.6:
        severity = "Low"  # low-confidence detections get downgraded, never escalated

    steps = list(mapping.recommended_controls) if mapping else [
        "No specific playbook on file for this label -- route to analyst for manual triage."
    ]
    steps.append(
        "ALL actions above require human SOC analyst approval before execution "
        "(ARGUS does not have firewall/EDR execution permissions by design)."
    )

    return Playbook(
        incident_label=label,
        severity=severity,
        requires_human_approval=True,
        steps=steps,
        attack_reference=f"{mapping.technique_id} {mapping.technique_name}" if mapping else None,
    )


if __name__ == "__main__":
    pb = propose_playbook("DDoS", confidence=0.97)
    print(pb)
