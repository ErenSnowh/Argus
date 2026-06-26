"""
mitre_mapping.py
-----------------
Maps ARGUS detection labels to MITRE ATT&CK (Enterprise) tactics/techniques.

This intentionally uses a small local, hand-curated lookup table rather than
calling the live MITRE ATT&CK API/STIX bundle, for two reasons:
  1. Offline reliability -- the mapping must work in an air-gapped SOC demo.
  2. Determinism -- judges should see the same technique IDs every run.

In a production deployment, swap `TECHNIQUE_MAP` for a loader that pulls the
current STIX bundle from https://github.com/mitre/cti and caches it locally;
the rest of the agent pipeline is unaffected because it only depends on the
`map_to_attack()` function signature below.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AttackMapping:
    tactic: str
    technique_id: str
    technique_name: str
    description: str
    recommended_controls: list[str] = field(default_factory=list)


TECHNIQUE_MAP: dict[str, AttackMapping] = {
    "DDoS": AttackMapping(
        tactic="Impact",
        technique_id="T1498",
        technique_name="Network Denial of Service",
        description=(
            "High packet-per-second / low-payload flood pattern consistent "
            "with volumetric or protocol-level DoS flooding a host or service."
        ),
        recommended_controls=[
            "Enable upstream rate-limiting / scrubbing (e.g. cloud DDoS protection)",
            "Apply SYN-cookie / connection-rate throttling at the edge firewall",
            "Alert network team to validate upstream ISP blackhole routing if sustained",
        ],
    ),
    "PortScan": AttackMapping(
        tactic="Reconnaissance",
        technique_id="T1595.001",
        technique_name="Active Scanning: Scanning IP Blocks / Vulnerability Scanning",
        description=(
            "Single source contacting many distinct destination ports with "
            "minimal payload -- classic horizontal/vertical port-scan signature."
        ),
        recommended_controls=[
            "Throttle or temporarily block the source IP at the perimeter",
            "Cross-check source IP against threat-intel reputation feeds",
            "Review exposed services on the scanned host(s) for hardening",
        ],
    ),
    "BruteForce": AttackMapping(
        tactic="Credential Access",
        technique_id="T1110",
        technique_name="Brute Force",
        description=(
            "Repeated short-lived connections with regular timing and PSH/ACK "
            "bursts consistent with automated credential-guessing against an "
            "authentication service (SSH/RDP/web login)."
        ),
        recommended_controls=[
            "Enforce account lockout / exponential backoff on the targeted service",
            "Require MFA on the affected account(s) if not already enforced",
            "Block source IP after N failed attempts; review for credential stuffing",
        ],
    ),
    "WebAttack": AttackMapping(
        tactic="Initial Access",
        technique_id="T1190",
        technique_name="Exploit Public-Facing Application",
        description=(
            "Asymmetric request/response sizes with elevated forward-packet "
            "variance, consistent with SQLi/XSS/parameter-tampering probes "
            "against a web application."
        ),
        recommended_controls=[
            "Inspect WAF logs for the same source/time window",
            "Validate input sanitization on the targeted endpoint",
            "Patch / virtually patch the public-facing application if vulnerable",
        ],
    ),
    "Botnet": AttackMapping(
        tactic="Command and Control",
        technique_id="T1071",
        technique_name="Application Layer Protocol (C2)",
        description=(
            "Long-lived, low-throughput, highly-regular beaconing interval -- "
            "classic C2 check-in pattern (cf. Cobalt Strike beacon analysis)."
        ),
        recommended_controls=[
            "Isolate host from the network pending forensic memory capture",
            "Extract and hunt for the beacon's C2 domain/IP across the fleet",
            "Run YARA / Volatility memory analysis for known C2 implants",
        ],
    ),
}


def map_to_attack(label: str) -> AttackMapping | None:
    """Return the ATT&CK mapping for a detection label, or None for BENIGN/unknown."""
    return TECHNIQUE_MAP.get(label)


if __name__ == "__main__":
    for lbl, mapping in TECHNIQUE_MAP.items():
        print(f"{lbl:12s} -> {mapping.technique_id} {mapping.technique_name} ({mapping.tactic})")
