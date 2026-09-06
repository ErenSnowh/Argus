"""
threat_score.py
-----------------
Composite Threat Score Engine for ARGUS — computes a 0-100 risk score by
fusing multiple signal dimensions into a single analyst-actionable number.

Why this matters for Indian SOCs:
    CERT-In's 2022 directive mandates incident reporting within 6 hours.
    Indian SOC teams at NCIIPC, CERT-In, and sector CERTs (finance, power,
    telecom) process thousands of alerts daily. Without risk-based
    prioritization, analysts waste time on low-severity noise while
    high-severity incidents breach the 6-hour reporting window. This engine
    solves that by fusing five signal dimensions into a single, explainable
    0-100 score.

Score dimensions (configurable weights):
    1. ML Detection Confidence    (0-100)  — how certain is the classifier?
    2. IOC Reputation Severity    (0-100)  — is the indicator known-malicious?
    3. MITRE ATT&CK Position      (0-100)  — where in the kill chain?
    4. Temporal Risk Multiplier   (1.0-2.0) — repeated alerts escalate score
    5. Asset Criticality Tier     (1.0-2.0) — crown jewels get higher priority

Output:
    ThreatScore dataclass with score, severity band, human-readable breakdown,
    and SLA recommendation per CERT-In timelines.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Severity bands and SLA recommendations (aligned with CERT-In timelines)
# ---------------------------------------------------------------------------

SEVERITY_BANDS = [
    (90, "CRITICAL", "Immediate escalation - report to CERT-In within 6 hours"),
    (70, "HIGH", "Investigate within 1 hour - potential reportable incident"),
    (40, "MEDIUM", "Investigate within 4 hours - monitor for escalation"),
    (0, "LOW", "Log and monitor - no immediate action required"),
]


def _severity_band(score: float) -> tuple[str, str]:
    for threshold, band, sla in SEVERITY_BANDS:
        if score >= threshold:
            return band, sla
    return "LOW", SEVERITY_BANDS[-1][2]


# ---------------------------------------------------------------------------
# MITRE ATT&CK kill-chain position weights
# (later stages = higher risk — attacker is deeper in the network)
# ---------------------------------------------------------------------------

_TACTIC_WEIGHTS: dict[str, float] = {
    "Reconnaissance": 25.0,
    "Initial Access": 60.0,
    "Credential Access": 70.0,
    "Lateral Movement": 80.0,
    "Command and Control": 85.0,
    "Exfiltration": 92.0,
    "Impact": 90.0,
    # Fallback for unmapped tactics
}

# Label-to-tactic quick lookup (mirrors mitre_mapping.py)
_LABEL_TACTIC: dict[str, str] = {
    "PortScan": "Reconnaissance",
    "WebAttack": "Initial Access",
    "BruteForce": "Credential Access",
    "LateralMovement": "Lateral Movement",
    "Botnet": "Command and Control",
    "Exfiltration": "Exfiltration",
    "DDoS": "Impact",
}


# ---------------------------------------------------------------------------
# IOC reputation → score mapping
# ---------------------------------------------------------------------------

_IOC_VERDICT_SCORES: dict[str, float] = {
    "malicious": 95.0,
    "suspicious": 55.0,
    "clean": 5.0,
    "unknown": 30.0,
}


# ---------------------------------------------------------------------------
# Temporal alert memory (in-process sliding window)
# ---------------------------------------------------------------------------

class TemporalMemory:
    """Track recent alerts by source IP to compute temporal risk multiplier.
    If the same source IP triggers multiple alerts within `window_sec`,
    the risk multiplier increases (indicating a sustained/escalating attack)."""

    def __init__(self, window_sec: float = 300.0, max_multiplier: float = 2.0):
        self.window_sec = window_sec
        self.max_multiplier = max_multiplier
        self._history: dict[str, list[float]] = defaultdict(list)

    def record_and_get_multiplier(self, source_ip: str | None) -> float:
        if not source_ip:
            return 1.0
        now = time.time()
        # Prune old entries
        self._history[source_ip] = [
            t for t in self._history[source_ip]
            if now - t < self.window_sec
        ]
        self._history[source_ip].append(now)
        count = len(self._history[source_ip])
        if count <= 1:
            return 1.0
        # Logarithmic escalation: 2 alerts → 1.3x, 5 alerts → 1.7x, 10+ → 2.0x
        import math
        multiplier = 1.0 + min(math.log2(count) * 0.3, self.max_multiplier - 1.0)
        return round(multiplier, 2)

    def get_alert_count(self, source_ip: str | None) -> int:
        if not source_ip:
            return 0
        now = time.time()
        return sum(1 for t in self._history.get(source_ip, []) if now - t < self.window_sec)


# ---------------------------------------------------------------------------
# Asset criticality tiers
# ---------------------------------------------------------------------------

ASSET_TIERS: dict[str, float] = {
    "crown_jewel": 2.0,    # Domain controllers, DB servers, payment systems
    "high": 1.5,           # Application servers, email gateways
    "standard": 1.0,       # Workstations, printers
    "low": 0.8,            # IoT devices, guest network
}


# ---------------------------------------------------------------------------
# Threat Score computation
# ---------------------------------------------------------------------------

@dataclass
class ThreatScore:
    """Composite threat score with full breakdown for analyst transparency."""
    score: float                     # 0-100 composite score
    severity: str                    # CRITICAL / HIGH / MEDIUM / LOW
    sla_recommendation: str          # CERT-In aligned SLA guidance
    breakdown: dict[str, float]      # Individual dimension scores
    explanation: str                 # Human-readable summary
    temporal_multiplier: float       # How much temporal context escalated
    asset_multiplier: float          # How much asset criticality escalated
    alert_count_window: int          # How many alerts from this source recently


# Singleton temporal memory (shared across the process lifetime)
_temporal_memory = TemporalMemory()


# Default dimension weights (sum to 1.0)
DEFAULT_WEIGHTS = {
    "ml_confidence": 0.35,
    "ioc_reputation": 0.25,
    "attack_position": 0.25,
    "base_severity": 0.15,
}


def compute_threat_score(
    predicted_label: str,
    ml_confidence: float,
    ioc_verdict: str | None = None,
    source_ip: str | None = None,
    asset_tier: str = "standard",
    weights: dict[str, float] | None = None,
) -> ThreatScore:
    """Compute the composite 0-100 threat score for an alert.

    Parameters
    ----------
    predicted_label : str
        ML classifier output (e.g. 'DDoS', 'PortScan', 'BENIGN')
    ml_confidence : float
        Classifier confidence (0.0 - 1.0)
    ioc_verdict : str or None
        IOC enrichment verdict ('malicious', 'suspicious', 'clean', 'unknown')
    source_ip : str or None
        Source IP for temporal correlation
    asset_tier : str
        Asset criticality tier ('crown_jewel', 'high', 'standard', 'low')
    weights : dict or None
        Override default dimension weights
    """
    w = weights or DEFAULT_WEIGHTS

    # --- Dimension 1: ML confidence score ---
    if predicted_label == "BENIGN":
        # For benign, high confidence = LOW risk
        ml_score = (1.0 - ml_confidence) * 100.0
    else:
        ml_score = ml_confidence * 100.0

    # --- Dimension 2: IOC reputation ---
    ioc_score = _IOC_VERDICT_SCORES.get(ioc_verdict or "unknown", 30.0)

    # --- Dimension 3: MITRE ATT&CK kill-chain position ---
    tactic = _LABEL_TACTIC.get(predicted_label, "")
    attack_score = _TACTIC_WEIGHTS.get(tactic, 15.0)

    # --- Dimension 4: Base severity from label ---
    _LABEL_BASE: dict[str, float] = {
        "BENIGN": 5.0,
        "PortScan": 30.0,
        "BruteForce": 55.0,
        "WebAttack": 70.0,
        "LateralMovement": 75.0,
        "DDoS": 80.0,
        "Botnet": 90.0,
        "Exfiltration": 95.0,
    }
    base_score = _LABEL_BASE.get(predicted_label, 40.0)

    # --- Weighted composite ---
    raw_score = (
        ml_score * w["ml_confidence"]
        + ioc_score * w["ioc_reputation"]
        + attack_score * w["attack_position"]
        + base_score * w["base_severity"]
    )

    # --- Multipliers ---
    temporal_mult = _temporal_memory.record_and_get_multiplier(source_ip)
    asset_mult = ASSET_TIERS.get(asset_tier, 1.0)

    final_score = min(raw_score * temporal_mult * asset_mult, 100.0)
    final_score = round(final_score, 1)

    severity, sla = _severity_band(final_score)

    # --- Build explanation ---
    parts = [f"Threat Score: {final_score}/100 ({severity})"]
    parts.append(f"  ML detection: {predicted_label} @ {ml_confidence:.0%} -> {ml_score:.0f}/100")
    if ioc_verdict:
        parts.append(f"  IOC reputation: {ioc_verdict} -> {ioc_score:.0f}/100")
    if tactic:
        parts.append(f"  Kill-chain position: {tactic} -> {attack_score:.0f}/100")
    if temporal_mult > 1.0:
        alert_count = _temporal_memory.get_alert_count(source_ip)
        parts.append(f"  Temporal escalation: {alert_count} alerts in 5min -> {temporal_mult}x multiplier")
    if asset_mult != 1.0:
        parts.append(f"  Asset criticality: {asset_tier} -> {asset_mult}x multiplier")
    parts.append(f"  SLA: {sla}")

    return ThreatScore(
        score=final_score,
        severity=severity,
        sla_recommendation=sla,
        breakdown={
            "ml_confidence_score": round(ml_score, 1),
            "ioc_reputation_score": round(ioc_score, 1),
            "attack_position_score": round(attack_score, 1),
            "base_severity_score": round(base_score, 1),
            "raw_composite": round(raw_score, 1),
        },
        explanation="\n".join(parts),
        temporal_multiplier=temporal_mult,
        asset_multiplier=asset_mult,
        alert_count_window=_temporal_memory.get_alert_count(source_ip),
    )


def reset_temporal_memory() -> None:
    """Reset the temporal memory (useful for tests)."""
    global _temporal_memory
    _temporal_memory = TemporalMemory()


if __name__ == "__main__":
    # Demo: escalating threat scores from the same source
    for i in range(3):
        ts = compute_threat_score(
            predicted_label="DDoS",
            ml_confidence=0.94,
            ioc_verdict="malicious",
            source_ip="185.220.101.45",
            asset_tier="high",
        )
        print(ts.explanation)
        print()
