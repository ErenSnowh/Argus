"""
correlation.py
-----------------
Alert Correlation Engine — groups isolated alerts into coherent attack
campaigns by detecting temporal, source-based, and kill-chain patterns.

Why this matters:
    Attackers don't launch single-flow attacks. A port scan precedes
    exploitation. Brute force precedes lateral movement. A national SOC
    tool (NCIIPC/CERT-In) must correlate individual alerts into campaigns
    to understand the full scope of an intrusion and respond appropriately.

Correlation strategies:
    1. Source-IP grouping   — alerts from the same source within a window
    2. Kill-chain staging   — PortScan → BruteForce → Botnet = escalation
    3. Target convergence   — multiple attack types hitting the same target
    4. Campaign scoring     — composite campaign severity from member alerts

The engine maintains an in-memory sliding window of recent alerts. Each new
alert is checked against the window to find correlated siblings. When a
correlation is found, a Campaign object is emitted.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from collections import defaultdict


# ---------------------------------------------------------------------------
# Kill-chain stage ordering (lower number = earlier in the kill chain)
# ---------------------------------------------------------------------------

KILL_CHAIN_STAGES: dict[str, int] = {
    "BENIGN": 0,
    "PortScan": 1,       # Reconnaissance
    "WebAttack": 2,       # Initial Access
    "BruteForce": 3,      # Credential Access
    "LateralMovement": 4, # Lateral Movement
    "Botnet": 5,          # Command & Control
    "Exfiltration": 6,    # Exfiltration
    "DDoS": 7,            # Impact
}

KILL_CHAIN_TACTICS: dict[str, str] = {
    "PortScan": "Reconnaissance",
    "WebAttack": "Initial Access",
    "BruteForce": "Credential Access",
    "LateralMovement": "Lateral Movement",
    "Botnet": "Command and Control",
    "Exfiltration": "Exfiltration",
    "DDoS": "Impact",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AlertRecord:
    """A single alert stored in the correlation window."""
    alert_id: str
    timestamp: float
    source_ip: str | None
    target_ip: str | None
    predicted_label: str
    confidence: float
    threat_score: float


@dataclass
class Campaign:
    """A correlated group of alerts forming an attack campaign."""
    campaign_id: str
    alerts: list[AlertRecord]
    source_ips: set[str]
    target_ips: set[str]
    attack_types: list[str]
    kill_chain_progression: list[str]
    campaign_severity: str
    campaign_score: float
    first_seen: float
    last_seen: float
    duration_sec: float
    description: str


# ---------------------------------------------------------------------------
# Correlation Engine
# ---------------------------------------------------------------------------

class CorrelationEngine:
    """In-memory sliding-window alert correlator.

    Parameters
    ----------
    window_sec : float
        How far back to look for correlated alerts (default 5 minutes).
    min_alerts_for_campaign : int
        Minimum number of correlated alerts to declare a campaign.
    """

    def __init__(self, window_sec: float = 300.0, min_alerts_for_campaign: int = 2):
        self.window_sec = window_sec
        self.min_alerts_for_campaign = min_alerts_for_campaign
        self._alerts: list[AlertRecord] = []
        self._campaigns: list[Campaign] = []

    def _prune(self) -> None:
        """Remove alerts outside the sliding window."""
        cutoff = time.time() - self.window_sec
        self._alerts = [a for a in self._alerts if a.timestamp > cutoff]

    def ingest_alert(
        self,
        source_ip: str | None,
        target_ip: str | None,
        predicted_label: str,
        confidence: float,
        threat_score: float = 0.0,
    ) -> AlertRecord:
        """Add a new alert to the correlation window and return the record."""
        self._prune()
        record = AlertRecord(
            alert_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            source_ip=source_ip,
            target_ip=target_ip,
            predicted_label=predicted_label,
            confidence=confidence,
            threat_score=threat_score,
        )
        self._alerts.append(record)
        return record

    def correlate(self, current_alert: AlertRecord) -> Campaign | None:
        """Check if the current alert correlates with existing alerts
        in the window. Returns a Campaign if a pattern is found."""
        self._prune()

        # Find correlated alerts (same source IP OR same target IP)
        correlated: list[AlertRecord] = []
        for alert in self._alerts:
            if alert.alert_id == current_alert.alert_id:
                continue
            # Source-IP correlation
            if (current_alert.source_ip and alert.source_ip
                    and current_alert.source_ip == alert.source_ip):
                correlated.append(alert)
            # Target-IP correlation
            elif (current_alert.target_ip and alert.target_ip
                  and current_alert.target_ip == alert.target_ip):
                correlated.append(alert)

        if len(correlated) < self.min_alerts_for_campaign - 1:
            return None

        # Build campaign from correlated alerts + current
        all_alerts = correlated + [current_alert]
        all_alerts.sort(key=lambda a: a.timestamp)

        source_ips = {a.source_ip for a in all_alerts if a.source_ip}
        target_ips = {a.target_ip for a in all_alerts if a.target_ip}
        attack_types = list(dict.fromkeys(
            a.predicted_label for a in all_alerts if a.predicted_label != "BENIGN"
        ))

        # Detect kill-chain progression
        kill_chain = self._detect_kill_chain(all_alerts)

        # Campaign severity scoring
        campaign_score = self._compute_campaign_score(all_alerts, kill_chain)
        campaign_severity = self._score_to_severity(campaign_score)

        first_seen = min(a.timestamp for a in all_alerts)
        last_seen = max(a.timestamp for a in all_alerts)

        description = self._build_description(
            all_alerts, source_ips, target_ips, attack_types, kill_chain
        )

        campaign = Campaign(
            campaign_id=f"CMP-{str(uuid.uuid4())[:8].upper()}",
            alerts=all_alerts,
            source_ips=source_ips,
            target_ips=target_ips,
            attack_types=attack_types,
            kill_chain_progression=kill_chain,
            campaign_severity=campaign_severity,
            campaign_score=campaign_score,
            first_seen=first_seen,
            last_seen=last_seen,
            duration_sec=round(last_seen - first_seen, 1),
            description=description,
        )
        self._campaigns.append(campaign)
        return campaign

    def _detect_kill_chain(self, alerts: list[AlertRecord]) -> list[str]:
        """Detect kill-chain stage progression in a set of alerts."""
        stages_seen: dict[str, int] = {}
        for alert in alerts:
            stage = KILL_CHAIN_STAGES.get(alert.predicted_label, -1)
            if stage > 0 and alert.predicted_label not in stages_seen:
                stages_seen[alert.predicted_label] = stage

        if len(stages_seen) < 2:
            return []

        # Sort by kill-chain order
        progression = sorted(stages_seen.keys(), key=lambda l: stages_seen[l])

        # Check if this is an actual progression (stages increase)
        stage_values = [stages_seen[l] for l in progression]
        is_progression = all(
            stage_values[i] < stage_values[i + 1]
            for i in range(len(stage_values) - 1)
        )

        if is_progression:
            return [
                f"{label} ({KILL_CHAIN_TACTICS.get(label, '?')})"
                for label in progression
            ]
        return []

    def _compute_campaign_score(
        self, alerts: list[AlertRecord], kill_chain: list[str]
    ) -> float:
        """Compute a campaign-level severity score."""
        if not alerts:
            return 0.0

        # Base: max threat score among member alerts
        max_score = max(a.threat_score for a in alerts)

        # Bonus for kill-chain progression (multi-stage attacks are more severe)
        chain_bonus = len(kill_chain) * 8.0

        # Bonus for alert count (more alerts = sustained attack)
        import math
        count_bonus = min(math.log2(len(alerts)) * 5.0, 15.0)

        return min(max_score + chain_bonus + count_bonus, 100.0)

    @staticmethod
    def _score_to_severity(score: float) -> str:
        if score >= 90:
            return "CRITICAL"
        if score >= 70:
            return "HIGH"
        if score >= 40:
            return "MEDIUM"
        return "LOW"

    def _build_description(
        self,
        alerts: list[AlertRecord],
        source_ips: set[str],
        target_ips: set[str],
        attack_types: list[str],
        kill_chain: list[str],
    ) -> str:
        parts = [
            f"Campaign detected: {len(alerts)} correlated alerts",
            f"from {len(source_ips)} source(s) targeting {len(target_ips)} host(s).",
        ]
        if attack_types:
            parts.append(f"Attack types observed: {', '.join(attack_types)}.")
        if kill_chain:
            parts.append(
                f"Kill-chain progression detected: {' -> '.join(kill_chain)}. "
                "This indicates a multi-stage attack — the adversary is advancing "
                "through the kill chain."
            )
        else:
            parts.append("No kill-chain progression detected (parallel/repeated attacks).")

        return " ".join(parts)

    def get_recent_campaigns(self, n: int = 10) -> list[Campaign]:
        """Return the N most recent campaigns."""
        return sorted(self._campaigns, key=lambda c: c.last_seen, reverse=True)[:n]

    def get_alert_count(self) -> int:
        self._prune()
        return len(self._alerts)

    def get_all_alerts(self) -> list[AlertRecord]:
        self._prune()
        return list(self._alerts)

    def to_dict(self, campaign: Campaign) -> dict:
        """Serialize a Campaign to a JSON-safe dict."""
        return {
            "campaign_id": campaign.campaign_id,
            "alert_count": len(campaign.alerts),
            "source_ips": sorted(campaign.source_ips),
            "target_ips": sorted(campaign.target_ips),
            "attack_types": campaign.attack_types,
            "kill_chain_progression": campaign.kill_chain_progression,
            "campaign_severity": campaign.campaign_severity,
            "campaign_score": round(campaign.campaign_score, 1),
            "first_seen": campaign.first_seen,
            "last_seen": campaign.last_seen,
            "duration_sec": campaign.duration_sec,
            "description": campaign.description,
        }


# Module-level singleton for use by MCP tools and agents
correlation_engine = CorrelationEngine()


def correlate_alerts(
    source_ip: str | None,
    target_ip: str | None,
    predicted_label: str,
    confidence: float,
    threat_score: float = 0.0,
) -> dict:
    """Public API: ingest an alert and check for campaign correlation.

    Returns a dict with the alert record and, if found, the campaign details.
    """
    record = correlation_engine.ingest_alert(
        source_ip=source_ip,
        target_ip=target_ip,
        predicted_label=predicted_label,
        confidence=confidence,
        threat_score=threat_score,
    )
    campaign = correlation_engine.correlate(record)

    result: dict = {
        "alert_id": record.alert_id,
        "alerts_in_window": correlation_engine.get_alert_count(),
        "campaign_detected": campaign is not None,
    }
    if campaign:
        result["campaign"] = correlation_engine.to_dict(campaign)
    return result


if __name__ == "__main__":
    import json

    # Simulate a multi-stage attack from the same source
    stages = [
        ("PortScan", 0.92, 35.0),
        ("BruteForce", 0.88, 62.0),
        ("Botnet", 0.95, 88.0),
    ]
    for label, conf, score in stages:
        result = correlate_alerts(
            source_ip="185.220.101.45",
            target_ip="10.0.4.50",
            predicted_label=label,
            confidence=conf,
            threat_score=score,
        )
        print(json.dumps(result, indent=2))
        print()
