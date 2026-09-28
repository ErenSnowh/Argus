"""
pcap_forensics.py
-------------------
Advanced PCAP triage and forensics utilities for ARGUS.

Capabilities:
  1. Basic PCAP summary (packet counts, protocol mix, top talkers)
  2. DNS tunneling detection (high-entropy subdomain analysis)
  3. C2 beaconing detection (periodic callback pattern analysis)
  4. Payload anomaly flags (non-standard protocols on standard ports)
  5. File hash verification against known-malicious sets

These forensics capabilities are exposed as MCP tools that the
Forensics Agent calls during investigation.
"""

from __future__ import annotations

import hashlib
import logging
import math
import statistics
from collections import Counter
from pathlib import Path

# Suppress harmless Scapy runtime warnings when libpcap/Npcap is not installed
import warnings
logging.getLogger("scapy").setLevel(logging.ERROR)
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="scapy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scapy")

from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.dns import DNS, DNSQR
from scapy.utils import rdpcap


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def analyze_pcap_summary(pcap_path: str, max_packets: int = 20000) -> dict:
    """Return a structured, analyst-readable summary of a PCAP file:
    packet/byte counts, top talkers, port distribution, protocol mix, and
    capture duration. Designed to be cheap enough to run synchronously
    inside an agent tool call for files in the low tens of thousands of
    packets; larger captures should be pre-filtered (e.g. with tshark)
    before being handed to this tool.
    """
    p = Path(pcap_path)
    if not p.exists():
        return {"error": f"file not found: {pcap_path}"}

    file_hash = sha256_file(p)
    packets = rdpcap(str(p))
    if len(packets) > max_packets:
        packets = packets[:max_packets]

    src_ips: Counter = Counter()
    dst_ips: Counter = Counter()
    dst_ports: Counter = Counter()
    protocols: Counter = Counter()
    total_bytes = 0
    timestamps = []

    for pkt in packets:
        total_bytes += len(pkt)
        if hasattr(pkt, "time"):
            timestamps.append(float(pkt.time))
        if IP in pkt:
            src_ips[pkt[IP].src] += 1
            dst_ips[pkt[IP].dst] += 1
            if TCP in pkt:
                protocols["TCP"] += 1
                dst_ports[pkt[TCP].dport] += 1
            elif UDP in pkt:
                protocols["UDP"] += 1
                dst_ports[pkt[UDP].dport] += 1
            else:
                protocols["OTHER_IP"] += 1
        else:
            protocols["NON_IP"] += 1

    duration = (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0

    # Run deep analysis
    dns_tunneling = detect_dns_tunneling(packets)
    beaconing = detect_beaconing(packets, timestamps)
    payload_anomalies = detect_payload_anomalies(packets)

    result = {
        "file": str(p.name),
        "sha256": file_hash,
        "packet_count": len(packets),
        "total_bytes": total_bytes,
        "duration_sec": round(duration, 3),
        "top_src_ips": src_ips.most_common(5),
        "top_dst_ips": dst_ips.most_common(5),
        "top_dst_ports": dst_ports.most_common(10),
        "protocol_mix": dict(protocols),
        "unique_dst_ports_contacted": len(dst_ports),
    }

    # Attach deep forensics results
    if dns_tunneling["suspicious"]:
        result["dns_tunneling"] = dns_tunneling
    if beaconing["beaconing_detected"]:
        result["beaconing"] = beaconing
    if payload_anomalies["anomalies_found"]:
        result["payload_anomalies"] = payload_anomalies

    # Overall forensic risk assessment
    risk_flags = []
    if dns_tunneling["suspicious"]:
        risk_flags.append("DNS_TUNNELING")
    if beaconing["beaconing_detected"]:
        risk_flags.append("C2_BEACONING")
    if payload_anomalies["anomalies_found"]:
        risk_flags.append("PAYLOAD_ANOMALY")
    result["forensic_risk_flags"] = risk_flags

    return result


# ---------------------------------------------------------------------------
# DNS Tunneling Detection
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string (higher = more random = suspicious)."""
    if not s:
        return 0.0
    freq: Counter = Counter(s)
    length = len(s)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in freq.values()
    )


def detect_dns_tunneling(packets, entropy_threshold: float = 3.8, length_threshold: int = 40) -> dict:
    """Detect potential DNS tunneling by analyzing DNS query names.

    DNS tunneling encodes data in DNS query labels, producing:
    - Unusually long subdomain labels
    - High Shannon entropy (random-looking characters)
    - High query volume to a single authoritative domain

    Parameters
    ----------
    entropy_threshold : float
        Shannon entropy above this value is flagged (normal domains ≈ 2.5-3.5).
    length_threshold : int
        Query names longer than this are flagged.
    """
    dns_queries = []
    query_domains: Counter = Counter()
    suspicious_queries = []

    for pkt in packets:
        if DNS in pkt and pkt[DNS].qr == 0 and DNSQR in pkt:  # DNS query (not response)
            try:
                qname = pkt[DNSQR].qname.decode("utf-8", errors="ignore").rstrip(".")
            except (AttributeError, UnicodeDecodeError):
                continue

            dns_queries.append(qname)

            # Extract base domain (last 2 labels)
            labels = qname.split(".")
            base_domain = ".".join(labels[-2:]) if len(labels) >= 2 else qname
            query_domains[base_domain] += 1

            # Check for tunneling indicators
            entropy = _shannon_entropy(qname)
            is_suspicious = False
            reasons = []

            if entropy > entropy_threshold:
                is_suspicious = True
                reasons.append(f"high entropy ({entropy:.2f})")

            if len(qname) > length_threshold:
                is_suspicious = True
                reasons.append(f"long query ({len(qname)} chars)")

            # Check for hex/base64-like subdomain labels
            if len(labels) > 2:
                subdomain = ".".join(labels[:-2])
                if len(subdomain) > 20 and _shannon_entropy(subdomain) > 3.5:
                    is_suspicious = True
                    reasons.append("high-entropy subdomain (possible data exfiltration)")

            if is_suspicious:
                suspicious_queries.append({
                    "query": qname[:80],  # Truncate for display
                    "entropy": round(entropy, 2),
                    "length": len(qname),
                    "reasons": reasons,
                })

    return {
        "suspicious": len(suspicious_queries) > 0,
        "total_dns_queries": len(dns_queries),
        "unique_domains": len(query_domains),
        "top_queried_domains": query_domains.most_common(5),
        "suspicious_queries": suspicious_queries[:10],  # Limit output
        "verdict": (
            f"DNS tunneling suspected: {len(suspicious_queries)} suspicious queries detected"
            if suspicious_queries
            else "No DNS tunneling indicators found"
        ),
    }


# ---------------------------------------------------------------------------
# C2 Beaconing Detection
# ---------------------------------------------------------------------------

def detect_beaconing(
    packets,
    timestamps: list[float] | None = None,
    jitter_threshold: float = 0.15,
    min_beacons: int = 5,
) -> dict:
    """Detect C2 beaconing patterns in network traffic.

    C2 implants (e.g., Cobalt Strike, Metasploit Meterpreter) typically
    beacon at regular intervals. We detect this by:
    1. Grouping connections by (src_ip, dst_ip) pair
    2. Computing inter-arrival times for each pair
    3. Flagging pairs with low coefficient of variation (regular timing)

    Parameters
    ----------
    jitter_threshold : float
        CoV (std/mean) below this threshold is flagged as beaconing.
        Real C2 tools typically have 0-10% jitter.
    min_beacons : int
        Minimum number of connections to consider for beaconing analysis.
    """
    # Group packet timestamps by (src, dst) pair
    pair_times: dict[tuple[str, str], list[float]] = {}

    for pkt in packets:
        if IP in pkt and hasattr(pkt, "time"):
            src = pkt[IP].src
            dst = pkt[IP].dst
            pair = (src, dst)
            if pair not in pair_times:
                pair_times[pair] = []
            pair_times[pair].append(float(pkt.time))

    beaconing_pairs = []

    for pair, times in pair_times.items():
        if len(times) < min_beacons:
            continue

        times.sort()
        intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]

        if not intervals:
            continue

        mean_interval = statistics.mean(intervals)
        if mean_interval < 0.01:  # Too fast to be meaningful beaconing
            continue

        std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
        cov = std_interval / mean_interval if mean_interval > 0 else float("inf")

        if cov < jitter_threshold:
            beaconing_pairs.append({
                "source": pair[0],
                "destination": pair[1],
                "beacon_count": len(times),
                "mean_interval_sec": round(mean_interval, 2),
                "jitter_cov": round(cov, 3),
                "estimated_sleep_sec": round(mean_interval, 1),
            })

    beaconing_pairs.sort(key=lambda x: x["beacon_count"], reverse=True)

    return {
        "beaconing_detected": len(beaconing_pairs) > 0,
        "beaconing_pairs": beaconing_pairs[:5],  # Top 5 most active
        "total_pairs_analyzed": len(pair_times),
        "verdict": (
            f"C2 beaconing suspected: {len(beaconing_pairs)} pair(s) with regular callback intervals"
            if beaconing_pairs
            else "No beaconing patterns detected"
        ),
    }


# ---------------------------------------------------------------------------
# Payload Anomaly Detection
# ---------------------------------------------------------------------------

# Standard port → expected protocol mapping
_STANDARD_PORTS: dict[int, str] = {
    80: "HTTP",
    443: "HTTPS/TLS",
    53: "DNS",
    22: "SSH",
    21: "FTP",
    25: "SMTP",
    3389: "RDP",
}


def detect_payload_anomalies(packets) -> dict:
    """Detect non-standard payloads on standard ports.

    Attackers often use standard ports (80, 443) for C2 traffic to evade
    basic firewall rules. This detector flags connections where the payload
    doesn't match the expected protocol for the port.
    """
    anomalies = []

    for pkt in packets:
        if IP not in pkt or TCP not in pkt:
            continue

        dport = pkt[TCP].dport
        expected = _STANDARD_PORTS.get(dport)
        if not expected:
            continue

        # Extract payload
        payload = bytes(pkt[TCP].payload) if pkt[TCP].payload else b""
        if len(payload) < 4:
            continue

        anomaly = None

        # Port 80 should have HTTP
        if dport == 80:
            http_methods = [b"GET", b"POST", b"PUT", b"DELETE", b"HEAD", b"OPTIONS", b"HTTP"]
            if not any(payload[:10].startswith(m) for m in http_methods):
                anomaly = f"Non-HTTP payload on port 80 (first bytes: {payload[:8].hex()})"

        # Port 443 should have TLS ClientHello (0x16 0x03)
        elif dport == 443:
            if not (payload[0] == 0x16 and payload[1] == 0x03):
                anomaly = f"Non-TLS payload on port 443 (first bytes: {payload[:8].hex()})"

        # Port 53 should have DNS
        elif dport == 53:
            if DNS not in pkt:
                anomaly = f"Non-DNS payload on port 53 (first bytes: {payload[:8].hex()})"

        # Port 22 should have SSH banner
        elif dport == 22:
            if not payload[:4].startswith(b"SSH-"):
                # Only flag if this looks like it could be the connection start
                if pkt[TCP].flags & 0x08:  # PSH flag set (data push)
                    anomaly = f"Non-SSH payload on port 22"

        if anomaly:
            anomalies.append({
                "source": pkt[IP].src if IP in pkt else "?",
                "destination": pkt[IP].dst if IP in pkt else "?",
                "dst_port": dport,
                "expected_protocol": expected,
                "anomaly": anomaly,
            })

    return {
        "anomalies_found": len(anomalies) > 0,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies[:10],  # Limit output
        "verdict": (
            f"Payload anomalies detected: {len(anomalies)} connection(s) with unexpected protocols"
            if anomalies
            else "No payload anomalies detected"
        ),
    }


def verify_file_hash(file_path: str, known_malicious_hashes: dict[str, str] | None = None) -> dict:
    """Hash a file and check it against a small known-malicious hash set.
    `known_malicious_hashes` maps sha256 -> a label/description; callers can
    pass in a hash-set pulled from MalwareBazaar / internal IOC feeds for
    production use. Includes the EICAR test file hash by default as a safe,
    universally-known positive-control sample.
    """
    p = Path(file_path)
    if not p.exists():
        return {"error": f"file not found: {file_path}"}

    try:
        digest = sha256_file(p)
    except OSError as e:
        return {
            "file": p.name,
            "error": f"unable to read file ({e})",
            "match_found": False,
        }
    known = known_malicious_hashes or {
        # EICAR standard antivirus test file -- safe, well-known positive control
        "275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f": "EICAR-Test-File (benign test signature)",
    }
    match = known.get(digest)
    return {
        "file": p.name,
        "sha256": digest,
        "match_found": match is not None,
        "match_label": match,
    }


if __name__ == "__main__":
    # Quick self-test against the EICAR string written to a temp file.
    import tempfile

    eicar = rb"X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(eicar)
        tmp_path = f.name
    print(verify_file_hash(tmp_path))

