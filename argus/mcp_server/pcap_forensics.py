"""
pcap_forensics.py
-------------------
Lightweight PCAP triage utilities -- a small, focused slice of the author's
"CTF Forensics Automation Toolkit" (PCAP metadata parsing, protocol/IP
summarization) repurposed as an MCP tool the Forensics Agent can call when
the Triage Agent flags a suspicious flow and an analyst (or the agent itself)
wants a first-pass look at the underlying capture.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from scapy.layers.inet import IP, TCP, UDP
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

    return {
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

    digest = sha256_file(p)
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
