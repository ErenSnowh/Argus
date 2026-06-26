#!/usr/bin/env python3
"""Generates a small synthetic PCAP (SYN-scan-like traffic) for demo purposes
so the forensics tool and dashboard have a real file to operate on without
requiring the user to supply their own capture."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scapy.layers.inet import IP, TCP
from scapy.utils import wrpcap

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_portscan.pcap"


def main():
    packets = []
    src = "10.0.4.12"
    dst = "10.0.4.50"
    for port in range(20, 220):
        pkt = IP(src=src, dst=dst) / TCP(sport=44000 + (port % 50), dport=port, flags="S")
        packets.append(pkt)
    wrpcap(str(OUT), packets)
    print(f"Wrote {len(packets)} packets to {OUT}")


if __name__ == "__main__":
    main()
