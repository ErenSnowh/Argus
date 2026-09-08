"""
features.py
-------------
Extended feature schema for the World Model covering both flow-level AND
packet-level attributes as required by SIH26153 §1.

Flow-level features capture aggregate behaviour (a SYN flood), while
packet-level features expose timing and sequencing patterns (a slow
reconnaissance scan designed to evade flow-based thresholds).

This module also provides functions to:
  - Extract features from PCAP files (using Scapy)
  - Generate synthetic temporal attack sequences for training
  - Build the combined feature matrix from any supported source
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Combined feature schema: 30 flow-level + 8 packet-level = 38 features
# ---------------------------------------------------------------------------

# Original 24 flow-level features from the RF classifier
FLOW_FEATURE_COLUMNS = [
    "flow_duration_ms",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "fwd_packet_len_mean",
    "fwd_packet_len_std",
    "bwd_packet_len_mean",
    "bwd_packet_len_std",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "flow_iat_mean",
    "flow_iat_std",
    "fwd_iat_mean",
    "bwd_iat_mean",
    "syn_flag_count",
    "ack_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "fin_flag_count",
    "unique_dst_ports_per_src",
    "packets_per_flow",
    "avg_packet_size",
    "down_up_ratio",
]

# Additional flow-level features for world model
EXTENDED_FLOW_COLUMNS = [
    "urg_flag_count",
    "flow_iat_max",
    "fwd_iat_std",
    "bwd_iat_std",
    "fwd_header_len",
    "bwd_header_len",
]

# Packet-level features (required by SIH26153 §1)
PACKET_LEVEL_COLUMNS = [
    "ttl_mean",
    "ttl_std",
    "tcp_window_mean",
    "tcp_window_std",
    "ip_fragment_flag_count",
    "payload_size_mean",
    "payload_size_std",
    "retransmission_count",
]

# Topology-derived features (P1-B: lightweight graph signals without full GNN)
# These capture host-to-host communication patterns critical for detecting
# lateral movement (T1021), internal reconnaissance (T1046), and propagation.
TOPOLOGY_FEATURE_COLUMNS = [
    "src_fanout",              # Unique destinations contacted by source
    "dst_fanin",               # Unique sources contacting destination
    "unique_dst_hosts",        # Distinct destination IPs in window
    "unique_src_hosts",        # Distinct source IPs in window
    "new_host_edges",          # New host-to-host connections not seen before
    "cross_subnet_edges",      # Connections crossing /24 subnet boundaries
    "new_dst_ports",           # Newly observed destination ports in window
    "connection_repetition",   # Repeated connection attempts to same dst
]

# Full world model feature set
WORLD_MODEL_FEATURES = (
    FLOW_FEATURE_COLUMNS
    + EXTENDED_FLOW_COLUMNS
    + PACKET_LEVEL_COLUMNS
    + TOPOLOGY_FEATURE_COLUMNS
)

# All attack labels including new stages for full kill-chain coverage
ATTACK_LABELS = [
    "BENIGN",
    "PortScan",       # Reconnaissance
    "WebAttack",      # Initial Access
    "BruteForce",     # Credential Access
    "LateralMovement",# Lateral Movement
    "Botnet",         # Command & Control
    "Exfiltration",   # Exfiltration
    "DDoS",           # Impact
]

# MITRE ATT&CK stage ordering (kill-chain position index)
ATTACK_STAGE_INDEX = {
    "BENIGN": 0,
    "PortScan": 1,         # Reconnaissance
    "WebAttack": 2,        # Initial Access
    "BruteForce": 3,       # Credential Access
    "LateralMovement": 4,  # Lateral Movement
    "Botnet": 5,           # Command & Control
    "Exfiltration": 6,     # Exfiltration
    "DDoS": 7,             # Impact
}

ATTACK_STAGE_NAMES = {
    0: "BENIGN",
    1: "Reconnaissance",
    2: "Initial Access",
    3: "Credential Access",
    4: "Lateral Movement",
    5: "Command & Control",
    6: "Exfiltration",
    7: "Impact",
}

NUM_FEATURES = len(WORLD_MODEL_FEATURES)
NUM_ATTACK_STAGES = len(ATTACK_LABELS)


# ---------------------------------------------------------------------------
# Synthetic temporal data generation for world model training
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _generate_flow_vector(label: str, rng: np.random.Generator) -> np.ndarray:
    """Generate a single feature vector for a given label.

    The statistics are designed to mimic the distributional properties of
    CIC-IDS-2017/2018 flow records for each attack type.
    """
    profiles = {
        "BENIGN": {
            "flow_duration_ms": (800, 300), "total_fwd_packets": (12, 4),
            "total_bwd_packets": (11, 4), "total_fwd_bytes": (1400, 500),
            "total_bwd_bytes": (1300, 480), "fwd_packet_len_mean": (450, 120),
            "fwd_packet_len_std": (80, 30), "bwd_packet_len_mean": (420, 110),
            "bwd_packet_len_std": (75, 28), "flow_bytes_per_sec": (3500, 1200),
            "flow_packets_per_sec": (15, 6), "flow_iat_mean": (60, 20),
            "flow_iat_std": (20, 8), "fwd_iat_mean": (55, 18),
            "bwd_iat_mean": (58, 19), "syn_flag_count": (1, 0.8),
            "ack_flag_count": (10, 3), "rst_flag_count": (0.2, 0.3),
            "psh_flag_count": (3, 1.5), "fin_flag_count": (1, 0.7),
            "unique_dst_ports_per_src": (2, 1), "packets_per_flow": (23, 7),
            "avg_packet_size": (430, 90), "down_up_ratio": (0.95, 0.2),
            "urg_flag_count": (0, 0.1), "flow_iat_max": (200, 80),
            "fwd_iat_std": (18, 7), "bwd_iat_std": (19, 7),
            "fwd_header_len": (20, 2), "bwd_header_len": (20, 2),
            "ttl_mean": (64, 4), "ttl_std": (1, 0.5),
            "tcp_window_mean": (65535, 5000), "tcp_window_std": (1000, 500),
            "ip_fragment_flag_count": (0, 0.1), "payload_size_mean": (430, 90),
            "payload_size_std": (80, 30), "retransmission_count": (0.5, 0.5),
            # Topology features — benign: low diversity, few new edges
            "src_fanout": (2, 1), "dst_fanin": (2, 1),
            "unique_dst_hosts": (3, 1.5), "unique_src_hosts": (2, 1),
            "new_host_edges": (0.5, 0.5), "cross_subnet_edges": (0.2, 0.3),
            "new_dst_ports": (0.5, 0.5), "connection_repetition": (1, 0.5),
        },
        "PortScan": {
            "flow_duration_ms": (8, 5), "total_fwd_packets": (2, 1),
            "total_bwd_packets": (1, 0.8), "total_fwd_bytes": (60, 20),
            "total_bwd_bytes": (40, 30), "fwd_packet_len_mean": (40, 8),
            "fwd_packet_len_std": (2, 1), "bwd_packet_len_mean": (40, 20),
            "bwd_packet_len_std": (2, 1), "flow_bytes_per_sec": (8000, 3000),
            "flow_packets_per_sec": (250, 90), "flow_iat_mean": (4, 2),
            "flow_iat_std": (1, 0.5), "fwd_iat_mean": (4, 2),
            "bwd_iat_mean": (6, 3), "syn_flag_count": (1.8, 0.8),
            "ack_flag_count": (0.3, 0.4), "rst_flag_count": (0.9, 0.6),
            "psh_flag_count": (0.1, 0.2), "fin_flag_count": (0.1, 0.2),
            "unique_dst_ports_per_src": (180, 50), "packets_per_flow": (3, 1.5),
            "avg_packet_size": (40, 10), "down_up_ratio": (0.6, 0.3),
            "urg_flag_count": (0, 0.1), "flow_iat_max": (15, 8),
            "fwd_iat_std": (1, 0.5), "bwd_iat_std": (2, 1),
            "fwd_header_len": (20, 1), "bwd_header_len": (20, 1),
            "ttl_mean": (128, 10), "ttl_std": (0.5, 0.3),
            "tcp_window_mean": (1024, 200), "tcp_window_std": (50, 30),
            "ip_fragment_flag_count": (0, 0.1), "payload_size_mean": (0, 2),
            "payload_size_std": (0, 1), "retransmission_count": (0.2, 0.3),
            # Topology — PortScan: very high fanout, many new ports
            "src_fanout": (120, 40), "dst_fanin": (1, 0.5),
            "unique_dst_hosts": (80, 30), "unique_src_hosts": (1, 0.5),
            "new_host_edges": (60, 20), "cross_subnet_edges": (15, 8),
            "new_dst_ports": (150, 50), "connection_repetition": (1, 0.5),
        },
        "BruteForce": {
            "flow_duration_ms": (300, 100), "total_fwd_packets": (9, 3),
            "total_bwd_packets": (9, 3), "total_fwd_bytes": (700, 200),
            "total_bwd_bytes": (650, 190), "fwd_packet_len_mean": (75, 20),
            "fwd_packet_len_std": (8, 4), "bwd_packet_len_mean": (70, 18),
            "bwd_packet_len_std": (8, 4), "flow_bytes_per_sec": (4500, 1500),
            "flow_packets_per_sec": (60, 25), "flow_iat_mean": (15, 6),
            "flow_iat_std": (5, 2), "fwd_iat_mean": (14, 5),
            "bwd_iat_mean": (16, 6), "syn_flag_count": (8, 3),
            "ack_flag_count": (8, 3), "rst_flag_count": (2, 1),
            "psh_flag_count": (7, 2), "fin_flag_count": (0.5, 0.5),
            "unique_dst_ports_per_src": (1, 0.5), "packets_per_flow": (18, 5),
            "avg_packet_size": (72, 15), "down_up_ratio": (0.9, 0.2),
            "urg_flag_count": (0, 0.1), "flow_iat_max": (50, 20),
            "fwd_iat_std": (5, 2), "bwd_iat_std": (6, 2),
            "fwd_header_len": (20, 1), "bwd_header_len": (20, 1),
            "ttl_mean": (64, 5), "ttl_std": (2, 1),
            "tcp_window_mean": (32768, 4000), "tcp_window_std": (500, 200),
            "ip_fragment_flag_count": (0, 0.1), "payload_size_mean": (72, 15),
            "payload_size_std": (8, 4), "retransmission_count": (1, 0.8),
            # Topology — BruteForce: single target, high repetition
            "src_fanout": (1, 0.5), "dst_fanin": (1, 0.5),
            "unique_dst_hosts": (1, 0.5), "unique_src_hosts": (1, 0.5),
            "new_host_edges": (0.2, 0.3), "cross_subnet_edges": (0.5, 0.5),
            "new_dst_ports": (0.5, 0.5), "connection_repetition": (25, 10),
        },
        "WebAttack": {
            "flow_duration_ms": (220, 90), "total_fwd_packets": (6, 2),
            "total_bwd_packets": (5, 2), "total_fwd_bytes": (2200, 800),
            "total_bwd_bytes": (900, 400), "fwd_packet_len_mean": (380, 140),
            "fwd_packet_len_std": (120, 50), "bwd_packet_len_mean": (180, 70),
            "bwd_packet_len_std": (50, 25), "flow_bytes_per_sec": (14000, 5000),
            "flow_packets_per_sec": (50, 20), "flow_iat_mean": (35, 15),
            "flow_iat_std": (12, 5), "fwd_iat_mean": (33, 14),
            "bwd_iat_mean": (38, 16), "syn_flag_count": (1, 0.7),
            "ack_flag_count": (6, 2), "rst_flag_count": (0.3, 0.4),
            "psh_flag_count": (5, 2), "fin_flag_count": (0.8, 0.6),
            "unique_dst_ports_per_src": (1, 0.5), "packets_per_flow": (11, 3),
            "avg_packet_size": (310, 100), "down_up_ratio": (0.45, 0.2),
            "urg_flag_count": (0, 0.1), "flow_iat_max": (80, 35),
            "fwd_iat_std": (12, 5), "bwd_iat_std": (15, 6),
            "fwd_header_len": (20, 2), "bwd_header_len": (20, 2),
            "ttl_mean": (64, 5), "ttl_std": (1, 0.5),
            "tcp_window_mean": (65535, 5000), "tcp_window_std": (2000, 800),
            "ip_fragment_flag_count": (0, 0.1), "payload_size_mean": (310, 100),
            "payload_size_std": (120, 50), "retransmission_count": (0.3, 0.4),
            # Topology — WebAttack: single target, moderate repetition
            "src_fanout": (1, 0.5), "dst_fanin": (2, 1),
            "unique_dst_hosts": (1, 0.5), "unique_src_hosts": (1, 0.5),
            "new_host_edges": (0.3, 0.3), "cross_subnet_edges": (0.3, 0.3),
            "new_dst_ports": (1, 0.5), "connection_repetition": (8, 3),
        },
        "LateralMovement": {
            "flow_duration_ms": (500, 200), "total_fwd_packets": (8, 3),
            "total_bwd_packets": (7, 3), "total_fwd_bytes": (900, 300),
            "total_bwd_bytes": (800, 300), "fwd_packet_len_mean": (110, 40),
            "fwd_packet_len_std": (25, 10), "bwd_packet_len_mean": (115, 40),
            "bwd_packet_len_std": (28, 12), "flow_bytes_per_sec": (3400, 1200),
            "flow_packets_per_sec": (30, 12), "flow_iat_mean": (30, 12),
            "flow_iat_std": (10, 5), "fwd_iat_mean": (28, 10),
            "bwd_iat_mean": (32, 12), "syn_flag_count": (2, 1),
            "ack_flag_count": (7, 3), "rst_flag_count": (0.5, 0.5),
            "psh_flag_count": (5, 2), "fin_flag_count": (1, 0.7),
            "unique_dst_ports_per_src": (3, 1.5), "packets_per_flow": (15, 5),
            "avg_packet_size": (112, 35), "down_up_ratio": (0.88, 0.15),
            "urg_flag_count": (0, 0.1), "flow_iat_max": (100, 40),
            "fwd_iat_std": (10, 4), "bwd_iat_std": (12, 5),
            "fwd_header_len": (20, 2), "bwd_header_len": (20, 2),
            "ttl_mean": (127, 5), "ttl_std": (3, 1.5),
            "tcp_window_mean": (32768, 4000), "tcp_window_std": (800, 300),
            "ip_fragment_flag_count": (0, 0.1), "payload_size_mean": (110, 35),
            "payload_size_std": (25, 10), "retransmission_count": (0.8, 0.6),
            # Topology — LateralMovement: high cross-subnet, many new edges
            "src_fanout": (8, 3), "dst_fanin": (3, 1.5),
            "unique_dst_hosts": (6, 2), "unique_src_hosts": (3, 1.5),
            "new_host_edges": (5, 2), "cross_subnet_edges": (4, 2),
            "new_dst_ports": (3, 1.5), "connection_repetition": (3, 1.5),
        },
        "Botnet": {
            "flow_duration_ms": (5000, 2000), "total_fwd_packets": (4, 2),
            "total_bwd_packets": (4, 2), "total_fwd_bytes": (320, 100),
            "total_bwd_bytes": (300, 95), "fwd_packet_len_mean": (80, 25),
            "fwd_packet_len_std": (10, 5), "bwd_packet_len_mean": (75, 24),
            "bwd_packet_len_std": (10, 5), "flow_bytes_per_sec": (60, 30),
            "flow_packets_per_sec": (0.8, 0.4), "flow_iat_mean": (1200, 400),
            "flow_iat_std": (300, 120), "fwd_iat_mean": (1150, 390),
            "bwd_iat_mean": (1250, 410), "syn_flag_count": (0.5, 0.5),
            "ack_flag_count": (3, 1.5), "rst_flag_count": (0.1, 0.2),
            "psh_flag_count": (2, 1), "fin_flag_count": (0.1, 0.2),
            "unique_dst_ports_per_src": (1, 0.5), "packets_per_flow": (8, 3),
            "avg_packet_size": (78, 20), "down_up_ratio": (0.93, 0.15),
            "urg_flag_count": (0, 0.1), "flow_iat_max": (3000, 1000),
            "fwd_iat_std": (300, 100), "bwd_iat_std": (320, 110),
            "fwd_header_len": (20, 2), "bwd_header_len": (20, 2),
            "ttl_mean": (128, 8), "ttl_std": (0.3, 0.2),
            "tcp_window_mean": (8192, 2000), "tcp_window_std": (200, 100),
            "ip_fragment_flag_count": (0, 0.1), "payload_size_mean": (78, 20),
            "payload_size_std": (10, 5), "retransmission_count": (0.2, 0.3),
            # Topology — Botnet: low fanout, periodic beaconing to C2
            "src_fanout": (1, 0.5), "dst_fanin": (1, 0.5),
            "unique_dst_hosts": (2, 1), "unique_src_hosts": (1, 0.5),
            "new_host_edges": (0.3, 0.3), "cross_subnet_edges": (1, 0.5),
            "new_dst_ports": (0.5, 0.5), "connection_repetition": (15, 5),
        },
        "Exfiltration": {
            "flow_duration_ms": (3000, 1500), "total_fwd_packets": (20, 8),
            "total_bwd_packets": (5, 2), "total_fwd_bytes": (15000, 6000),
            "total_bwd_bytes": (400, 150), "fwd_packet_len_mean": (750, 200),
            "fwd_packet_len_std": (150, 60), "bwd_packet_len_mean": (80, 25),
            "bwd_packet_len_std": (15, 8), "flow_bytes_per_sec": (5000, 2000),
            "flow_packets_per_sec": (8, 3), "flow_iat_mean": (150, 60),
            "flow_iat_std": (50, 20), "fwd_iat_mean": (100, 40),
            "bwd_iat_mean": (600, 200), "syn_flag_count": (1, 0.7),
            "ack_flag_count": (15, 5), "rst_flag_count": (0.1, 0.2),
            "psh_flag_count": (12, 4), "fin_flag_count": (1, 0.7),
            "unique_dst_ports_per_src": (1, 0.5), "packets_per_flow": (25, 8),
            "avg_packet_size": (600, 180), "down_up_ratio": (0.04, 0.03),
            "urg_flag_count": (0, 0.1), "flow_iat_max": (500, 200),
            "fwd_iat_std": (40, 15), "bwd_iat_std": (200, 80),
            "fwd_header_len": (20, 2), "bwd_header_len": (20, 2),
            "ttl_mean": (64, 5), "ttl_std": (1, 0.5),
            "tcp_window_mean": (65535, 5000), "tcp_window_std": (3000, 1000),
            "ip_fragment_flag_count": (0.5, 0.5), "payload_size_mean": (750, 200),
            "payload_size_std": (150, 60), "retransmission_count": (0.5, 0.5),
            # Topology — Exfiltration: single external target, high data volume
            "src_fanout": (1, 0.5), "dst_fanin": (1, 0.5),
            "unique_dst_hosts": (1, 0.5), "unique_src_hosts": (1, 0.5),
            "new_host_edges": (0.5, 0.5), "cross_subnet_edges": (1, 0.7),
            "new_dst_ports": (0.3, 0.3), "connection_repetition": (10, 4),
        },
        "DDoS": {
            "flow_duration_ms": (40, 25), "total_fwd_packets": (600, 150),
            "total_bwd_packets": (2, 1.5), "total_fwd_bytes": (30000, 9000),
            "total_bwd_bytes": (80, 60), "fwd_packet_len_mean": (60, 15),
            "fwd_packet_len_std": (5, 3), "bwd_packet_len_mean": (40, 20),
            "bwd_packet_len_std": (3, 2), "flow_bytes_per_sec": (400000, 120000),
            "flow_packets_per_sec": (8000, 2000), "flow_iat_mean": (0.6, 0.4),
            "flow_iat_std": (0.3, 0.2), "fwd_iat_mean": (0.5, 0.3),
            "bwd_iat_mean": (5, 4), "syn_flag_count": (400, 100),
            "ack_flag_count": (5, 3), "rst_flag_count": (1, 0.8),
            "psh_flag_count": (0.5, 0.5), "fin_flag_count": (0.2, 0.3),
            "unique_dst_ports_per_src": (1, 0.5), "packets_per_flow": (600, 150),
            "avg_packet_size": (58, 12), "down_up_ratio": (0.02, 0.02),
            "urg_flag_count": (0, 0.1), "flow_iat_max": (2, 1.5),
            "fwd_iat_std": (0.2, 0.15), "bwd_iat_std": (3, 2),
            "fwd_header_len": (20, 1), "bwd_header_len": (20, 1),
            "ttl_mean": (255, 20), "ttl_std": (15, 8),
            "tcp_window_mean": (512, 200), "tcp_window_std": (50, 30),
            "ip_fragment_flag_count": (0.1, 0.2), "payload_size_mean": (0, 2),
            "payload_size_std": (0, 1), "retransmission_count": (5, 3),
            # Topology — DDoS: many sources, single target, high repetition
            "src_fanout": (1, 0.5), "dst_fanin": (50, 20),
            "unique_dst_hosts": (1, 0.5), "unique_src_hosts": (40, 15),
            "new_host_edges": (30, 10), "cross_subnet_edges": (20, 8),
            "new_dst_ports": (0.5, 0.5), "connection_repetition": (80, 30),
        },
    }

    profile = profiles[label]
    vec = []
    for col in WORLD_MODEL_FEATURES:
        mean, std = profile[col]
        val = rng.normal(mean, max(std, 0.01))
        vec.append(max(val, 0.0))
    return np.array(vec, dtype=np.float32)


# Pre-defined multi-stage attack scenarios for temporal training data
ATTACK_SCENARIOS = [
    # Classic reconnaissance → exploitation → C2 → exfiltration kill-chain
    ["BENIGN", "BENIGN", "PortScan", "PortScan", "WebAttack", "BruteForce",
     "LateralMovement", "Botnet", "Exfiltration", "Exfiltration"],

    # DDoS campaign preceded by reconnaissance
    ["BENIGN", "BENIGN", "BENIGN", "PortScan", "PortScan", "DDoS",
     "DDoS", "DDoS", "DDoS", "DDoS"],

    # Brute force → lateral movement → exfiltration
    ["BENIGN", "BENIGN", "BruteForce", "BruteForce", "BruteForce",
     "LateralMovement", "LateralMovement", "Botnet", "Exfiltration", "BENIGN"],

    # Slow reconnaissance → web exploitation → C2
    ["BENIGN", "BENIGN", "BENIGN", "BENIGN", "PortScan", "PortScan",
     "WebAttack", "WebAttack", "Botnet", "Botnet"],

    # All benign (negative sample)
    ["BENIGN", "BENIGN", "BENIGN", "BENIGN", "BENIGN",
     "BENIGN", "BENIGN", "BENIGN", "BENIGN", "BENIGN"],

    # Short burst: scan → brute force (no lateral)
    ["BENIGN", "BENIGN", "BENIGN", "PortScan", "BruteForce",
     "BruteForce", "BENIGN", "BENIGN", "BENIGN", "BENIGN"],

    # Immediate DDoS (no recon)
    ["BENIGN", "DDoS", "DDoS", "DDoS", "DDoS",
     "DDoS", "DDoS", "DDoS", "BENIGN", "BENIGN"],

    # Botnet C2 beaconing → slow exfiltration
    ["BENIGN", "BENIGN", "Botnet", "Botnet", "Botnet",
     "Botnet", "Exfiltration", "Exfiltration", "Exfiltration", "BENIGN"],
]


def generate_temporal_dataset(
    n_sequences: int = 500,
    seq_len: int = 10,
    seed: int = 42,
    noise: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic temporal attack sequences for world model training.

    Returns
    -------
    X : ndarray of shape (n_sequences, seq_len, n_features)
        Feature vectors for each time step in each sequence.
    y_labels : ndarray of shape (n_sequences, seq_len)
        Integer attack stage label at each time step.
    y_infiltration : ndarray of shape (n_sequences, seq_len)
        Binary infiltration flag (1.0 if any attack is present or imminent).
    """
    rng = np.random.default_rng(seed)

    X = np.zeros((n_sequences, seq_len, NUM_FEATURES), dtype=np.float32)
    y_labels = np.zeros((n_sequences, seq_len), dtype=np.int64)
    y_infiltration = np.zeros((n_sequences, seq_len), dtype=np.float32)

    for i in range(n_sequences):
        # Pick a random scenario (or generate a random one)
        if rng.random() < 0.7:
            scenario = ATTACK_SCENARIOS[rng.integers(0, len(ATTACK_SCENARIOS))]
        else:
            # Random scenario
            scenario = [rng.choice(ATTACK_LABELS) for _ in range(seq_len)]

        # Extend or trim to seq_len
        if len(scenario) < seq_len:
            scenario = scenario + ["BENIGN"] * (seq_len - len(scenario))
        scenario = scenario[:seq_len]

        for t, label in enumerate(scenario):
            # Generate feature vector for this time step
            vec = _generate_flow_vector(label, rng)
            # Add noise for realism
            noise_vec = rng.normal(0, noise, NUM_FEATURES).astype(np.float32)
            vec = np.maximum(vec + vec * noise_vec, 0.0)
            X[i, t] = vec

            # Label
            y_labels[i, t] = ATTACK_STAGE_INDEX[label]

            # Infiltration: 1.0 if this step or any future step is malicious
            future_labels = scenario[t:]
            has_attack = any(l != "BENIGN" for l in future_labels)
            y_infiltration[i, t] = 1.0 if has_attack else 0.0

    return X, y_labels, y_infiltration


def generate_temporal_dataset_with_horizons(
    n_sequences: int = 500,
    seq_len: int = 15,
    seed: int = 42,
    noise: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Generate synthetic temporal sequences along with explicit multi-horizon targets.

    Horizon offsets (in 30s steps):
      - h30:  +1 step  (+30s)
      - h60:  +2 steps (+60s)
      - h120: +4 steps (+120s)
      - h300: +10 steps (+300s)

    Returns
    -------
    X : ndarray of shape (n_sequences, seq_len, n_features)
    y_labels : ndarray of shape (n_sequences, seq_len)
    y_infiltration : ndarray of shape (n_sequences, seq_len)
    horizon_labels : dict[str, ndarray]
        Mapping horizon name -> array of shape (n_sequences, seq_len) with future stage index,
        or -1 where future is beyond sequence boundary.
    """
    X, y_labels, y_infiltration = generate_temporal_dataset(
        n_sequences=n_sequences, seq_len=seq_len, seed=seed, noise=noise
    )

    horizon_offsets = {"h30": 1, "h60": 2, "h120": 4, "h300": 10}
    horizon_labels: dict[str, np.ndarray] = {}

    for h_name, offset in horizon_offsets.items():
        h_arr = np.full((n_sequences, seq_len), -1, dtype=np.int64)
        if offset < seq_len:
            h_arr[:, : seq_len - offset] = y_labels[:, offset:]
        horizon_labels[h_name] = h_arr

    return X, y_labels, y_infiltration, horizon_labels


def extract_features_from_pcap(pcap_path: str, window_size: int = 10) -> np.ndarray:
    """Extract world model features from a PCAP file.

    Groups packets into temporal windows and computes flow-level + packet-level
    features for each window. Returns a (n_windows, n_features) matrix suitable
    for world model inference.
    """
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.utils import rdpcap

    p = Path(pcap_path)
    if not p.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")

    packets = rdpcap(str(p))
    if not packets:
        return np.zeros((1, NUM_FEATURES), dtype=np.float32)

    # Group packets into time windows
    timestamps = [float(pkt.time) for pkt in packets if hasattr(pkt, "time")]
    if not timestamps:
        return np.zeros((1, NUM_FEATURES), dtype=np.float32)

    min_t, max_t = min(timestamps), max(timestamps)
    duration = max_t - min_t
    if duration < 0.001:
        n_windows = 1
        window_dur = 1.0
    else:
        window_dur = duration / window_size
        n_windows = window_size

    features = np.zeros((n_windows, NUM_FEATURES), dtype=np.float32)

    for w in range(n_windows):
        w_start = min_t + w * window_dur
        w_end = w_start + window_dur
        w_pkts = [pkt for pkt, ts in zip(packets, timestamps)
                  if w_start <= ts < w_end]
        if not w_pkts:
            continue

        # Compute flow-level stats
        total_bytes = sum(len(pkt) for pkt in w_pkts)
        fwd_pkts = [pkt for pkt in w_pkts if IP in pkt]
        fwd_sizes = [len(pkt) for pkt in fwd_pkts]
        w_dur_ms = window_dur * 1000

        # TCP flags
        syn_c = sum(1 for pkt in w_pkts if TCP in pkt and pkt[TCP].flags & 0x02)
        ack_c = sum(1 for pkt in w_pkts if TCP in pkt and pkt[TCP].flags & 0x10)
        rst_c = sum(1 for pkt in w_pkts if TCP in pkt and pkt[TCP].flags & 0x04)
        psh_c = sum(1 for pkt in w_pkts if TCP in pkt and pkt[TCP].flags & 0x08)
        fin_c = sum(1 for pkt in w_pkts if TCP in pkt and pkt[TCP].flags & 0x01)
        urg_c = sum(1 for pkt in w_pkts if TCP in pkt and pkt[TCP].flags & 0x20)

        # Dst ports
        dst_ports = set()
        for pkt in w_pkts:
            if TCP in pkt:
                dst_ports.add(pkt[TCP].dport)
            elif UDP in pkt:
                dst_ports.add(pkt[UDP].dport)

        # Packet-level features
        ttls = [pkt[IP].ttl for pkt in w_pkts if IP in pkt]
        tcp_windows = [pkt[TCP].window for pkt in w_pkts if TCP in pkt]
        payload_sizes = [len(bytes(pkt[TCP].payload)) for pkt in w_pkts
                         if TCP in pkt and pkt[TCP].payload]
        frag_flags = sum(1 for pkt in w_pkts if IP in pkt and pkt[IP].flags & 0x01)
        retrans = sum(1 for pkt in w_pkts if TCP in pkt and pkt[TCP].flags & 0x04)

        # Inter-arrival times
        w_timestamps = sorted([float(pkt.time) for pkt in w_pkts if hasattr(pkt, "time")])
        if len(w_timestamps) > 1:
            iats = [w_timestamps[j+1] - w_timestamps[j] for j in range(len(w_timestamps)-1)]
            iat_mean = statistics.mean(iats) * 1000
            iat_std = statistics.stdev(iats) * 1000 if len(iats) > 1 else 0
            iat_max = max(iats) * 1000
        else:
            iat_mean = iat_std = iat_max = 0

        n_pkts = len(w_pkts)
        avg_size = total_bytes / n_pkts if n_pkts else 0

        # Topology features from packet src/dst IP pairs
        src_ips = set()
        dst_ips = set()
        host_edges = set()
        for pkt in w_pkts:
            if IP in pkt:
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                src_ips.add(src_ip)
                dst_ips.add(dst_ip)
                host_edges.add((src_ip, dst_ip))

        # Cross-subnet: different /24 prefix
        cross_subnet = 0
        for s, d in host_edges:
            s_prefix = ".".join(s.split(".")[:3])
            d_prefix = ".".join(d.split(".")[:3])
            if s_prefix != d_prefix:
                cross_subnet += 1

        # Connection repetition: how many packets go to same dst
        dst_counter: dict[str, int] = {}
        for pkt in w_pkts:
            if IP in pkt:
                dst_counter[pkt[IP].dst] = dst_counter.get(pkt[IP].dst, 0) + 1
        max_repetition = max(dst_counter.values()) if dst_counter else 0

        # Fill feature vector
        vec = [
            w_dur_ms,                                          # flow_duration_ms
            n_pkts * 0.5,                                      # total_fwd_packets
            n_pkts * 0.5,                                      # total_bwd_packets
            total_bytes * 0.5,                                 # total_fwd_bytes
            total_bytes * 0.5,                                 # total_bwd_bytes
            np.mean(fwd_sizes) if fwd_sizes else 0,            # fwd_packet_len_mean
            np.std(fwd_sizes) if len(fwd_sizes) > 1 else 0,   # fwd_packet_len_std
            np.mean(fwd_sizes) if fwd_sizes else 0,            # bwd_packet_len_mean
            np.std(fwd_sizes) if len(fwd_sizes) > 1 else 0,   # bwd_packet_len_std
            total_bytes / (window_dur + 1e-9),                 # flow_bytes_per_sec
            n_pkts / (window_dur + 1e-9),                      # flow_packets_per_sec
            iat_mean,                                          # flow_iat_mean
            iat_std,                                           # flow_iat_std
            iat_mean,                                          # fwd_iat_mean
            iat_mean,                                          # bwd_iat_mean
            syn_c, ack_c, rst_c, psh_c, fin_c,                # flag counts
            len(dst_ports),                                    # unique_dst_ports_per_src
            n_pkts,                                            # packets_per_flow
            avg_size,                                          # avg_packet_size
            0.5,                                               # down_up_ratio
            urg_c,                                             # urg_flag_count
            iat_max,                                           # flow_iat_max
            iat_std,                                           # fwd_iat_std
            iat_std,                                           # bwd_iat_std
            20,                                                # fwd_header_len
            20,                                                # bwd_header_len
            np.mean(ttls) if ttls else 64,                     # ttl_mean
            np.std(ttls) if len(ttls) > 1 else 0,             # ttl_std
            np.mean(tcp_windows) if tcp_windows else 0,        # tcp_window_mean
            np.std(tcp_windows) if len(tcp_windows) > 1 else 0, # tcp_window_std
            frag_flags,                                        # ip_fragment_flag_count
            np.mean(payload_sizes) if payload_sizes else 0,    # payload_size_mean
            np.std(payload_sizes) if len(payload_sizes) > 1 else 0, # payload_size_std
            retrans,                                           # retransmission_count
            # Topology features
            len(dst_ips),                                      # src_fanout (unique dsts per src)
            len(src_ips),                                      # dst_fanin (unique srcs per dst)
            len(dst_ips),                                      # unique_dst_hosts
            len(src_ips),                                      # unique_src_hosts
            len(host_edges),                                   # new_host_edges
            cross_subnet,                                      # cross_subnet_edges
            len(dst_ports),                                    # new_dst_ports
            max_repetition,                                    # connection_repetition
        ]
        features[w] = np.array(vec, dtype=np.float32)

    return features


if __name__ == "__main__":
    X, y_labels, y_inf = generate_temporal_dataset(n_sequences=10, seq_len=10, seed=42)
    print(f"X shape: {X.shape}")
    print(f"y_labels shape: {y_labels.shape}")
    print(f"y_infiltration shape: {y_inf.shape}")
    print(f"Feature count: {NUM_FEATURES}")
    print(f"Feature names: {WORLD_MODEL_FEATURES}")
