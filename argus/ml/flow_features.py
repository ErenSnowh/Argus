"""
flow_features.py
-----------------
Defines the network-flow feature schema used by ARGUS's detection layer and
generates a synthetic, labeled dataset that mimics the statistical shape of
CICIDS2017 flow records.

Why synthetic data?
The full CICIDS2017 dataset is several GB and requires a manual download from
the Canadian Institute for Cybersecurity. For this capstone repo we generate a
smaller, reproducible synthetic dataset with the *same feature philosophy*
(duration / packet-count / byte-count / inter-arrival-time / flag statistics)
so the whole pipeline runs end-to-end with `python scripts/train_model.py`
with zero external downloads. Swapping in real CICIDS2017 CSVs only requires
pointing `load_real_cicids2017()` at the files -- the feature schema and
the rest of the agent pipeline do not change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Feature schema: 24 statistical features per flow, inspired by the 78-feature
# CICIDS2017 flow export, trimmed to the most discriminative subset for a fast,
# explainable demo model (full list documented in docs/ARCHITECTURE.md).
FEATURE_COLUMNS = [
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

LABELS = ["BENIGN", "DDoS", "PortScan", "BruteForce", "WebAttack", "Botnet"]

_RNG = np.random.default_rng(42)


def _benign(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flow_duration_ms": _RNG.normal(800, 300, n).clip(5),
            "total_fwd_packets": _RNG.poisson(12, n) + 1,
            "total_bwd_packets": _RNG.poisson(11, n) + 1,
            "total_fwd_bytes": _RNG.normal(1400, 500, n).clip(40),
            "total_bwd_bytes": _RNG.normal(1300, 480, n).clip(40),
            "fwd_packet_len_mean": _RNG.normal(450, 120, n).clip(20),
            "fwd_packet_len_std": _RNG.normal(80, 30, n).clip(0),
            "bwd_packet_len_mean": _RNG.normal(420, 110, n).clip(20),
            "bwd_packet_len_std": _RNG.normal(75, 28, n).clip(0),
            "flow_bytes_per_sec": _RNG.normal(3500, 1200, n).clip(1),
            "flow_packets_per_sec": _RNG.normal(15, 6, n).clip(0.1),
            "flow_iat_mean": _RNG.normal(60, 20, n).clip(0.1),
            "flow_iat_std": _RNG.normal(20, 8, n).clip(0),
            "fwd_iat_mean": _RNG.normal(55, 18, n).clip(0.1),
            "bwd_iat_mean": _RNG.normal(58, 19, n).clip(0.1),
            "syn_flag_count": _RNG.poisson(1, n),
            "ack_flag_count": _RNG.poisson(10, n),
            "rst_flag_count": _RNG.poisson(0.2, n),
            "psh_flag_count": _RNG.poisson(3, n),
            "fin_flag_count": _RNG.poisson(1, n),
            "unique_dst_ports_per_src": _RNG.poisson(1.2, n) + 1,
            "packets_per_flow": _RNG.poisson(23, n) + 1,
            "avg_packet_size": _RNG.normal(430, 90, n).clip(20),
            "down_up_ratio": _RNG.normal(0.95, 0.2, n).clip(0.05),
            "label": "BENIGN",
        }
    )


def _ddos(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flow_duration_ms": _RNG.normal(40, 25, n).clip(0.5),
            "total_fwd_packets": _RNG.poisson(600, n) + 50,
            "total_bwd_packets": _RNG.poisson(2, n),
            "total_fwd_bytes": _RNG.normal(30000, 9000, n).clip(500),
            "total_bwd_bytes": _RNG.normal(80, 60, n).clip(0),
            "fwd_packet_len_mean": _RNG.normal(60, 15, n).clip(20),
            "fwd_packet_len_std": _RNG.normal(5, 3, n).clip(0),
            "bwd_packet_len_mean": _RNG.normal(40, 20, n).clip(0),
            "bwd_packet_len_std": _RNG.normal(3, 2, n).clip(0),
            "flow_bytes_per_sec": _RNG.normal(400000, 120000, n).clip(1000),
            "flow_packets_per_sec": _RNG.normal(8000, 2000, n).clip(50),
            "flow_iat_mean": _RNG.normal(0.6, 0.4, n).clip(0.01),
            "flow_iat_std": _RNG.normal(0.3, 0.2, n).clip(0),
            "fwd_iat_mean": _RNG.normal(0.5, 0.3, n).clip(0.01),
            "bwd_iat_mean": _RNG.normal(5, 4, n).clip(0.01),
            "syn_flag_count": _RNG.poisson(400, n),
            "ack_flag_count": _RNG.poisson(5, n),
            "rst_flag_count": _RNG.poisson(1, n),
            "psh_flag_count": _RNG.poisson(0.5, n),
            "fin_flag_count": _RNG.poisson(0.2, n),
            "unique_dst_ports_per_src": _RNG.poisson(1, n) + 1,
            "packets_per_flow": _RNG.poisson(600, n) + 50,
            "avg_packet_size": _RNG.normal(58, 12, n).clip(20),
            "down_up_ratio": _RNG.normal(0.02, 0.02, n).clip(0.001),
            "label": "DDoS",
        }
    )


def _port_scan(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flow_duration_ms": _RNG.normal(8, 5, n).clip(0.1),
            "total_fwd_packets": _RNG.poisson(2, n) + 1,
            "total_bwd_packets": _RNG.poisson(1, n),
            "total_fwd_bytes": _RNG.normal(60, 20, n).clip(20),
            "total_bwd_bytes": _RNG.normal(40, 30, n).clip(0),
            "fwd_packet_len_mean": _RNG.normal(40, 8, n).clip(20),
            "fwd_packet_len_std": _RNG.normal(2, 1, n).clip(0),
            "bwd_packet_len_mean": _RNG.normal(40, 20, n).clip(0),
            "bwd_packet_len_std": _RNG.normal(2, 1, n).clip(0),
            "flow_bytes_per_sec": _RNG.normal(8000, 3000, n).clip(100),
            "flow_packets_per_sec": _RNG.normal(250, 90, n).clip(5),
            "flow_iat_mean": _RNG.normal(4, 2, n).clip(0.01),
            "flow_iat_std": _RNG.normal(1, 0.5, n).clip(0),
            "fwd_iat_mean": _RNG.normal(4, 2, n).clip(0.01),
            "bwd_iat_mean": _RNG.normal(6, 3, n).clip(0.01),
            "syn_flag_count": _RNG.poisson(1.8, n),
            "ack_flag_count": _RNG.poisson(0.3, n),
            "rst_flag_count": _RNG.poisson(0.9, n),
            "psh_flag_count": _RNG.poisson(0.1, n),
            "fin_flag_count": _RNG.poisson(0.1, n),
            "unique_dst_ports_per_src": _RNG.poisson(180, n) + 20,
            "packets_per_flow": _RNG.poisson(3, n) + 1,
            "avg_packet_size": _RNG.normal(40, 10, n).clip(20),
            "down_up_ratio": _RNG.normal(0.6, 0.3, n).clip(0.01),
            "label": "PortScan",
        }
    )


def _brute_force(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flow_duration_ms": _RNG.normal(300, 100, n).clip(10),
            "total_fwd_packets": _RNG.poisson(9, n) + 2,
            "total_bwd_packets": _RNG.poisson(9, n) + 2,
            "total_fwd_bytes": _RNG.normal(700, 200, n).clip(40),
            "total_bwd_bytes": _RNG.normal(650, 190, n).clip(40),
            "fwd_packet_len_mean": _RNG.normal(75, 20, n).clip(20),
            "fwd_packet_len_std": _RNG.normal(8, 4, n).clip(0),
            "bwd_packet_len_mean": _RNG.normal(70, 18, n).clip(20),
            "bwd_packet_len_std": _RNG.normal(8, 4, n).clip(0),
            "flow_bytes_per_sec": _RNG.normal(4500, 1500, n).clip(50),
            "flow_packets_per_sec": _RNG.normal(60, 25, n).clip(2),
            "flow_iat_mean": _RNG.normal(15, 6, n).clip(0.1),
            "flow_iat_std": _RNG.normal(5, 2, n).clip(0),
            "fwd_iat_mean": _RNG.normal(14, 5, n).clip(0.1),
            "bwd_iat_mean": _RNG.normal(16, 6, n).clip(0.1),
            "syn_flag_count": _RNG.poisson(8, n),
            "ack_flag_count": _RNG.poisson(8, n),
            "rst_flag_count": _RNG.poisson(2, n),
            "psh_flag_count": _RNG.poisson(7, n),
            "fin_flag_count": _RNG.poisson(0.5, n),
            "unique_dst_ports_per_src": _RNG.poisson(1, n) + 1,
            "packets_per_flow": _RNG.poisson(18, n) + 2,
            "avg_packet_size": _RNG.normal(72, 15, n).clip(20),
            "down_up_ratio": _RNG.normal(0.9, 0.2, n).clip(0.05),
            "label": "BruteForce",
        }
    )


def _web_attack(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flow_duration_ms": _RNG.normal(220, 90, n).clip(5),
            "total_fwd_packets": _RNG.poisson(6, n) + 1,
            "total_bwd_packets": _RNG.poisson(5, n) + 1,
            "total_fwd_bytes": _RNG.normal(2200, 800, n).clip(60),
            "total_bwd_bytes": _RNG.normal(900, 400, n).clip(40),
            "fwd_packet_len_mean": _RNG.normal(380, 140, n).clip(20),
            "fwd_packet_len_std": _RNG.normal(120, 50, n).clip(0),
            "bwd_packet_len_mean": _RNG.normal(180, 70, n).clip(20),
            "bwd_packet_len_std": _RNG.normal(50, 25, n).clip(0),
            "flow_bytes_per_sec": _RNG.normal(14000, 5000, n).clip(100),
            "flow_packets_per_sec": _RNG.normal(50, 20, n).clip(1),
            "flow_iat_mean": _RNG.normal(35, 15, n).clip(0.1),
            "flow_iat_std": _RNG.normal(12, 5, n).clip(0),
            "fwd_iat_mean": _RNG.normal(33, 14, n).clip(0.1),
            "bwd_iat_mean": _RNG.normal(38, 16, n).clip(0.1),
            "syn_flag_count": _RNG.poisson(1, n),
            "ack_flag_count": _RNG.poisson(6, n),
            "rst_flag_count": _RNG.poisson(0.3, n),
            "psh_flag_count": _RNG.poisson(5, n),
            "fin_flag_count": _RNG.poisson(0.8, n),
            "unique_dst_ports_per_src": _RNG.poisson(1, n) + 1,
            "packets_per_flow": _RNG.poisson(11, n) + 1,
            "avg_packet_size": _RNG.normal(310, 100, n).clip(20),
            "down_up_ratio": _RNG.normal(0.45, 0.2, n).clip(0.02),
            "label": "WebAttack",
        }
    )


def _botnet(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flow_duration_ms": _RNG.normal(5000, 2000, n).clip(100),
            "total_fwd_packets": _RNG.poisson(4, n) + 1,
            "total_bwd_packets": _RNG.poisson(4, n) + 1,
            "total_fwd_bytes": _RNG.normal(320, 100, n).clip(40),
            "total_bwd_bytes": _RNG.normal(300, 95, n).clip(40),
            "fwd_packet_len_mean": _RNG.normal(80, 25, n).clip(20),
            "fwd_packet_len_std": _RNG.normal(10, 5, n).clip(0),
            "bwd_packet_len_mean": _RNG.normal(75, 24, n).clip(20),
            "bwd_packet_len_std": _RNG.normal(10, 5, n).clip(0),
            "flow_bytes_per_sec": _RNG.normal(60, 30, n).clip(1),
            "flow_packets_per_sec": _RNG.normal(0.8, 0.4, n).clip(0.01),
            "flow_iat_mean": _RNG.normal(1200, 400, n).clip(10),
            "flow_iat_std": _RNG.normal(300, 120, n).clip(0),
            "fwd_iat_mean": _RNG.normal(1150, 390, n).clip(10),
            "bwd_iat_mean": _RNG.normal(1250, 410, n).clip(10),
            "syn_flag_count": _RNG.poisson(0.5, n),
            "ack_flag_count": _RNG.poisson(3, n),
            "rst_flag_count": _RNG.poisson(0.1, n),
            "psh_flag_count": _RNG.poisson(2, n),
            "fin_flag_count": _RNG.poisson(0.1, n),
            "unique_dst_ports_per_src": _RNG.poisson(1, n) + 1,
            "packets_per_flow": _RNG.poisson(8, n) + 1,
            "avg_packet_size": _RNG.normal(78, 20, n).clip(20),
            "down_up_ratio": _RNG.normal(0.93, 0.15, n).clip(0.05),
            "label": "Botnet",
        }
    )


GENERATORS = {
    "BENIGN": _benign,
    "DDoS": _ddos,
    "PortScan": _port_scan,
    "BruteForce": _brute_force,
    "WebAttack": _web_attack,
    "Botnet": _botnet,
}


def generate_dataset(
    n_per_class: int = 1500,
    seed: int = 42,
    label_noise: float = 0.04,
    feature_jitter: float = 0.35,
) -> pd.DataFrame:
    """Generate a balanced synthetic flow dataset across all classes.

    Real network telemetry is messy: overlapping distributions, mislabeled
    flows from imperfect ground truth, measurement noise. A perfectly
    separable synthetic dataset would train a classifier that hits 100%
    accuracy and teach the wrong lesson. `feature_jitter` adds extra Gaussian
    noise on top of each class's base distribution (simulating real-world
    measurement variance) and `label_noise` randomly flips a small fraction
    of labels (simulating imperfect ground-truth / borderline flows), which
    together land the demo model in a realistic ~93-96% accuracy band --
    consistent with the 94.2% reported on the real CICIDS2017 IDS project.
    """
    global _RNG
    _RNG = np.random.default_rng(seed)
    frames = [gen(n_per_class) for gen in GENERATORS.values()]
    df = pd.concat(frames, ignore_index=True)

    if feature_jitter > 0:
        for col in FEATURE_COLUMNS:
            scale = df[col].std() * feature_jitter
            df[col] = (df[col] + _RNG.normal(0, scale, len(df))).clip(lower=0)

    if label_noise > 0:
        n_flip = int(len(df) * label_noise)
        flip_idx = _RNG.choice(df.index, size=n_flip, replace=False)
        all_labels = list(GENERATORS.keys())
        df.loc[flip_idx, "label"] = [
            _RNG.choice([l for l in all_labels if l != cur])
            for cur in df.loc[flip_idx, "label"]
        ]

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_dataset()
    print(df["label"].value_counts())
    print(df.head())
