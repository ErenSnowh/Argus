"""
graph_signals.py
-----------------
Lightweight graph-theoretic anomaly signals for lateral movement detection.

Computes 3 derived graph features from the 8 topology columns already in the
WORLD_MODEL_FEATURES schema, without requiring a full GNN training run:

    graph_anomaly_score     Composite score [0, 1] combining all three signals
    degree_centrality_delta Change in src_fanout relative to its recent mean
    new_edge_burst_score    Burst of new_host_edges vs historical rate

These signals are particularly effective for:
  - Lateral movement (T1021): sudden increase in cross-subnet connections
  - Reconnaissance (T1046): fanout spike across many new destination hosts
  - Botnet C2 (T1071): very regular new_host_edges with low variance

They are computed from features already in each time-step vector, so they
can be added to any forecast context window without a separate data source.
"""

from __future__ import annotations

import numpy as np


# Feature index lookup (populated once at import time)
def _build_index() -> dict[str, int]:
    from ml.world_model.features import WORLD_MODEL_FEATURES
    return {name: i for i, name in enumerate(WORLD_MODEL_FEATURES)}


_IDX: dict[str, int] | None = None


def _idx() -> dict[str, int]:
    global _IDX
    if _IDX is None:
        _IDX = _build_index()
    return _IDX


def compute_graph_signals(
    context_window: np.ndarray,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Compute graph anomaly signals from a context window of feature vectors.

    Parameters
    ----------
    context_window : ndarray of shape (seq_len, n_features)
        A sequence of feature vectors (normalized or raw).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    dict with:
        graph_anomaly_score     float in [0, 1]
        degree_centrality_delta float (positive = fanout increasing)
        new_edge_burst_score    float in [0, 1]
        cross_subnet_ratio      float in [0, 1]
        lateral_movement_signal float in [0, 1]
    """
    idx = _idx()
    window = np.asarray(context_window, dtype=np.float32)
    if window.ndim == 1:
        window = window.reshape(1, -1)

    seq_len = window.shape[0]

    def _get(name: str) -> np.ndarray:
        i = idx.get(name)
        if i is None or i >= window.shape[-1]:
            return np.zeros(seq_len, dtype=np.float32)
        return window[:, i]

    src_fanout = _get("src_fanout")
    new_host_edges = _get("new_host_edges")
    cross_subnet = _get("cross_subnet_edges")
    unique_dst = _get("unique_dst_hosts")
    connection_rep = _get("connection_repetition")
    unique_dst_ports = _get("unique_dst_ports_per_src")

    # --- Signal 1: Degree centrality delta ---
    # How much has the src_fanout grown compared to the window mean?
    # A sudden spike in fanout is a hallmark of port-scan and lateral movement.
    if seq_len > 1:
        mean_fanout = float(src_fanout[:-1].mean()) + eps
        current_fanout = float(src_fanout[-1])
        degree_delta = (current_fanout - mean_fanout) / mean_fanout
    else:
        degree_delta = 0.0

    # --- Signal 2: New edge burst score ---
    # Rate of new host edges in the final step vs the window baseline.
    # Burst = current new_host_edges >> window mean → reconnaissance or propagation.
    if seq_len > 1:
        mean_edges = float(new_host_edges[:-1].mean()) + eps
        current_edges = float(new_host_edges[-1])
        burst_ratio = current_edges / mean_edges
        # Clip to [0, 1] using logistic transform
        new_edge_burst = float(1.0 / (1.0 + np.exp(-0.5 * (burst_ratio - 3.0))))
    else:
        new_edge_burst = float(np.clip(new_host_edges[-1] / 20.0, 0.0, 1.0))

    # --- Signal 3: Cross-subnet ratio ---
    # Proportion of edges that cross /24 subnet boundaries.
    # High values in a single host's traffic signal lateral movement.
    total_edges_last = max(float(unique_dst[-1]), 1.0)
    cross_ratio = float(np.clip(cross_subnet[-1] / total_edges_last, 0.0, 1.0))

    # --- Signal 4: Lateral movement composite ---
    # Combines: cross-subnet activity + low connection repetition
    # (attacker touches many hosts once, not one host many times)
    low_rep = float(np.clip(1.0 - connection_rep[-1] / max(float(connection_rep.max()), 1.0), 0.0, 1.0))
    high_port_diversity = float(np.clip(unique_dst_ports[-1] / 50.0, 0.0, 1.0))
    lateral_signal = float(np.clip((cross_ratio + low_rep + high_port_diversity) / 3.0, 0.0, 1.0))

    # --- Composite anomaly score ---
    # Weighted combination: degree delta (40%), new edge burst (30%), lateral (30%)
    degree_contrib = float(np.clip(abs(degree_delta) / 5.0, 0.0, 1.0)) * 0.40
    graph_anomaly = float(np.clip(
        degree_contrib + new_edge_burst * 0.30 + lateral_signal * 0.30,
        0.0, 1.0,
    ))

    return {
        "graph_anomaly_score": round(graph_anomaly, 4),
        "degree_centrality_delta": round(degree_delta, 4),
        "new_edge_burst_score": round(new_edge_burst, 4),
        "cross_subnet_ratio": round(cross_ratio, 4),
        "lateral_movement_signal": round(lateral_signal, 4),
    }


def interpret_graph_signals(signals: dict[str, float]) -> str:
    """Return a human-readable interpretation of graph anomaly signals."""
    score = signals.get("graph_anomaly_score", 0.0)
    delta = signals.get("degree_centrality_delta", 0.0)
    burst = signals.get("new_edge_burst_score", 0.0)
    cross = signals.get("cross_subnet_ratio", 0.0)
    lateral = signals.get("lateral_movement_signal", 0.0)

    parts = []

    if score >= 0.7:
        parts.append(f"[CRITICAL] Graph anomaly score {score:.0%} — strong structural deviation")
    elif score >= 0.4:
        parts.append(f"[HIGH] Graph anomaly score {score:.0%} — notable structural change")
    elif score >= 0.2:
        parts.append(f"[MEDIUM] Graph anomaly score {score:.0%} — moderate graph activity")
    else:
        parts.append(f"[LOW] Graph anomaly score {score:.0%} — normal graph topology")

    if delta > 2.0:
        parts.append(f"Fanout spike: {delta:.1f}x above baseline — possible reconnaissance sweep")
    elif delta > 0.5:
        parts.append(f"Fanout increase: {delta:.1f}x above baseline — expanding contact list")

    if burst >= 0.7:
        parts.append(f"New edge burst: {burst:.0%} — rapid propagation to unseen hosts")

    if cross >= 0.5:
        parts.append(f"Cross-subnet ratio: {cross:.0%} — traffic crossing network boundaries (lateral movement indicator)")

    if lateral >= 0.6:
        parts.append(f"Lateral movement signal: {lateral:.0%} — pattern consistent with T1021 (SMB/RDP spread)")

    return " | ".join(parts) if parts else "Normal graph topology"


def augment_forecast_with_graph(
    forecast_result: dict,
    context_window: np.ndarray,
) -> dict:
    """Add graph anomaly signals to an existing forecast result dict.

    Called by forecast_infiltration() when a context window is available.
    The signals are added under the 'graph_signals' key and the
    forecast_explanation is appended with the graph interpretation.
    """
    signals = compute_graph_signals(context_window)
    interpretation = interpret_graph_signals(signals)

    forecast_result["graph_signals"] = signals
    forecast_result["graph_interpretation"] = interpretation

    # Boost the risk level if graph anomaly is high
    existing_risk = forecast_result.get("risk_level", "LOW")
    score = signals["graph_anomaly_score"]

    risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    graph_risk = "CRITICAL" if score >= 0.75 else "HIGH" if score >= 0.5 else "MEDIUM" if score >= 0.25 else "LOW"

    current_level = risk_order.get(existing_risk, 0)
    graph_level = risk_order.get(graph_risk, 0)
    if graph_level > current_level:
        forecast_result["risk_level"] = graph_risk
        forecast_result["risk_elevated_by"] = "graph_anomaly"

    return forecast_result


if __name__ == "__main__":
    # Demo: compute signals from a PortScan feature vector
    from ml.world_model.features import _generate_flow_vector, ATTACK_LABELS

    rng = np.random.default_rng(42)

    print("Graph Anomaly Signals by Attack Stage:")
    print("=" * 60)
    for label in ATTACK_LABELS:
        # Build a 5-step window where the last step is the attack
        window = np.stack([
            _generate_flow_vector("BENIGN", rng),
            _generate_flow_vector("BENIGN", rng),
            _generate_flow_vector("BENIGN", rng),
            _generate_flow_vector("BENIGN", rng),
            _generate_flow_vector(label, rng),
        ])
        signals = compute_graph_signals(window)
        interp = interpret_graph_signals(signals)
        print(f"\n{label}:")
        print(f"  Score: {signals['graph_anomaly_score']:.2%}")
        print(f"  {interp}")
