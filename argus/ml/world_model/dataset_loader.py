"""
dataset_loader.py
-------------------
Loaders for real public cybersecurity datasets and temporal sequence builder.

Supports:
  - CIC-IDS-2017 CSV flow records
  - CIC-IDS-2018 CSV flow records
  - UNSW-NB15 CSV records
  - CTU-13 BiNetFlow data
  - CICIoT2023 CSV records
  - LANL Authentication Dataset
  - DARPA Intrusion Detection datasets
  - Synthetic temporal data (default when real datasets are unavailable)

The key function is `generate_state_sequences()` which converts flat flow
records into temporal (S_t, S_t+1) training pairs for the world model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.world_model.features import (
    ATTACK_LABELS,
    ATTACK_STAGE_INDEX,
    ATTACK_STAGE_NAMES,
    NUM_FEATURES,
    WORLD_MODEL_FEATURES,
    generate_temporal_dataset,
)


# ---------------------------------------------------------------------------
# Column name mapping: CIC-IDS-2018 → ARGUS schema
# The actual CSE-CIC-IDS-2018 CSV headers use abbreviated column names.
# Both the old (CICFlowMeter v3) and new (v4) naming variants are included
# to handle both formats transparently.
# ---------------------------------------------------------------------------

_CICIDS_COLUMN_MAP = {
    # --- New CSE-CIC-2018 format (abbreviated names) ---
    "Flow Duration": "flow_duration_ms",
    "Tot Fwd Pkts": "total_fwd_packets",
    "Tot Bwd Pkts": "total_bwd_packets",
    "TotLen Fwd Pkts": "total_fwd_bytes",
    "TotLen Bwd Pkts": "total_bwd_bytes",
    "Fwd Pkt Len Mean": "fwd_packet_len_mean",
    "Fwd Pkt Len Std": "fwd_packet_len_std",
    "Bwd Pkt Len Mean": "bwd_packet_len_mean",
    "Bwd Pkt Len Std": "bwd_packet_len_std",
    "Flow Byts/s": "flow_bytes_per_sec",
    "Flow Pkts/s": "flow_packets_per_sec",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Bwd IAT Mean": "bwd_iat_mean",
    "SYN Flag Cnt": "syn_flag_count",
    "ACK Flag Cnt": "ack_flag_count",
    "RST Flag Cnt": "rst_flag_count",
    "PSH Flag Cnt": "psh_flag_count",
    "FIN Flag Cnt": "fin_flag_count",
    "URG Flag Cnt": "urg_flag_count",
    "Fwd Header Len": "fwd_header_len",
    "Bwd Header Len": "bwd_header_len",
    "Down/Up Ratio": "down_up_ratio",
    "Pkt Size Avg": "avg_packet_size",
    "Fwd IAT Std": "fwd_iat_std",
    "Bwd IAT Std": "bwd_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Fwd Seg Size Avg": "fwd_packet_len_mean",  # alias
    "Bwd Seg Size Avg": "bwd_packet_len_mean",  # alias
    "Init Fwd Win Byts": "tcp_window_mean",     # initial TCP window
    "Init Bwd Win Byts": "tcp_window_std",      # secondary TCP window stat
    "Timestamp": "timestamp",
    "Label": "label",
    # --- Old CICFlowMeter v3 format (longer names) ---
    "Total Fwd Packet": "total_fwd_packets",
    "Total Bwd packets": "total_bwd_packets",
    "Total Length of Fwd Packet": "total_fwd_bytes",
    "Total Length of Bwd Packet": "total_bwd_bytes",
    "Fwd Packet Length Mean": "fwd_packet_len_mean",
    "Fwd Packet Length Std": "fwd_packet_len_std",
    "Bwd Packet Length Mean": "bwd_packet_len_mean",
    "Bwd Packet Length Std": "bwd_packet_len_std",
    "Flow Bytes/s": "flow_bytes_per_sec",
    "Flow Packets/s": "flow_packets_per_sec",
    "SYN Flag Count": "syn_flag_count",
    "ACK Flag Count": "ack_flag_count",
    "RST Flag Count": "rst_flag_count",
    "PSH Flag Count": "psh_flag_count",
    "FIN Flag Count": "fin_flag_count",
    "URG Flag Count": "urg_flag_count",
    "Average Packet Size": "avg_packet_size",
}

# CIC-IDS-2018 attack label → our label mapping
_CICIDS_LABEL_MAP = {
    "Benign": "BENIGN",
    "BENIGN": "BENIGN",
    "DoS attacks-Hulk": "DDoS",
    "DoS attacks-SlowHTTPTest": "DDoS",
    "DoS attacks-Slowloris": "DDoS",
    "DoS attacks-GoldenEye": "DDoS",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "DDoS attack-LOIC-UDP": "DDoS",
    "DDoS attack-HOIC": "DDoS",
    "SSH-Bruteforce": "BruteForce",
    "FTP-BruteForce": "BruteForce",
    "Brute Force -Web": "BruteForce",
    "Brute Force -XSS": "WebAttack",
    "SQL Injection": "WebAttack",
    "Bot": "Botnet",
    "Infilteration": "LateralMovement",
}


def load_cicids2018(csv_dir: str, max_rows: int | None = None) -> pd.DataFrame:
    """Load CIC-IDS-2018 CSV files and map to our feature schema.

    Parameters
    ----------
    csv_dir : str
        Directory containing CIC-IDS-2018 CSV files.
    max_rows : int or None
        If set, limit total rows loaded (for faster iteration).

    Returns
    -------
    DataFrame with columns matching WORLD_MODEL_FEATURES + 'label'.
    """
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {csv_dir}")

    if csv_path.is_file():
        csv_files = [csv_path]
    else:
        csv_files = sorted(csv_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    dfs = []
    if len(csv_files) == 1:
        nrows = max_rows
        df = pd.read_csv(csv_files[0], nrows=nrows, low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df)
    else:
        effective_max = max_rows if max_rows is not None else 25000
        per_file = max(20, effective_max // len(csv_files))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, nrows=per_file, low_memory=False)
                df.columns = df.columns.str.strip()
                dfs.append(df)
            except Exception as e:
                print(f"Warning: could not load {csv_file.name}: {e}")
                continue

    if not dfs:
        raise ValueError("No data loaded from CSV files")

    combined = pd.concat(dfs, ignore_index=True)

    # Rename columns and remove duplicate column names created by aliases
    combined = combined.rename(columns=_CICIDS_COLUMN_MAP)
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]

    # Filter out repeated header rows if present
    if "label" in combined.columns:
        combined = combined[combined["label"].astype(str).str.strip() != "Label"]
        combined["label"] = combined["label"].astype(str).str.strip().map(
            lambda x: _CICIDS_LABEL_MAP.get(x, "BENIGN")
        )

    # Clean: convert to numeric, replace inf/nan with 0, and clip bounds
    for col in WORLD_MODEL_FEATURES:
        if col not in combined.columns:
            combined[col] = 0.0
        else:
            combined[col] = (
                pd.to_numeric(combined[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .clip(lower=0.0, upper=1e9)
                .astype(np.float32)
            )

    extra_cols = [c for c in ["label", "timestamp"] if c in combined.columns]
    if "label" not in extra_cols:
        combined["label"] = "BENIGN"
        extra_cols.append("label")

    return combined[WORLD_MODEL_FEATURES + extra_cols]


def load_ctu13(binetflow_path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Load CTU-13 BiNetFlow data and map to our feature schema.

    CTU-13 uses a different format, so this loader extracts what it can
    and fills the rest with synthetic defaults.

    Parameters
    ----------
    binetflow_path : str
        Path to a CTU-13 .binetflow file.
    max_rows : int or None
        Limit rows loaded.

    Returns
    -------
    DataFrame with columns matching WORLD_MODEL_FEATURES + 'label'.
    """
    p = Path(binetflow_path)
    if not p.exists():
        raise FileNotFoundError(f"BiNetFlow file not found: {binetflow_path}")

    df = pd.read_csv(str(p), nrows=max_rows, low_memory=False)
    df.columns = df.columns.str.strip()

    # CTU-13 label mapping
    label_map = {
        "Normal": "BENIGN",
        "Botnet": "Botnet",
        "Background": "BENIGN",
    }

    result = pd.DataFrame()
    for col in WORLD_MODEL_FEATURES:
        result[col] = 0.0

    # Map available CTU-13 columns
    if "Dur" in df.columns:
        result["flow_duration_ms"] = df["Dur"].astype(float) * 1000
    if "TotPkts" in df.columns:
        result["packets_per_flow"] = df["TotPkts"].astype(float)
        result["total_fwd_packets"] = df["TotPkts"].astype(float) * 0.5
        result["total_bwd_packets"] = df["TotPkts"].astype(float) * 0.5
    if "TotBytes" in df.columns:
        result["total_fwd_bytes"] = df["TotBytes"].astype(float) * 0.5
        result["total_bwd_bytes"] = df["TotBytes"].astype(float) * 0.5
    if "SrcBytes" in df.columns:
        result["total_fwd_bytes"] = df["SrcBytes"].astype(float)

    # Map labels
    if "Label" in df.columns:
        result["label"] = df["Label"].str.strip().map(
            lambda x: next((v for k, v in label_map.items() if k in str(x)), "BENIGN")
        )
    else:
        result["label"] = "BENIGN"

    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in WORLD_MODEL_FEATURES:
        result[col] = result[col].clip(lower=0)

    return result[WORLD_MODEL_FEATURES + ["label"]]


def generate_state_sequences(
    df: pd.DataFrame,
    window_size: int = 10,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a flat DataFrame of flows into temporal state sequences.

    Groups flows into overlapping windows of `window_size` consecutive flows
    and creates the (S_t, S_t+1) training pairs needed for world model
    supervised dynamics learning.

    Parameters
    ----------
    df : DataFrame
        Must contain WORLD_MODEL_FEATURES columns and 'label' column.
    window_size : int
        Number of time steps per sequence.
    stride : int
        Step size between consecutive windows.

    Returns
    -------
    X : ndarray of shape (n_sequences, window_size, n_features)
    y_labels : ndarray of shape (n_sequences, window_size)
    y_infiltration : ndarray of shape (n_sequences, window_size)
    """
    feature_matrix = df[WORLD_MODEL_FEATURES].values.astype(np.float32)
    labels = df["label"].map(ATTACK_STAGE_INDEX).values.astype(np.int64)

    n_total = len(df)
    sequences_X = []
    sequences_y = []
    sequences_inf = []

    for start in range(0, n_total - window_size + 1, stride):
        end = start + window_size
        X_seq = feature_matrix[start:end]
        y_seq = labels[start:end]

        # Infiltration: 1.0 if any attack present at or after this step
        inf_seq = np.zeros(window_size, dtype=np.float32)
        for t in range(window_size):
            if np.any(y_seq[t:] > 0):
                inf_seq[t] = 1.0

        sequences_X.append(X_seq)
        sequences_y.append(y_seq)
        sequences_inf.append(inf_seq)

    if not sequences_X:
        return (
            np.zeros((0, window_size, NUM_FEATURES), dtype=np.float32),
            np.zeros((0, window_size), dtype=np.int64),
            np.zeros((0, window_size), dtype=np.float32),
        )

    return (
        np.stack(sequences_X),
        np.stack(sequences_y),
        np.stack(sequences_inf),
    )


def train_test_split_temporal(
    X: np.ndarray,
    y_labels: np.ndarray,
    y_infiltration: np.ndarray,
    test_ratio: float = 0.2,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Time-respecting train/test split (no future data leaking into training).

    Unlike random splitting, this preserves temporal ordering by using the
    first (1-test_ratio) sequences for training and the last test_ratio for
    testing. This is critical for evaluating whether the model generalises
    to future traffic, not just interpolates within known patterns.
    """
    n = len(X)
    split_idx = int(n * (1 - test_ratio))

    train = (X[:split_idx], y_labels[:split_idx], y_infiltration[:split_idx])
    test = (X[split_idx:], y_labels[split_idx:], y_infiltration[split_idx:])

    return train, test


def train_test_split_by_day(
    df: pd.DataFrame,
    train_days: list[str] | None = None,
    test_days: list[str] | None = None,
    timestamp_col: str = "timestamp",
    window_size: int = 10,
    stride: int = 1,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Chronological split by calendar day to evaluate out-of-distribution generalization.

    Ensures zero temporal leakage between training and testing days.
    Sequences are generated independently within each split so that sequence
    windows never cross day boundaries.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing WORLD_MODEL_FEATURES, 'label', and optionally timestamp_col.
    train_days : list of str, optional
        Date strings (e.g. ['2018-02-14', '2018-02-15']) for training.
    test_days : list of str, optional
        Date strings for testing.
    timestamp_col : str
        Column name for timestamp.
    window_size : int
        Sequence length.
    stride : int
        Window stride.
    """
    if timestamp_col not in df.columns:
        # Fallback to temporal split on generated sequences
        X, y_labels, y_inf = generate_state_sequences(df, window_size=window_size, stride=stride)
        return train_test_split_temporal(X, y_labels, y_inf, test_ratio=0.2)

    df_copy = df.copy()
    ts = pd.to_datetime(df_copy[timestamp_col], errors="coerce")
    df_copy["_day"] = ts.dt.strftime("%Y-%m-%d")

    unique_days = sorted([d for d in df_copy["_day"].dropna().unique()])
    if len(unique_days) >= 2:
        if train_days is None and test_days is None:
            n_train = max(1, int(len(unique_days) * 0.75))
            train_days = unique_days[:n_train]
            test_days = unique_days[n_train:]
        elif train_days is None and test_days is not None:
            train_days = [d for d in unique_days if d not in test_days]
        elif test_days is None and train_days is not None:
            test_days = [d for d in unique_days if d not in train_days]
    else:
        # Not enough distinct days, fall back to chronological split
        X, y_labels, y_inf = generate_state_sequences(df, window_size=window_size, stride=stride)
        return train_test_split_temporal(X, y_labels, y_inf, test_ratio=0.2)

    df_train = df_copy[df_copy["_day"].isin(train_days)]
    df_test = df_copy[df_copy["_day"].isin(test_days)]

    train_data = generate_state_sequences(df_train, window_size=window_size, stride=stride)
    test_data = generate_state_sequences(df_test, window_size=window_size, stride=stride)

    return train_data, test_data


def get_temporal_split_info(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict:
    """Return diagnostic metadata about temporal splits."""
    X_tr, y_tr, yi_tr = train
    X_te, y_te, yi_te = test

    train_stages = sorted(list(set(int(s) for s in y_tr.flatten()))) if len(y_tr) > 0 else []
    test_stages = sorted(list(set(int(s) for s in y_te.flatten()))) if len(y_te) > 0 else []
    unseen_stages = sorted(list(set(test_stages) - set(train_stages)))

    # Leakage check: verify no exact match between first steps of test and train sequences
    leakage = False
    if len(X_tr) > 0 and len(X_te) > 0:
        tr_sample = X_tr[:min(100, len(X_tr)), 0]
        te_sample = X_te[:min(100, len(X_te)), 0]
        for te_row in te_sample:
            if np.any(np.all(np.isclose(tr_sample, te_row, atol=1e-5), axis=-1)):
                leakage = True
                break

    return {
        "train_sequences": int(len(X_tr)),
        "test_sequences": int(len(X_te)),
        "seq_len": int(X_tr.shape[1]) if len(X_tr) > 0 else 0,
        "n_features": int(X_tr.shape[2]) if len(X_tr) > 0 else 0,
        "train_stages": [ATTACK_STAGE_NAMES.get(s, str(s)) for s in train_stages],
        "test_stages": [ATTACK_STAGE_NAMES.get(s, str(s)) for s in test_stages],
        "unseen_test_stages": [ATTACK_STAGE_NAMES.get(s, str(s)) for s in unseen_stages],
        "train_infiltration_ratio": round(float(np.mean(yi_tr)), 4) if len(yi_tr) > 0 else 0.0,
        "test_infiltration_ratio": round(float(np.mean(yi_te)), 4) if len(yi_te) > 0 else 0.0,
        "leakage_detected": leakage,
    }


SAMPLE_DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample_datasets"


def get_sample_path(source: str) -> Path | None:
    """Return the path to the bundled sample fixture for a dataset if available."""
    mapping = {
        "cicids2018": SAMPLE_DATASETS_DIR / "cicids2018",
        "cicids2017": SAMPLE_DATASETS_DIR / "cicids2017",
        "unsw_nb15": SAMPLE_DATASETS_DIR / "unsw_nb15",
        "ctu13": SAMPLE_DATASETS_DIR / "ctu13" / "sample_ctu13.binetflow",
        "ciciot2023": SAMPLE_DATASETS_DIR / "ciciot2023",
        "lanl": SAMPLE_DATASETS_DIR / "lanl" / "sample_auth.txt",
        "darpa": SAMPLE_DATASETS_DIR / "darpa" / "sample_nsl_kdd.csv",
    }
    p = mapping.get(source)
    if p and p.exists():
        return p
    return None


def load_dataset(
    source: str = "synthetic",
    path: str | None = None,
    n_sequences: int = 500,
    seq_len: int = 10,
    seed: int = 42,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unified dataset loading interface.

    Parameters
    ----------
    source : str
        One of: 'synthetic', 'cicids2017', 'cicids2018', 'unsw_nb15',
        'ctu13', 'ciciot2023', 'lanl', 'darpa'
    path : str or None
        Path to dataset directory or file. If None, automatically falls back
        to the bundled representative sample fixture in argus/data/sample_datasets/.
    n_sequences : int
        Number of sequences for synthetic data
    seq_len : int
        Sequence length
    max_rows : int or None
        Limit rows loaded (for faster iteration)

    Returns
    -------
    X, y_labels, y_infiltration
    """
    if source == "synthetic":
        return generate_temporal_dataset(n_sequences, seq_len, seed)

    effective_path = path
    if not effective_path:
        sample = get_sample_path(source)
        if sample:
            effective_path = str(sample)

    if source == "cicids2017":
        if not effective_path:
            raise ValueError("path required for cicids2017 dataset (no bundled sample found)")
        df = load_cicids2017(effective_path, max_rows=max_rows)
        return generate_state_sequences(df, window_size=seq_len)

    elif source == "cicids2018":
        if not effective_path:
            raise ValueError("path required for cicids2018 dataset (no bundled sample found)")
        df = load_cicids2018(effective_path, max_rows=max_rows)
        return generate_state_sequences(df, window_size=seq_len)

    elif source == "unsw_nb15":
        if not effective_path:
            raise ValueError("path required for unsw_nb15 dataset (no bundled sample found)")
        df = load_unsw_nb15(effective_path, max_rows=max_rows)
        return generate_state_sequences(df, window_size=seq_len)

    elif source == "ctu13":
        if not effective_path:
            raise ValueError("path required for ctu13 dataset (no bundled sample found)")
        df = load_ctu13(effective_path, max_rows=max_rows)
        return generate_state_sequences(df, window_size=seq_len)

    elif source == "ciciot2023":
        if not effective_path:
            raise ValueError("path required for ciciot2023 dataset (no bundled sample found)")
        df = load_ciciot2023(effective_path, max_rows=max_rows)
        return generate_state_sequences(df, window_size=seq_len)

    elif source == "lanl":
        if not effective_path:
            raise ValueError("path required for lanl dataset (no bundled sample found)")
        df = load_lanl_auth(effective_path, max_rows=max_rows)
        return generate_state_sequences(df, window_size=seq_len)

    elif source == "darpa":
        if not effective_path:
            raise ValueError("path required for darpa dataset (no bundled sample found)")
        df = load_darpa(effective_path, max_rows=max_rows)
        return generate_state_sequences(df, window_size=seq_len)

    else:
        valid = "synthetic, cicids2017, cicids2018, unsw_nb15, ctu13, ciciot2023, lanl, darpa"
        raise ValueError(f"Unknown dataset source: {source}. Valid: {valid}")



# ---------------------------------------------------------------------------
# CIC-IDS-2017 loader
# ---------------------------------------------------------------------------

# CIC-IDS-2017 uses slightly different column names than 2018
_CICIDS2017_COLUMN_MAP = {
    " Flow Duration": "flow_duration_ms",
    "Flow Duration": "flow_duration_ms",
    " Total Fwd Packets": "total_fwd_packets",
    "Total Fwd Packets": "total_fwd_packets",
    " Total Backward Packets": "total_bwd_packets",
    "Total Backward Packets": "total_bwd_packets",
    "Total Length of Fwd Packets": "total_fwd_bytes",
    " Total Length of Fwd Packets": "total_fwd_bytes",
    "Total Length of Bwd Packets": "total_bwd_bytes",
    " Total Length of Bwd Packets": "total_bwd_bytes",
    "Fwd Packet Length Mean": "fwd_packet_len_mean",
    " Fwd Packet Length Mean": "fwd_packet_len_mean",
    "Fwd Packet Length Std": "fwd_packet_len_std",
    " Fwd Packet Length Std": "fwd_packet_len_std",
    "Bwd Packet Length Mean": "bwd_packet_len_mean",
    " Bwd Packet Length Mean": "bwd_packet_len_mean",
    "Bwd Packet Length Std": "bwd_packet_len_std",
    " Bwd Packet Length Std": "bwd_packet_len_std",
    "Flow Bytes/s": "flow_bytes_per_sec",
    " Flow Bytes/s": "flow_bytes_per_sec",
    "Flow Packets/s": "flow_packets_per_sec",
    " Flow Packets/s": "flow_packets_per_sec",
    "Flow IAT Mean": "flow_iat_mean",
    " Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    " Flow IAT Std": "flow_iat_std",
    "Fwd IAT Mean": "fwd_iat_mean",
    " Fwd IAT Mean": "fwd_iat_mean",
    "Bwd IAT Mean": "bwd_iat_mean",
    " Bwd IAT Mean": "bwd_iat_mean",
    "SYN Flag Count": "syn_flag_count",
    " SYN Flag Count": "syn_flag_count",
    "ACK Flag Count": "ack_flag_count",
    " ACK Flag Count": "ack_flag_count",
    "RST Flag Count": "rst_flag_count",
    " RST Flag Count": "rst_flag_count",
    "PSH Flag Count": "psh_flag_count",
    " PSH Flag Count": "psh_flag_count",
    "FIN Flag Count": "fin_flag_count",
    " FIN Flag Count": "fin_flag_count",
    "URG Flag Count": "urg_flag_count",
    " URG Flag Count": "urg_flag_count",
    "Down/Up Ratio": "down_up_ratio",
    " Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "avg_packet_size",
    " Average Packet Size": "avg_packet_size",
    "Fwd Header Length": "fwd_header_len",
    " Fwd Header Length": "fwd_header_len",
    "Bwd Header Length": "bwd_header_len",
    " Bwd Header Length": "bwd_header_len",
    "Fwd IAT Std": "fwd_iat_std",
    " Fwd IAT Std": "fwd_iat_std",
    "Bwd IAT Std": "bwd_iat_std",
    " Bwd IAT Std": "bwd_iat_std",
    "Flow IAT Max": "flow_iat_max",
    " Flow IAT Max": "flow_iat_max",
    " Label": "label",
    "Label": "label",
}

_CICIDS2017_LABEL_MAP = {
    "BENIGN": "BENIGN",
    "Bot": "Botnet",
    "DDoS": "DDoS",
    "DoS Hulk": "DDoS",
    "DoS GoldenEye": "DDoS",
    "DoS slowloris": "DDoS",
    "DoS Slowhttptest": "DDoS",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Heartbleed": "WebAttack",
    "Infiltration": "LateralMovement",
    "PortScan": "PortScan",
    "Web Attack \u2013 Brute Force": "BruteForce",
    "Web Attack \u2013 XSS": "WebAttack",
    "Web Attack \u2013 Sql Injection": "WebAttack",
    "Web Attack - Brute Force": "BruteForce",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
}


def load_cicids2017(csv_dir: str, max_rows: int | None = None) -> pd.DataFrame:
    """Load CIC-IDS-2017 CSV files (MachineLearningCSV format).

    The CIC-IDS-2017 dataset contains flow-level features exported from
    CICFlowMeter. CSV files are typically named by day (Monday, Tuesday, etc.).

    Download: https://www.unb.ca/cic/datasets/ids-2017.html
    """
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {csv_dir}")

    csv_files = sorted(csv_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    dfs = []
    rows_loaded = 0
    for csv_file in csv_files:
        nrows = None
        if max_rows is not None:
            remaining = max_rows - rows_loaded
            if remaining <= 0:
                break
            nrows = remaining
        try:
            df = pd.read_csv(csv_file, nrows=nrows, low_memory=False, encoding="utf-8")
            df.columns = df.columns.str.strip()
            dfs.append(df)
            rows_loaded += len(df)
        except Exception as e:
            print(f"Warning: could not load {csv_file.name}: {e}")
            continue

    if not dfs:
        raise ValueError("No data loaded from CIC-IDS-2017 CSV files")

    combined = pd.concat(dfs, ignore_index=True)
    combined.columns = combined.columns.str.strip()
    combined = combined.rename(columns=_CICIDS2017_COLUMN_MAP)

    if "label" in combined.columns:
        combined["label"] = combined["label"].str.strip().map(
            lambda x: _CICIDS2017_LABEL_MAP.get(x, "BENIGN")
        )
    else:
        combined["label"] = "BENIGN"

    for col in WORLD_MODEL_FEATURES:
        if col not in combined.columns:
            combined[col] = 0.0

    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in WORLD_MODEL_FEATURES:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0).clip(lower=0)

    return combined[WORLD_MODEL_FEATURES + ["label"]]


# ---------------------------------------------------------------------------
# UNSW-NB15 loader
# ---------------------------------------------------------------------------

_UNSW_LABEL_MAP = {
    "Normal": "BENIGN",
    "Exploits": "WebAttack",
    "Reconnaissance": "PortScan",
    "DoS": "DDoS",
    "Generic": "DDoS",
    "Fuzzers": "WebAttack",
    "Analysis": "PortScan",
    "Backdoor": "Botnet",
    "Backdoors": "Botnet",
    "Shellcode": "LateralMovement",
    "Worms": "Exfiltration",
}


def load_unsw_nb15(csv_path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Load UNSW-NB15 dataset CSV files.

    The UNSW-NB15 dataset was created by the Australian Centre for Cyber
    Security (ACCS) using the IXIA PerfectStorm tool. It contains 49 features
    across 9 attack families + normal traffic.

    Download: https://research.unsw.edu.au/projects/unsw-nb15-dataset

    Expected files: UNSW-NB15_1.csv through UNSW-NB15_4.csv, or the
    combined training/testing CSVs.
    """
    p = Path(csv_path)
    if p.is_dir():
        csv_files = sorted(p.glob("*.csv"))
    elif p.is_file():
        csv_files = [p]
    else:
        raise FileNotFoundError(f"UNSW-NB15 path not found: {csv_path}")

    dfs = []
    rows_loaded = 0
    for csv_file in csv_files:
        nrows = None
        if max_rows is not None:
            remaining = max_rows - rows_loaded
            if remaining <= 0:
                break
            nrows = remaining
        try:
            df = pd.read_csv(csv_file, nrows=nrows, low_memory=False)
            df.columns = df.columns.str.strip().str.lower()
            dfs.append(df)
            rows_loaded += len(df)
        except Exception as e:
            print(f"Warning: could not load {csv_file.name}: {e}")

    if not dfs:
        raise ValueError("No data loaded from UNSW-NB15 files")

    combined = pd.concat(dfs, ignore_index=True)

    # Map UNSW-NB15 columns to our schema
    result = pd.DataFrame()
    for col in WORLD_MODEL_FEATURES:
        result[col] = 0.0

    # Direct mappings where available
    col_map = {
        "dur": "flow_duration_ms",
        "spkts": "total_fwd_packets",
        "dpkts": "total_bwd_packets",
        "sbytes": "total_fwd_bytes",
        "dbytes": "total_bwd_bytes",
        "sttl": "ttl_mean",
        "dttl": "ttl_std",
        "swin": "tcp_window_mean",
        "dwin": "tcp_window_std",
        "smean": "fwd_packet_len_mean",
        "dmean": "bwd_packet_len_mean",
        "sjit": "fwd_iat_std",
        "djit": "bwd_iat_std",
        "sinpkt": "fwd_iat_mean",
        "dinpkt": "bwd_iat_mean",
        "tcprtt": "flow_iat_mean",
        "ct_dst_sport_ltm": "unique_dst_ports_per_src",
    }

    for unsw_col, argus_col in col_map.items():
        if unsw_col in combined.columns:
            result[argus_col] = pd.to_numeric(combined[unsw_col], errors="coerce").fillna(0)

    # Duration: UNSW-NB15 stores in seconds, convert to ms
    if "dur" in combined.columns:
        result["flow_duration_ms"] = pd.to_numeric(combined["dur"], errors="coerce").fillna(0) * 1000

    # Compute derived features
    if "spkts" in combined.columns and "dpkts" in combined.columns:
        total_pkts = pd.to_numeric(combined["spkts"], errors="coerce").fillna(0) + \
                     pd.to_numeric(combined["dpkts"], errors="coerce").fillna(0)
        result["packets_per_flow"] = total_pkts

    if "sbytes" in combined.columns and "dbytes" in combined.columns:
        total_bytes = pd.to_numeric(combined["sbytes"], errors="coerce").fillna(0) + \
                      pd.to_numeric(combined["dbytes"], errors="coerce").fillna(0)
        dur = result["flow_duration_ms"].clip(lower=1) / 1000
        result["flow_bytes_per_sec"] = total_bytes / dur
        result["flow_packets_per_sec"] = result["packets_per_flow"] / dur
        avg_size = total_bytes / result["packets_per_flow"].clip(lower=1)
        result["avg_packet_size"] = avg_size

    # TCP flags from ct_flw_http_mthd and is_ftp_login columns
    if "ct_flw_http_mthd" in combined.columns:
        result["psh_flag_count"] = pd.to_numeric(combined["ct_flw_http_mthd"], errors="coerce").fillna(0)

    # Down/up ratio
    fwd = result["total_fwd_bytes"].clip(lower=1)
    bwd = result["total_bwd_bytes"].clip(lower=1)
    result["down_up_ratio"] = bwd / fwd

    # Label mapping
    label_col = None
    for candidate in ["attack_cat", "label", "Label"]:
        if candidate in combined.columns:
            label_col = candidate
            break

    if label_col == "attack_cat":
        result["label"] = combined[label_col].str.strip().map(
            lambda x: _UNSW_LABEL_MAP.get(str(x).strip(), "BENIGN") if pd.notna(x) else "BENIGN"
        )
    elif label_col:
        # Binary label: 0 = normal, 1 = attack
        result["label"] = combined[label_col].apply(
            lambda x: "BENIGN" if str(x).strip() in ("0", "Normal") else "WebAttack"
        )
    else:
        result["label"] = "BENIGN"

    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in WORLD_MODEL_FEATURES:
        result[col] = result[col].clip(lower=0)

    return result[WORLD_MODEL_FEATURES + ["label"]]


# ---------------------------------------------------------------------------
# CICIoT2023 loader
# ---------------------------------------------------------------------------

_CICIOT_LABEL_MAP = {
    "BenignTraffic": "BENIGN",
    "Benign": "BENIGN",
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DoS-TCP_Flood": "DDoS",
    "DoS-UDP_Flood": "DDoS",
    "DoS-HTTP_Flood": "DDoS",
    "DoS-SYN_Flood": "DDoS",
    "Recon-PingSweep": "PortScan",
    "Recon-OSScan": "PortScan",
    "Recon-PortScan": "PortScan",
    "Recon-HostDiscovery": "PortScan",
    "VulnerabilityScan": "PortScan",
    "BruteForce": "BruteForce",
    "Mirai-greeth_flood": "Botnet",
    "Mirai-greip_flood": "Botnet",
    "Mirai-udpplain": "Botnet",
    "Spoofing": "LateralMovement",
    "DNS_Spoofing": "LateralMovement",
    "MITM-ArpSpoofing": "LateralMovement",
    "SqlInjection": "WebAttack",
    "CommandInjection": "WebAttack",
    "XSS": "WebAttack",
    "Backdoor_Malware": "Botnet",
    "Uploading_Attack": "Exfiltration",
}


def load_ciciot2023(csv_path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Load CICIoT2023 dataset.

    The CICIoT2023 dataset contains network traffic from 33 IoT attacks
    in 7 categories. It uses CICFlowMeter features similar to CIC-IDS-2017.

    Download: https://www.unb.ca/cic/datasets/iotdataset-2023.html
    """
    p = Path(csv_path)
    if p.is_dir():
        csv_files = sorted(p.glob("*.csv"))
    elif p.is_file():
        csv_files = [p]
    else:
        raise FileNotFoundError(f"CICIoT2023 path not found: {csv_path}")

    dfs = []
    rows_loaded = 0
    for csv_file in csv_files:
        nrows = None
        if max_rows is not None:
            remaining = max_rows - rows_loaded
            if remaining <= 0:
                break
            nrows = remaining
        try:
            df = pd.read_csv(csv_file, nrows=nrows, low_memory=False)
            df.columns = df.columns.str.strip()
            dfs.append(df)
            rows_loaded += len(df)
        except Exception as e:
            print(f"Warning: could not load {csv_file.name}: {e}")

    if not dfs:
        raise ValueError("No data loaded from CICIoT2023 files")

    combined = pd.concat(dfs, ignore_index=True)

    # CICIoT2023 uses CICFlowMeter-like columns — reuse the 2017 mapping
    combined = combined.rename(columns=_CICIDS2017_COLUMN_MAP)

    # Also try direct column name matches
    direct_maps = {
        "flow_duration": "flow_duration_ms",
        "Header_Length": "fwd_header_len",
        "Duration": "flow_duration_ms",
        "Rate": "flow_packets_per_sec",
        "Srate": "flow_bytes_per_sec",
        "fin_flag_number": "fin_flag_count",
        "syn_flag_number": "syn_flag_count",
        "rst_flag_number": "rst_flag_count",
        "psh_flag_number": "psh_flag_count",
        "ack_flag_number": "ack_flag_count",
        "urg_flag_number": "urg_flag_count",
        "Tot sum": "total_fwd_bytes",
        "Min": "bwd_packet_len_mean",
        "Max": "fwd_packet_len_mean",
        "AVG": "avg_packet_size",
        "Std": "fwd_packet_len_std",
        "Tot size": "total_bwd_bytes",
        "IAT": "flow_iat_mean",
        "Number": "packets_per_flow",
        "Magnitue": "flow_bytes_per_sec",
        "Radius": "ttl_mean",
        "Covariance": "ttl_std",
        "Variance": "flow_iat_std",
        "Weight": "down_up_ratio",
    }
    combined = combined.rename(columns=direct_maps)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    # Label
    label_col = None
    for candidate in ["label", "Label", "attack_type"]:
        if candidate in combined.columns:
            label_col = candidate
            break

    if label_col:
        combined["label"] = combined[label_col].str.strip().map(
            lambda x: _CICIOT_LABEL_MAP.get(x, "BENIGN")
        )
    else:
        combined["label"] = "BENIGN"

    for col in WORLD_MODEL_FEATURES:
        if col not in combined.columns:
            combined[col] = 0.0

    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in WORLD_MODEL_FEATURES:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0).clip(lower=0)

    return combined[WORLD_MODEL_FEATURES + ["label"]]


# ---------------------------------------------------------------------------
# LANL Authentication Dataset loader
# ---------------------------------------------------------------------------

def load_lanl_auth(data_path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Load LANL Unified Host and Network Dataset (authentication events).

    The LANL dataset contains authentication events from ~12,000 users over
    58 consecutive days. It's primarily an authentication log, so we synthesize
    flow-like features from the temporal authentication patterns.

    Download: https://csr.lanl.gov/data/cyber1/

    Expected file: auth.txt.gz or auth.txt (tab/comma separated)
    Columns: time, source_user@domain, dest_user@domain, source_computer,
             dest_computer, auth_type, logon_type, auth_orientation, success/failure
    """
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"LANL auth file not found: {data_path}")

    # LANL auth.txt uses comma-separated values
    df = pd.read_csv(
        str(p), nrows=max_rows, low_memory=False,
        names=["time", "src_user", "dst_user", "src_computer",
               "dst_computer", "auth_type", "logon_type",
               "auth_orientation", "success"],
    )

    # Group auth events into time windows and create flow-like features
    # Each row represents a window of authentication events
    df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(0)
    df["is_failure"] = df["success"].str.strip().str.lower().isin(["fail", "failure", "0"])

    # Window size: 60 seconds
    window_sec = 60
    df["window"] = (df["time"] // window_sec).astype(int)

    grouped = df.groupby("window").agg(
        n_events=pd.NamedAgg(column="time", aggfunc="count"),
        n_failures=pd.NamedAgg(column="is_failure", aggfunc="sum"),
        n_unique_src=pd.NamedAgg(column="src_computer", aggfunc="nunique"),
        n_unique_dst=pd.NamedAgg(column="dst_computer", aggfunc="nunique"),
        n_unique_users=pd.NamedAgg(column="src_user", aggfunc="nunique"),
        duration=pd.NamedAgg(column="time", aggfunc=lambda x: x.max() - x.min()),
    ).reset_index()

    result = pd.DataFrame()
    for col in WORLD_MODEL_FEATURES:
        result[col] = 0.0

    result["flow_duration_ms"] = grouped["duration"].clip(lower=1) * 1000
    result["total_fwd_packets"] = grouped["n_events"]
    result["total_bwd_packets"] = grouped["n_events"] * 0.8
    result["packets_per_flow"] = grouped["n_events"]
    result["unique_dst_ports_per_src"] = grouped["n_unique_dst"]
    result["syn_flag_count"] = grouped["n_events"]  # Each auth attempt is like a SYN
    result["ack_flag_count"] = grouped["n_events"] - grouped["n_failures"]
    result["rst_flag_count"] = grouped["n_failures"]  # Failed auths are like RSTs
    result["flow_packets_per_sec"] = grouped["n_events"] / (grouped["duration"].clip(lower=1))

    # Label based on failure patterns
    failure_rate = grouped["n_failures"] / grouped["n_events"].clip(lower=1)
    n_unique_dst = grouped["n_unique_dst"]

    conditions = [
        (failure_rate > 0.7) & (grouped["n_events"] > 10),  # BruteForce
        (n_unique_dst > 5) & (failure_rate > 0.3),           # LateralMovement
        (n_unique_dst > 15),                                  # PortScan-like recon
    ]
    choices = ["BruteForce", "LateralMovement", "PortScan"]
    result["label"] = np.select(
        conditions, choices, default="BENIGN"
    )

    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in WORLD_MODEL_FEATURES:
        result[col] = result[col].clip(lower=0)

    return result[WORLD_MODEL_FEATURES + ["label"]]


# ---------------------------------------------------------------------------
# DARPA Intrusion Detection loader
# ---------------------------------------------------------------------------

def load_darpa(data_path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Load DARPA Intrusion Detection evaluation data.

    Supports the KDD Cup 1999 / NSL-KDD format (derived from DARPA 1998/1999
    intrusion detection evaluation). These are the most widely used benchmark
    datasets in intrusion detection research.

    Download:
      - KDD Cup 1999: http://kdd.ics.uci.edu/databases/kddcup99/
      - NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html

    Expected: CSV file with 41 features + label (no header in original KDD,
    header present in NSL-KDD).
    """
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"DARPA/KDD dataset not found: {data_path}")

    # KDD Cup column names
    kdd_columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
        "num_failed_logins", "logged_in", "num_compromised", "root_shell",
        "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "label",
    ]

    # Try reading with header first (NSL-KDD format)
    try:
        df = pd.read_csv(str(p), nrows=max_rows, low_memory=False)
        if "label" not in df.columns and "Label" not in df.columns:
            # No header — use KDD column names
            df = pd.read_csv(str(p), nrows=max_rows, low_memory=False,
                             header=None, names=kdd_columns)
    except Exception:
        df = pd.read_csv(str(p), nrows=max_rows, low_memory=False,
                         header=None, names=kdd_columns)

    df.columns = df.columns.str.strip().str.lower()

    # KDD/DARPA attack type → ARGUS label mapping
    _kdd_label_map = {
        "normal": "BENIGN", "normal.": "BENIGN",
        # DoS attacks
        "back": "DDoS", "back.": "DDoS", "land": "DDoS", "land.": "DDoS",
        "neptune": "DDoS", "neptune.": "DDoS", "pod": "DDoS", "pod.": "DDoS",
        "smurf": "DDoS", "smurf.": "DDoS", "teardrop": "DDoS", "teardrop.": "DDoS",
        "apache2": "DDoS", "udpstorm": "DDoS", "processtable": "DDoS",
        "mailbomb": "DDoS",
        # Probe attacks
        "ipsweep": "PortScan", "ipsweep.": "PortScan",
        "nmap": "PortScan", "nmap.": "PortScan",
        "portsweep": "PortScan", "portsweep.": "PortScan",
        "satan": "PortScan", "satan.": "PortScan",
        "mscan": "PortScan", "saint": "PortScan",
        # R2L attacks
        "ftp_write": "BruteForce", "ftp_write.": "BruteForce",
        "guess_passwd": "BruteForce", "guess_passwd.": "BruteForce",
        "imap": "BruteForce", "imap.": "BruteForce",
        "multihop": "LateralMovement", "multihop.": "LateralMovement",
        "phf": "WebAttack", "phf.": "WebAttack",
        "spy": "Exfiltration", "spy.": "Exfiltration",
        "warezclient": "Exfiltration", "warezclient.": "Exfiltration",
        "warezmaster": "Exfiltration", "warezmaster.": "Exfiltration",
        "xlock": "BruteForce", "xsnoop": "Exfiltration",
        "snmpgetattack": "PortScan", "snmpguess": "BruteForce",
        "sendmail": "WebAttack", "named": "WebAttack",
        "httptunnel": "Exfiltration", "worm": "Botnet",
        # U2R attacks
        "buffer_overflow": "WebAttack", "buffer_overflow.": "WebAttack",
        "loadmodule": "LateralMovement", "loadmodule.": "LateralMovement",
        "perl": "WebAttack", "perl.": "WebAttack",
        "rootkit": "Botnet", "rootkit.": "Botnet",
        "sqlattack": "WebAttack", "xterm": "LateralMovement",
        "ps": "PortScan",
    }

    result = pd.DataFrame()
    for col in WORLD_MODEL_FEATURES:
        result[col] = 0.0

    # Map KDD features to our schema
    if "duration" in df.columns:
        result["flow_duration_ms"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0) * 1000
    if "src_bytes" in df.columns:
        result["total_fwd_bytes"] = pd.to_numeric(df["src_bytes"], errors="coerce").fillna(0)
    if "dst_bytes" in df.columns:
        result["total_bwd_bytes"] = pd.to_numeric(df["dst_bytes"], errors="coerce").fillna(0)
    if "count" in df.columns:
        result["total_fwd_packets"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)
        result["packets_per_flow"] = result["total_fwd_packets"]
    if "srv_count" in df.columns:
        result["total_bwd_packets"] = pd.to_numeric(df["srv_count"], errors="coerce").fillna(0)
    if "urgent" in df.columns:
        result["urg_flag_count"] = pd.to_numeric(df["urgent"], errors="coerce").fillna(0)
    if "wrong_fragment" in df.columns:
        result["ip_fragment_flag_count"] = pd.to_numeric(df["wrong_fragment"], errors="coerce").fillna(0)
    if "num_failed_logins" in df.columns:
        result["rst_flag_count"] = pd.to_numeric(df["num_failed_logins"], errors="coerce").fillna(0)
    if "dst_host_count" in df.columns:
        result["unique_dst_ports_per_src"] = pd.to_numeric(df["dst_host_count"], errors="coerce").fillna(0)

    # Derived features
    dur_sec = result["flow_duration_ms"].clip(lower=1) / 1000
    total_bytes = result["total_fwd_bytes"] + result["total_bwd_bytes"]
    result["flow_bytes_per_sec"] = total_bytes / dur_sec
    result["flow_packets_per_sec"] = result["packets_per_flow"] / dur_sec
    result["avg_packet_size"] = total_bytes / result["packets_per_flow"].clip(lower=1)
    fwd = result["total_fwd_bytes"].clip(lower=1)
    bwd = result["total_bwd_bytes"].clip(lower=1)
    result["down_up_ratio"] = bwd / fwd
    result["fwd_packet_len_mean"] = result["total_fwd_bytes"] / result["total_fwd_packets"].clip(lower=1)
    result["bwd_packet_len_mean"] = result["total_bwd_bytes"] / result["total_bwd_packets"].clip(lower=1)

    # SYN/ACK flags from serror_rate and rerror_rate
    if "serror_rate" in df.columns:
        result["syn_flag_count"] = pd.to_numeric(df["serror_rate"], errors="coerce").fillna(0) * 100
    if "rerror_rate" in df.columns:
        result["fin_flag_count"] = pd.to_numeric(df["rerror_rate"], errors="coerce").fillna(0) * 10

    # Label mapping
    if "label" in df.columns:
        result["label"] = df["label"].str.strip().str.lower().str.rstrip(".").map(
            lambda x: _kdd_label_map.get(x, _kdd_label_map.get(x + ".", "BENIGN"))
        )
    else:
        result["label"] = "BENIGN"

    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in WORLD_MODEL_FEATURES:
        result[col] = result[col].clip(lower=0)

    return result[WORLD_MODEL_FEATURES + ["label"]]


# ---------------------------------------------------------------------------
# Dataset Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "synthetic": {
        "name": "Synthetic Temporal Data",
        "description": "Generated temporal attack sequences mimicking CIC-IDS-2017/2018",
        "url": None,
        "format": "Built-in generator",
        "loader": "generate_temporal_dataset",
        "sample_available": True,
    },
    "cicids2017": {
        "name": "CIC-IDS-2017",
        "description": "Canadian Institute for Cybersecurity IDS dataset (2017). "
                       "5 days of traffic: Monday-Friday with 14 attack types.",
        "url": "https://www.unb.ca/cic/datasets/ids-2017.html",
        "format": "CSV (CICFlowMeter exports)",
        "loader": "load_cicids2017",
        "size": "~50GB (full), ~2GB (ML subset)",
        "sample_available": True,
        "sample_path": "data/sample_datasets/cicids2017",
    },
    "cicids2018": {
        "name": "CIC-IDS-2018",
        "description": "Updated version with 7 attack scenarios over 10 days.",
        "url": "https://www.unb.ca/cic/datasets/ids-2018.html",
        "format": "CSV (CICFlowMeter exports)",
        "loader": "load_cicids2018",
        "size": "~16GB",
        "sample_available": True,
        "sample_path": "data/sample_datasets/cicids2018",
    },
    "unsw_nb15": {
        "name": "UNSW-NB15",
        "description": "49 features, 9 attack families + normal. Created by ACCS "
                       "using IXIA PerfectStorm.",
        "url": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "format": "CSV",
        "loader": "load_unsw_nb15",
        "size": "~2GB",
        "sample_available": True,
        "sample_path": "data/sample_datasets/unsw_nb15",
    },
    "ctu13": {
        "name": "CTU-13",
        "description": "13 botnet traffic scenarios captured at CTU University, Prague.",
        "url": "https://www.stratosphereips.org/datasets-ctu13",
        "format": "BiNetFlow",
        "loader": "load_ctu13",
        "size": "~3GB",
        "sample_available": True,
        "sample_path": "data/sample_datasets/ctu13/sample_ctu13.binetflow",
    },
    "ciciot2023": {
        "name": "CICIoT2023",
        "description": "33 IoT attack types across 7 categories from 105 IoT devices.",
        "url": "https://www.unb.ca/cic/datasets/iotdataset-2023.html",
        "format": "CSV",
        "loader": "load_ciciot2023",
        "size": "~15GB",
        "sample_available": True,
        "sample_path": "data/sample_datasets/ciciot2023",
    },
    "lanl": {
        "name": "LANL Authentication Dataset",
        "description": "58 days of authentication events from ~12,000 users "
                       "at Los Alamos National Laboratory.",
        "url": "https://csr.lanl.gov/data/cyber1/",
        "format": "CSV (auth.txt.gz)",
        "loader": "load_lanl_auth",
        "size": "~12GB compressed",
        "sample_available": True,
        "sample_path": "data/sample_datasets/lanl/sample_auth.txt",
    },
    "darpa": {
        "name": "DARPA / KDD Cup 1999 / NSL-KDD",
        "description": "Classic IDS benchmark from DARPA 1998/1999 evaluation. "
                       "41 features, 4 attack categories (DoS, Probe, R2L, U2R).",
        "url": "https://www.unb.ca/cic/datasets/nsl.html",
        "format": "CSV",
        "loader": "load_darpa",
        "size": "~150MB (NSL-KDD)",
        "sample_available": True,
        "sample_path": "data/sample_datasets/darpa/sample_nsl_kdd.csv",
    },
}


def list_datasets() -> dict:
    """Return the dataset registry with metadata."""
    return DATASET_REGISTRY


if __name__ == "__main__":
    print("ARGUS Dataset Registry")
    print("=" * 60)
    for key, info in DATASET_REGISTRY.items():
        print(f"\n  {key}:")
        print(f"    Name: {info['name']}")
        print(f"    URL:  {info.get('url', 'Built-in')}")
        print(f"    Size: {info.get('size', 'N/A')}")

    # Demo: load synthetic data
    X, y_labels, y_inf = load_dataset("synthetic", n_sequences=100)
    print(f"\nLoaded synthetic dataset:")
    print(f"  X: {X.shape}")
    print(f"  y_labels: {y_labels.shape}")
    print(f"  y_infiltration: {y_inf.shape}")

    (X_train, yl_train, yi_train), (X_test, yl_test, yi_test) = \
        train_test_split_temporal(X, y_labels, y_inf)
    print(f"  Train: {X_train.shape[0]} sequences")
    print(f"  Test:  {X_test.shape[0]} sequences")
