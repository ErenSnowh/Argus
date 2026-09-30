"""
binning.py
----------
Host/time binning for the CIC-IDS-2017 flow tier (plan D1, WP2).

A state S_t is one source host aggregated over one fixed wall-clock bin:

    key        (src_ip, bin_start), bin_start = floor(unix_seconds / bin_seconds) * bin_seconds
    features   the 24 FLOW_FEATURE_COLUMNS + 6 EXTENDED_FLOW_COLUMNS aggregated over the
               bin's flows, the 8 TOPOLOGY_FEATURE_COLUMNS computed from the bin's real
               dst_ip / dst_port / timestamp values, plus n_flows = 39 columns
               (feature_list, in that order)
    label      the most severe non-BENIGN stage in the bin (SEVERITY = ATTACK_STAGE_INDEX,
               features.py:117-126); BENIGN only when every flow in the bin is BENIGN

The 8 PACKET_LEVEL_COLUMNS are NOT part of this schema and are never zero-filled into
it: CICFlowMeter tables have no packet-level columns, so those values exist only for the
packet tier and stay NaN-with-a-coverage-flag there (amended D2).

Aggregation rules (D1): `sum` for counters (`*_packets`, `*_bytes`, `*_flag_count`),
`mean` over the bin's flows for durations / lengths / IATs / rates / ratios, `max` for
the one column whose frozen name is a maximum (`flow_iat_max`), and two derived columns:

    unique_dst_ports_per_src = nunique(dst_port) over the bin's flows
                               (CICFlowMeter rows carry no per-flow distinct-port column,
                               so this is measured per bin and NEVER summed)
    packets_per_flow         = (total_fwd_packets + total_bwd_packets) / n_flows

`unique_dst_ports_per_src` keeps its frozen FLOW_FEATURE_COLUMNS name; it is a per-bin
nunique even though the name says "per_src" (the src is the bin key).

Two D3 amendments are implemented here. Infiltration (36 flows) and Heartbleed (11)
keep their mapped stage for provenance and count as attacks in the binary target. WP2
only MARKS their stage targets: a bin whose only malicious evidence is one of those two
families carries `stage_valid = False`, and `make_sequences` writes STAGE_TARGET_IGNORE
(-1) as that bin's `y_stage_*` value.

Nothing masks those markers yet, so this module does not claim the model is protected.
`MultiTaskLoss` builds a default `nn.CrossEntropyLoss()` (model.py:324) and passes stage
targets to it at model.py:342 (primary head) and model.py:368 (horizon heads), where
`ignore_index` is -100 - so -1 is NOT ignored there and would be read as a class index.
Those samples are excluded from the stage targets in the DATA, not from the loss. WP3
must add and test masking in both stage losses before any training run reads a parquet
produced here.

The binary target is unaffected: `y_attack_within_h{k}` still counts those flows, and
-1 != 0 keeps `y_attack_within_h1 == (y_stage_h1 != 0)` true.

Nothing here trains, benchmarks or reads a PCAP; it is pure data shaping.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ml.world_model.features import (
    ATTACK_STAGE_INDEX,
    EXTENDED_FLOW_COLUMNS,
    FLOW_FEATURE_COLUMNS,
    TOPOLOGY_FEATURE_COLUMNS,
)

# Severity order used to label a bin: the kill-chain position of features.py:117-126.
SEVERITY = ATTACK_STAGE_INDEX

# The value written into `y_stage_*` for a bin whose only malicious evidence is one of
# the two families D3 keeps out of the stage head. It is a MARK, not a mask: nothing in
# this repository ignores it (the losses at model.py:342 and model.py:368 use a default
# `nn.CrossEntropyLoss()` whose ignore_index is -100), and it is deliberately NOT -100 so
# that a training run started before WP3 adds masking fails loudly instead of silently
# dropping those samples. WP3 owns the masking.
STAGE_TARGET_IGNORE = -1
STAGE_EXCLUDED_FAMILIES = frozenset({"Infiltration", "Heartbleed"})

# The 39 model columns, in order: 24 flow + 6 extended + 8 topology + n_flows.
FLOW_TIER_FEATURE_LIST = (
    list(FLOW_FEATURE_COLUMNS)
    + list(EXTENDED_FLOW_COLUMNS)
    + list(TOPOLOGY_FEATURE_COLUMNS)
    + ["n_flows"]
)

# Non-feature columns carried in the bin table alongside the 39 features.
BIN_KEY_COLUMNS = ("src_ip", "bin_start")
BIN_META_COLUMNS = ("day", "bin_index", "label", "label_index", "label_flows", "stage_valid")

SUM_COLUMNS = (
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "syn_flag_count",
    "ack_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "fin_flag_count",
    "urg_flag_count",
)
MEAN_COLUMNS = (
    "flow_duration_ms",
    "fwd_packet_len_mean",
    "fwd_packet_len_std",
    "bwd_packet_len_mean",
    "bwd_packet_len_std",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "flow_iat_mean",
    "flow_iat_std",
    "fwd_iat_mean",
    "fwd_iat_std",
    "bwd_iat_mean",
    "bwd_iat_std",
    "avg_packet_size",
    "down_up_ratio",
    "fwd_header_len",
    "bwd_header_len",
)
MAX_COLUMNS = ("flow_iat_max",)
DERIVED_COLUMNS = ("unique_dst_ports_per_src", "packets_per_flow")

REQUIRED_FLOW_COLUMNS = (
    "src_ip",
    "dst_ip",
    "dst_port",
    "timestamp",
    "label",
)

# `flow_duration_ms` is a mean of the raw `Flow Duration` column; CICFlowMeter writes
# microseconds and dataset_loader.py:566-567 renames the column without rescaling, so no
# rescale is applied here either - the frozen name is kept, the unit is documented.
FLOW_DURATION_NOTE = "mean of raw Flow Duration (microseconds in CICFlowMeter; name frozen)"


# ---------------------------------------------------------------------------
# Label normalisation (V12)
# ---------------------------------------------------------------------------
# The pinned snapshot spells every Web-Attack family with U+0096 ("Web Attack \x96
# Brute Force"). dataset_loader.py:639-644 has ASCII-hyphen and en-dash (U+2013)
# spellings but NOT U+0096, so on this snapshot those rows fall through
# `.get(x, "BENIGN")` at dataset_loader.py:691 and are silently relabelled benign.
# These entries are the ones measured on the snapshot; the hyphen / en-dash /
# U+FFFD spellings are kept as defensive entries. Unknown labels now RAISE.
_MOJIBAKE_JOINERS = ("\u0096", "\u2013", "\u2014", "\ufffd")

# canonical dataset family -> stage name (features.py ATTACK_LABELS vocabulary)
CICIDS2017_FAMILY_TO_STAGE = {
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
    "Web Attack - Brute Force": "BruteForce",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
}

def _canon_label(raw: object) -> str:
    """Mojibake joiners -> "-", whitespace collapsed, dash SPACING normalised.

    Only spacing around a dash is collapsed: "FTP-Patator" must stay "FTP-Patator" (a
    regex that rewrote every dash to " - " would turn it into "FTP - Patator" and lose
    two families), while "Web Attack \\u0096 Brute Force" becomes "Web Attack-Brute Force".
    """
    text = str(raw).strip()
    for joiner in _MOJIBAKE_JOINERS:
        text = text.replace(joiner, "-")
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s*-\s*", "-", text)


# accepted spellings -> canonical family, keyed by the canonical spelling so the
# mojibake / dash / case variants all resolve to one key.
_RAW_LABEL_ALIASES = {
    "BENIGN": "BENIGN",
    "Benign": "BENIGN",
    "Bot": "Bot",
    "DDoS": "DDoS",
    "DoS Hulk": "DoS Hulk",
    "DoS GoldenEye": "DoS GoldenEye",
    "DoS slowloris": "DoS slowloris",
    "DoS Slowhttptest": "DoS Slowhttptest",
    "FTP-Patator": "FTP-Patator",
    "SSH-Patator": "SSH-Patator",
    "Heartbleed": "Heartbleed",
    "Infiltration": "Infiltration",
    "PortScan": "PortScan",
    "Web Attack - Brute Force": "Web Attack - Brute Force",
    "Web Attack - XSS": "Web Attack - XSS",
    "Web Attack - Sql Injection": "Web Attack - Sql Injection",
    "Web Attack - SQL Injection": "Web Attack - Sql Injection",
    "Web Attack - Sql injection": "Web Attack - Sql Injection",
}

CICIDS2017_FAMILY_ALIASES = {
    _canon_label(key): value for key, value in _RAW_LABEL_ALIASES.items()
}

CICIDS2017_FAMILIES = tuple(sorted(CICIDS2017_FAMILY_TO_STAGE))


class UnmappedLabelError(ValueError):
    """A raw label that is not in the pinned CIC-IDS-2017 vocabulary.

    Raised instead of falling back to BENIGN: silently relabelling an attack as
    benign is the failure V12 documents. The message names every distinct offender
    so a caller can print them in a manifest (`unmapped_labels` must be empty).
    """


def normalise_cicids2017_label(raw: object) -> str:
    """Raw CIC-IDS-2017 label -> canonical family key (whitespace + mojibake).

    Raises UnmappedLabelError for anything outside the pinned vocabulary.
    """
    if raw is None:
        raise UnmappedLabelError("unmapped CIC-IDS-2017 label: None")
    try:
        if isinstance(raw, float) and np.isnan(raw):
            raise UnmappedLabelError("unmapped CIC-IDS-2017 label: NaN")
    except TypeError:
        pass
    text = _canon_label(raw)
    family = CICIDS2017_FAMILY_ALIASES.get(text)
    if family is None:
        raise UnmappedLabelError(f"unmapped CIC-IDS-2017 label: {raw!r}")
    return family


def normalise_cicids2017_labels(values: Iterable[object]) -> tuple[list[str], list[str]]:
    """Vectorised-friendly helper: (families, distinct unmapped raw labels).

    Never raises, so a caller can report every offender at once instead of dying on
    the first one.
    """
    families: list[str] = []
    unmapped: list[str] = []
    seen_unmapped: set[str] = set()
    for value in values:
        try:
            families.append(normalise_cicids2017_label(value))
        except UnmappedLabelError:
            key = repr(value)
            if key not in seen_unmapped:
                seen_unmapped.add(key)
                unmapped.append(str(value))
            families.append("")
    return families, unmapped


def cicids2017_stage(family: str) -> str:
    """Canonical family -> stage name (raises UnmappedLabelError if unknown)."""
    stage = CICIDS2017_FAMILY_TO_STAGE.get(family)
    if stage is None:
        raise UnmappedLabelError(f"unmapped CIC-IDS-2017 family: {family!r}")
    return stage


def cicids2017_stage_index(family: str) -> int:
    """Canonical family -> kill-chain severity index (SEVERITY)."""
    return SEVERITY[cicids2017_stage(family)]


def is_stage_excluded(family: str) -> bool:
    """True for the two families excluded from y_stage_* by the D3 amendment."""
    return family in STAGE_EXCLUDED_FAMILIES


STAGE_BY_INDEX = {index: stage for stage, index in SEVERITY.items()}


def _subnet24(values: pd.Series) -> pd.Series:
    """IPv4 /24 prefix of a string IP column (CIC-IDS-2017 is IPv4-only)."""
    return values.str.rsplit(".", n=1).str[0]


def _unique_frozenset(values: pd.Series) -> frozenset:
    return frozenset(values)


def prepare_flow_frame(df: pd.DataFrame, entity: str = "src_ip",
                       bin_seconds: int = 60) -> pd.DataFrame:
    """Validate a flow table and derive the bin key plus per-flow label columns.

    Mutates nothing: returns a new frame carrying the canonical columns, the bin key
    (`bin_index`, monotone across days, and `bin_start` / `day`), the per-flow family
    and stage, and two integer flags used by the aggregation (`__attack`,
    `__cross_subnet`).

    Fails loudly instead of filling:
      * a missing required column raises KeyError naming it (a real CIC export has
        all 30 feature columns; an absent one is a wrong variant, not a zero),
      * an unparsable or absent timestamp raises ValueError (all-null padding rows
        must be dropped by the caller BEFORE binning, never binned at epoch 0),
      * a label outside the pinned vocabulary raises UnmappedLabelError listing
        every distinct offender.
    """
    missing = [c for c in REQUIRED_FLOW_COLUMNS if c not in df.columns]
    missing += [c for c in (*SUM_COLUMNS, *MEAN_COLUMNS, *MAX_COLUMNS) if c not in df.columns]
    if missing:
        raise KeyError("flow table is missing required column(s): " + ", ".join(sorted(set(missing))))

    families, unmapped = normalise_cicids2017_labels(df["label"].tolist())
    if unmapped:
        raise UnmappedLabelError(
            "unmapped CIC-IDS-2017 label(s): " + ", ".join(repr(u) for u in unmapped)
        )

    work = df.copy()
    work["label_family"] = families
    work["label_stage"] = [cicids2017_stage(f) for f in families]

    timestamp = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    bad = int(timestamp.isna().sum())
    if bad:
        raise ValueError(
            f"{bad} flow row(s) have a missing or unparsable timestamp; drop the "
            "all-null padding rows before binning (never fill them)"
        )
    work["timestamp"] = timestamp

    epoch_sec = (timestamp - pd.Timestamp("1970-01-01T00:00:00Z")) // pd.Timedelta(seconds=1)
    bin_index = (epoch_sec // bin_seconds).astype("int64")
    work["bin_index"] = bin_index
    work["bin_start"] = pd.to_datetime(bin_index * bin_seconds, unit="s", utc=True)
    work["day"] = work["bin_start"].dt.strftime("%Y-%m-%d")

    stage_index = np.array([SEVERITY[s] for s in work["label_stage"]], dtype=np.int64)
    excluded = np.array([is_stage_excluded(f) for f in families], dtype=bool)
    work["__stage_index"] = stage_index
    work["__excluded"] = excluded.astype(np.int64)
    work["__valid_stage"] = np.where(excluded, -1, stage_index)
    work["__excluded_stage"] = np.where(excluded, stage_index, -1)
    work["__attack"] = (stage_index != SEVERITY["BENIGN"]).astype(np.int64)
    work["__cross_subnet"] = (
        _subnet24(work["src_ip"]) != _subnet24(work["dst_ip"])
    ).astype(np.int64)

    # Non-finite cells are dropped for THAT column only (pandas mean/sum skip NaN); they
    # are never replaced with 0. `Flow Bytes/s` / `Flow Packets/s` produce inf on
    # zero-duration flows. Infinity and pre-existing missingness are counted SEPARATELY:
    # an infinity is a value this pipeline destroyed, while a NaN that was already in the
    # export is a value it never had - lumping them into one "dropped" number would make
    # the manifest overstate what was cleaned.
    numeric = work[list((*SUM_COLUMNS, *MEAN_COLUMNS, *MAX_COLUMNS))]
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    infinite = int(np.isinf(values).sum())
    missing = int(np.isnan(values).sum())
    work[list(numeric.columns)] = numeric.replace([np.inf, -np.inf], np.nan)
    work.attrs["infinite_cells"] = infinite
    work.attrs["missing_cells"] = missing

    dst_port = pd.to_numeric(work["dst_port"], errors="coerce")
    work["dst_port"] = dst_port
    work["__edge"] = work["dst_ip"].astype(str) + "|" + dst_port.astype("Int64").astype(str)
    return work


def _aggregate_bins(work: pd.DataFrame, entity: str = "src_ip",
                    bin_seconds: int = 60) -> pd.DataFrame:
    """One row per (entity, bin_index): the 30 aggregated features and the bin label."""
    spec = {}
    for column in SUM_COLUMNS:
        # min_count=1 is required, not cosmetic: pandas' plain sum() of an all-NaN group
        # returns 0.0, which would silently re-introduce the "fill missing with 0" policy
        # this pipeline exists to avoid. With min_count=1 an all-NaN group stays NaN, so
        # "the counter could not be computed" is visible in the artifact instead of
        # looking like "no packets happened".
        spec[column] = pd.NamedAgg(
            column=column, aggfunc=lambda values: values.sum(min_count=1))
    for column in MEAN_COLUMNS:
        spec[column] = pd.NamedAgg(column=column, aggfunc="mean")
    for column in MAX_COLUMNS:
        spec[column] = pd.NamedAgg(column=column, aggfunc="max")
    spec["n_flows"] = pd.NamedAgg(column="dst_port", aggfunc="size")
    spec["unique_dst_ports_per_src"] = pd.NamedAgg(column="dst_port", aggfunc="nunique")
    spec["n_dst_hosts"] = pd.NamedAgg(column="dst_ip", aggfunc="nunique")
    spec["n_edges"] = pd.NamedAgg(column="__edge", aggfunc="nunique")
    spec["cross_subnet_edges"] = pd.NamedAgg(column="__cross_subnet", aggfunc="sum")
    spec["label_flows"] = pd.NamedAgg(column="__attack", aggfunc="sum")
    spec["max_valid_stage"] = pd.NamedAgg(column="__valid_stage", aggfunc="max")
    spec["max_excluded_stage"] = pd.NamedAgg(column="__excluded_stage", aggfunc="max")

    bins = work.groupby([entity, "bin_index"], sort=True).agg(**spec).reset_index()

    bins["packets_per_flow"] = (
        (bins["total_fwd_packets"] + bins["total_bwd_packets"]) / bins["n_flows"]
    )
    # "repeated connection attempts to the same destination": flows per distinct
    # (dst_ip, dst_port) edge in the bin, so 1.0 means every flow was a new edge.
    bins["connection_repetition"] = bins["n_flows"] / bins["n_edges"]
    # On the flow tier src_fanout and unique_dst_hosts are the same bin-local count of
    # distinct destinations. The frozen schema names both, and the packet tier differs
    # (a PCAP bin can see hosts that CICFlowMeter's flow table missed).
    bins["src_fanout"] = bins["n_dst_hosts"].astype("int64")
    bins["unique_dst_hosts"] = bins["n_dst_hosts"].astype("int64")

    # label = the most severe VALID stage in the bin. A bin whose only malicious
    # evidence is an excluded family (Infiltration / Heartbleed) keeps its mapped stage
    # for provenance but is flagged stage_valid=False, and make_sequences writes
    # STAGE_TARGET_IGNORE (-1) into its stage targets - a mark the loss does not mask yet
    # (see the module docstring; the masking is WP3's work).
    has_valid = bins["max_valid_stage"] > 0
    has_excluded = bins["max_excluded_stage"] > 0
    label_index = np.where(
        has_valid, bins["max_valid_stage"],
        np.where(has_excluded, bins["max_excluded_stage"], 0),
    )
    bins["label_index"] = label_index
    bins["stage_valid"] = (~has_excluded) | has_valid
    bins["label"] = [STAGE_BY_INDEX[int(value)] for value in bins["label_index"]]

    bins["bin_start"] = pd.to_datetime(bins["bin_index"] * bin_seconds, unit="s", utc=True)
    bins["day"] = bins["bin_start"].dt.strftime("%Y-%m-%d")
    bins["n_flows"] = bins["n_flows"].astype("int64")
    bins["label_flows"] = bins["label_flows"].astype("int64")
    return bins


def _add_topology(bins: pd.DataFrame, work: pd.DataFrame,
                  entity: str = "src_ip") -> pd.DataFrame:
    """dst_fanin, unique_src_hosts, new_host_edges and new_dst_ports.

    `unique_src_hosts` and `dst_fanin` are bin-GLOBAL: they describe the network during
    that wall-clock bin, which is what makes them graph signal rather than a rescaled
    flow count. `new_host_edges` and `new_dst_ports` are measured against every PRIOR
    bin of the same host on the same day, so a host can only be credited once for an
    edge it has already used, and a new day starts with an empty history instead of
    inheriting yesterday's edges. `cross_subnet_edges` is already summed per bin.
    """
    keys = [entity, "day", "bin_index"]

    src_per_bin = work.groupby("bin_index", sort=False)["src_ip"].nunique()
    fanin_map = (work.groupby(["bin_index", "dst_ip"], sort=False)["src_ip"]
                     .nunique().rename("dst_fanin"))
    pairs = work[[entity, "bin_index", "dst_ip"]].drop_duplicates()
    pairs = pairs.merge(fanin_map.reset_index(), on=["bin_index", "dst_ip"], how="left")
    fanin = (pairs.groupby([entity, "bin_index"], sort=False)["dst_fanin"]
                  .mean().rename("dst_fanin"))

    observed = (work.groupby(keys, sort=False)
                    .agg(dst_hosts=pd.NamedAgg(column="dst_ip", aggfunc=_unique_frozenset),
                         dst_ports=pd.NamedAgg(column="dst_port", aggfunc=_unique_frozenset))
                    .reset_index())
    novelty_rows: list[tuple] = []
    for (host, day), group in observed.groupby([entity, "day"], sort=True):
        group = group.sort_values("bin_index")
        seen_hosts: set = set()
        seen_ports: set = set()
        for bin_index, hosts, ports in zip(group["bin_index"], group["dst_hosts"],
                                           group["dst_ports"]):
            novelty_rows.append((host, day, int(bin_index), len(hosts - seen_hosts),
                                 len(ports - seen_ports)))
            seen_hosts |= hosts
            seen_ports |= ports
    novelty = pd.DataFrame(
        novelty_rows,
        columns=[entity, "day", "bin_index", "new_host_edges", "new_dst_ports"],
    )

    bins = bins.merge(src_per_bin.rename("unique_src_hosts").reset_index(),
                      on="bin_index", how="left")
    bins = bins.merge(fanin.reset_index(), on=[entity, "bin_index"], how="left")
    bins = bins.merge(novelty, on=keys, how="left")
    return bins



def build_host_time_bins(df: pd.DataFrame, bin_seconds: int = 60,
                         entity: str = "src_ip") -> pd.DataFrame:
    """D1 binning: one row per (source host, fixed wall-clock bin).

    Returns the bin table described at the top of this module: the key columns, the 39
    `feature_list` columns in order, then `label` / `label_index` / `label_flows` /
    `stage_valid`. Pass one day of flows at a time: the topologies that look back at
    "prior bins of this host" then stay inside the file's own day, which is also what
    the per-day split protocols (P-A / P-B) assume.

    Empty bins are NOT materialised: a host with no flows in a bin simply has no row.
    That is why make_sequences demands contiguous bin_index runs instead of treating a
    missing bin as an empty (zero-filled) state.
    """
    if bin_seconds <= 0:
        raise ValueError(f"bin_seconds must be positive, got {bin_seconds}")
    work = prepare_flow_frame(df, entity=entity, bin_seconds=bin_seconds)
    bins = _aggregate_bins(work, entity=entity, bin_seconds=bin_seconds)
    bins = _add_topology(bins, work, entity=entity)

    columns = (
        [entity, "bin_start", "day", "bin_index"]
        + list(FLOW_TIER_FEATURE_LIST)
        + ["label", "label_index", "label_flows", "stage_valid"]
    )
    missing = [column for column in columns if column not in bins.columns]
    if missing:
        raise RuntimeError("bin table is missing column(s): " + ", ".join(missing))

    out = bins[columns].sort_values([entity, "bin_index"]).reset_index(drop=True)
    out["bin_index"] = out["bin_index"].astype("int32")
    out["n_flows"] = out["n_flows"].astype("int32")
    out["label_flows"] = out["label_flows"].astype("int32")
    out["label_index"] = out["label_index"].astype("int8")
    for column in FLOW_TIER_FEATURE_LIST:
        if column != "n_flows":
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    # Carried for the manifest: how many feature cells were infinite (a value destroyed
    # here) versus already missing in the export (a value never had). Kept as two fields
    # so neither number is mistaken for the other.
    out.attrs["infinite_cells"] = int(work.attrs.get("infinite_cells", 0))
    out.attrs["missing_cells"] = int(work.attrs.get("missing_cells", 0))
    out.attrs["rows_binned"] = int(len(work))
    return out

# ---------------------------------------------------------------------------
# Sequences and targets
# ---------------------------------------------------------------------------


@dataclass
class BinnedSequences:
    """W consecutive bins of one host, plus the k-step targets (plan D1/agent-prompt WP2).

    `X` is `(N, W, F) float32` over `feature_list`. `y_stage_now` is the stage at t;
    `y_stage_h[k]` / `y_attack_within_h[k]` are the stage at t+k and "any non-BENIGN in
    (t, t+k]".

    Stage targets carry STAGE_TARGET_IGNORE (-1) where the target bin belongs to one of
    the two families D3 keeps out of the stage head (Infiltration, Heartbleed). That
    marker is masked by NOTHING in this repository yet: model.py:324 builds a default
    `nn.CrossEntropyLoss()` (ignore_index=-100), used at model.py:342 and model.py:368, so
    a consumer has to pass `ignore_index=stage_ignore_index` or apply
    `stage_target_masks()` - that is WP3's work and it must be tested there.
    -1 != 0 keeps `y_attack_within_h1 == (y_stage_h1 != 0)` true for every sequence.
    """

    X: np.ndarray
    y_stage_now: np.ndarray
    y_stage_h: dict[int, np.ndarray]
    y_attack_within_h: dict[int, np.ndarray]
    meta: pd.DataFrame
    feature_list: list[str]
    bin_seconds: int
    W: int
    horizons: tuple[int, ...]
    stage_ignore_index: int = STAGE_TARGET_IGNORE

    @property
    def n_sequences(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[2]) if self.X.ndim == 3 else 0

    def stage_target_masks(self) -> dict[str, np.ndarray]:
        """True where a stage target is a real stage index (i.e. not the -1 marker).

        This is the mask WP3 needs - `nn.CrossEntropyLoss(ignore_index=-1)` or an explicit
        `where()` in both stage losses. Nothing in this repository consumes it yet: the
        losses at model.py:342 and model.py:368 do not ignore -1 today.
        """
        masks = {"y_stage_now": self.y_stage_now != self.stage_ignore_index}
        for horizon, values in self.y_stage_h.items():
            masks[f"y_stage_h{horizon}"] = values != self.stage_ignore_index
        return masks


SEQUENCE_META_COLUMNS = (
    "host", "day", "bin_index", "bin_index_start", "bin_index_end", "t_start",
    "window_start", "n_flows",
)


def make_sequences(bins: pd.DataFrame, W: int = 10, horizons: Sequence[int] = (1, 2, 4),
                   stride: int = 1, bin_seconds: int = 60,
                   feature_list: Sequence[str] | None = None,
                   entity: str = "src_ip") -> BinnedSequences:
    """W consecutive bins of one host -> sequences and k-step targets.

    A sequence exists only when the bins t-W+1 .. t+max(horizons) all exist for that
    host AND are contiguous (`bin_index` step of exactly 1), i.e. the window really is W
    contiguous minutes of one host's history and every target bin is present. A gap ends
    a run: bins are never stitched across a hole and a missing bin is never treated as
    an all-zero state. Because the split is per (host, day), no sequence can cross a
    day boundary or a host boundary - that is asserted in the tests.

    Targets: `y_stage_now` = stage at t, `y_stage_h[k]` = stage at t+k, and
    `y_attack_within_h[k]` = any non-BENIGN bin in (t, t+k] - so for k=1 it is exactly
    `y_stage_h1 != 0`, including for target bins that carry STAGE_TARGET_IGNORE.
    """
    if W < 1:
        raise ValueError(f"W must be >= 1, got {W}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    horizon_tuple = tuple(sorted({int(k) for k in horizons}))
    if not horizon_tuple or horizon_tuple[0] < 1:
        raise ValueError(f"horizons must be positive k values, got {tuple(horizons)}")

    features = list(FLOW_TIER_FEATURE_LIST if feature_list is None else feature_list)
    required = (entity, "day", "bin_index", "bin_start", "n_flows", "label_index",
                "stage_valid")
    missing = [c for c in required if c not in bins.columns]
    missing += [c for c in features if c not in bins.columns]
    if missing:
        raise KeyError("bin table is missing column(s): " + ", ".join(sorted(set(missing))))
    # `bin_seconds` is an explicit input on purpose: it CANNOT be recovered from the bin
    # table, because bin_index is a bin counter (unix_seconds // bin_seconds), so two
    # adjacent minutes differ by 1 whether the grid was 30 s or 60 s. An earlier draft
    # inferred it from the smallest positive bin_index delta and returned 1 - a silently
    # wrong bin_seconds in every BinnedSequences.
    if bin_seconds <= 0:
        raise ValueError(f"bin_seconds must be positive, got {bin_seconds}")

    need = W + max(horizon_tuple)
    X_parts: list[np.ndarray] = []
    y_now: list[int] = []
    y_stage: dict[int, list[int]] = {k: [] for k in horizon_tuple}
    y_attack: dict[int, list[int]] = {k: [] for k in horizon_tuple}
    meta_rows: list[dict] = []

    for (host, day), group in bins.groupby([entity, "day"], sort=True):
        group = group.sort_values("bin_index")
        index = group["bin_index"].to_numpy(dtype="int64")
        stage = group["label_index"].to_numpy(dtype="int64")
        valid = group["stage_valid"].to_numpy(dtype=bool)
        malicious = stage != SEVERITY["BENIGN"]
        matrix = group[features].to_numpy(dtype=np.float32)
        starts = group["bin_start"]
        counts = group["n_flows"].to_numpy(dtype="int64")

        for start in range(0, len(index) - need + 1, stride):
            if index[start + need - 1] - index[start] != need - 1:
                continue  # a hole in the run: not W contiguous minutes of history
            current = start + W - 1
            X_parts.append(matrix[start:start + W])
            y_now.append(int(stage[current]) if valid[current] else STAGE_TARGET_IGNORE)
            for horizon in horizon_tuple:
                target = current + horizon
                y_stage[horizon].append(
                    int(stage[target]) if valid[target] else STAGE_TARGET_IGNORE
                )
                y_attack[horizon].append(int(malicious[current + 1:target + 1].any()))
            meta_rows.append({
                "host": host,
                "day": day,
                "bin_index": int(index[current]),
                "bin_index_start": int(index[start]),
                "bin_index_end": int(index[current]),
                "t_start": starts.iloc[current],
                "window_start": starts.iloc[start],
                "n_flows": int(counts[current]),
            })

    if X_parts:
        X = np.stack(X_parts).astype(np.float32)
    else:
        X = np.empty((0, W, len(features)), dtype=np.float32)

    return BinnedSequences(
        X=X,
        y_stage_now=np.asarray(y_now, dtype=np.int64),
        y_stage_h={k: np.asarray(v, dtype=np.int64) for k, v in y_stage.items()},
        y_attack_within_h={k: np.asarray(v, dtype=np.int8) for k, v in y_attack.items()},
        meta=pd.DataFrame(meta_rows, columns=list(SEQUENCE_META_COLUMNS)),
        feature_list=features,
        bin_seconds=int(bin_seconds),
        W=int(W),
        horizons=horizon_tuple,
    )
