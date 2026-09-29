"""
test_pcap_bins.py
-----------------
Tests for the streaming fixed-60 s-bin packet-tier extractor (D2 amendment)
and the demo-extractor RST fix. Synthetic captures are built in-process with
explicit packet times - no network, no large downloads.

Run: pytest argus/tests/test_pcap_bins.py -q
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scapy.layers.inet import IP, TCP, UDP  # noqa: E402
from scapy.packet import Raw  # noqa: E402
from scapy.utils import wrpcap  # noqa: E402

from ml.world_model.pcap_bins import (  # noqa: E402
    DEFAULT_REORDER_GRACE_SEC,
    BackwardsTimestampError,
    bin_start_for,
    extract_pcap_bins,
    join_packet_features,
    measure_bin_coverage,
    measure_clock_offset,
)

# 1_500_000_000 is exactly divisible by 60, so bin edges are trivial to
# reason about in assertions.
BASE = 1_500_000_000
BIN = 60


def _tcp(src, dst, sport, dport, seq, flags="PA", payload=b"x" * 10):
    return IP(src=src, dst=dst) / TCP(
        sport=sport, dport=dport, flags=flags, seq=seq) / Raw(payload)


# --------------------------------------------------------------------------
# fixed bins keyed like D1
# --------------------------------------------------------------------------
def test_fixed_60s_bins_keyed_by_src(tmp_path):
    p = tmp_path / "bins.pcap"
    pkts = []
    for ts, src in [
        (BASE + 0.5, "10.0.0.1"),
        (BASE + 30.0, "10.0.0.2"),   # other src, same bin
        (BASE + 59.9, "10.0.0.1"),   # same bin as first packet
        (BASE + 60.1, "10.0.0.1"),   # next bin
    ]:
        pkt = IP(src=src, dst="10.0.0.9") / UDP(sport=1234, dport=53)
        pkt.time = ts
        pkts.append(pkt)
    wrpcap(str(p), pkts)

    result = extract_pcap_bins(p, bin_seconds=BIN)
    keys = {(r["src_ip"], r["bin_start"]) for r in result["records"]}
    assert keys == {
        ("10.0.0.1", BASE),
        ("10.0.0.1", BASE + BIN),
        ("10.0.0.2", BASE),
    }
    for r in result["records"]:
        assert r["bin_start"] % BIN == 0  # fixed wall-clock grid
    assert result["stats"]["non_monotonic_steps"] == 0
    assert bin_start_for(BASE + 119.9) == BASE + 60


# --------------------------------------------------------------------------
# RST must never be a retransmission; true duplicate seq must count
# --------------------------------------------------------------------------
def test_rst_not_counted_true_retransmission_counted(tmp_path):
    p = tmp_path / "retrans.pcap"
    pkts = []
    seq_base = BASE + 5.0
    # packet 1: data at seq 1000
    pkt = _tcp("10.0.0.1", "10.0.0.2", 44000, 80, seq=1000)
    pkt.time = seq_base
    pkts.append(pkt)
    # packet 2: RST echoing the same seq - the OLD code counted this
    pkt = _tcp("10.0.0.1", "10.0.0.2", 44000, 80, seq=1000, flags="RA")
    pkt.time = seq_base + 0.01
    pkts.append(pkt)
    # packet 3: genuine retransmission (same seq, new data)
    pkt = _tcp("10.0.0.1", "10.0.0.2", 44000, 80, seq=1000)
    pkt.time = seq_base + 0.02
    pkts.append(pkt)
    # packet 4: pure ACK with a stale seq - not a retransmission either
    pkt = _tcp("10.0.0.1", "10.0.0.2", 44000, 80, seq=1000,
               flags="A", payload=b"")
    pkt.time = seq_base + 0.03
    pkts.append(pkt)
    wrpcap(str(p), pkts)

    result = extract_pcap_bins(p, bin_seconds=BIN)
    assert len(result["records"]) == 1
    feats = result["records"][0]["features"]
    assert feats["retransmission_count"] == 1.0  # only packet 3


def test_demo_extractor_no_longer_counts_rst_as_retrans(tmp_path):
    """The dashboard demo path had the same RST bug - regression guard."""
    from ml.world_model.features import (
        WORLD_MODEL_FEATURES,
        extract_features_from_pcap,
    )

    idx = WORLD_MODEL_FEATURES.index("retransmission_count")
    p = tmp_path / "demo.pcap"
    pkts = []
    for flags in ["PA", "RA", "PA"]:
        pkt = _tcp("10.0.0.1", "10.0.0.2", 44000, 80, seq=1000, flags=flags)
        pkt.time = BASE + 0.1  # identical times -> one window
        pkts.append(pkt)
    wrpcap(str(p), pkts)

    feats = extract_features_from_pcap(str(p))
    # data, RST, data: exactly one retransmission, RST excluded
    assert feats[0, idx] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# no zero-fill: uncovered flow bins stay null + flagged
# --------------------------------------------------------------------------
def test_uncovered_flow_bins_are_null_never_zero(tmp_path):
    p = tmp_path / "cover.pcap"
    pkt = IP(src="10.0.0.1", dst="10.0.0.9") / UDP(sport=1, dport=2)
    pkt.time = BASE + 10
    wrpcap(str(p), [pkt])

    result = extract_pcap_bins(p, bin_seconds=BIN)
    flow_keys = [
        ("10.0.0.1", BASE),        # covered
        ("10.0.0.7", BASE),        # NOT in the PCAP
        ("10.0.0.1", BASE + 600),  # NOT in the PCAP
    ]
    rows = join_packet_features(flow_keys, result["records"])
    by_key = {(r["src_ip"], r["bin_start"]): r for r in rows}

    covered = by_key[("10.0.0.1", BASE)]
    assert covered["packet_features_covered"] is True
    assert covered["packet_features"]["ttl_mean"] is not None

    for key in [("10.0.0.7", BASE), ("10.0.0.1", BASE + 600)]:
        row = by_key[key]
        assert row["packet_features_covered"] is False
        assert row["packet_features"] is None  # null, not zeros


def test_tcp_window_null_when_no_tcp_packets(tmp_path):
    """A covered bin with no TCP must report null, not 0."""
    p = tmp_path / "udp.pcap"
    pkt = IP(src="10.0.0.1", dst="10.0.0.9") / UDP(sport=1, dport=2)
    pkt.time = BASE + 10
    wrpcap(str(p), [pkt])

    result = extract_pcap_bins(p, bin_seconds=BIN)
    feats = result["records"][0]["features"]
    assert feats["tcp_window_mean"] is None
    assert feats["tcp_window_std"] is None
    assert feats["ttl_mean"] is not None  # observed from IP headers


# --------------------------------------------------------------------------
# clock offset is measured, not assumed
# --------------------------------------------------------------------------
def test_clock_offset_measured_on_matched_5tuple(tmp_path):
    p = tmp_path / "offset.pcap"
    pkt = _tcp("10.0.0.1", "10.0.0.2", 44000, 80, seq=1)
    pkt.time = BASE + 10.0
    wrpcap(str(p), [pkt])

    flow_rows = [{
        "src": "10.0.0.1", "dst": "10.0.0.2",
        "sport": 44000, "dport": 80,
        "ts": BASE + 10.0 + 7.5,  # flow clock runs 7.5 s ahead
    }]
    offset = measure_clock_offset(p, flow_rows)
    assert offset["status"] == "measured"
    assert offset["n_matched_pairs"] == 1
    assert offset["observed_delta"][
        "median_sec_biased_under_quantization"] == pytest.approx(7.5)
    # measured is not applicable: one pair and undetermined precision cannot
    # establish an offset, and the guard says so instead of defaulting to 0
    assert offset["apply"]["eligible"] is False
    assert offset["apply"]["correction_sec"] is None


def test_clock_offset_not_measured_when_no_match(tmp_path):
    p = tmp_path / "nomatch.pcap"
    pkt = _tcp("10.0.0.1", "10.0.0.2", 44000, 80, seq=1)
    pkt.time = BASE + 10.0
    wrpcap(str(p), [pkt])

    empty = measure_clock_offset(p, [])
    assert empty["status"] == "not_measured"

    unmatched = measure_clock_offset(p, [{
        "src": "10.9.9.9", "dst": "10.9.9.8",
        "sport": 1, "dport": 2, "ts": BASE + 10.0,
    }])
    assert unmatched["status"] == "not_measured"


# --------------------------------------------------------------------------
# flow/PCAP bin coverage measurement
# --------------------------------------------------------------------------
def test_bin_coverage_percentages():
    flow_keys = [("h1", BASE), ("h1", BASE + 60),
                 ("h2", BASE), ("h3", BASE)]
    pcap_keys = [("h1", BASE), ("h2", BASE)]  # covers 2 of 4
    attack_keys = [("h1", BASE), ("h3", BASE)]  # 1 of 2 covered

    cov = measure_bin_coverage(flow_keys, pcap_keys, attack_keys)
    assert cov["bins_flow_total"] == 4
    assert cov["bins_covered"] == 2
    assert cov["coverage_pct"] == pytest.approx(50.0)
    assert cov["bins_attack_total"] == 2
    assert cov["attack_bins_covered"] == 1
    assert cov["attack_coverage_pct"] == pytest.approx(50.0)
    assert "never zero-filled" in cov["uncovered_handling"]

    empty = measure_bin_coverage([], [], [])
    assert empty["coverage_pct"] is None  # no denominator -> not 0%


# --------------------------------------------------------------------------
# streaming implementation guard
# --------------------------------------------------------------------------
def test_extractor_streams_with_pcapreader_never_rdpcap():
    src = (ROOT / "ml" / "world_model" / "pcap_bins.py").read_text(
        encoding="utf-8")
    assert "PcapReader" in src
    assert "import rdpcap" not in src
    assert "rdpcap(" not in src


# --------------------------------------------------------------------------
# flow-table timestamps: pandas unit drift (us vs ns) must not corrupt epochs
# --------------------------------------------------------------------------
def _load_cli():
    """Load the CLI script as a module (scripts/ is not an importable pkg)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pcap_stream_bins",
        ROOT / "scripts" / "pcap_stream_bins.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_flow_rows_epoch_not_corrupted_by_pandas_unit(tmp_path):
    """pandas 3 parses to datetime64[us]; the old /1e9 gave ~1e3x-too-small
    epochs, which silently destroyed clock-offset and coverage joins."""
    mod = _load_cli()
    csv_path = tmp_path / "flows.csv"
    csv_path.write_text(
        "Source IP,Destination IP,Source Port,Destination Port,Timestamp,"
        " Label\n"
        "10.0.0.1,10.0.0.2,1111,80,26/06/2026 12:39:24,PortScan\n",
        encoding="utf-8")
    rows, stats = mod.load_flow_rows(csv_path)

    assert stats["timestamp_unparsable"] == 0
    # ~2026 epoch: far above 1e7 (the value a unit bug produces) and far
    # below 1e16 (the value a missing /1e9 produces)
    assert 1.7e9 < rows[0]["ts"] < 2.2e9


# --------------------------------------------------------------------------
# backwards timestamps: fail closed by default, counted merge on request
# --------------------------------------------------------------------------
def _backwards_pcap(tmp_path, name="backwards.pcap"):
    """Two bins, then a packet belonging to the bin already closed."""
    p = tmp_path / name
    pkts = []
    for ts in (BASE + 10.0, BASE + 120.0, BASE + 20.0):
        pkt = IP(src="10.0.0.1", dst="10.0.0.9") / UDP(sport=1, dport=2)
        pkt.time = ts
        pkts.append(pkt)
    wrpcap(str(p), pkts)
    return p


def test_backwards_packet_fails_closed_by_default(tmp_path):
    p = _backwards_pcap(tmp_path)
    with pytest.raises(BackwardsTimestampError) as exc:
        extract_pcap_bins(p, bin_seconds=BIN)
    # the error must say how to opt into reordering - never drop silently
    assert "on_backwards='merge'" in str(exc.value)


def test_backwards_merge_keeps_and_counts_the_packet(tmp_path):
    p = _backwards_pcap(tmp_path, "merge.pcap")
    result = extract_pcap_bins(p, bin_seconds=BIN, on_backwards="merge")
    keys = [(r["src_ip"], r["bin_start"]) for r in result["records"]]
    assert len(keys) == len(set(keys))  # exactly one record per key
    by_key = {(r["src_ip"], r["bin_start"]): r for r in result["records"]}
    assert by_key[("10.0.0.1", BASE)]["packet_count"] == 2  # late one kept
    assert by_key[("10.0.0.1", BASE + 120)]["packet_count"] == 1
    assert result["stats"]["backwards_packets_merged"] == 1
    assert result["stats"]["non_monotonic_steps"] == 1


def test_reorder_grace_keeps_a_recent_bin_open_without_failing(tmp_path):
    p = _backwards_pcap(tmp_path, "grace.pcap")
    # 90 s grace: bin BASE is still open when the late packet arrives, so the
    # default fail policy never triggers and nothing needs merging.
    result = extract_pcap_bins(p, bin_seconds=BIN, reorder_grace_sec=90.0)
    by_key = {(r["src_ip"], r["bin_start"]): r for r in result["records"]}
    assert by_key[("10.0.0.1", BASE)]["packet_count"] == 2
    assert result["stats"]["backwards_packets_merged"] == 0
    assert result["stats"]["reorder_grace_sec"] == 90.0


def test_invalid_backwards_policy_and_grace_are_rejected(tmp_path):
    p = _backwards_pcap(tmp_path, "invalid.pcap")
    with pytest.raises(ValueError):
        extract_pcap_bins(p, bin_seconds=BIN, on_backwards="ignore")
    with pytest.raises(ValueError):
        extract_pcap_bins(p, bin_seconds=BIN, reorder_grace_sec=-1.0)
    # an unbounded grace IS the backlog this extractor exists to avoid
    with pytest.raises(ValueError):
        extract_pcap_bins(p, bin_seconds=BIN, reorder_grace_sec=100000.0)


# --------------------------------------------------------------------------
# reordering is memory-bounded: retention horizon, not an unbounded backlog
# --------------------------------------------------------------------------
def _timed_pcap(tmp_path, times, name):
    p = tmp_path / name
    pkts = []
    for ts in times:
        pkt = IP(src="10.0.0.1", dst="10.0.0.9") / UDP(sport=1, dport=2)
        pkt.time = ts
        pkts.append(pkt)
    wrpcap(str(p), pkts)
    return p


def test_default_reorder_grace_is_nonzero_and_bounded(tmp_path):
    p = _timed_pcap(tmp_path, [BASE + 10.0], "default-grace.pcap")
    stats = extract_pcap_bins(p, bin_seconds=BIN)["stats"]
    assert DEFAULT_REORDER_GRACE_SEC == 1.0  # the review's recommendation
    assert stats["reorder_grace_sec"] == DEFAULT_REORDER_GRACE_SEC
    assert stats["retention_horizon_sec"] == BIN + DEFAULT_REORDER_GRACE_SEC
    assert stats["backwards_policy"] == "fail"


def test_late_packet_inside_retention_merges_and_is_counted(tmp_path):
    """A 50 s-late packet is a counted merge, not an abort, in merge mode."""
    p = _timed_pcap(tmp_path, [BASE + 5.0, BASE + 70.0, BASE + 20.0],
                    "in-window.pcap")
    result = extract_pcap_bins(p, bin_seconds=BIN, on_backwards="merge")
    by_key = {(r["src_ip"], r["bin_start"]): r for r in result["records"]}
    assert by_key[("10.0.0.1", BASE)]["packet_count"] == 2
    stats = result["stats"]
    assert stats["backwards_packets_merged"] == 1
    assert stats["backwards_packets_late_dropped"] == 0


def test_hours_late_packet_is_counted_as_dropped_not_retained(tmp_path):
    """Merge mode must never become an unbounded backlog of old accumulators."""
    p = _timed_pcap(
        tmp_path, [BASE + 5.0, BASE + 10.0, BASE + 7205.0, BASE + 12.0],
        "very-late.pcap")
    result = extract_pcap_bins(p, bin_seconds=BIN, on_backwards="merge")
    stats = result["stats"]
    by_key = {(r["src_ip"], r["bin_start"]): r for r in result["records"]}
    assert by_key[("10.0.0.1", BASE)]["packet_count"] == 2  # early pair kept
    assert ("10.0.0.1", BASE + 7200) in by_key
    assert stats["backwards_packets_late_dropped"] == 1
    assert stats["backwards_packets_merged"] == 0


def _growth_stats(tmp_path, n_bins, on_backwards="merge"):
    p = _timed_pcap(tmp_path, [BASE + i * 3.0 for i in range(n_bins * 20)],
                    f"growth-{n_bins}-{on_backwards}.pcap")
    result = extract_pcap_bins(p, bin_seconds=BIN,
                               on_backwards=on_backwards)
    stats = result["stats"]
    assert stats["retention_horizon_sec"] == BIN + DEFAULT_REORDER_GRACE_SEC
    assert sum(r["packet_count"] for r in result["records"]) == n_bins * 20
    assert stats["watermark_scans"] <= n_bins + 1  # scans, not per packet
    return stats


def test_retained_bins_do_not_grow_with_capture_length(tmp_path):
    """The whole point of the retention horizon: memory tracks the grace."""
    small = _growth_stats(tmp_path, 20)
    big = _growth_stats(tmp_path, 200)
    assert small["peak_retained_bins"] <= 4
    assert big["peak_retained_bins"] == small["peak_retained_bins"]


def test_unsupported_or_null_names_only_null_statistics(tmp_path):
    from ml.world_model.pcap_bins import unsupported_or_null

    p = tmp_path / "udp-only.pcap"
    pkt = IP(src="10.0.0.1", dst="10.0.0.9") / UDP(sport=1, dport=2)
    pkt.time = BASE + 10
    wrpcap(str(p), [pkt])
    record = extract_pcap_bins(p, bin_seconds=BIN)["records"][0]
    assert unsupported_or_null(record) == [
        "tcp_window_mean", "tcp_window_std"]


# --------------------------------------------------------------------------
# watermark: bins close on boundary crossings, not once per packet
# --------------------------------------------------------------------------
def test_watermark_scans_scale_with_bins_not_packets(tmp_path):
    p = tmp_path / "many.pcap"
    pkts = []
    for i in range(600):  # 10 bins x 60 packets, same emitting host
        pkt = IP(src="10.0.0.1", dst="10.0.0.9") / UDP(sport=1, dport=2)
        pkt.time = BASE + i
        pkts.append(pkt)
    wrpcap(str(p), pkts)

    result = extract_pcap_bins(p, bin_seconds=BIN)
    stats = result["stats"]
    assert stats["packets_considered"] == 600
    assert len(result["records"]) == 10
    assert stats["watermark_scans"] <= 11  # boundary crossings, not 600
    assert sum(r["packet_count"] for r in result["records"]) == 600


# --------------------------------------------------------------------------
# timestamp precision: determined per source, never assumed
# --------------------------------------------------------------------------
def _infer(values, **kw):
    from ml.world_model.pcap_bins import infer_timestamp_quantization
    return infer_timestamp_quantization(values, **kw)


def test_precision_reads_subsecond_second_and_minute_grids():
    sub = _infer([BASE + 10.5 + i * 0.3 for i in range(30)], kind="t")
    assert sub["known"] is True and sub["quantization_sec"] == 0.001

    sec = _infer([float(BASE + 7 + i * 37) for i in range(30)], kind="t")
    assert sec["known"] is True and sec["quantization_sec"] == 1.0
    assert sec["evidence"]["off_minute_count"] > 0

    # the trap the review called out: values that all sit on the minute grid
    # are MINUTE precision and must never be labelled second precision
    minute = _infer([float(BASE + 60 * i) for i in range(30)], kind="t")
    assert minute["known"] is True and minute["quantization_sec"] == 60.0
    assert minute["evidence"]["off_minute_count"] == 0


def test_precision_stays_unknown_from_too_few_values():
    few = _infer([float(BASE + i) for i in range(3)], kind="t")
    assert few["known"] is False and few["quantization_sec"] is None
    assert "unknown" in few["reason"]
    grid = _infer([float(BASE + 60 * i) for i in range(3)], kind="t")
    assert grid["known"] is False  # 3 grid values prove nothing either


def test_precision_counts_values_not_min_max():
    """min/max look minute-aligned; the values in between say otherwise."""
    values = [float(BASE)] + [float(BASE + 60 * i + 30) for i in range(30)]
    got = _infer(values, kind="t")
    assert got["quantization_sec"] == 1.0
    assert got["evidence"]["off_minute_count"] == 30


def test_declared_precision_never_beats_the_measured_values():
    # the data proves sub-second; a declared "1 s" cannot make the estimate
    # finer, and the coarser candidate wins because it is the safe direction
    got = _infer([BASE + 10.5 + i * 0.3 for i in range(30)], kind="t",
                 declared_sec=1.0)
    assert got["quantization_sec"] == 1.0
    assert "declared_precision_differs_from_values" in got["flags"]

    # a declaration with too few values to verify is allowed but labelled
    alone = _infer([float(BASE + i) for i in range(3)], kind="t",
                   declared_sec=1.0)
    assert alone["quantization_sec"] == 1.0
    assert "declared_precision_unverified" in alone["flags"]

    with pytest.raises(ValueError):
        _infer([float(BASE + i) for i in range(30)], kind="t",
               declared_sec=5.0)  # not a supported step


def test_feasible_interval_is_the_intersection_of_per_pair_bounds():
    from ml.world_model.pcap_bins import feasible_offset_interval as iv

    got = iv([7.0, 7.5], flow_quantization_sec=1.0,
             packet_quantization_sec=0.001)
    assert got["computable"] is True
    assert got["lo_sec"] == pytest.approx(7.5 - 0.001)
    assert got["hi_sec"] == pytest.approx(7.0 + 1.0)
    assert got["width_sec"] == pytest.approx(8.0 - (7.5 - 0.001))
    assert got["midpoint_sec"] == pytest.approx((7.499 + 8.0) / 2.0)
    assert got["empty"] is False

    # pairs that cannot share one constant offset -> empty, not "average them"
    assert iv([0.0, 12.0], flow_quantization_sec=1.0,
              packet_quantization_sec=0.001)["empty"] is True

    unknown = iv([1.0, 2.0], flow_quantization_sec=None,
                 packet_quantization_sec=0.001)
    assert unknown["computable"] is False
    assert unknown["unknown_inputs"] == ["flow table"]
    assert unknown["midpoint_sec"] is None


# --------------------------------------------------------------------------
# retransmission_count is a sequence-regression detector, NOT a retrans count
# --------------------------------------------------------------------------
def _seq_pcap(tmp_path, segments, name):
    """One direction, segments = [(seq, payload)], 10 ms apart in one bin."""
    p = tmp_path / name
    pkts = []
    for i, (seq, payload) in enumerate(segments):
        pkt = _tcp("10.0.0.1", "10.0.0.2", 44000, 80, seq=seq, payload=payload)
        pkt.time = BASE + 5.0 + i * 0.01
        pkts.append(pkt)
    wrpcap(str(p), pkts)
    return p


def _regression(tmp_path, segments, name):
    result = extract_pcap_bins(_seq_pcap(tmp_path, segments, name),
                               bin_seconds=BIN)
    count = result["records"][0]["features"]["retransmission_count"]
    return count, result


def test_in_window_reordering_is_not_a_sequence_regression(tmp_path):
    """1000, 1020, then the 1010 that fills the gap: zero, not one.

    The review's counter-example: an earlier draft of this fix counted any
    "seq below the highest end seen" as a retransmission, which mislabels
    in-window reordering as loss.
    """
    count, _ = _regression(tmp_path, [(1000, b"x" * 10), (1020, b"x" * 10),
                                      (1010, b"x" * 10)], "reorder.pcap")
    assert count == 0.0


def test_duplicate_byte_range_is_a_sequence_regression(tmp_path):
    count, _ = _regression(tmp_path, [(1000, b"x" * 10), (1000, b"x" * 10)],
                           "dup.pcap")
    assert count == 1.0


def test_partial_overlap_that_adds_new_bytes_is_not_counted(tmp_path):
    count, _ = _regression(tmp_path, [(1000, b"x" * 10), (1005, b"x" * 10)],
                           "partial.pcap")
    assert count == 0.0


def test_keep_alives_are_never_counted(tmp_path):
    """Fifty repeated zero-length ACKs are the old false-positive family."""
    count, _ = _regression(
        tmp_path, [(1000, b"x" * 10)] + [(1010, b"") for _ in range(50)],
        "keepalive.pcap")
    assert count == 0.0


def test_out_of_order_range_memory_is_capped(tmp_path):
    from ml.world_model.pcap_bins import SEQ_RANGES_CAP

    segments = [(1000 + i * 10000, b"x" * 10)
                for i in range(SEQ_RANGES_CAP * 3)]
    count, result = _regression(tmp_path, segments, "cap.pcap")
    assert result["stats"]["seq_range_sets_capped"] >= 1  # bounded, and said
    assert count == 0.0                                  # none are duplicates
    assert result["records"][0]["packet_count"] == len(segments)


def test_column_name_is_kept_and_its_semantics_are_reported(tmp_path):
    from ml.world_model.features import PACKET_LEVEL_COLUMNS

    assert "retransmission_count" in PACKET_LEVEL_COLUMNS  # the frozen column
    _, result = _regression(tmp_path, [(1000, b"x" * 10)], "sem.pcap")
    semantics = result["stats"]["feature_semantics"]["retransmission_count"]
    assert "sequence-regression" in semantics
    assert "not a retransmission count" in semantics.lower()
    assert any("sequence-regression" in a for a in result["approximations"])


def test_join_key_semantics_say_what_matches_and_what_does_not():
    from ml.world_model.pcap_bins import BIN_KEY_SEMANTICS

    assert "emitting host" in BIN_KEY_SEMANTICS
    # the join aligns host + bucket only; direction and unit do not match
    assert "counting unit" in BIN_KEY_SEMANTICS
    assert "not between equivalent observations" in BIN_KEY_SEMANTICS


def test_packet_tier_has_no_training_or_benchmark_consumer_yet():
    """The claim under review is extraction + measurement, not a trained model:
    nothing outside the CLI and these tests may import the extractor."""
    import re

    consumers = []
    for path in ROOT.rglob("*.py"):
        if path.name in {"pcap_bins.py", "test_pcap_bins.py"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^\s*(from|import)\s+[\w.]*pcap_bins", text, re.M):
            consumers.append(str(path.relative_to(ROOT)))
    assert consumers == ["scripts\\pcap_stream_bins.py"] or consumers == [
        "scripts/pcap_stream_bins.py"]


# --------------------------------------------------------------------------
# clock offset: feasible interval, guarded apply, never the biased median
# --------------------------------------------------------------------------
def _floor_to(value, step):
    return math.floor(value / step) * step


def _offset_scenario(tmp_path, *, q_flow, theta, step, n=40, q_pkt=0.001,
                     name="offset.pcap", base=BASE + 10.0, outliers=()):
    """Build a PCAP + flow rows for one (q_flow, q_pkt, theta) scenario.

    Connection i's first packet sits at true time ``t_i`` as recorded by a
    clock that truncates to ``q_pkt``; its flow row carries the same event on a
    clock running ``theta`` ahead that truncates to ``q_flow``. That is exactly
    the floor-quantization model the estimator assumes, so the feasible
    interval must contain ``theta`` - and its midpoint must be closer to
    ``theta`` than the sample median ever is.

    ``outliers`` is a sequence of ``(index, extra_sec)`` pairs that push the
    flow row of connection ``index`` ``extra_sec`` seconds off. Those are real
    rows on real 5-tuples, so they still match and still produce a delta - they
    are exactly what a stepped clock, a DST jump or a mis-paired connection
    looks like from the estimator's side.

    Returns ``(pcap_path, flow_rows, true_times)``.
    """
    shifted = dict(outliers)
    p = tmp_path / name
    trues = [base + i * step for i in range(n)]
    pkts, rows = [], []
    for i, t in enumerate(trues):
        pkt = _tcp("10.0.0.1", "10.0.0.2", 44000 + i, 80, seq=1)
        pkt.time = _floor_to(t, q_pkt)
        pkts.append(pkt)
        rows.append({"src": "10.0.0.1", "dst": "10.0.0.2",
                     "sport": 44000 + i, "dport": 80, "proto": "tcp",
                     "ts": _floor_to(t + theta + shifted.get(i, 0.0), q_flow)})
    wrpcap(str(p), pkts)
    return p, rows, trues


def _offset_pcap(tmp_path, n=4, name="offset.pcap", pkt_ts=BASE + 10.0,
                 frac_step=0.0):
    """``n`` distinct TCP connections, first packets ``frac_step`` apart."""
    p = tmp_path / name
    pkts = []
    for i in range(n):
        pkt = _tcp("10.0.0.1", "10.0.0.2", 44000 + i, 80, seq=1)
        pkt.time = pkt_ts + i * frac_step
        pkts.append(pkt)
    wrpcap(str(p), pkts)
    return p


def _offset_flows(deltas):
    """Flow rows starting ``deltas[i]`` after the matching packet."""
    return [{"src": "10.0.0.1", "dst": "10.0.0.2", "sport": 44000 + i,
             "dport": 80, "proto": "tcp", "ts": BASE + 10.0 + d}
            for i, d in enumerate(deltas)]


THETA = 7.34  # deliberately non-round: a median cannot recover this exactly
TOL = 1e-6    # float resolution at unix-epoch magnitude (~2.4e-7 s)

# (q_flow, q_pkt, n, step, appliable): eligible exactly when the intersection
# of the per-pair constraints really is at most OFFSET_MAX_INTERVAL_WIDTH_SEC.
_APPLY_MATRIX = [
    (0.001, 0.001, 25, 0.4167, True),    # both clocks sub-second: pinned
    (1.0, 0.001, 200, 3.0157, True),     # second-precision, widely sampled
    (1.0, 0.001, 25, 0.01, False),       # fracs too clustered to pin it down
    (1.0, 1.0, 200, 3.0157, False),      # both clocks coarse: >= 1 s left over
    (60.0, 0.001, 200, 6.03, False),     # minute precision cannot pin 0.1 s
]


@pytest.mark.parametrize(
    "q_flow,q_pkt,n,step,appliable", _APPLY_MATRIX,
    ids=[f"qf-{m[0]}-qp-{m[1]}-n{m[2]}" for m in _APPLY_MATRIX])
def test_offset_interval_contains_theta_and_width_decides_application(
        tmp_path, q_flow, q_pkt, n, step, appliable):
    """Checked against ground truth on every supported q combination.

    The two clocks are generated by the exact floor model the estimator
    assumes, so the feasible interval must contain the true offset - and
    application must follow the width rule, never the median.
    """
    pcap, rows, _ = _offset_scenario(
        tmp_path, q_flow=q_flow, q_pkt=q_pkt, theta=THETA, step=step, n=n,
        name=f"off-{q_flow}-{q_pkt}-{n}.pcap")
    off = measure_clock_offset(pcap, rows)
    obs = off["observed_delta"]
    interval = obs["feasible_interval"]

    assert off["status"] == "measured"
    assert off["n_matched_pairs"] == n
    # precision is read from each source's own values, for every input
    assert obs["flow_timestamp_precision"]["quantization_sec"] == q_flow
    assert obs["packet_timestamp_precision"]["quantization_sec"] == q_pkt
    assert interval["computable"] is True
    assert interval["empty"] is False
    assert interval["lo_sec"] <= THETA + TOL
    assert interval["hi_sec"] > THETA - TOL

    ap = off["apply"]
    assert ap["eligible"] is appliable
    if appliable:
        assert abs(ap["correction_sec"] - THETA) <= ap["max_error_sec"] + TOL
        assert ap["max_error_sec"] == pytest.approx(
            interval["width_sec"] / 2.0, abs=TOL)
        assert ap["max_error_sec"] <= 0.05   # decision B: error <= 0.05 s
    else:
        assert ap["correction_sec"] is None  # never applied, never silently 0
        assert interval["width_sec"] > 0.1
        assert "refusing to apply" in ap["reason"]


# A step of 15.7 ms: every pass through the 1 s quantization cell lands 4.8 ms
# later, so a few hundred pairs walk the whole cell and pin the feasible
# interval to a few milliseconds. That makes an outlier's effect on it visible
# instead of buried in quantization slack.
_DENSE_STEP = 0.0157
_DENSE_N = 600

_OUTLIER_MATRIX = [
    # (outliers out of 600 pairs, expected eligible). 3 % is inside the 5 %
    # tolerance and must not move the answer; 10 % is not and must refuse.
    (18, True),
    (60, False),
]


@pytest.mark.parametrize(
    "n_out,eligible", _OUTLIER_MATRIX,
    ids=[f"{m[0] * 100 // _DENSE_N}pct-outliers" for m in _OUTLIER_MATRIX])
def test_outlier_deltas_are_counted_not_averaged_and_capped(
        tmp_path, n_out, eligible):
    """A minority of wrong pairs cannot move the correction, a bigger one can.

    Both scenarios carry the same honest pairs under a planted offset
    ``THETA`` (582 and 540 of them) plus outliers pointing both ways (+600 s, a
    clock stepped by hand; -45 s, a mis-paired connection). The outliers are
    visible in the reported spread, invisible to the interval, and the deciding
    rule is the fraction of them - not a trimmed mean, not a median.
    """
    outliers = [(int(_DENSE_N * (j + 1) / (n_out + 1)),
                 600.0 if j % 2 == 0 else -45.0) for j in range(n_out)]
    pcap, rows, _ = _offset_scenario(
        tmp_path, q_flow=1.0, q_pkt=0.001, theta=THETA, step=_DENSE_STEP,
        n=_DENSE_N, name=f"outliers-{n_out}.pcap", outliers=outliers)
    off = measure_clock_offset(pcap, rows)
    obs = off["observed_delta"]
    interval = obs["feasible_interval"]
    ap = off["apply"]

    assert off["status"] == "measured"
    assert off["n_matched_pairs"] == _DENSE_N
    # every outlier was seen, none of them was believed
    assert obs["n_inconsistent"] == n_out
    assert obs["n_consistent"] == _DENSE_N - n_out
    assert obs["min_sec"] < THETA - 40.0        # the -45 s pairs are reported
    assert obs["max_sec"] > THETA + 500.0       # and the +600 s pairs
    assert obs["consistent_min_sec"] > THETA - 1.01
    assert obs["consistent_max_sec"] < THETA + 0.01
    # ... but they do not reach the interval the decision is made on
    assert interval["n_pairs_used"] == _DENSE_N - n_out
    assert interval["n_pairs_total"] == _DENSE_N
    assert interval["lo_sec"] <= THETA + TOL
    assert interval["hi_sec"] > THETA - TOL
    assert interval["width_sec"] <= 0.1
    assert ap["n_inconsistent"] == n_out

    assert ap["eligible"] is eligible
    if eligible:
        # decision B: applied within 0.05 s of the planted offset
        assert abs(ap["correction_sec"] - THETA) <= 0.05
        assert ap["reason"] is None
    else:
        assert ap["correction_sec"] is None
        assert "inconsistent" in ap["reason"]
        assert "inconsistent_pairs" in ap["flags"]


def test_inconsistent_pair_tolerance_is_inclusive_at_the_bound(tmp_path):
    """Exactly 5 % inconsistent is tolerated; one pair more is not.

    The rule is ``n_inconsistent <= 0.05 * n``, so the boundary itself must be
    on the eligible side - a tolerance that refuses its own documented bound
    would be a rule nobody can satisfy.
    """
    n = 120   # 5 % is exactly 6 pairs; 120 pairs also walk the whole 1 s cell
    for n_out, eligible in ((6, True), (7, False)):
        pcap, rows, _ = _offset_scenario(
            tmp_path, q_flow=1.0, q_pkt=0.001, theta=THETA, step=_DENSE_STEP,
            n=n, name=f"bound-{n_out}.pcap",
            outliers=[(17 * (j + 1), 600.0) for j in range(n_out)])
        off = measure_clock_offset(pcap, rows)
        obs, ap = off["observed_delta"], off["apply"]
        assert obs["n_inconsistent"] == n_out
        assert ap["eligible"] is eligible, ap["reason"]
        if eligible:
            assert abs(ap["correction_sec"] - THETA) <= 0.05
        else:
            assert "inconsistent" in ap["reason"]


def test_minute_precision_is_pinnable_only_by_dense_sampling(tmp_path):
    """The 0.1 s bound is a width rule, not a ban on coarse clocks.

    With packet times every 0.05 s, 60 s-quantized flow rows intersect to well
    under a tenth of a second and the estimator says so; sampled every 6 s (the
    matrix case above) the same q=60 input is refused.
    """
    pcap, rows, _ = _offset_scenario(
        tmp_path, q_flow=60.0, q_pkt=0.001, theta=THETA, step=0.05, n=2400,
        name="dense.pcap")
    off = measure_clock_offset(pcap, rows)
    interval = off["observed_delta"]["feasible_interval"]
    assert interval["q_flow_sec"] == 60.0
    assert interval["width_sec"] <= 0.1
    assert off["apply"]["eligible"] is True
    assert (abs(off["apply"]["correction_sec"] - THETA)
            <= off["apply"]["max_error_sec"] + TOL)


def test_applied_correction_keeps_packet_bins_on_the_flow_bins(tmp_path):
    """The decision that matters is the BIN, not the nanosecond.

    Every packet whose true flow time sits further from a 60 s edge than the
    claimed worst-case error must land in exactly the flow table's bin - and
    the interval midpoint must beat the quantization-biased median.
    """
    pcap, rows, trues = _offset_scenario(
        tmp_path, q_flow=1.0, q_pkt=0.001, theta=THETA, step=3.0157, n=200)
    off = measure_clock_offset(pcap, rows)
    ap = off["apply"]
    assert ap["eligible"] is True
    corr, err = ap["correction_sec"], ap["max_error_sec"]

    biased = off["observed_delta"]["median_sec_biased_under_quantization"]
    # floor quantization biases the median low by the mean sub-step, so it sits
    # further from the truth than the midpoint - the point of this fix
    assert abs(biased - THETA) > 0.15
    assert err < abs(biased - THETA) - TOL

    checked = 0
    for t, row in zip(trues, rows):
        flow_true = t + THETA
        flow_bin = bin_start_for(row["ts"])
        if flow_bin != bin_start_for(flow_true):
            continue  # the flow's own truncation crossed the edge: not ours
        if flow_true % 60 < err or flow_true % 60 > 60 - err:
            continue  # inside the claimed error band of an edge: no promise
        assert bin_start_for(_floor_to(t, 0.001) + corr) == flow_bin
        checked += 1
    assert checked > 150  # the bin agreement is not vacuous


def test_offset_reports_evidence_and_never_dumps_every_delta(tmp_path):
    from ml.world_model.pcap_bins import MAX_DELTA_SAMPLES

    pcap, rows, _ = _offset_scenario(
        tmp_path, q_flow=1.0, q_pkt=0.001, theta=THETA, step=3.0157, n=200)
    obs = measure_clock_offset(pcap, rows)["observed_delta"]
    assert "deltas_sec" not in obs  # 200 raw deltas must not reach the JSON
    assert len(obs["deltas_sample_sec"]) == MAX_DELTA_SAMPLES
    assert obs["deltas_sample_truncated"] is True
    assert "NEVER applied" in obs["bias_note"]
    assert "ADDED to" in obs["sign_convention"]  # the sign is spelled out
    assert (obs["flow_timestamp_precision"]["evidence"]
            ["n_values_checked"]) == 200
    assert obs["packet_timestamp_precision"]["known"] is True


def test_offset_refused_when_precision_cannot_be_determined(tmp_path):
    """Few values: the offset is still measured, but never applied."""
    p = _offset_pcap(tmp_path, n=5, frac_step=0.2)
    off = measure_clock_offset(p, _offset_flows(
        [7.0 - i * 0.2 for i in range(5)]))
    assert off["status"] == "measured"
    ap = off["apply"]
    assert ap["eligible"] is False and ap["correction_sec"] is None
    assert "not determined" in ap["reason"]
    assert "unknown_flow_timestamp_precision" in ap["flags"]
    assert "unknown_packet_timestamp_precision" in ap["flags"]


def test_offset_apply_refused_when_too_few_matches(tmp_path):
    p = _offset_pcap(tmp_path, n=2)
    off = measure_clock_offset(p, _offset_flows([7.0] * 2))
    # the biased median is still reported ...
    assert off["observed_delta"][
        "median_sec_biased_under_quantization"] == pytest.approx(7.0)
    assert off["apply"]["eligible"] is False
    assert off["apply"]["correction_sec"] is None  # ... never applied as 0
    assert "too few" in off["apply"]["reason"]


def test_one_contradictory_pair_is_excluded_not_averaged_into_the_interval(
        tmp_path):
    """A single mis-paired instance cannot widen the interval or stop it.

    The pre-item-2 behaviour was to build the interval from every pair: one
    mismatched row then made ``lo > hi``, the interval came back empty and the
    whole measurement was refused. Fail-closed, but it also let one junk row
    deny the join for a 200-pair capture. The bad pair is now named
    (``n_inconsistent``), held out of the interval, and the honest pairs still
    answer the question. The contradiction rule itself did not go anywhere - it
    lives on in :func:`feasible_offset_interval` (see
    ``test_feasible_interval_is_the_intersection_of_per_pair_bounds``) and in
    the bimodal case below.
    """
    pcap, rows, _ = _offset_scenario(
        tmp_path, q_flow=1.0, q_pkt=0.001, theta=THETA, step=_DENSE_STEP,
        n=_DENSE_N, name="one-bad.pcap")
    rows[7]["ts"] += 30.0   # a mismatched instance, or a clock that drifted
    off = measure_clock_offset(pcap, rows)
    obs = off["observed_delta"]
    interval = obs["feasible_interval"]

    assert obs["n_inconsistent"] == 1
    assert obs["n_consistent"] == _DENSE_N - 1
    assert interval["empty"] is False
    assert interval["n_pairs_used"] == _DENSE_N - 1
    assert interval["lo_sec"] <= THETA + TOL
    assert interval["hi_sec"] > THETA - TOL
    assert off["apply"]["eligible"] is True
    assert abs(off["apply"]["correction_sec"] - THETA) <= 0.05


def test_two_offset_populations_are_refused_as_inconsistent(tmp_path):
    """A contradicted constant offset still refuses; it is never averaged.

    Half the pairs say the flow clock runs ``THETA`` ahead, half say ``THETA +
    120`` - two populations, no single offset. The midpoint of the two would be
    wrong for every pair, so the answer must be a refusal that names the
    inconsistency and applies nothing.
    """
    pcap, rows, _ = _offset_scenario(
        tmp_path, q_flow=1.0, q_pkt=0.001, theta=THETA, step=_DENSE_STEP,
        n=_DENSE_N, name="bimodal.pcap")
    for row in rows[::2]:
        row["ts"] += 120.0
    off = measure_clock_offset(pcap, rows)
    obs, ap = off["observed_delta"], off["apply"]

    assert obs["n_inconsistent"] >= _DENSE_N // 2 - 1
    assert ap["eligible"] is False
    assert ap["correction_sec"] is None      # never the midpoint of two groups
    assert "inconsistent" in ap["reason"]
    assert "inconsistent_pairs" in ap["flags"]


def test_offset_flags_timezone_scale_without_correcting_it(tmp_path):
    pcap, rows, _ = _offset_scenario(
        tmp_path, q_flow=1.0, q_pkt=0.001, theta=3600.34, step=3.0157,
        n=200, name="tz.pcap")
    off = measure_clock_offset(pcap, rows)
    assert "timezone_scale" in off["apply"]["flags"]
    assert off["apply"]["eligible"] is False
    assert "timezone scale" in off["apply"]["reason"]
    assert off["apply"]["correction_sec"] is None


def test_offset_matching_is_protocol_aware(tmp_path):
    """A UDP packet must never pair with a TCP flow row on equal ports."""
    p = tmp_path / "proto.pcap"
    pkt = IP(src="10.0.0.1", dst="10.0.0.2") / UDP(sport=44000, dport=80)
    pkt.time = BASE + 10.0
    wrpcap(str(p), [pkt])
    off = measure_clock_offset(p, _offset_flows([7.0]))
    assert off["status"] == "not_measured"
    assert off["apply"]["correction_sec"] is None


def test_offset_pairs_per_connection_instance_not_earliest(tmp_path):
    """A reused 5-tuple pairs each flow start with its own connection."""
    p = tmp_path / "reuse.pcap"
    pkts = []
    for ts in (BASE + 10.0, BASE + 11.0, BASE + 300.0, BASE + 301.0):
        pkt = _tcp("10.0.0.1", "10.0.0.2", 44000, 80, seq=1)
        pkt.time = ts
        pkts.append(pkt)
    wrpcap(str(p), pkts)

    flows = [{"src": "10.0.0.1", "dst": "10.0.0.2", "sport": 44000,
              "dport": 80, "proto": "tcp", "ts": ts}
             for ts in (BASE + 17.0, BASE + 340.0)]
    off = measure_clock_offset(p, flows)
    assert off["status"] == "measured"
    assert off["n_matched_pairs"] == 2
    # two distinct deltas prove the instances were not collapsed to the
    # earliest flow of the tuple (which would yield one bogus correction)
    assert off["observed_delta"]["deltas_sample_sec"] == [7.0, 40.0]

    # extra unpaired instances are counted, never guessed at or fabricated
    extra = measure_clock_offset(p, flows + [{
        "src": "10.0.0.1", "dst": "10.0.0.2", "sport": 44000, "dport": 80,
        "proto": "tcp", "ts": BASE + 400.0}])
    assert extra["n_matched_pairs"] == 2
    assert extra["n_unpaired_instances"] == 1


# --------------------------------------------------------------------------
# flow-table loading: null cells, padding vs garbage, protocol, many files
# --------------------------------------------------------------------------
def _ts_str(offset_sec):
    """CIC ``dd/mm/yyyy hh:mm:ss`` (UTC) for BASE + offset_sec.

    ``None`` yields an empty cell (a real null), a str passes through
    (for genuinely unparsable values).
    """
    import datetime

    if offset_sec is None:
        return ""
    if isinstance(offset_sec, str):
        return offset_sec
    dt = datetime.datetime.fromtimestamp(BASE + offset_sec,
                                         datetime.timezone.utc)
    return dt.strftime("%d/%m/%Y %H:%M:%S")


def _flow_csv(path, rows):
    """Write a CIC-shaped CSV from src/dst/ports/proto/timestamp/label rows."""
    lines = ["Source IP,Destination IP,Source Port,Destination Port,"
             " Protocol ,Timestamp,Label"]  # padded header, as in CIC files
    for src, dst, sport, dport, proto, off, label in rows:
        lines.append(f"{src},{dst},{sport},{dport},{proto},{_ts_str(off)},"
                     f"{label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_flow_rows_null_label_and_missing_identity_never_become_nan(
        tmp_path):
    mod = _load_cli()
    csv_path = _flow_csv(tmp_path / "f.csv", [
        ("10.0.0.1", "10.0.0.2", 1111, 80, 6, 30.0, ""),        # null label
        ("", "10.0.0.2", 1112, 80, 6, 31.0, "BENIGN"),          # no src IP
    ])
    rows, stats = mod.load_flow_rows(csv_path)

    assert stats["flow_rows"] == 1
    assert stats["rows_dropped_missing_identity"] == 1
    assert rows[0]["label"] is None            # never the string "nan"
    assert stats["labels_null"] == 1
    assert "nan" not in {str(v).lower() for v in rows[0].values()}
    assert rows[0]["proto"] == "tcp"


def test_load_flow_rows_all_null_timestamp_column_is_padding_not_garbage(
        tmp_path):
    """The pinned snapshot's 288,602 padding rows: null timestamps are
    reported as null (all-null), never counted as unparsable values."""
    mod = _load_cli()
    csv_path = _flow_csv(tmp_path / "pad.csv", [
        ("10.0.0.1", "10.0.0.2", 1111, 80, 6, None, "BENIGN"),
        ("10.0.0.1", "10.0.0.2", 1112, 80, 6, None, "BENIGN"),
        ("10.0.0.1", "10.0.0.2", 1113, 80, 6, None, ""),
    ])
    rows, stats = mod.load_flow_rows(csv_path)

    assert rows == []
    assert stats["timestamp_null"] == 3
    assert stats["timestamp_unparsable"] == 0
    assert stats["timestamp_all_null"] is True


def test_load_flow_rows_separates_unparsable_from_null_timestamps(tmp_path):
    mod = _load_cli()
    csv_path = _flow_csv(tmp_path / "mix.csv", [
        ("10.0.0.1", "10.0.0.2", 1111, 80, 6, "not-a-date", "BENIGN"),
        ("10.0.0.1", "10.0.0.2", 1112, 80, 6, 30.0, "BENIGN"),
    ])
    rows, stats = mod.load_flow_rows(csv_path)

    assert stats["flow_rows"] == 1
    assert stats["timestamp_null"] == 0
    assert stats["timestamp_unparsable"] == 1
    assert stats["timestamp_all_null"] is False


def test_load_flow_rows_normalises_protocol_codes(tmp_path):
    mod = _load_cli()
    csv_path = _flow_csv(tmp_path / "proto.csv", [
        ("10.0.0.1", "10.0.0.2", 1111, 53, 17, 30.0, "BENIGN"),
        ("10.0.0.1", "10.0.0.2", 1112, 80, 6, 31.0, "BENIGN"),
    ])
    rows, _ = mod.load_flow_rows(csv_path)
    assert [r["proto"] for r in rows] == ["udp", "tcp"]


def test_load_flow_tables_concatenates_files_and_directories(tmp_path):
    import pandas as pd

    mod = _load_cli()
    flows_dir = tmp_path / "flows"
    flows_dir.mkdir()
    _flow_csv(flows_dir / "a.csv",
              [("10.0.0.1", "10.0.0.9", 1111, 80, 6, 30.0, "PortScan")])
    pd.DataFrame([{
        "Source IP": "10.0.0.2", "Destination IP": "10.0.0.9",
        "Source Port": 2222, "Destination Port": 80, "Protocol": 6,
        "Timestamp": _ts_str(30), "Label": "BENIGN",
    }]).to_parquet(flows_dir / "x.parquet")
    extra = _flow_csv(tmp_path / "c.csv",
                      [("10.0.0.3", "10.0.0.9", 3333, 80, 6, 30.0, "BENIGN")])

    rows, stats = mod.load_flow_tables([flows_dir, extra])

    assert rows and len(rows) == 3
    # deterministic order: parquet first, then csv, each sorted by name
    assert stats["flow_files"] == ["x.parquet", "a.csv", "c.csv"]
    assert stats["flow_files_count"] == 3
    assert stats["flow_rows"] == 3
    assert len(stats["files"]) == 3


def test_resolve_flow_paths_rejects_missing_or_empty_inputs(tmp_path):
    mod = _load_cli()
    with pytest.raises(SystemExit):
        mod.resolve_flow_paths([tmp_path / "absent.csv"])
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(SystemExit):
        mod.resolve_flow_paths([empty_dir])


# --------------------------------------------------------------------------
# CLI end-to-end: apply guard, coverage reporting, backwards policy
# --------------------------------------------------------------------------
def test_cli_applies_measured_offset_so_flow_bins_join(tmp_path):
    """40 matched second-precision flow rows pin the offset to <0.03 s.

    Applying that measured correction moves every packet bin onto the flow bin;
    without it the same join covers nothing. The difference is measured, and the
    run reports the worst-case error it is claiming.
    """
    mod = _load_cli()
    pcap = _offset_pcap(tmp_path, n=40, name="shift.pcap",
                        pkt_ts=BASE + 55.0, frac_step=0.025)
    flows = _flow_csv(tmp_path / "shift.csv", [
        ("10.0.0.1", "10.0.0.2", 44000 + i, 80, 6, 62.0, "PortScan")
        for i in range(40)])

    applied = tmp_path / "applied.json"
    assert mod.main(["--pcap", str(pcap), "--flows", str(flows),
                     "--apply-offset", "--out", str(applied)]) == 0
    payload = json.loads(applied.read_text(encoding="utf-8"))
    apply_block = payload["clock_offset"]["apply"]
    assert apply_block["eligible"] is True
    assert apply_block["max_error_sec"] <= 0.05
    # the truth here is +7.000 s: the midpoint lands inside its own bound
    assert payload["offset_applied_sec"] == pytest.approx(7.0, abs=0.05)
    assert payload["pcap"]["apply_offset_sec"] == pytest.approx(7.0, abs=0.05)
    assert payload["coverage"]["bins_flow_total"] == 1
    assert payload["coverage"]["coverage_pct"] == pytest.approx(100.0)
    assert payload["coverage"]["attack_coverage_pct"] == pytest.approx(100.0)
    assert payload["rows"][0]["bin_start"] == BASE + 60
    assert payload["rows"][0]["packet_features_covered"] is True
    # precisions were determined per source, from the values themselves
    assert payload["timestamp_precision"]["flow_table_sec"] == 1.0
    assert payload["timestamp_precision"]["pcap_sec"] == 0.001

    unapplied = tmp_path / "unapplied.json"
    assert mod.main(["--pcap", str(pcap), "--flows", str(flows),
                     "--out", str(unapplied)]) == 0
    second = json.loads(unapplied.read_text(encoding="utf-8"))
    assert second["offset_applied_sec"] == 0.0
    assert second["coverage"]["coverage_pct"] == 0.0


def test_cli_refuses_apply_offset_and_writes_no_output(tmp_path, capsys):
    mod = _load_cli()
    pcap = _offset_pcap(tmp_path, n=4, name="guard.pcap")

    too_few = _flow_csv(tmp_path / "two.csv", [
        ("10.0.0.1", "10.0.0.2", 44000 + i, 80, 6, 17.0, "BENIGN")
        for i in range(2)])
    out1 = tmp_path / "o1.json"
    assert mod.main(["--pcap", str(pcap), "--flows", str(too_few),
                     "--apply-offset", "--out", str(out1)]) == 1
    assert not out1.exists()  # a refused apply must write nothing
    assert "too few" in capsys.readouterr().err

    unmatched = _flow_csv(tmp_path / "other.csv", [
        ("10.9.9.1", "10.9.9.2", 44000, 80, 6, 17.0, "BENIGN")])
    out2 = tmp_path / "o2.json"
    assert mod.main(["--pcap", str(pcap), "--flows", str(unmatched),
                     "--apply-offset", "--out", str(out2)]) == 1
    assert not out2.exists()


def test_cli_all_benign_flow_table_reports_null_attack_coverage(
        tmp_path, capsys):
    """No attack bins -> attack coverage is null / 'n/a', never 0.0%."""
    mod = _load_cli()
    pcap = tmp_path / "benign.pcap"
    pkt = IP(src="10.0.0.1", dst="10.0.0.9") / UDP(sport=1, dport=2)
    pkt.time = BASE + 10
    wrpcap(str(pcap), [pkt])
    flows = _flow_csv(tmp_path / "benign.csv", [
        ("10.0.0.1", "10.0.0.9", 1, 2, 17, 30.0, "BENIGN")])
    out = tmp_path / "benign.json"

    assert mod.main(["--pcap", str(pcap), "--flows", str(flows),
                     "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["coverage"]["bins_attack_total"] == 0
    assert payload["coverage"]["attack_coverage_pct"] is None
    assert payload["coverage"]["coverage_pct"] == pytest.approx(100.0)
    assert payload["clock_offset"]["status"] == "measured"  # proto-aware match
    assert "n/a" in capsys.readouterr().out


def test_cli_backwards_packet_exits_1_until_merge_is_requested(
        tmp_path, capsys):
    mod = _load_cli()
    pcap = _backwards_pcap(tmp_path, "cli.pcap")
    out = tmp_path / "c.json"

    assert mod.main(["--pcap", str(pcap), "--out", str(out)]) == 1
    assert "closed bin" in capsys.readouterr().err
    assert not out.exists()

    merged = tmp_path / "m.json"
    assert mod.main(["--pcap", str(pcap), "--on-backwards", "merge",
                     "--out", str(merged)]) == 0
    payload = json.loads(merged.read_text(encoding="utf-8"))
    assert payload["backwards_policy"]["on_backwards"] == "merge"
    assert payload["backwards_policy"]["retention_horizon_sec"] == BIN + 1.0
    assert payload["pcap"]["backwards_packets_merged"] == 1
    assert "late packets merged" in capsys.readouterr().out


def test_cli_rejects_bad_arguments(tmp_path):
    mod = _load_cli()
    pcap = _backwards_pcap(tmp_path, "args.pcap")
    with pytest.raises(SystemExit):
        mod.main(["--pcap", str(pcap), "--bin-seconds", "0"])
    with pytest.raises(SystemExit):
        mod.main(["--pcap", str(pcap), "--reorder-grace-sec", "-1"])
    with pytest.raises(SystemExit):  # grace would bound no memory at all
        mod.main(["--pcap", str(pcap), "--reorder-grace-sec", "100000"])
    with pytest.raises(SystemExit):  # --apply-offset needs --flows
        mod.main(["--pcap", str(pcap), "--apply-offset"])
    # a missing PCAP is a clean exit 1, not a traceback
    assert mod.main(["--pcap", str(tmp_path / "absent.pcap")]) == 1


