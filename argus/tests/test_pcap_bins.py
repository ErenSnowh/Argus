"""
test_pcap_bins.py
-----------------
Tests for the streaming fixed-60 s-bin packet-tier extractor (D2 amendment)
and the demo-extractor RST fix. Synthetic captures are built in-process with
explicit packet times - no network, no large downloads.

Run: pytest argus/tests/test_pcap_bins.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Raw
from scapy.utils import wrpcap

from ml.world_model.pcap_bins import (
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
    assert offset["median_sec"] == pytest.approx(7.5)
    assert offset["n_matched_tuples"] == 1


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
def test_load_flow_rows_epoch_not_corrupted_by_pandas_unit(tmp_path):
    """pandas 3 parses to datetime64[us]; the old /1e9 gave ~1e3x-too-small
    epochs, which silently destroyed clock-offset and coverage joins."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pcap_stream_bins",
        ROOT / "scripts" / "pcap_stream_bins.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

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


