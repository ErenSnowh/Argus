"""
pcap_bins.py
------------
Streaming, fixed-60 s-bin packet-tier extractor (D2 amendment, D1 keying).

Computes the 8 ``PACKET_LEVEL_COLUMNS`` (SIH 26153 section 1) directly from a
PCAP so they can be joined onto the flow-tier state of plan D1. Design rules,
each fixing a defect of the old demo path:

* **Streaming.** Reads with ``scapy.utils.PcapReader`` packet-by-packet -
  never ``rdpcap`` - so a multi-GB capture never materialises in memory.
  Bins are flushed as soon as the capture time moves past their end
  (captures are expected to be time-ordered; violations are counted).
* **Fixed wall-clock bins, not relative windows.** Bin edges are
  ``bin_start = floor(unix_seconds / 60) * 60`` (UTC) and the bin key is
  exactly the D1 key ``(src_ip, bin_start)`` - never ten equal-duration
  slices of the whole capture - so flow bins and PCAP bins join 1:1.
* **RST is not a retransmission.** ``retransmission_count`` uses a
  within-bin duplicate-sequence-number heuristic per direction; packets with
  the RST flag are resets and are skipped, and pure ACKs (zero-length
  segments) are not counted. The old code counted ``flags & 0x04`` (the RST
  flag) and labelled every reset a retransmission.
* **No zero-fill.** Every statistic is computed only from packets actually
  observed in that bin. Means/stds with no observations are ``None``
  (JSON ``null``), and bins of the flow table that the PCAP does not cover
  are joined as ``null`` + ``packet_features_covered: false`` - never as 0.

Known approximations (disclosed in every output's ``approximations`` block):
    * retransmission detection is a within-bin duplicate-sequence heuristic,
      not full TCP stream reassembly;
    * forward/backward directionality is not resolved (statistics are over
      all packets of the source host in the bin);
    * payload size is the transport payload for TCP/UDP and the L3 payload
      otherwise; ``tcp_window_*`` are ``null`` in bins with no TCP packets;
    * clock offset between the flow table and the PCAP is *measured*
      (``measure_clock_offset``), never assumed.

This module is the D1/WP2 implementation path. The dashboard demo path
(``features.extract_features_from_pcap``) is separate and must never be
joined to D1 bins.
"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Iterable, Optional

from ml.world_model.features import PACKET_LEVEL_COLUMNS

# The single place a packet-tier bin width is defined; D1's flow tier uses
# the same value (docs/plan/world-model-core.md, D1).
BIN_SECONDS = 60

APPROXIMATIONS = [
    "retransmission_count is a within-bin duplicate-sequence heuristic per "
    "direction (RST packets and pure ACKs are never counted); full TCP "
    "stream reassembly is not performed",
    "forward/backward directionality is not resolved from the raw capture",
    "payload_size is the transport payload for TCP/UDP, the L3 payload "
    "otherwise; tcp_window_* are null in bins without TCP packets",
    "means/stds are null (never 0) when a bin contains no observations of "
    "the underlying quantity",
    "clock offset between flow table and PCAP is measured per run, never "
    "assumed",
]

FlowKey = tuple[str, int]             # (src_ip, bin_start)
FiveTuple = tuple[str, str, int, int]  # directional (src, dst, sport, dport)


def bin_start_for(ts: float, bin_seconds: int = BIN_SECONDS) -> int:
    """D1 bin key component: floor of the unix timestamp to the bin grid."""
    return int(ts) // bin_seconds * bin_seconds


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile of an already sorted list (q in [0,1])."""
    if not sorted_values:
        raise ValueError("empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


class _BinAccumulator:
    """Packet-tier statistics for one (src_ip, bin_start) key."""

    __slots__ = (
        "src_ip", "bin_start", "packet_count", "byte_count",
        "first_pkt_ts", "last_pkt_ts", "ttls", "tcp_windows",
        "payloads", "frag_count", "retrans_count", "_seq_end",
    )

    def __init__(self, src_ip: str, bin_start: int) -> None:
        self.src_ip = src_ip
        self.bin_start = bin_start
        self.packet_count = 0
        self.byte_count = 0
        self.first_pkt_ts: Optional[float] = None
        self.last_pkt_ts: Optional[float] = None
        self.ttls: list[int] = []
        self.tcp_windows: list[int] = []
        self.payloads: list[int] = []
        self.frag_count = 0
        self.retrans_count = 0
        # directional 5-tuple -> highest end-seen sequence number (mod 2^32)
        self._seq_end: dict[FiveTuple, int] = {}

    def _observe_tcp(self, tcp: Any, key: FiveTuple) -> None:
        if int(tcp.flags) & 0x04:  # RST: a reset, never a retransmission
            return
        length = (
            len(bytes(tcp.payload))
            + bool(int(tcp.flags) & 0x02)  # SYN consumes a sequence number
            + bool(int(tcp.flags) & 0x01)  # FIN consumes a sequence number
        )
        if not length:
            return  # pure ACK: carries no data, not a retransmission
        seq = int(tcp.seq)
        prev = self._seq_end.get(key)
        if prev is not None and ((seq - prev) & 0xFFFFFFFF) >= 0x80000000:
            self.retrans_count += 1  # seq strictly before the highest end seen
        end = (seq + length) & 0xFFFFFFFF
        if prev is None or ((end - prev) & 0xFFFFFFFF) < 0x80000000:
            self._seq_end[key] = end

    def add(self, pkt: Any, ts: float, ip: Any, tcp: Any, udp: Any) -> None:
        self.packet_count += 1
        self.byte_count += len(pkt)
        self.first_pkt_ts = ts if self.first_pkt_ts is None else min(
            self.first_pkt_ts, ts)
        self.last_pkt_ts = ts if self.last_pkt_ts is None else max(
            self.last_pkt_ts, ts)

        self.ttls.append(int(ip.ttl))
        if int(ip.flags) & 0x01:  # More-Fragments bit (same as demo path)
            self.frag_count += 1
        if tcp is not None:
            self.tcp_windows.append(int(tcp.window))
            self.payloads.append(len(bytes(tcp.payload)))
            self._observe_tcp(
                tcp, (ip.src, ip.dst, int(tcp.sport), int(tcp.dport)))
        elif udp is not None:
            self.payloads.append(len(bytes(udp.payload)))
        else:
            self.payloads.append(len(bytes(ip.payload)))

    @staticmethod
    def _mean_std(values: list[float]) -> tuple[Optional[float], Optional[float]]:
        if not values:
            return None, None  # absent observation - never zero-filled
        return statistics.fmean(values), (
            statistics.pstdev(values) if len(values) > 1 else 0.0)

    def record(self) -> dict[str, Any]:
        ttl_mean, ttl_std = self._mean_std(self.ttls)
        win_mean, win_std = self._mean_std(self.tcp_windows)
        pay_mean, pay_std = self._mean_std(self.payloads)
        return {
            "src_ip": self.src_ip,
            "bin_start": self.bin_start,
            "packet_count": self.packet_count,
            "byte_count": self.byte_count,
            "first_pkt_ts": self.first_pkt_ts,
            "last_pkt_ts": self.last_pkt_ts,
            "features": {
                "ttl_mean": ttl_mean,
                "ttl_std": ttl_std,
                "tcp_window_mean": win_mean,
                "tcp_window_std": win_std,
                # counts are real observations: 0 means "none seen", which
                # is true, unlike a zero-filled absent statistic
                "ip_fragment_flag_count": float(self.frag_count),
                "payload_size_mean": pay_mean,
                "payload_size_std": pay_std,
                "retransmission_count": float(self.retrans_count),
            },
        }


def extract_pcap_bins(
    pcap_path: str | Path,
    bin_seconds: int = BIN_SECONDS,
    apply_offset_sec: float = 0.0,
) -> dict[str, Any]:
    """Stream a PCAP and return packet-tier records on the D1 bin grid.

    Returns a dict with ``records`` (list of per-``(src_ip, bin_start)``
    packet-tier rows, sorted by bin then src), ``stats`` (capture-level
    provenance) and ``approximations``. ``apply_offset_sec`` shifts packet
    timestamps before binning - only ever use the value returned by
    :func:`measure_clock_offset`, never a hand-picked number.
    """
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.utils import PcapReader

    path = Path(pcap_path)
    if not path.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")

    open_bins: dict[FlowKey, _BinAccumulator] = {}
    flushed: list[dict[str, Any]] = []
    packets = 0
    non_monotonic = 0
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    monotonic = True

    with PcapReader(str(path)) as reader:
        for pkt in reader:
            if not pkt.haslayer(IP):
                continue  # packet tier is defined over IP traffic
            if not hasattr(pkt, "time"):
                continue
            ts = float(pkt.time) + apply_offset_sec
            packets += 1
            if last_ts is not None and ts < last_ts:
                non_monotonic += 1
                monotonic = False  # stop flushing: a late packet may still
                # belong to an already-"closed" bin in an unordered capture
            last_ts = ts if last_ts is None else max(last_ts, ts)
            first_ts = ts if first_ts is None else min(first_ts, ts)

            ip = pkt[IP]
            tcp = pkt[TCP] if pkt.haslayer(TCP) else None
            udp = pkt[UDP] if pkt.haslayer(UDP) else None

            key = (ip.src, bin_start_for(ts, bin_seconds))
            acc = open_bins.get(key)
            if acc is None:
                acc = open_bins[key] = _BinAccumulator(key[0], key[1])
            acc.add(pkt, ts, ip, tcp, udp)

            if monotonic:
                # every bin whose window has ended can never receive another
                # packet from a time-ordered capture
                closed = [k for k in open_bins
                          if k[1] + bin_seconds <= ts]
                for k in closed:
                    flushed.append(open_bins.pop(k).record())

    flushed.extend(acc.record() for acc in open_bins.values())
    flushed.sort(key=lambda r: (r["bin_start"], r["src_ip"]))

    return {
        "records": flushed,
        "stats": {
            "pcap_file": path.name,
            "pcap_path": str(path),
            "packets_considered": packets,
            "first_pkt_ts": first_ts,
            "last_pkt_ts": last_ts,
            "non_monotonic_steps": non_monotonic,
            "bin_seconds": bin_seconds,
            "bin_key": "(src_ip, floor(unix_seconds / bin_seconds) * "
                       "bin_seconds)",
            "apply_offset_sec": apply_offset_sec,
        },
        "approximations": list(APPROXIMATIONS),
    }


def measure_clock_offset(
    pcap_path: str | Path,
    flow_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Measure PCAP-clock vs flow-table-clock offset on matched 5-tuples.

    ``flow_rows`` must yield dicts with keys ``src``, ``dst``, ``sport``,
    ``dport`` and ``ts`` (unix seconds, flow table clock). For every
    directional 5-tuple present in both sources the delta
    ``flow_ts - first_packet_ts`` is computed; the reported offset is the
    median of those deltas with the spread reported alongside. Returns
    ``status: "not_measured"`` when nothing matches - an unmeasured offset
    must be reported as such, never assumed to be zero.
    """
    from scapy.layers.inet import IP, TCP
    from scapy.utils import PcapReader

    flows = list(flow_rows)
    if not flows:
        return {
            "status": "not_measured",
            "reason": "no flow rows supplied",
            "method": "median(flow_ts - first_packet_ts) over matched "
                      "directional 5-tuples",
        }

    needed: dict[FiveTuple, float] = {}
    for row in flows:
        key = (str(row["src"]), str(row["dst"]),
               int(row["sport"]), int(row["dport"]))
        ts = float(row["ts"])
        # earliest flow start per 5-tuple (a 5-tuple may repeat across files)
        if key not in needed or ts < needed[key]:
            needed[key] = ts

    first_pkt: dict[FiveTuple, float] = {}
    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            if not pkt.haslayer(IP) or not pkt.haslayer(TCP):
                continue  # CICFlowMeter flow starts are TCP-handshake based
            ip, tcp = pkt[IP], pkt[TCP]
            key = (ip.src, ip.dst, int(tcp.sport), int(tcp.dport))
            if key in needed and key not in first_pkt:
                first_pkt[key] = float(pkt.time)

    deltas = sorted(needed[k] - first_pkt[k] for k in first_pkt)
    if not deltas:
        return {
            "status": "not_measured",
            "reason": "no directional 5-tuple matched between flow table "
                      "and PCAP",
            "method": "median(flow_ts - first_packet_ts) over matched "
                      "directional 5-tuples",
            "n_candidate_tuples": len(needed),
        }

    median = statistics.median(deltas)
    return {
        "status": "measured",
        "method": "median(flow_ts - first_packet_ts) over matched "
                  "directional 5-tuples",
        "n_matched_tuples": len(deltas),
        "n_candidate_tuples": len(needed),
        "median_sec": median,
        "p25_sec": _percentile(deltas, 0.25),
        "p75_sec": _percentile(deltas, 0.75),
        "min_sec": deltas[0],
        "max_sec": deltas[-1],
        "dispersion_iqr_sec": _percentile(deltas, 0.75)
                              - _percentile(deltas, 0.25),
        "note": "applied to binning only when --apply-offset is passed; "
                "a large |median| (e.g. whole hours) usually means a "
                "timezone-labelled flow clock, which must be reported, "
                "not silently corrected",
    }

def measure_bin_coverage(
    flow_bin_keys: Iterable[FlowKey],
    pcap_bin_keys: Iterable[FlowKey],
    attack_flow_bin_keys: Iterable[FlowKey] = (),
) -> dict[str, Any]:
    """Coverage of flow bins by PCAP bins, overall and for attack bins.

    This is the measurement D2's PCAP gate requires: both percentages must
    appear in the results/provenance JSON before any packet-tier number is
    reported, and a bin that is not covered is *excluded* (NaN), never
    zero-filled.
    """
    flow_keys = set(flow_bin_keys)
    pcap_keys = set(pcap_bin_keys)
    attack_keys = set(attack_flow_bin_keys)
    covered = flow_keys & pcap_keys
    attack_covered = attack_keys & covered

    def pct(part: int, whole: int) -> Optional[float]:
        return (100.0 * part / whole) if whole else None

    return {
        "bins_flow_total": len(flow_keys),
        "bins_covered": len(covered),
        "coverage_pct": pct(len(covered), len(flow_keys)),
        "bins_attack_total": len(attack_keys),
        "attack_bins_covered": len(attack_covered),
        "attack_coverage_pct": pct(len(attack_covered), len(attack_keys)),
        "pcap_bins_without_flow": len(pcap_keys - flow_keys),
        "uncovered_handling": "excluded from the packet-tier arm (NaN), "
                              "never zero-filled",
    }


def join_packet_features(
    flow_bin_keys: Iterable[FlowKey],
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Left-join packet-tier records onto flow bins (D1 keys).

    Every flow bin appears exactly once. Covered bins carry their measured
    features; uncovered bins carry ``features: null`` and
    ``packet_features_covered: false``. There is no code path here that can
    substitute 0 for a missing packet observation.
    """
    by_key = {(r["src_ip"], r["bin_start"]): r for r in records}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(flow_bin_keys)):
        rec = by_key.get(key)
        rows.append({
            "src_ip": key[0],
            "bin_start": key[1],
            "packet_features": rec["features"] if rec else None,
            "packet_features_covered": rec is not None,
        })
    return rows


def unsupported_or_null(record: dict[str, Any]) -> list[str]:
    """Names of packet-tier features that are null in a covered bin record."""
    return [c for c in PACKET_LEVEL_COLUMNS
            if record["features"].get(c) is None]

