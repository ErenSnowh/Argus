"""
pcap_bins.py
------------
Streaming, fixed-60 s-bin packet-tier extractor (D2 amendment, D1 keying).

Computes the 8 ``PACKET_LEVEL_COLUMNS`` (SIH 26153 section 1) directly from a
PCAP so they can be joined onto the flow-tier state of plan D1. Design rules,
each fixing a defect of the old demo path:

* **Streaming with online accumulators.** Reads with ``scapy.utils.PcapReader``
  packet-by-packet - never ``rdpcap`` - so a multi-GB capture never
  materialises in memory. Per-bin statistics are computed with O(1)
  count/mean/population-variance accumulators (Welford) - never per-packet
  TTL/window/payload lists - and open bins are closed by a **watermark scan
  that runs only when the capture crosses a new bin boundary**, not once per
  packet.
* **Backwards timestamps are never silent.** A packet that targets a bin the
  watermark already closed is either an error (default ``on_backwards="fail"``
  raises :class:`BackwardsTimestampError`) or an explicitly requested, counted
  merge (``on_backwards="merge"``, bounded by ``reorder_grace_sec``); data is
  never dropped or deduplicated without a trace. Exactly one record is emitted
  per ``(src_ip, bin_start)`` by construction (records come from a dict keyed
  by that tuple).
* **Bin identity = emitted-by host.** The bin key ``(src_ip, bin_start)``
  always names the host that *emitted* the packets: for the packet tier that
  is the raw ``IP.src`` of each packet (reverse-direction packets are keyed
  under the responder, not flipped onto the flow initiator); for the flow tier
  it is ``Source IP`` - the host that emitted the flow's first packet. See
  ``BIN_KEY_SEMANTICS``.
* **Fixed wall-clock bins, not relative windows.** Bin edges are
  ``bin_start = floor(unix_seconds / 60) * 60`` (UTC) - never ten
  equal-duration slices of the whole capture - so flow bins and PCAP bins
  join 1:1 on the identical D1 key.
* **RST is not a retransmission.** ``retransmission_count`` is an
  **approximate duplicate-sequence proxy**: a within-bin, per-direction
  duplicate-sequence-number heuristic; packets with the RST flag are resets
  and are skipped, and pure ACKs (zero-length segments) are not counted. The
  old code counted ``flags & 0x04`` (the RST flag) and labelled every reset a
  retransmission.
* **Clock offset: observed vs applied are separate.** ``measure_clock_offset``
  reports the raw observed delta (``observed_delta`` block, quantization-
  aware) and a *separate* ``apply`` decision (``eligible``, ``reason``,
  ``correction_sec``); an offset is applied only when it was measured with
  enough matches and unambiguous dispersion - never assumed, never silently 0.
* **No zero-fill.** Every statistic is computed only from packets actually
  observed in that bin. Means/stds with no observations are ``None``
  (JSON ``null``), and bins of the flow table that the PCAP does not cover
  are joined as ``null`` + ``packet_features_covered: false`` - never as 0.

Known approximations (disclosed in every output's ``approximations`` block):
    * retransmission detection is an approximate duplicate-sequence proxy
      (within-bin), not full TCP stream reassembly;
    * forward/backward directionality is not resolved (statistics are over
      all packets the source host emitted in the bin);
    * payload size is the transport payload for TCP/UDP and the L3 payload
      otherwise; ``tcp_window_*`` are ``null`` in bins with no TCP packets;
    * clock offset between the flow table and the PCAP is *measured*
      (``measure_clock_offset``), never assumed, and application is a
      separate, guarded decision.

This module is the D1/WP2 implementation path. The dashboard demo path
(``features.extract_features_from_pcap``) is separate and must never be
joined to D1 bins.
"""
from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Optional

from ml.world_model.features import PACKET_LEVEL_COLUMNS

# The single place a packet-tier bin width is defined; D1's flow tier uses
# the same value (docs/plan/world-model-core.md, D1).
BIN_SECONDS = 60

# Clock-offset application guards. Flow-table timestamps are second-precision,
# so a single measured delta carries up to 1 s of quantization: a *spread*
# larger than one quantization step means the matches do not agree on one
# offset (ambiguous), and fewer than OFFSET_MIN_MATCHES_TO_APPLY pairs cannot
# establish one either. Both conditions refuse application instead of
# silently applying 0 or a bad number.
OFFSET_MIN_MATCHES_TO_APPLY = 3
OFFSET_MAX_IQR_TO_APPLY_SEC = 1.0

# An |median| at or above one hour is the signature of a timezone-labelled
# flow clock (e.g. whole-hour offsets); it is flagged, never silently fixed.
TIMEZONE_SCALE_SEC = 3600.0

# Clock-matcher connection splitting: a reused (proto, src, dst, sport, dport)
# tuple starts a new connection after this idle gap (CICFlowMeter's flow
# timeout scale) or on a fresh SYN after this short gap. Matching is per
# connection instance - never a collapse to the earliest flow of a tuple.
FLOW_IDLE_GAP_SEC = 120.0
SYN_RESTART_GAP_SEC = 30.0

APPROXIMATIONS = [
    "retransmission_count is an approximate duplicate-sequence proxy: a "
    "within-bin, per-direction duplicate-sequence heuristic (RST packets and "
    "pure ACKs are never counted); it is not a true retransmission detector "
    "and full TCP stream reassembly is not performed",
    "forward/backward directionality is not resolved from the raw capture; "
    "statistics are over the packets the keyed host emitted in the bin",
    "payload_size is the transport payload for TCP/UDP, the L3 payload "
    "otherwise; tcp_window_* are null in bins without TCP packets",
    "means/stds are null (never 0) when a bin contains no observations of "
    "the underlying quantity",
    "clock offset between flow table and PCAP is measured per run, never "
    "assumed; the observed delta and the applied correction are reported "
    "separately, and application is refused on too few or ambiguous matches",
]

FlowKey = tuple[str, int]             # (src_ip, bin_start)
FiveTuple = tuple[str, str, int, int]  # directional (src, dst, sport, dport)
# Clock-matcher key: protocol FIRST so TCP and UDP cannot collide on the
# same (ip, port) tuple; directional, per connection instance.
OffsetKey = tuple[str, str, str, int, int]  # (proto, src, dst, sport, dport)

# What the bin key means on BOTH tiers (D1 flow tier and this packet tier):
# the host that emitted the record. For packets that is the raw IP.src; for
# flows it is Source IP (the emitter of the flow's first packet). Reverse
# packets therefore stay under the responder's own key - never flipped onto
# the flow initiator - and joins are between like concepts.
BIN_KEY_SEMANTICS = (
    "emitted-by host: src_ip names the host that emitted the record "
    "(packet tier: raw IP.src of each packet, so reverse-direction packets "
    "stay under the responder; flow tier: Source IP, the emitter of the "
    "flow's first packet)"
)


class BackwardsTimestampError(ValueError):
    """A packet targeted a bin the watermark already closed.

    Raised by :func:`extract_pcap_bins` with the default
    ``on_backwards="fail"`` policy so that out-of-order capture data is an
    explicit, visible failure instead of silent deduplication loss. Pass
    ``on_backwards="merge"`` to accept and count such packets instead.
    """


def bin_start_for(ts: float, bin_seconds: int = BIN_SECONDS) -> int:
    """D1 bin key component: floor of the unix timestamp to the bin grid."""
    return int(ts) // bin_seconds * bin_seconds


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile of a sorted list (q in [0, 1])."""
    if not sorted_values:
        raise ValueError("empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


class _OnlineStats:
    """O(1) count/mean/population-variance accumulator (Welford).

    Replaces the old per-packet ``ttls``/``tcp_windows``/``payloads`` lists:
    memory per bin stays constant no matter how many packets land in it.
    Returns ``(None, None)`` with no observations and ``std = 0.0`` with a
    single observation - identical semantics to the previous
    ``_mean_std`` over lists (``statistics.pstdev`` is the population
    std, which is what ``M2 / n`` computes here).
    """

    __slots__ = ("n", "mean", "_m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def add(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (value - self.mean)

    @property
    def mean_std(self) -> tuple[Optional[float], Optional[float]]:
        if self.n == 0:
            return None, None  # absent observation - never zero-filled
        variance = self._m2 / self.n
        return self.mean, (math.sqrt(variance) if self.n > 1 else 0.0)


class _BinAccumulator:
    """Packet-tier statistics for one (src_ip, bin_start) key.

    All per-quantity state is O(1): online accumulators plus the per-flow
    sequence-end map needed by the duplicate-sequence proxy. No per-packet
    lists are retained.
    """

    __slots__ = (
        "src_ip", "bin_start", "packet_count", "byte_count",
        "first_pkt_ts", "last_pkt_ts", "_ttl", "_window",
        "_payload", "frag_count", "retrans_count", "_seq_end",
    )

    def __init__(self, src_ip: str, bin_start: int) -> None:
        self.src_ip = src_ip
        self.bin_start = bin_start
        self.packet_count = 0
        self.byte_count = 0
        self.first_pkt_ts: Optional[float] = None
        self.last_pkt_ts: Optional[float] = None
        self._ttl = _OnlineStats()
        self._window = _OnlineStats()
        self._payload = _OnlineStats()
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

        self._ttl.add(float(int(ip.ttl)))
        # Fragments: the More-Fragments bit OR a non-zero fragment offset
        # (non-first fragments have MF clear but frag != 0).
        if (int(ip.flags) & 0x01) or int(ip.frag) != 0:
            self.frag_count += 1
        if tcp is not None:
            self._window.add(float(int(tcp.window)))
            self._payload.add(float(len(bytes(tcp.payload))))
            self._observe_tcp(
                tcp, (ip.src, ip.dst, int(tcp.sport), int(tcp.dport)))
        elif udp is not None:
            self._payload.add(float(len(bytes(udp.payload))))
        else:
            self._payload.add(float(len(bytes(ip.payload))))

    def record(self) -> dict[str, Any]:
        ttl_mean, ttl_std = self._ttl.mean_std
        win_mean, win_std = self._window.mean_std
        pay_mean, pay_std = self._payload.mean_std
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
    on_backwards: str = "fail",
    reorder_grace_sec: float = 0.0,
) -> dict[str, Any]:
    """Stream a PCAP and return packet-tier records on the D1 bin grid.

    Returns a dict with ``records`` (list of per-``(src_ip, bin_start)``
    packet-tier rows, sorted by bin then src - **exactly one row per key**,
    guaranteed because records are produced from a dict keyed by that tuple),
    ``stats`` (capture-level provenance) and ``approximations``.
    ``apply_offset_sec`` shifts packet timestamps before binning - only ever
    use the value returned by :func:`measure_clock_offset`'s ``apply``
    block, never a hand-picked number.

    Ordering policy (backwards timestamps are never silently lost):

    * The extractor keeps ``open_bins`` and closes them with a **watermark
      scan that runs only when the capture crosses a bin boundary** (tracked
      as ``stats["watermark_scans"]``), not on every packet.
    * ``on_backwards="fail"`` (default): a packet whose target bin is
      already closed raises :class:`BackwardsTimestampError`.
    * ``on_backwards="merge"``: such packets are merged into their (still
      held) bin accumulator and counted in
      ``stats["backwards_packets_merged"]`` - bounded reordering; bins are
      never discarded until every packet has been consumed.
    """
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.utils import PcapReader

    if on_backwards not in ("fail", "merge"):
        raise ValueError(
            f"on_backwards must be 'fail' or 'merge', got {on_backwards!r}")
    if reorder_grace_sec < 0:
        raise ValueError("reorder_grace_sec must be >= 0")

    path = Path(pcap_path)
    if not path.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")

    open_bins: dict[FlowKey, _BinAccumulator] = {}
    # Closed accumulators are retained ONLY in "merge" mode so backwards
    # packets can still be folded in; "fail" mode converts them to records
    # on close and frees the per-flow sequence map (streaming memory).
    closed_bins: dict[FlowKey, _BinAccumulator] = {}
    flushed: list[dict[str, Any]] = []
    packets = 0
    non_monotonic = 0
    watermark_scans = 0
    backwards_merged = 0
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    watermark: Optional[float] = None
    last_boundary: Optional[int] = None  # bin_start of latest watermark

    def _close_past(t: float) -> None:
        """Close every bin whose window (plus reorder grace) has ended.

        Runs only when the capture crosses a bin boundary - not per packet -
        which is the point of the watermark.
        """
        nonlocal watermark_scans
        watermark_scans += 1
        cutoff = t - reorder_grace_sec
        closed = [k for k in open_bins if k[1] + bin_seconds <= cutoff]
        for k in closed:
            acc = open_bins.pop(k)
            if on_backwards == "merge":
                closed_bins[k] = acc  # kept: late packets merge into it
            else:
                flushed.append(acc.record())

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
            last_ts = ts if last_ts is None else max(last_ts, ts)
            first_ts = ts if first_ts is None else min(first_ts, ts)

            ip = pkt[IP]
            tcp = pkt[TCP] if pkt.haslayer(TCP) else None
            udp = pkt[UDP] if pkt.haslayer(UDP) else None

            key = (ip.src, bin_start_for(ts, bin_seconds))
            acc = open_bins.get(key)
            if acc is None:
                if _is_closed(key, watermark, bin_seconds, reorder_grace_sec):
                    # The watermark already moved past this bin: a packet
                    # arriving after its bin closed. Never silent - either
                    # fail explicitly or merge into the retained accumulator.
                    if on_backwards == "fail":
                        raise BackwardsTimestampError(
                            f"packet at ts={ts} targets closed bin {key} "
                            f"(watermark={watermark}); capture moved "
                            f"backwards beyond reorder_grace_sec="
                            f"{reorder_grace_sec}. Re-run with "
                            f"on_backwards='merge' to accept bounded "
                            f"reordering instead of failing.")
                    acc = closed_bins.get(key)
                    if acc is None:
                        # Bin never seen before (old bin, first packet now):
                        # create it; the next boundary scan closes it.
                        acc = open_bins[key] = _BinAccumulator(key[0], key[1])
                    else:
                        backwards_merged += 1
                else:
                    acc = open_bins[key] = _BinAccumulator(key[0], key[1])
            acc.add(pkt, ts, ip, tcp, udp)

            # Watermark advances with capture time; the close scan runs only
            # when the capture crosses into a different bin - not per packet.
            if watermark is None or ts > watermark:
                watermark = ts
                boundary = bin_start_for(ts, bin_seconds)
                if last_boundary is None or boundary != last_boundary:
                    _close_past(watermark)
                    last_boundary = boundary

    records = flushed
    records.extend(acc.record() for acc in open_bins.values())
    records.extend(acc.record() for acc in closed_bins.values())
    records.sort(key=lambda r: (r["bin_start"], r["src_ip"]))

    return {
        "records": records,
        "stats": {
            "pcap_file": path.name,
            "pcap_path": str(path),
            "packets_considered": packets,
            "first_pkt_ts": first_ts,
            "last_pkt_ts": last_ts,
            "non_monotonic_steps": non_monotonic,
            "backwards_policy": on_backwards,
            "backwards_packets_merged": backwards_merged,
            "reorder_grace_sec": reorder_grace_sec,
            "watermark_scans": watermark_scans,
            "bin_seconds": bin_seconds,
            "bin_key": "(src_ip, floor(unix_seconds / bin_seconds) * "
                       "bin_seconds)",
            "bin_key_semantics": BIN_KEY_SEMANTICS,
            "apply_offset_sec": apply_offset_sec,
        },
        "approximations": list(APPROXIMATIONS),
    }


def _is_closed(
    key: FlowKey,
    watermark: Optional[float],
    bin_seconds: int,
    reorder_grace_sec: float,
) -> bool:
    """True when the watermark has moved past ``key``'s bin window."""
    if watermark is None:
        return False
    return key[1] + bin_seconds <= watermark - reorder_grace_sec


def measure_clock_offset(
    pcap_path: str | Path,
    flow_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Measure PCAP-clock vs flow-table-clock offset on matched 5-tuples.

    ``flow_rows`` must yield dicts with keys ``src``, ``dst``, ``sport``,
    ``dport`` and ``ts`` (unix seconds, flow table clock); ``proto`` is
    optional (``"tcp"``/``"udp"``, default ``"tcp"`` because CICFlowMeter
    flow starts are TCP-handshake based). The match key is the *protocol-
    aware directional 5-tuple* ``(proto, src, dst, sport, dport)`` so a UDP
    flow can never collide with a TCP flow on the same ports, and matching
    is *per connection instance*: a reused tuple is split into connection
    instances (idle gap >= :data:`FLOW_IDLE_GAP_SEC` or a fresh SYN after
    :data:`SYN_RESTART_GAP_SEC`) and each flow start is paired with the
    matching instance's first packet - never collapsed to the earliest flow
    of the tuple.

    The result separates two things the old API conflated:

    * ``observed_delta`` - the raw measurement (``flow_ts -
      first_packet_ts`` per matched pair), with median, spread, min/max,
      pair count and the flow table's timestamp quantization step. Always
      reported when anything matched.
    * ``apply`` - an explicit decision block: ``eligible``, ``reason``
      (``None`` when eligible), ``correction_sec`` (the value to add to
      packet timestamps when eligible, ``None`` otherwise) and ``flags``.
      Application is refused when fewer than
      :data:`OFFSET_MIN_MATCHES_TO_APPLY` pairs matched or when the
      dispersion exceeds :data:`OFFSET_MAX_IQR_TO_APPLY_SEC` (the
      quantization-aware ambiguity bound) - a refused offset must never be
      silently applied as 0.

    ``status: "not_measured"`` is returned when nothing matches - an
    unmeasured offset is reported as such, never assumed to be zero. The
    legacy flat fields (``median_sec`` etc.) mirror ``observed_delta``.
    """
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.utils import PcapReader

    method = ("median(flow_ts - first_packet_ts) over matched "
              "protocol-aware directional 5-tuples, paired per "
              "connection instance")

    flows = list(flow_rows)
    if not flows:
        return {
            "status": "not_measured",
            "reason": "no flow rows supplied",
            "method": method,
            "apply": {
                "eligible": False,
                "reason": "offset not measured (no flow rows)",
                "correction_sec": None,
                "flags": [],
            },
        }

    # Protocol-aware directional key -> flow starts (all instances kept;
    # no earliest-flow collapse).
    needed: dict[OffsetKey, list[float]] = {}
    for row in flows:
        proto = str(row.get("proto", "tcp")).lower()
        key = (proto, str(row["src"]), str(row["dst"]),
               int(row["sport"]), int(row["dport"]))
        needed.setdefault(key, []).append(float(row["ts"]))

    # PCAP side: first packet of every connection instance per key.
    conn_starts: dict[OffsetKey, list[float]] = {}
    last_seen: dict[OffsetKey, float] = {}
    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            if not pkt.haslayer(IP):
                continue
            ip = pkt[IP]
            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                key = ("tcp", ip.src, ip.dst,
                       int(tcp.sport), int(tcp.dport))
                is_syn = (bool(int(tcp.flags) & 0x02)
                          and not bool(int(tcp.flags) & 0x10))  # SYN w/o ACK
            elif pkt.haslayer(UDP):
                udp = pkt[UDP]
                key = ("udp", ip.src, ip.dst,
                       int(udp.sport), int(udp.dport))
                is_syn = False
            else:
                continue
            if key not in needed:
                continue
            ts = float(pkt.time)
            starts = conn_starts.setdefault(key, [])
            prev_start = starts[-1] if starts else None
            prev_seen = last_seen.get(key)
            new_conn = (
                prev_start is None
                or (prev_seen is not None
                    and ts - prev_seen >= FLOW_IDLE_GAP_SEC)
                or (is_syn and prev_start is not None
                    and ts - prev_start >= SYN_RESTART_GAP_SEC)
            )
            if new_conn:
                starts.append(ts)
            last_seen[key] = ts

    # Pair i-th flow start with i-th connection start per key (both in
    # chronological order). Extra instances on either side are counted but
    # not paired - no fabricated matches.
    deltas: list[float] = []
    n_unpaired = 0
    for key, flow_starts in needed.items():
        flow_starts.sort()
        conns = conn_starts.get(key, [])
        for flow_ts, conn_ts in zip(flow_starts, conns):
            deltas.append(flow_ts - conn_ts)
        n_unpaired += abs(len(flow_starts) - len(conns))
    deltas.sort()
    if not deltas:
        return {
            "status": "not_measured",
            "reason": "no protocol-aware directional 5-tuple matched "
                      "between flow table and PCAP",
            "method": method,
            "n_candidate_tuples": len(needed),
            "n_unpaired_instances": n_unpaired,
            "apply": {
                "eligible": False,
                "reason": "offset not measured (no matches)",
                "correction_sec": None,
                "flags": [],
            },
        }

    median = statistics.median(deltas)
    iqr = _percentile(deltas, 0.75) - _percentile(deltas, 0.25)
    quantization = _flow_timestamp_quantization(flows)

    flags: list[str] = []
    if abs(median) >= TIMEZONE_SCALE_SEC:
        flags.append("timezone_scale")
    if quantization >= 1.0:
        flags.append("second_precision_flow_timestamps")

    observed = {
        "median_sec": median,
        "p25_sec": _percentile(deltas, 0.25),
        "p75_sec": _percentile(deltas, 0.75),
        "min_sec": deltas[0],
        "max_sec": deltas[-1],
        "dispersion_iqr_sec": iqr,
        "n_matched_pairs": len(deltas),
        "n_unpaired_instances": n_unpaired,
        "quantization_sec": quantization,
        "deltas_sec": deltas,
        "note": "raw observed delta (flow_ts - packet_ts); application is "
                "a separate, guarded decision in the 'apply' block",
    }

    # --- apply decision (never implicit, never silently 0) ---
    if len(deltas) < OFFSET_MIN_MATCHES_TO_APPLY:
        apply_block = {
            "eligible": False,
            "reason": (
                f"too few matched connection instances "
                f"({len(deltas)} < {OFFSET_MIN_MATCHES_TO_APPLY}); "
                f"refusing to apply - report the observed delta only"),
            "correction_sec": None,
            "flags": flags + ["too_few_matches"],
        }
    elif iqr > OFFSET_MAX_IQR_TO_APPLY_SEC:
        apply_block = {
            "eligible": False,
            "reason": (
                f"ambiguous offset: dispersion iqr={iqr:.3f}s exceeds "
                f"quantization-aware bound "
                f"{OFFSET_MAX_IQR_TO_APPLY_SEC:.1f}s (matches disagree; "
                f"flow timestamps quantized to {quantization:g}s); "
                f"refusing to apply"),
            "correction_sec": None,
            "flags": flags + ["ambiguous_dispersion"],
        }
    else:
        apply_block = {
            "eligible": True,
            "reason": None,
            "correction_sec": median,
            "flags": flags,
        }

    return {
        "status": "measured",
        "method": method,
        "n_matched_tuples": len(deltas),      # legacy alias
        "n_candidate_tuples": len(needed),    # legacy alias
        "median_sec": median,                 # legacy alias
        "p25_sec": observed["p25_sec"],
        "p75_sec": observed["p75_sec"],
        "min_sec": observed["min_sec"],
        "max_sec": observed["max_sec"],
        "dispersion_iqr_sec": iqr,
        "observed_delta": observed,
        "apply": apply_block,
        "note": "observed_delta is the raw measurement; apply.correction_sec "
                "is the only value --apply-offset may use, and only when "
                "apply.eligible is true. A large |median| (e.g. whole hours) "
                "usually means a timezone-labelled flow clock: reported via "
                "the timezone_scale flag, never silently corrected",
    }


def _flow_timestamp_quantization(flows: list[dict[str, Any]]) -> float:
    """Timestamp quantization step of the flow table (seconds).

    Flow timestamps are second-precision in this snapshot, which quantizes
    every measured delta by up to 1 s - the reason the apply-dispersion
    bound is quantization-aware. Sub-second values lower the bound.
    """
    ts = [float(f["ts"]) for f in flows]
    if not ts:
        return 1.0
    frac_span = max(abs(t - round(t)) for t in ts)
    return 1.0 if frac_span < 1e-9 else 0.001


def measure_bin_coverage(
    flow_bin_keys: Iterable[FlowKey],
    pcap_bin_keys: Iterable[FlowKey],
    attack_flow_bin_keys: Iterable[FlowKey] = (),
) -> dict[str, Any]:
    """Coverage of flow bins by PCAP bins, overall and for attack bins.

    Both sides are de-duplicated to *unique* ``(src_ip, bin_start)``
    host-minute bins first, so repeated flow rows for the same host-minute
    (or duplicate PCAP records - which the extractor cannot produce) never
    inflate the denominators. This is the measurement D2's PCAP gate
    requires: both percentages must appear in the results/provenance JSON
    before any packet-tier number is reported, and a bin that is not covered
    is *excluded* (NaN), never zero-filled. A zero denominator (e.g. an
    all-benign flow table with no attack bins) yields ``None`` - never
    ``0.0`` and never a division error.
    """
    flow_keys = set(flow_bin_keys)
    pcap_keys = set(pcap_bin_keys)
    attack_keys = set(attack_flow_bin_keys)
    covered = flow_keys & pcap_keys
    attack_covered = attack_keys & covered

    def pct(part: int, whole: int) -> Optional[float]:
        return (100.0 * part / whole) if whole else None

    return {
        "counting_unit": "unique (src_ip, bin_start) host-minute bins "
                         "(de-duplicated before counting)",
        "bins_flow_total": len(flow_keys),
        "bins_covered": len(covered),
        "coverage_pct": pct(len(covered), len(flow_keys)),
        "bins_attack_total": len(attack_keys),
        "attack_bins_covered": len(attack_covered),
        "attack_coverage_pct": pct(len(attack_covered), len(attack_keys)),
        "pcap_bins_without_flow": len(pcap_keys - flow_keys),
        "uncovered_handling": "excluded from the packet-tier arm (NaN), "
                              "never zero-filled",
        "zero_denominator_handling": "null (JSON) / 'n/a' when printing - "
                                     "never 0.0 masquerading as a rate",
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

