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
* **Backwards timestamps are never silent, and reordering is memory-bounded.**
  A packet that targets a bin the watermark already closed is either an error
  (default ``on_backwards="fail"`` raises :class:`BackwardsTimestampError`) or
  an explicitly requested, counted merge (``on_backwards="merge"``, bounded by
  ``reorder_grace_sec``). Retaining a closed accumulator is bounded too: a bin
  is held for at most ``bin_seconds + reorder_grace_sec`` of capture time
  (``stats["retention_horizon_sec"]``) before it is finalised and freed, so a
  packet arriving hours late can never keep an old accumulator alive - in
  ``merge`` mode such a packet is counted in
  ``backwards_packets_late_dropped``, in ``fail`` mode it aborts the run.
  Exactly one record is emitted per ``(src_ip, bin_start)`` by construction
  (records come from a dict keyed by that tuple).
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
* **``retransmission_count`` is a sequence-regression detector, not a
  retransmission count.** The column name is fixed by ``PACKET_LEVEL_COLUMNS``;
  what it counts is a within-bin, per-direction **duplicate byte range**: a TCP
  segment whose entire sequence range was already observed on that directional
  5-tuple. RST packets are resets and are never counted, pure ACKs (zero-length
  segments) are never counted, and **in-window reordering is not counted** - a
  segment that arrives early and later fills a gap advances the contiguous
  frontier instead of being called a duplicate. The old code counted
  ``flags & 0x04`` (the RST flag) and labelled every reset a retransmission; a
  naive "seq before the highest end seen" rule would have called 1000, 1020,
  1010 one retransmission, which it is not.
* **Clock offset: an interval, never a biased point estimate.** Under floor
  quantization ``q`` of the flow clock, one matched pair only proves that the
  true offset lies in ``[delta, delta + q)``, so ``measure_clock_offset``
  reports the **feasible interval** ``[max(delta_i) - q_packet, min(delta_i) +
  q_flow)`` (``observed_delta.feasible_interval``) and only applies a
  correction when that interval is non-empty and narrow enough
  (``OFFSET_MAX_INTERVAL_WIDTH_SEC``), the precision of *both* inputs is
  determined (see :func:`infer_timestamp_quantization` - minute-precision input
  is never labelled second precision), and enough valid, correctly paired
  connection instances matched. The applied value is the interval midpoint
  with ``max_error_sec`` = half its width; the sample median is kept only as a
  labelled-biased dispersion statistic and is never applied.
* **No zero-fill.** Every statistic is computed only from packets actually
  observed in that bin. Means/stds with no observations are ``None``
  (JSON ``null``), and bins of the flow table that the PCAP does not cover
  are joined as ``null`` + ``packet_features_covered: false`` - never as 0.

Known approximations (disclosed in every output's ``approximations`` block):
    * ``retransmission_count`` is a sequence-regression detector (duplicate
      byte ranges, within-bin, per direction), not a retransmission count and
      not full TCP stream reassembly;
    * forward/backward directionality is not resolved (statistics are over
      all packets the source host emitted in the bin);
    * payload size is the transport payload for TCP/UDP and the L3 payload
      otherwise; ``tcp_window_*`` are ``null`` in bins with no TCP packets;
    * clock offset between the flow table and the PCAP is *measured* as a
      feasible interval (``measure_clock_offset``), never assumed, and is
      applied only when that interval is narrow and both inputs' timestamp
      precision is determined - otherwise application is refused.

This module is the D1/WP2 **extraction** path and a **standalone CLI**
(``scripts/pcap_stream_bins.py``): no training, binning or benchmark code
consumes its output yet, because no real CIC PCAP has been measured and no
combined flow+packet feature scope has been approved. The dashboard demo path
(``features.extract_features_from_pcap``) is separate and must never be joined
to D1 bins.
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

# --- clock-offset application guards --------------------------------------
# Under floor quantization with step q, one matched pair of timestamps proves
# only that the true offset lies in [delta, delta + q). Application is
# therefore decided on the *width of the feasible interval*, not on the spread
# of a median (which is biased low by the sub-quantization-step part of the
# packet clock). OFFSET_MAX_INTERVAL_WIDTH_SEC is the widest interval the
# reviewer accepts as "the offset is pinned": applying its midpoint then
# carries a known worst-case error of half that width (max_error_sec). It is
# deliberately tighter than one second, so minute-precision flow tables are
# refused unless their matched packet starts span almost a whole
# quantization cell.
OFFSET_MIN_PAIRS_TO_APPLY = 3
OFFSET_MAX_INTERVAL_WIDTH_SEC = 0.1

# Timestamp precisions the estimator understands. Any input whose precision
# cannot be *determined* (too few values, or a declared step contradicted by
# the values themselves) makes the offset not-applicable - never assumed.
SUPPORTED_QUANTIZATION_SEC = (0.001, 1.0, 60.0)
MIN_VALUES_TO_INFER_QUANTIZATION = 20
# Bounded number of raw deltas written into the JSON output; the full list is
# never serialised (a 2.8M-pair measurement would produce a ~50 MB artifact).
MAX_DELTA_SAMPLES = 10

# An |interval midpoint| at or above one hour is the signature of a
# timezone-labelled flow clock (e.g. whole-hour offsets); it is flagged and
# refused, never silently fixed.
TIMEZONE_SCALE_SEC = 3600.0

# Clock-matcher connection splitting: a reused (proto, src, dst, sport, dport)
# tuple starts a new connection after this idle gap (CICFlowMeter's flow
# timeout scale) or on a fresh SYN after this short gap. Matching is per
# connection instance - never a collapse to the earliest flow of a tuple.
FLOW_IDLE_GAP_SEC = 120.0
SYN_RESTART_GAP_SEC = 30.0

# --- reordering bounds -----------------------------------------------------
# Nonzero default grace: capture files are routinely written a little out of
# order, and a 0 s grace made every such packet an error. One second is the
# review's recommendation and is still far below a 60 s bin.
DEFAULT_REORDER_GRACE_SEC = 1.0
# A grace larger than this would itself be the memory leak it is meant to
# handle (retention horizon = bin_seconds + grace), so it is rejected.
MAX_REORDER_GRACE_SEC = 300.0

# Per-direction bound on the out-of-order sequence ranges remembered by the
# sequence-regression detector. Beyond it, ranges are dropped and counted
# (``seq_range_sets_capped``) instead of growing without limit.
SEQ_RANGES_CAP = 64

FEATURE_SEMANTICS = {
    "retransmission_count": (
        "sequence-regression detector, NOT a retransmission count: counts TCP "
        "segments whose entire sequence range was already observed on that "
        "directional 5-tuple inside this bin (duplicate byte ranges). RST "
        "packets and pure ACKs are never counted; in-window reordering and "
        "gap-filling are never counted. The column name is fixed by "
        "PACKET_LEVEL_COLUMNS, the semantics are not"),
}

APPROXIMATIONS = [
    "retransmission_count is a sequence-regression detector: a within-bin, "
    "per-direction duplicate-byte-range count (RST packets and pure ACKs are "
    "never counted, in-window reordering is never counted). It is not a "
    "retransmission count, and full TCP stream reassembly is not performed",
    "forward/backward directionality is not resolved from the raw capture; "
    "statistics are over the packets the keyed host emitted in the bin",
    "payload_size is the transport payload for TCP/UDP, the L3 payload "
    "otherwise; tcp_window_* are null in bins without TCP packets",
    "means/stds are null (never 0) when a bin contains no observations of "
    "the underlying quantity",
    "clock offset between flow table and PCAP is measured as a feasible "
    "interval under each input's determined timestamp precision, never "
    "assumed; the sample median is biased by floor quantization and is "
    "reported as a dispersion statistic only - a correction is applied only "
    "when the interval is non-empty and at most "
    f"{OFFSET_MAX_INTERVAL_WIDTH_SEC} s wide, and its midpoint's worst-case "
    "error (max_error_sec) is reported with it",
]

FlowKey = tuple[str, int]             # (src_ip, bin_start)
FiveTuple = tuple[str, str, int, int]  # directional (src, dst, sport, dport)
# Clock-matcher key: protocol FIRST so TCP and UDP cannot collide on the
# same (ip, port) tuple; directional, per connection instance.
OffsetKey = tuple[str, str, str, int, int]  # (proto, src, dst, sport, dport)

# What the bin key names on BOTH tiers: the host that emitted the record.
# That is the ONLY thing the two tiers share - the join is not a join between
# equivalent observations, and must never be described as one:
#   * packet tier: raw ``IP.src`` of each packet, so a host's bin mixes every
#     direction it emitted (a responder's replies sit under the responder) and
#     the counting unit is one packet;
#   * flow tier: CICFlowMeter ``Source IP`` = the flow *initiator*, and the
#     row's features already mix BOTH directions of that flow; the counting
#     unit is one flow.
# So the join aligns host identity and time bucket only - not direction
# semantics and not the counting unit.
BIN_KEY_SEMANTICS = (
    "src_ip names the emitting host on both tiers (packet tier: raw IP.src "
    "per packet, so reverse-direction packets stay under the responder; flow "
    "tier: CICFlowMeter Source IP, the flow initiator). Host identity and "
    "time bucket match; direction semantics and the counting unit do NOT "
    "(per-direction packets vs bidirectional flow rows) - the join is not "
    "between equivalent observations"
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


def _seq_lt(a: int, b: int) -> bool:
    """Wrap-aware strict ``a < b`` on the 32-bit sequence circle.

    Uses the half-window convention: ``a`` precedes ``b`` when the forward
    distance from ``a`` to ``b`` is under 2^31.
    """
    return 0 < ((b - a) & 0xFFFFFFFF) < 0x80000000


def _seq_le(a: int, b: int) -> bool:
    """Wrap-aware non-strict comparison (``a`` does not follow ``b``)."""
    return not _seq_lt(b, a)


def _range_covered(ranges: list[tuple[int, int]], start: int,
                   end: int) -> bool:
    """True when [start, end) lies entirely inside one already-seen range."""
    return any(_seq_le(lo, start) and _seq_le(end, hi)
               for lo, hi in ranges)


def _range_insert(ranges: list[list[int]], start: int, end: int) -> int:
    """Insert/merge the sequence range [start, end); return the new count.

    Touching ranges are merged, so an in-window segment that closes a gap
    folds the gap into the contiguous run instead of adding state.
    """
    lo, hi = start, end
    kept: list[list[int]] = []
    for existing in ranges:
        s, e = existing[0], existing[1]
        if _seq_le(s, hi) and _seq_le(lo, e):  # overlaps or touches
            if _seq_lt(s, lo):
                lo = s
            if _seq_lt(hi, e):
                hi = e
        else:
            kept.append([s, e])
    kept.append([lo, hi])
    kept.sort(key=lambda r: r[0])
    ranges[:] = kept
    return len(ranges)


class _BinAccumulator:
    """Packet-tier statistics for one (src_ip, bin_start) key.

    All per-quantity state is O(1) online accumulators plus one bounded
    per-direction sequence-range set (at most :data:`SEQ_RANGES_CAP` merged
    ranges per directional 5-tuple) needed by the sequence-regression
    detector. No per-packet lists are retained.
    """

    __slots__ = (
        "src_ip", "bin_start", "packet_count", "byte_count",
        "first_pkt_ts", "last_pkt_ts", "_ttl", "_window",
        "_payload", "frag_count", "seq_regression_count",
        "seq_ranges_capped", "_ranges",
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
        # the column PACKET_LEVEL_COLUMNS calls "retransmission_count" - see
        # FEATURE_SEMANTICS: this counts duplicate byte ranges, not
        # retransmissions, and is never presented as one.
        self.seq_regression_count = 0
        self.seq_ranges_capped = 0
        # directional 5-tuple -> merged observed (start, end) sequence ranges
        self._ranges: dict[FiveTuple, list[list[int]]] = {}

    def _observe_tcp(self, tcp: Any, key: FiveTuple) -> None:
        """Sequence-regression check: was this segment's range seen before?

        A regression is a **duplicate byte range** on this directional
        5-tuple: the whole [seq, seq+len) interval was already observed in
        this bin. Deliberately NOT counted:

        * RST segments (resets carry no data of interest);
        * pure ACKs (zero sequence space);
        * in-window reordering - a segment that arrives *before* earlier data
          and the earlier data arriving afterwards are two distinct ranges, so
          seq 1000, 1020, then 1010 is zero regressions (the old
          "seq < highest end seen" rule called that one retransmission).
        """
        if int(tcp.flags) & 0x04:  # RST: a reset, never a regression
            return
        length = (
            len(bytes(tcp.payload))
            + bool(int(tcp.flags) & 0x02)  # SYN consumes a sequence number
            + bool(int(tcp.flags) & 0x01)  # FIN consumes a sequence number
        )
        if not length:
            return  # pure ACK: occupies no sequence space
        seq = int(tcp.seq)
        end = (seq + length) & 0xFFFFFFFF
        ranges = self._ranges.setdefault(key, [])
        if _range_covered(ranges, seq, end):
            self.seq_regression_count += 1
            return
        if _range_insert(ranges, seq, end) > SEQ_RANGES_CAP:
            # bounded memory beats perfect recall: drop the oldest ranges and
            # count it (disclosed in stats["seq_range_sets_capped"])
            del ranges[: len(ranges) - SEQ_RANGES_CAP]
            self.seq_ranges_capped += 1


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
                # column name kept for PACKET_LEVEL_COLUMNS compatibility;
                # this is a sequence-regression count - see FEATURE_SEMANTICS
                "retransmission_count": float(self.seq_regression_count),
            },
        }


def extract_pcap_bins(
    pcap_path: str | Path,
    bin_seconds: int = BIN_SECONDS,
    apply_offset_sec: float = 0.0,
    on_backwards: str = "fail",
    reorder_grace_sec: float = DEFAULT_REORDER_GRACE_SEC,
) -> dict[str, Any]:
    """Stream a PCAP and return packet-tier records on the D1 bin grid.

    Returns a dict with ``records`` (list of per-``(src_ip, bin_start)``
    packet-tier rows, sorted by bin then src - **exactly one row per key**,
    guaranteed because records are produced from a dict keyed by that tuple),
    ``stats`` (capture-level provenance, including the reported
    ``feature_semantics`` for ``retransmission_count``) and ``approximations``.
    ``apply_offset_sec`` shifts packet timestamps before binning - only ever
    use the ``apply.correction_sec`` value returned by
    :func:`measure_clock_offset` (whose ``apply.max_error_sec`` then bounds the
    worst-case per-packet bin error), never a hand-picked number.

    Ordering policy (backwards timestamps are never silent, and reordering is
    memory-bounded):

    * The extractor keeps one bounded set of accumulators and closes them with
      a **watermark scan that runs only when the capture crosses a bin
      boundary** (``stats["watermark_scans"]``), not on every packet.
    * ``on_backwards="fail"`` (default): a packet whose target bin is already
      closed raises :class:`BackwardsTimestampError`.
    * ``on_backwards="merge"``: such packets are folded into the bin's
      accumulator and counted in ``stats["backwards_packets_merged"]``.
    * Retention is bounded: an accumulator is held for at most
      ``bin_seconds + reorder_grace_sec`` (``stats["retention_horizon_sec"]``)
      of capture time past its own window and then finalised and freed, so
      retained memory depends on the grace, never on capture length
      (``stats["peak_retained_bins"]`` reports the high-water mark). A packet
      arriving after that is counted in
      ``stats["backwards_packets_late_dropped"]`` - never silently absorbed by
      an unbounded backlog.
    """
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.utils import PcapReader

    if on_backwards not in ("fail", "merge"):
        raise ValueError(
            f"on_backwards must be 'fail' or 'merge', got {on_backwards!r}")
    if not 0.0 <= reorder_grace_sec <= MAX_REORDER_GRACE_SEC:
        raise ValueError(
            f"reorder_grace_sec must be within [0, {MAX_REORDER_GRACE_SEC}] "
            f"(retained memory is bin_seconds + grace; larger values are the "
            f"backlog this extractor exists to avoid)")

    path = Path(pcap_path)
    if not path.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")

    # One dict holds every accumulator still in memory: bins still open, plus
    # (in "merge" mode) bins whose window closed but that are inside the
    # retention horizon and may still receive a late packet. A bin leaves this
    # dict - and its memory - once the watermark passes its window plus
    # retention_sec, so retained memory is bounded by the grace, never by the
    # length of the capture.
    bins: dict[FlowKey, _BinAccumulator] = {}
    flushed: list[dict[str, Any]] = []
    retention_sec = bin_seconds + reorder_grace_sec
    packets = 0
    non_monotonic = 0
    watermark_scans = 0
    backwards_merged = 0
    late_dropped = 0
    ranges_capped_total = 0
    peak_retained = 0
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    watermark: Optional[float] = None
    last_boundary: Optional[int] = None  # bin_start of latest watermark

    def _release(acc: _BinAccumulator) -> None:
        """Finalise an accumulator: emit its record, free its state."""
        nonlocal ranges_capped_total
        ranges_capped_total += acc.seq_ranges_capped
        flushed.append(acc.record())

    def _close_past(t: float) -> None:
        """Close bins whose window (plus grace) ended; free past retention.

        Runs only when the capture crosses a bin boundary - not per packet -
        which is the point of the watermark.
        """
        nonlocal watermark_scans, peak_retained
        watermark_scans += 1
        cutoff_close = t - reorder_grace_sec
        cutoff_free = t - retention_sec
        for k in [k for k in bins if k[1] + bin_seconds <= cutoff_close]:
            if on_backwards == "merge" and k[1] + bin_seconds > cutoff_free:
                continue  # retention horizon: keep accepting late merges
            _release(bins.pop(k))
        peak_retained = max(peak_retained, len(bins))

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
            acc = bins.get(key)
            if not _is_closed(key, watermark, bin_seconds, reorder_grace_sec):
                if acc is None:
                    acc = bins[key] = _BinAccumulator(key[0], key[1])
            elif on_backwards == "fail":
                # The watermark already moved past this bin's window: a
                # packet arriving after its bin closed. Never silent - fail.
                raise BackwardsTimestampError(
                    f"packet at ts={ts} targets closed bin {key} "
                    f"(watermark={watermark}); capture moved backwards beyond "
                    f"reorder_grace_sec={reorder_grace_sec}. Re-run with "
                    f"on_backwards='merge' to accept bounded reordering "
                    f"(retention horizon {retention_sec}s) instead of "
                    f"failing.")
            elif acc is None and (key[1] + bin_seconds
                                  <= watermark - retention_sec):
                # Its accumulator was finalised and freed with the retention
                # horizon: an hours-late packet can only be counted as a loss,
                # never absorbed by an unbounded backlog.
                late_dropped += 1
                continue
            else:
                if acc is None:
                    # Old bin seen for the first time, still inside the
                    # retention horizon: create it; the next scan finalises it.
                    acc = bins[key] = _BinAccumulator(key[0], key[1])
                else:
                    backwards_merged += 1  # late packet into a retained bin
            acc.add(pkt, ts, ip, tcp, udp)

            # Watermark advances with capture time; the close scan runs only
            # when the capture crosses into a different bin - not per packet.
            if watermark is None or ts > watermark:
                watermark = ts
                boundary = bin_start_for(ts, bin_seconds)
                if last_boundary is None or boundary != last_boundary:
                    _close_past(watermark)
                    last_boundary = boundary

    # End of capture: every accumulator still held (open bins, and retained
    # bins in merge mode) is finalised - nothing is dropped at the tail.
    for k in list(bins):
        _release(bins.pop(k))

    records = flushed
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
            "backwards_packets_late_dropped": late_dropped,
            "reorder_grace_sec": reorder_grace_sec,
            "retention_horizon_sec": retention_sec,
            "peak_retained_bins": peak_retained,
            "watermark_scans": watermark_scans,
            "seq_range_sets_capped": ranges_capped_total,
            "bin_seconds": bin_seconds,
            "bin_key": "(src_ip, floor(unix_seconds / bin_seconds) * "
                       "bin_seconds)",
            "bin_key_semantics": BIN_KEY_SEMANTICS,
            "apply_offset_sec": apply_offset_sec,
            "feature_semantics": dict(FEATURE_SEMANTICS),
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


# ---------------------------------------------------------------------------
# timestamp precision: determined per input from its own values
# ---------------------------------------------------------------------------
QUANTIZATION_RULE = (
    "the finest level consistent with the values: any sub-second component -> "
    "0.001 s; else any value off the 60 s grid -> 1.0 s; else every value on "
    "the 60 s grid -> 60.0 s. Counted over every value, never inferred from "
    "min/max, so minute-precision input can never be called second precision")
QUANTIZATION_MODEL = (
    "floor/truncation: a source with step q reports floor(true/q)*q, so its "
    "reading sits up to q below the true time")


class _PrecisionCounter:
    """O(1) evidence for one timestamp source's quantization step.

    Counts how many values carry a sub-second component, how many are
    second-aligned but off the 60 s grid, and how many sit exactly on the grid.
    Values are not retained, so a multi-million-row source costs the same as a
    handful.
    """

    __slots__ = ("n_valid", "n_ignored", "subsecond", "off_minute", "on_grid")

    def __init__(self) -> None:
        self.n_valid = 0
        self.n_ignored = 0
        self.subsecond = 0
        self.off_minute = 0
        self.on_grid = 0

    def add(self, raw: Any) -> None:
        if raw is None:
            self.n_ignored += 1
            return
        try:
            value = float(raw)
        except (TypeError, ValueError):
            self.n_ignored += 1
            return
        if not math.isfinite(value):
            self.n_ignored += 1
            return
        self.n_valid += 1
        whole = math.floor(value)
        if abs(value - whole) > 1e-9:
            self.subsecond += 1
        elif whole % 60:
            self.off_minute += 1
        else:
            self.on_grid += 1

    def result(self, *, kind: str,
               declared_sec: Optional[float] = None,
               min_values: int = MIN_VALUES_TO_INFER_QUANTIZATION) -> dict:
        """Evidence plus the step to use (``None`` when precision is unknown).

        ``declared_sec`` is a caller's claim about the source's format. It is
        never trusted over the data: the **coarser** of claim and measurement
        decides, because a too-small step would wrongly narrow the feasible
        offset interval, and that is the dangerous direction of error.
        """
        if (declared_sec is not None
                and declared_sec not in SUPPORTED_QUANTIZATION_SEC):
            raise ValueError(
                f"declared timestamp step {declared_sec!r} is not one of "
                f"{SUPPORTED_QUANTIZATION_SEC}")
        inferred = None
        if self.n_valid >= min_values:
            if self.subsecond:
                inferred = 0.001
            elif self.off_minute:
                inferred = 1.0
            else:
                inferred = 60.0

        flags: list[str] = []
        candidates = [v for v in (inferred, declared_sec) if v is not None]
        step = max(candidates) if candidates else None
        if (inferred is not None and declared_sec is not None
                and inferred != declared_sec):
            flags.append("declared_precision_differs_from_values")
        reason = None
        if step is None:
            reason = (f"only {self.n_valid} usable timestamp values "
                      f"(<{min_values}) and no declared step: precision is "
                      f"unknown, so no offset may be applied")
        elif declared_sec is not None and inferred is None:
            flags.append("declared_precision_unverified")
        return {
            "kind": kind,
            "known": step is not None,
            "quantization_sec": step,
            "inferred_sec": inferred,
            "declared_sec": declared_sec,
            "model": QUANTIZATION_MODEL,
            "reason": reason,
            "flags": flags,
            "evidence": {
                "n_values_checked": self.n_valid,
                "n_values_ignored": self.n_ignored,
                "n_values_required": min_values,
                "subsecond_count": self.subsecond,
                "off_minute_count": self.off_minute,
                "on_minute_grid_count": self.on_grid,
                "rule": QUANTIZATION_RULE,
            },
        }


def infer_timestamp_quantization(
    values: Iterable[Any],
    *,
    kind: str,
    declared_sec: float | None = None,
    min_values: int = MIN_VALUES_TO_INFER_QUANTIZATION,
) -> dict[str, Any]:
    """Determine a timestamp source's quantization step from its values.

    Returns a dict with ``known``, ``quantization_sec``, ``inferred_sec``,
    ``declared_sec``, ``model``, ``reason``, ``flags`` and ``evidence``.
    Precision is *known* only when at least ``min_values`` usable values were
    seen, or the caller declared a step that the values do not contradict;
    otherwise ``quantization_sec`` is ``None`` and any offset derived from that
    source must not be applied.
    """
    counter = _PrecisionCounter()
    for value in values:
        counter.add(value)
    return counter.result(kind=kind, declared_sec=declared_sec,
                          min_values=min_values)


def feasible_offset_interval(
    deltas: list[float],
    *,
    flow_quantization_sec: float | None,
    packet_quantization_sec: float | None,
) -> dict[str, Any]:
    """Intersect the per-pair constraints on the true flow-minus-packet offset.

    Each matched pair constrains the offset to ``[delta - q_packet, delta +
    q_flow)`` (see :func:`measure_clock_offset`), so the feasible set is
    ``lo = max(deltas) - q_packet``, ``hi = min(deltas) + q_flow``. An empty
    intersection means no single constant offset can explain the pairs.
    """
    unknown = []
    if flow_quantization_sec is None:
        unknown.append("flow table")
    if packet_quantization_sec is None:
        unknown.append("PCAP")
    base = {
        "formula": "lo = max(delta_i) - q_packet ; hi = min(delta_i) + q_flow "
                   "; feasible set = [lo, hi)",
        "assumption": QUANTIZATION_MODEL,
        "q_flow_sec": flow_quantization_sec,
        "q_packet_sec": packet_quantization_sec,
        "unknown_inputs": unknown,
    }
    if unknown:
        return {**base, "computable": False, "lo_sec": None, "hi_sec": None,
                "width_sec": None, "midpoint_sec": None, "empty": None,
                "reason": "quantization step unknown: " + " + ".join(unknown)}
    if not deltas:
        return {**base, "computable": False, "lo_sec": None, "hi_sec": None,
                "width_sec": None, "midpoint_sec": None, "empty": None,
                "reason": "no matched pairs"}
    lo = max(deltas) - packet_quantization_sec
    hi = min(deltas) + flow_quantization_sec
    width = hi - lo
    return {**base, "computable": True, "lo_sec": lo, "hi_sec": hi,
            "width_sec": width, "midpoint_sec": (lo + hi) / 2.0,
            "empty": width <= 0.0, "reason": None}


def _refused_apply(reason: str, *flags: str) -> dict[str, Any]:
    """A refused apply block: a refused offset is never an implicit zero."""
    return {
        "eligible": False,
        "reason": reason,
        "correction_sec": None,
        "max_error_sec": None,
        "selection_rule": "midpoint of the feasible interval; requires a "
                          "determined timestamp precision on both inputs",
        "max_interval_width_sec": OFFSET_MAX_INTERVAL_WIDTH_SEC,
        "min_pairs": OFFSET_MIN_PAIRS_TO_APPLY,
        "flags": list(flags),
    }


def measure_clock_offset(
    pcap_path: str | Path,
    flow_rows: Iterable[dict[str, Any]],
    *,
    flow_quantization_sec: float | None = None,
) -> dict[str, Any]:
    """Measure the flow-table vs PCAP clock offset as a *feasible interval*.

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
    of the tuple. Only valid, correctly paired instances produce deltas, and
    unpaired instances are counted, never guessed at.

    ``flow_quantization_sec`` optionally *declares* the flow table's timestamp
    step (one of :data:`SUPPORTED_QUANTIZATION_SEC`). The declaration is
    checked against the supplied values and the **coarser** of the two is
    used, so declaring can never make the estimate more confident than the
    data supports. If the step cannot be determined at all (too few values,
    nothing declared), the offset is still measured but application is
    refused - precision is never assumed.

    Why an interval and not a median
    --------------------------------
    Both clocks truncate (floor) their time to their own step, so for the same
    event with true time ``e`` and true flow-minus-packet offset ``theta``::

        packet_ts = e - eps_p,          0 <= eps_p < q_packet
        flow_ts   = e + theta - u,      0 <= u      < q_flow
        delta     = flow_ts - packet_ts = theta - u + eps_p

    One matched pair therefore proves only that ``delta`` lies in
    ``(theta - q_flow, theta + q_packet]``, i.e. that ``theta`` lies in
    ``[delta - q_packet, delta + q_flow)``. Averaging or taking the median of
    such deltas is biased low by the mean sub-step ``frac`` (which is why the
    old "median = measured offset" reading produced +7.0 s for a +7.34 s
    offset). This function instead intersects the per-pair constraints::

        lo = max(delta_i) - q_packet,   hi = min(delta_i) + q_flow

    and reports ``[lo, hi)``: empty means the constant-offset model is
    contradicted, wide means the offset is not pinned. Only a non-empty
    interval of at most :data:`OFFSET_MAX_INTERVAL_WIDTH_SEC` is applied, as
    its midpoint, with ``apply.max_error_sec`` = half the width.

    Output separates the measurement from the decision:
    ``observed_delta`` (statistics, both inputs' precision evidence, the
    feasible interval, and at most :data:`MAX_DELTA_SAMPLES` raw delta samples
    - never the full list) and ``apply`` (``eligible``, ``reason``,
    ``correction_sec``, ``max_error_sec``, ``flags``). ``status:
    "not_measured"`` is returned when nothing matches; an unmeasured offset is
    reported as such, never treated as zero.
    """
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.utils import PcapReader

    method = ("feasible interval of (flow_ts - first_packet_ts) over matched "
              "protocol-aware directional 5-tuples, paired per connection "
              "instance, under floor quantization of both clocks")

    flows = list(flow_rows)
    if not flows:
        return {
            "status": "not_measured",
            "reason": "no flow rows supplied",
            "method": method,
            "n_matched_pairs": 0,
            "apply": _refused_apply("offset not measured (no flow rows)"),
        }

    # Protocol-aware directional key -> flow starts (all instances kept;
    # no earliest-flow collapse).
    needed: dict[OffsetKey, list[float]] = {}
    for row in flows:
        proto = str(row.get("proto", "tcp")).lower()
        key = (proto, str(row["src"]), str(row["dst"]),
               int(row["sport"]), int(row["dport"]))
        needed.setdefault(key, []).append(float(row["ts"]))

    # Precision of the flow clock is measured from these values (never from
    # min/max, never assumed), optionally informed by a caller declaration.
    flow_prec = infer_timestamp_quantization(
        (row.get("ts") for row in flows),
        kind="flow_table", declared_sec=flow_quantization_sec)

    # PCAP side: first packet of every connection instance per key.
    conn_starts: dict[OffsetKey, list[float]] = {}
    last_seen: dict[OffsetKey, float] = {}
    pkt_prec = _PrecisionCounter()  # precision of the capture clock itself
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
            pkt_prec.add(ts)
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
            "n_matched_pairs": 0,
            "n_candidate_keys": len(needed),
            "n_unpaired_instances": n_unpaired,
            "flow_timestamp_precision": flow_prec,
            "apply": _refused_apply("offset not measured (no matched pairs)"),
        }

    pkt_prec = pkt_prec.result(kind="pcap_packet")
    interval = feasible_offset_interval(
        deltas, flow_quantization_sec=flow_prec["quantization_sec"],
        packet_quantization_sec=pkt_prec["quantization_sec"])
    mid = interval["midpoint_sec"]

    median = statistics.median(deltas)
    iqr = _percentile(deltas, 0.75) - _percentile(deltas, 0.25)

    flags: list[str] = list(flow_prec["flags"]) + list(pkt_prec["flags"])
    if not flow_prec["known"]:
        flags.append("unknown_flow_timestamp_precision")
    if not pkt_prec["known"]:
        flags.append("unknown_packet_timestamp_precision")
    if interval["computable"] and abs(mid) >= TIMEZONE_SCALE_SEC:
        flags.append("timezone_scale")

    observed = {
        "sign_convention": "delta = flow_ts - packet_ts (the flow clock runs "
                           "this much ahead), so correction_sec is ADDED to "
                           "packet timestamps to reach the flow clock",
        "n_matched_pairs": len(deltas),
        "n_unpaired_instances": n_unpaired,
        "min_sec": deltas[0],
        "max_sec": deltas[-1],
        "spread_sec": deltas[-1] - deltas[0],
        "p25_sec": _percentile(deltas, 0.25),
        "p75_sec": _percentile(deltas, 0.75),
        "dispersion_iqr_sec": iqr,
        "median_sec_biased_under_quantization": median,
        "bias_note": "under floor quantization delta = theta - u + eps with "
                     "0 <= u < q_flow, so the sample median is biased low by "
                     "up to one quantization step: it is reported as a "
                     "dispersion statistic and is NEVER applied",
        "deltas_sample_sec": deltas[:MAX_DELTA_SAMPLES],
        "deltas_sample_truncated": len(deltas) > MAX_DELTA_SAMPLES,
        "flow_timestamp_precision": flow_prec,
        "packet_timestamp_precision": pkt_prec,
        "feasible_interval": interval,
    }

    # --- apply decision: the feasible interval decides, never the median ---
    if len(deltas) < OFFSET_MIN_PAIRS_TO_APPLY:
        reason = (f"too few valid paired connection instances "
                  f"({len(deltas)} < {OFFSET_MIN_PAIRS_TO_APPLY}); refusing "
                  f"to apply - the observed delta is reported only")
    elif not interval["computable"]:
        reason = ("timestamp precision of "
                  + " + ".join(interval["unknown_inputs"])
                  + " is not determined, so the feasible interval cannot be "
                    "bounded and no offset can be applied")
    elif interval["empty"]:
        reason = (f"empty feasible interval (lo="
                  f"{interval['lo_sec']:.6f}s > hi="
                  f"{interval['hi_sec']:.6f}s): the matched pairs contradict "
                  f"a single constant offset (drifting clocks, or pairs that "
                  f"do not describe the same connection)")
    elif interval["width_sec"] > OFFSET_MAX_INTERVAL_WIDTH_SEC:
        reason = (f"feasible interval is {interval['width_sec']:.6f}s wide, "
                  f"over the {OFFSET_MAX_INTERVAL_WIDTH_SEC}s bound: "
                  f"quantization does not pin the offset (q_flow="
                  f"{interval['q_flow_sec']:g}s, q_packet="
                  f"{interval['q_packet_sec']:g}s, delta spread="
                  f"{deltas[-1] - deltas[0]:.6f}s); refusing to apply")
    elif abs(mid) >= TIMEZONE_SCALE_SEC:
        reason = (f"offset magnitude {abs(mid):.1f}s is timezone scale: "
                  f"reported, never silently corrected")
    else:
        reason = None

    apply_block = {
        "eligible": reason is None,
        "reason": reason,
        "correction_sec": mid if reason is None else None,
        "max_error_sec": (interval["width_sec"] / 2.0 if reason is None
                          else None),
        "selection_rule": "midpoint of the feasible interval "
                          "[max(delta) - q_packet, min(delta) + q_flow); its "
                          "worst-case error is half the interval width",
        "max_interval_width_sec": OFFSET_MAX_INTERVAL_WIDTH_SEC,
        "min_pairs": OFFSET_MIN_PAIRS_TO_APPLY,
        "flags": flags,
    }

    return {
        "status": "measured",
        "method": method,
        "n_matched_pairs": len(deltas),
        "n_candidate_keys": len(needed),
        "n_unpaired_instances": n_unpaired,
        "observed_delta": observed,
        "apply": apply_block,
        "note": "observed_delta is the measurement - its median is biased by "
                "floor quantization and is never applied; "
                "apply.correction_sec is the only value --apply-offset may "
                "use, and only when apply.eligible is true, with "
                "apply.max_error_sec as its worst-case per-packet time error. "
                "A whole-hour magnitude usually means a timezone-labelled "
                "flow clock: reported via the timezone_scale flag, never "
                "silently corrected",
    }


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

