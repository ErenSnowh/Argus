#!/usr/bin/env python3
"""
pcap_stream_bins.py
-------------------
CLI for the streaming fixed-60 s-bin packet-tier extractor (plan D2
amendment; D1 keying). Runs BEFORE any large PCAP download as the proposed and
tested extraction path, and again on real captures to measure clock offset and
flow/PCAP bin coverage. It is a STANDALONE CLI: no training, binning or
benchmark code consumes its output yet, because no real CIC PCAP has been
measured and no combined flow+packet feature scope has been approved.

    python scripts/pcap_stream_bins.py --pcap data/sample_portscan.pcap
    python scripts/pcap_stream_bins.py --pcap thu.pcap \
        --flows data/raw/cicids2017/ --out thu_packet_bins.json
    python scripts/pcap_stream_bins.py --pcap thu.pcap \
        --flows thu-thu.parquet thu-fri.parquet --apply-offset

Guarantees:
  * streams with PcapReader (never rdpcap), fixed floor(ts/60)*60 bins keyed
    (src_ip, bin_start) exactly as D1 keys flow bins, exactly one record per
    key; a packet that arrives after its bin closed FAILS CLOSED by default
    (exit 1, no output file, the error names ts/bin key/watermark and both
    remedies). Only --on-late drop accepts the loss, counting it in
    pcap.backwards_packets_late_dropped and naming the affected bins in
    pcap.bins_with_dropped_packets. Reordering inside --reorder-grace-sec
    (default 1 s, max 300 s) needs no opt-in - the bin is still open - and
    retained memory stays bounded at bin_seconds + grace of capture time, never
    tracking capture length;
  * retransmission_count is a SEQUENCE-REGRESSION detector (duplicate byte
    ranges per directional 5-tuple): RST and pure ACKs are never counted and
    in-window reordering is never counted. The column name comes from
    PACKET_LEVEL_COLUMNS; output carries feature_semantics saying what it
    measures, because it is not a retransmission count;
  * --flows accepts multiple parquet/csv files AND directories (expanded to
    *.parquet + *.csv, sorted); missing labels/NaN cells are real nulls -
    never the string "nan"; all-null timestamp padding is reported as
    timestamp_null, distinct from genuinely unparsable timestamps;
  * timestamp precision (0.001 / 1 / 60 s) is DETERMINED per flow file from
    its own values (never from min/max, never assumed), counted over every
    value; the run uses the coarsest file's step - the conservative bound -
    and reports mixed or undeterminable precision. A capture's own precision is
    measured the same way from its packet timestamps;
  * when --flows is given, output rows are the FLOW bins: uncovered bins get
    packet_features: null + packet_features_covered: false (never zero-filled)
    and coverage percentages (overall + attack bins) are written to the JSON
    over unique host-minute bins; an all-benign table yields attack coverage
    null (printed "n/a"), never 0.0;
  * clock offset is a FEASIBLE INTERVAL, not a median: under floor
    quantization a matched pair proves only delta <= theta + q_flow, so the
    observed statistics (whose median is biased low - reported, never applied)
    are kept separate from the apply decision. The interval is built from the
    MAJORITY WINDOW - the largest set of deltas one offset could explain - and
    the pairs outside it are counted in observed_delta.n_inconsistent; over
    OFFSET_MAX_INCONSISTENT_FRACTION of them is a refusal of its own.
    --apply-offset FAILS with exit 1 when the offset was not measured, matched
    too few connection instances, has undetermined precision on either input,
    is inconsistent beyond that tolerance, or leaves an interval that is empty
    or wider than OFFSET_MAX_INTERVAL_WIDTH_SEC; it never silently applies 0,
    and when it does apply it reports the worst-case error.

Exit status: 0 on success, 1 on error (including a refused --apply-offset).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.world_model.pcap_bins import (  # noqa: E402
    APPROXIMATIONS,
    BIN_KEY_SEMANTICS,
    BIN_SECONDS,
    BackwardsTimestampError,
    DEFAULT_REORDER_GRACE_SEC,
    MAX_REORDER_GRACE_SEC,
    bin_start_for,
    extract_pcap_bins,
    infer_timestamp_quantization,
    join_packet_features,
    measure_bin_coverage,
    measure_clock_offset,
)


def _find_col(columns, *candidates: str):
    """Case/whitespace-insensitive column lookup (CIC headers are padded)."""
    norm = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in norm:
            return norm[cand.lower()]
    return None


def _cell_str(value) -> str | None:
    """Stringify a cell, mapping NaN/NA/None/empty to None (never 'nan').

    The old ``str(r[col] or "")`` turned a float NaN into the literal string
    ``"nan"`` (NaN is truthy), which then polluted join keys and labels.
    """
    import pandas as pd

    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "nat"):
        return None
    return text


def _norm_proto(value) -> str:
    """Map a CIC Protocol cell (6/17 or tcp/udp) onto 'tcp'/'udp'."""
    text = (_cell_str(value) or "tcp").lower()
    if text in ("6", "tcp"):
        return "tcp"
    if text in ("17", "udp"):
        return "udp"
    return text


def load_flow_rows(path: Path) -> tuple[list[dict], dict]:
    """Read a flow table (parquet/csv) into offset/join inputs + stats.

    Null handling is explicit:
      * an all-null Timestamp column (the pinned CIC snapshot's 288,602
        padding rows) is reported as ``timestamp_null`` with
        ``timestamp_all_null: true`` - padding, not garbage values;
      * non-null timestamps that fail parsing are ``timestamp_unparsable``;
      * NaN labels/IPs/ports become ``None``/dropped rows - never the
        string ``"nan"``.
    """
    import pandas as pd

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    cols = {
        "src": _find_col(df.columns, "Source IP"),
        "dst": _find_col(df.columns, "Destination IP"),
        "sport": _find_col(df.columns, "Source Port"),
        "dport": _find_col(df.columns, "Destination Port"),
        "ts": _find_col(df.columns, "Timestamp"),
        "label": _find_col(df.columns, "Label"),
        "proto": _find_col(df.columns, "Protocol"),
    }
    missing = [k for k in ("src", "dst", "sport", "dport", "ts")
               if cols[k] is None]
    if missing:
        raise SystemExit(
            f"flow file {path.name} lacks required columns: {missing}")

    raw_ts = df[cols["ts"]]
    null_ts = int(raw_ts.isna().sum())
    ts = pd.to_datetime(raw_ts, dayfirst=True, errors="coerce")
    unparsable = int(ts.isna().sum()) - null_ts  # nulls are not "unparsable"
    ok = ts.notna()
    ok_ts = ts[ok]
    # pandas 3 parses to datetime64[us], pandas 2 to datetime64[ns]:
    # convert explicitly to ns (numpy handles the unit cast) so the epoch
    # scale is correct on both.
    epochs = (ok_ts.to_numpy(dtype="datetime64[ns]").astype("int64") / 1e9)
    epoch_by_idx = dict(zip(ok_ts.index, epochs))

    rows: list[dict] = []
    dropped_identity = 0
    for idx, r in df[ok].iterrows():
        src = _cell_str(r[cols["src"]])
        dst = _cell_str(r[cols["dst"]])
        sport = _cell_str(r[cols["sport"]])
        dport = _cell_str(r[cols["dport"]])
        if src is None or dst is None or sport is None or dport is None:
            dropped_identity += 1  # no join key without a real IP/port
            continue
        rows.append({
            "src": src,
            "dst": dst,
            "sport": int(float(sport)),
            "dport": int(float(dport)),
            "ts": float(epoch_by_idx[idx]),
            "label": (_cell_str(r[cols["label"]])
                      if cols["label"] is not None else None),
            "proto": (_norm_proto(r[cols["proto"]])
                      if cols["proto"] is not None else "tcp"),
        })

    # Timestamp precision is DETERMINED from this file's own values (counted
    # over every parsed value, never inferred from min/max, never assumed):
    # sub-second component -> 0.001 s, off the 60 s grid -> 1 s, otherwise the
    # values sit on the minute grid -> 60 s. Undeterminable (too few values)
    # stays None and blocks --apply-offset instead of defaulting to 1 s.
    precision = infer_timestamp_quantization(
        epochs.tolist(), kind=f"flow_table:{path.name}")

    stats = {
        "flow_file": path.name,
        "flow_rows": len(rows),
        "rows_dropped_missing_identity": dropped_identity,
        "timestamp_null": null_ts,
        "timestamp_unparsable": unparsable,
        "timestamp_all_null": bool(len(df)) and null_ts == len(df),
        "labels_null": (int(df[cols["label"]].isna().sum())
                        if cols["label"] is not None else None),
        "timestamp_precision": precision,
        "timestamp_precision_sec": precision["quantization_sec"],
        "timestamp_precision_known": precision["known"],
    }
    return rows, stats


def resolve_flow_paths(specs: list[Path]) -> list[Path]:
    """Expand --flows specs: directories -> sorted *.parquet + *.csv.

    A directory's files are taken in a deterministic order (parquet first,
    then csv, each sorted by name) so runs are reproducible.
    """
    files: list[Path] = []
    for spec in specs:
        if spec.is_dir():
            found = (sorted(spec.glob("*.parquet"))
                     + sorted(spec.glob("*.csv")))
            if not found:
                raise SystemExit(
                    f"--flows directory {spec} contains no parquet/csv files")
            files.extend(found)
        elif spec.is_file():
            files.append(spec)
        else:
            raise SystemExit(f"--flows path not found: {spec}")
    if not files:
        raise SystemExit("--flows matched no flow files")
    return files


def load_flow_tables(specs: list[Path]) -> tuple[list[dict], dict]:
    """Load one or many flow tables (files and/or directories).

    Rows are concatenated; per-file stats are kept in ``files`` and totals
    in the top-level keys, so multi-file joins report every input.
    """
    files = resolve_flow_paths(specs)
    rows: list[dict] = []
    per_file: list[dict] = []
    for f in files:
        file_rows, file_stats = load_flow_rows(f)
        rows.extend(file_rows)
        per_file.append(file_stats)
    known = [s for s in per_file if s["timestamp_precision_known"]]
    unknown_files = [s["flow_file"] for s in per_file
                     if not s["timestamp_precision_known"]]
    steps = sorted({s["timestamp_precision_sec"] for s in known})
    # Conservative aggregate: the COARSEST file's step, because assuming a
    # finer one would wrongly narrow the feasible offset interval. If any file
    # at all has undetermined precision the run cannot claim a step - an
    # unknown precision blocks application rather than being guessed at.
    precision_sec = max(steps) if (steps and not unknown_files) else None
    stats = {
        "flow_files": [s["flow_file"] for s in per_file],
        "flow_files_count": len(per_file),
        "flow_rows": len(rows),
        "rows_dropped_missing_identity": sum(
            s["rows_dropped_missing_identity"] for s in per_file),
        "timestamp_null": sum(s["timestamp_null"] for s in per_file),
        "timestamp_unparsable": sum(
            s["timestamp_unparsable"] for s in per_file),
        "timestamp_all_null_files": [
            s["flow_file"] for s in per_file if s["timestamp_all_null"]],
        "labels_null": sum(s["labels_null"] or 0 for s in per_file),
        "timestamp_precision_sec": precision_sec,
        "timestamp_precision_steps": steps,
        "timestamp_precision_mixed": len(steps) > 1,
        "timestamp_precision_unknown_files": unknown_files,
        "files": per_file,
    }
    return rows, stats


def _fmt_pct(value) -> str:
    """Format a percentage; a null (zero-denominator) rate prints 'n/a'."""
    return "n/a" if value is None else f"{value:.1f}%"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stream a PCAP into fixed-60 s packet-tier bins keyed "
                    "exactly as D1 keys flow bins.")
    ap.add_argument("--pcap", required=True, type=Path,
                    help="input PCAP (read packet-by-packet via PcapReader)")
    ap.add_argument("--flows", type=Path, nargs="+", default=None,
                    metavar="PATH",
                    help="optional flow table(s) for clock-offset "
                         "measurement, bin coverage and the D1 left-join: "
                         "parquet/csv files and/or directories (a directory "
                         "expands to sorted *.parquet + *.csv)")
    ap.add_argument("--bin-seconds", type=int, default=BIN_SECONDS)
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default: <pcap>_bins.json)")
    ap.add_argument("--apply-offset", action="store_true",
                    help="shift PCAP timestamps by the midpoint of the "
                         "MEASURED feasible clock-offset interval before "
                         "binning (requires --flows; refused with exit 1 "
                         "unless the offset is eligible)")
    ap.add_argument("--on-late", choices=("fail", "drop"), default="fail",
                    help="packet whose bin already closed (it arrived later "
                         "than bin_seconds + --reorder-grace-sec): 'fail' "
                         "(default - exit 1, no output file, nothing is ever "
                         "silently dropped) or 'drop', which discards it and "
                         "counts the loss per bin in "
                         "pcap.backwards_packets_late_dropped and "
                         "pcap.bins_with_dropped_packets")
    ap.add_argument("--reorder-grace-sec", type=float,
                    default=DEFAULT_REORDER_GRACE_SEC,
                    help=f"keep a bin open this long past its boundary before "
                         f"closing it (bounded reordering; default "
                         f"{DEFAULT_REORDER_GRACE_SEC}s, max "
                         f"{MAX_REORDER_GRACE_SEC}s - retained memory is "
                         f"bin_seconds + grace)")
    args = ap.parse_args(argv)

    if args.apply_offset and not args.flows:
        ap.error("--apply-offset requires --flows (offset must be measured)")
    if args.bin_seconds <= 0:
        ap.error("--bin-seconds must be > 0")
    if not 0 <= args.reorder_grace_sec <= MAX_REORDER_GRACE_SEC:
        ap.error(f"--reorder-grace-sec must be within "
                 f"[0, {MAX_REORDER_GRACE_SEC}] (retained memory is "
                 f"bin_seconds + grace)")
    if not args.pcap.is_file():
        print(f"ERROR: PCAP not found: {args.pcap}", file=sys.stderr)
        return 1

    out_path = args.out or args.pcap.with_name(args.pcap.stem + "_bins.json")

    flow_rows: list[dict] = []
    flow_stats: dict = {}
    if args.flows is not None:
        flow_rows, flow_stats = load_flow_tables(args.flows)

    # The flow-table timestamp step is passed in as the per-file measurement
    # (coarsest file wins, unknown blocks application) rather than being
    # re-guessed here from a handful of matched rows.
    offset = measure_clock_offset(
        args.pcap, flow_rows,
        flow_quantization_sec=(flow_stats.get("timestamp_precision_sec")
                               if flow_stats else None))
    apply_block = offset.get("apply") or {}
    apply_sec = 0.0
    if args.apply_offset:
        # Apply only what was MEASURED and cleared the guard: too few
        # matches or ambiguous dispersion exits 1 instead of quietly
        # binning with a hand-picked - or silently zero - correction.
        if offset["status"] != "measured" or not apply_block.get("eligible"):
            reason = apply_block.get("reason") or offset.get("reason")
            print(f"ERROR: --apply-offset refused: {reason}", file=sys.stderr)
            print("       the observed delta stays reported as measured but "
                  "unapplied; an unmeasured offset is never treated as 0",
                  file=sys.stderr)
            return 1
        apply_sec = float(apply_block["correction_sec"])

    try:
        extraction = extract_pcap_bins(
            args.pcap, bin_seconds=args.bin_seconds,
            apply_offset_sec=apply_sec,
            on_late=args.on_late,
            reorder_grace_sec=args.reorder_grace_sec)
    except BackwardsTimestampError as exc:
        # Fail closed: the run stops here, before any output file is written,
        # so a late packet can never produce a partial bin set that looks
        # complete. Both remedies are in the message.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    records = extraction["records"]
    stats = extraction["stats"]

    if flow_rows:
        flow_keys = [(r["src"], bin_start_for(r["ts"], args.bin_seconds))
                     for r in flow_rows]
        attack_keys = [key for key, r in zip(flow_keys, flow_rows)
                       if r["label"] and r["label"].upper() != "BENIGN"]
        pcap_keys = [(r["src_ip"], r["bin_start"]) for r in records]
        coverage = measure_bin_coverage(flow_keys, pcap_keys, attack_keys)
        rows = join_packet_features(flow_keys, records)
    else:
        coverage = {"status": "not_measured",
                    "reason": "no flow table supplied"}
        rows = [{
            "src_ip": r["src_ip"],
            "bin_start": r["bin_start"],
            "packet_features": r["features"],
            "packet_features_covered": True,
        } for r in records]

    covered_rows = [r for r in rows if r["packet_features_covered"]]
    null_features = sorted({
        c for r in covered_rows
        for c, v in r["packet_features"].items() if v is None})

    payload = {
        "extractor": "argus/scripts/pcap_stream_bins.py",
        "extractor_module": "ml.world_model.pcap_bins",
        "bin_seconds": args.bin_seconds,
        "bin_key": "(src_ip, floor(unix_seconds / bin_seconds) * "
                   "bin_seconds) in UTC",
        "late_policy": {
            "on_late": args.on_late,
            "reorder_grace_sec": args.reorder_grace_sec,
            "retention_horizon_sec": (args.bin_seconds
                                      + args.reorder_grace_sec),
            "note": "a packet is late when it arrives after bin_seconds + "
                    "reorder_grace_sec of capture time past its bin start. "
                    "'fail' (default) exits 1 with no output file and names "
                    "both remedies; 'drop' discards the packet and counts it "
                    "in pcap.backwards_packets_late_dropped, naming the bins "
                    "it would have filled in pcap.bins_with_dropped_packets. "
                    "Reordering inside the grace is not late: the bin is "
                    "open. Retained memory is bin_seconds + grace and never "
                    "tracks capture length",
        },
        "feature_semantics": stats["feature_semantics"],
        "timestamp_precision": {
            "flow_table_sec": (flow_stats.get("timestamp_precision_sec")
                               if flow_stats else None),
            "flow_table_steps": (flow_stats.get("timestamp_precision_steps")
                                 if flow_stats else []),
            "flow_table_mixed_files": (
                bool(flow_stats.get("timestamp_precision_mixed"))
                if flow_stats else False),
            "flow_table_unknown_files": (
                flow_stats.get("timestamp_precision_unknown_files", [])
                if flow_stats else []),
            "pcap_sec": ((offset.get("observed_delta") or {})
                         .get("packet_timestamp_precision") or {}).get(
                             "quantization_sec"),
            "note": "each source's step (0.001/1/60 s) is determined from its "
                    "own values, never assumed; an undetermined step blocks "
                    "--apply-offset rather than defaulting to 1 s",
        },
        "pcap": stats,
        "flows": flow_stats or {"status": "not_measured",
                                "reason": "no flow table supplied"},
        "clock_offset": offset,
        "offset_applied_sec": apply_sec,
        "coverage": coverage,
        "rows": rows,
        "null_features_within_covered_bins": null_features,
        "approximations": APPROXIMATIONS,
        "zero_fill": "never - uncovered bins are null + "
                     "packet_features_covered=false",
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"pcap      : {args.pcap}")
    print(f"bins      : {len(records)} packet-tier bins "
          f"({stats['packets_considered']} IP packets, "
          f"peak retained {stats['peak_retained_bins']})")
    if stats["backwards_packets_late_dropped"]:
        bins_hit = stats["bins_with_dropped_packets"]
        shown = ", ".join(f"{b['src_ip']}@{b['bin_start']}"
                          for b in bins_hit[:4])
        if len(bins_hit) > 4:
            shown += f" (+{len(bins_hit) - 4} more)"
        print(f"late      : {stats['backwards_packets_late_dropped']} packets "
              f"arrived after bin_seconds + grace = "
              f"{stats['retention_horizon_sec']}s and were DROPPED "
              f"(on_late=drop) from {len(bins_hit)} bin(s): {shown}",
              file=sys.stderr)
    if flow_stats:
        files = flow_stats["flow_files"]
        shown = ", ".join(files if len(files) <= 4
                          else files[:4] + [f"(+{len(files) - 4} more)"])
        print(f"flows     : {flow_stats['flow_rows']} rows from "
              f"{flow_stats['flow_files_count']} file(s): {shown}")
        print(f"          : null ts {flow_stats['timestamp_null']} "
              f"(all-null files: "
              f"{flow_stats['timestamp_all_null_files'] or 'none'}), "
              f"unparsable ts {flow_stats['timestamp_unparsable']}, "
              f"null labels {flow_stats['labels_null']}")
        print(f"          : ts precision "
              f"{flow_stats['timestamp_precision_sec']} (per-file steps "
              f"{flow_stats['timestamp_precision_steps']}, mixed "
              f"{flow_stats['timestamp_precision_mixed']}, unknown "
              f"{flow_stats['timestamp_precision_unknown_files'] or 'none'})")
    obs = offset.get("observed_delta") or {}
    interval = obs.get("feasible_interval") or {}
    if offset["status"] == "measured":
        q_flow = obs["flow_timestamp_precision"]["quantization_sec"]
        q_pcap = obs["packet_timestamp_precision"]["quantization_sec"]
        biased = obs["median_sec_biased_under_quantization"]
        print(f"offset    : measured n={offset['n_matched_pairs']} "
              f"consistent={obs['n_consistent']} "
              f"inconsistent={obs['n_inconsistent']} "
              f"delta=[{obs['min_sec']:+.3f},{obs['max_sec']:+.3f}]s "
              f"median(biased)={biased:+.3f}s")
        print(f"          : q_flow={q_flow}s q_pcap={q_pcap}s "
              f"flags={apply_block.get('flags') or 'none'}")
        if interval.get("computable"):
            state = "EMPTY" if interval["empty"] else "feasible"
            print(f"interval  : [{interval['lo_sec']:+.6f}, "
                  f"{interval['hi_sec']:+.6f}) {state} "
                  f"width={interval['width_sec']:.6f}s "
                  f"(from {interval.get('n_pairs_used')} consistent pairs)")
        else:
            print(f"interval  : not computable ({interval.get('reason')})")
    else:
        print(f"offset    : not measured ({offset.get('reason')})")
    err = apply_block.get("max_error_sec")
    err_note = (f"worst-case error {err:.6f}s" if isinstance(
        err, (int, float)) else "worst-case error unknown")
    if args.apply_offset:
        apply_note = f"applied {apply_sec:+.6f}s ({err_note})"
    elif apply_block.get("eligible"):
        apply_note = f"eligible, not applied (no --apply-offset); {err_note}"
    else:
        apply_note = ("refused: "
                      + str(apply_block.get("reason") or offset.get("reason")))
    print(f"apply     : {apply_note}")
    if "coverage_pct" in coverage:
        print(f"coverage  : {coverage['bins_covered']}/"
              f"{coverage['bins_flow_total']} flow bins "
              f"({_fmt_pct(coverage['coverage_pct'])}), attack "
              f"{coverage['attack_bins_covered']}/"
              f"{coverage['bins_attack_total']} "
              f"({_fmt_pct(coverage['attack_coverage_pct'])})")
    else:
        print("coverage  : not measured (no flow table)")
    print(f"written   : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

