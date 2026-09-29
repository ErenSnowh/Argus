#!/usr/bin/env python3
"""
pcap_stream_bins.py
-------------------
CLI for the streaming fixed-60 s-bin packet-tier extractor (plan D2
amendment; D1 keying). Runs BEFORE any large PCAP download as the proposed
and tested extraction path, and again on real captures to measure clock
offset and flow/PCAP bin coverage.

    python scripts/pcap_stream_bins.py --pcap data/sample_portscan.pcap
    python scripts/pcap_stream_bins.py --pcap thu.pcap \
        --flows data/raw/cicids2017/ --out thu_packet_bins.json
    python scripts/pcap_stream_bins.py --pcap thu.pcap \
        --flows thu-thu.parquet thu-fri.parquet --apply-offset

Guarantees:
  * streams with PcapReader (never rdpcap), fixed floor(ts/60)*60 bins keyed
    (src_ip, bin_start) exactly as D1 keys flow bins, exactly one record per
    key; backwards packets fail closed (default) or merge when explicitly
    requested (--on-backwards merge, bounded by --reorder-grace-sec);
  * RST packets are never counted as retransmissions (the count is an
    approximate duplicate-sequence proxy, stated in every output);
  * --flows accepts multiple parquet/csv files AND directories (expanded to
    *.parquet + *.csv, sorted); missing labels/NaN cells are real nulls -
    never the string "nan"; all-null timestamp padding is reported as
    timestamp_null, distinct from genuinely unparsable timestamps;
  * when --flows is given, output rows are the FLOW bins: uncovered bins get
    packet_features: null + packet_features_covered: false (never zero-filled)
    and coverage percentages (overall + attack bins) are written to the JSON
    over unique host-minute bins; an all-benign table yields attack coverage
    null (printed "n/a"), never 0.0;
  * clock offset: the observed delta (observed_delta) is reported separately
    from the apply decision (apply); --apply-offset FAILS with exit 1 when
    the offset was not measured, matched too few connection instances, or
    has ambiguous dispersion - it never silently applies 0.

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
    bin_start_for,
    extract_pcap_bins,
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

    stats = {
        "flow_file": path.name,
        "flow_rows": len(rows),
        "rows_dropped_missing_identity": dropped_identity,
        "timestamp_null": null_ts,
        "timestamp_unparsable": unparsable,
        "timestamp_all_null": bool(len(df)) and null_ts == len(df),
        "labels_null": (int(df[cols["label"]].isna().sum())
                        if cols["label"] is not None else None),
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
                    help="shift PCAP timestamps by the MEASURED median "
                         "clock offset before binning (requires --flows; "
                         "refused with exit 1 unless the offset is eligible)")
    ap.add_argument("--on-backwards", choices=("fail", "merge"),
                    default="fail",
                    help="packet whose target bin already closed: 'fail' "
                         "(default - abort, nothing is silently dropped) or "
                         "'merge' into the retained accumulator")
    ap.add_argument("--reorder-grace-sec", type=float, default=0.0,
                    help="keep a bin open this long past its boundary before "
                         "closing it (bounded reordering, default 0)")
    args = ap.parse_args(argv)

    if args.apply_offset and not args.flows:
        ap.error("--apply-offset requires --flows (offset must be measured)")
    if args.bin_seconds <= 0:
        ap.error("--bin-seconds must be > 0")
    if args.reorder_grace_sec < 0:
        ap.error("--reorder-grace-sec must be >= 0")
    if not args.pcap.is_file():
        print(f"ERROR: PCAP not found: {args.pcap}", file=sys.stderr)
        return 1

    out_path = args.out or args.pcap.with_name(args.pcap.stem + "_bins.json")

    flow_rows: list[dict] = []
    flow_stats: dict = {}
    if args.flows is not None:
        flow_rows, flow_stats = load_flow_tables(args.flows)

    offset = measure_clock_offset(args.pcap, flow_rows)
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
            on_backwards=args.on_backwards,
            reorder_grace_sec=args.reorder_grace_sec)
    except BackwardsTimestampError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    records = extraction["records"]

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
        "backwards_policy": {
            "on_backwards": args.on_backwards,
            "reorder_grace_sec": args.reorder_grace_sec,
            "note": "'fail' aborts rather than dropping late packets; "
                    "'merge' folds them into the retained bin and counts "
                    "them in pcap.backwards_packets_merged",
        },
        "pcap": extraction["stats"],
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

    stats = extraction["stats"]
    print(f"pcap      : {args.pcap}")
    print(f"bins      : {len(records)} packet-tier bins "
          f"({stats['packets_considered']} IP packets)")
    if stats["backwards_packets_merged"]:
        print(f"backwards : {stats['backwards_packets_merged']} late packets "
              f"merged (on_backwards=merge, "
              f"grace={stats['reorder_grace_sec']}s)")
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
    print(f"offset    : {offset['status']}"
          + (f" median={offset['median_sec']:+.3f}s "
             f"iqr={offset['dispersion_iqr_sec']:.3f}s "
             f"n={offset['n_matched_tuples']} "
             f"flags={apply_block.get('flags') or 'none'}"
             if offset["status"] == "measured"
             else f" ({offset.get('reason')})"))
    if args.apply_offset:
        apply_note = f"applied {apply_sec:+.3f}s"
    elif apply_block.get("eligible"):
        apply_note = "eligible but not applied (no --apply-offset given)"
    else:
        apply_note = ("refused: "
                      + str(apply_block.get("reason")
                            or offset.get("reason")))
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

