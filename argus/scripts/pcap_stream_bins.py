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
        --flows data/raw/cicids2017/Thursday-*.parquet \
        --out thu_packet_bins.json [--apply-offset]

Guarantees:
  * streams with PcapReader (never rdpcap), fixed floor(ts/60)*60 bins keyed
    (src_ip, bin_start) exactly as D1 keys flow bins;
  * RST packets are never counted as retransmissions;
  * when --flows is given, output rows are the FLOW bins: uncovered bins get
    packet_features: null + packet_features_covered: false (never zero-filled)
    and coverage percentages (overall + attack bins) are written to the JSON;
  * clock offset is measured (median over matched directional 5-tuples) and
    reported, status "not_measured" when it cannot be.

Exit status: 0 on success, 1 on error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.world_model.pcap_bins import (  # noqa: E402
    APPROXIMATIONS,
    BIN_SECONDS,
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


def load_flow_rows(path: Path) -> tuple[list[dict], dict]:
    """Read a flow table (parquet/csv) into offset/join inputs + stats."""
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
    }
    missing = [k for k in ("src", "dst", "sport", "dport", "ts")
               if cols[k] is None]
    if missing:
        raise SystemExit(
            f"flow file {path.name} lacks required columns: {missing}")

    ts = pd.to_datetime(df[cols["ts"]], dayfirst=True, errors="coerce")
    unparsable = int(ts.isna().sum())
    ok = ts.notna()

    rows: list[dict] = []
    for _, r in df[ok].iterrows():
        label = ""
        if cols["label"] is not None:
            label = str(r[cols["label"]] or "").strip()
        rows.append({
            "src": str(r[cols["src"]]).strip(),
            "dst": str(r[cols["dst"]]).strip(),
            "sport": int(r[cols["sport"]]),
            "dport": int(r[cols["dport"]]),
            "ts": 0.0,  # set from the vectorised epoch conversion below
            "label": label,
        })
    # vectorised epoch conversion (naive timestamps are taken as-is; the
    # measured clock offset is what reveals any timezone labelling)
    # pandas 3 parses to datetime64[us], pandas 2 to datetime64[ns]:
    # convert explicitly to ns (numpy handles the unit cast) so the
    # epoch scale is correct on both.
    epochs = (ts[ok].to_numpy(dtype="datetime64[ns]").astype("int64")
              / 1e9)
    for row, epoch in zip(rows, epochs):
        row["ts"] = float(epoch)

    stats = {
        "flow_file": path.name,
        "flow_rows": len(rows),
        "timestamp_unparsable": unparsable,
    }
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stream a PCAP into fixed-60 s packet-tier bins keyed "
                    "exactly as D1 keys flow bins.")
    ap.add_argument("--pcap", required=True, type=Path,
                    help="input PCAP (read packet-by-packet via PcapReader)")
    ap.add_argument("--flows", type=Path, default=None,
                    help="optional flow table (parquet/csv) for clock-offset "
                         "measurement, bin coverage and the D1 left-join")
    ap.add_argument("--bin-seconds", type=int, default=BIN_SECONDS)
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default: <pcap>_bins.json)")
    ap.add_argument("--apply-offset", action="store_true",
                    help="shift PCAP timestamps by the MEASURED median "
                         "clock offset before binning (requires --flows)")
    args = ap.parse_args()

    if args.apply_offset and args.flows is None:
        ap.error("--apply-offset requires --flows (offset must be measured)")

    out_path = args.out or args.pcap.with_name(args.pcap.stem + "_bins.json")

    flow_rows: list[dict] = []
    flow_stats: dict = {}
    if args.flows is not None:
        flow_rows, flow_stats = load_flow_rows(args.flows)

    offset = measure_clock_offset(args.pcap, flow_rows)
    apply_sec = 0.0
    if args.apply_offset and offset["status"] == "measured":
        apply_sec = float(offset["median_sec"])

    extraction = extract_pcap_bins(
        args.pcap, bin_seconds=args.bin_seconds, apply_offset_sec=apply_sec)
    records = extraction["records"]

    if flow_rows:
        flow_keys = [(r["src"], bin_start_for(r["ts"], args.bin_seconds))
                     for r in flow_rows]
        attack_keys = [(r["src"], bin_start_for(r["ts"], args.bin_seconds))
                       for r in flow_rows
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
        "pcap": extraction["stats"],
        "flows": flow_stats or {"status": "not_measured",
                                "reason": "no flow table supplied"},
        "clock_offset": offset,
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
          f"({extraction['stats']['packets_considered']} IP packets)")
    print(f"offset    : {offset['status']}"
          + (f" median={offset['median_sec']:+.3f}s "
             f"iqr={offset['dispersion_iqr_sec']:.3f}s "
             f"n={offset['n_matched_tuples']}"
             if offset["status"] == "measured" else ""))
    if "coverage_pct" in coverage:
        print(f"coverage  : {coverage['bins_covered']}/"
              f"{coverage['bins_flow_total']} flow bins "
              f"({coverage['coverage_pct']:.1f}%), attack "
              f"{coverage['attack_bins_covered']}/"
              f"{coverage['bins_attack_total']} "
              f"({coverage['attack_coverage_pct']:.1f}%)")
    else:
        print("coverage  : not measured (no flow table)")
    print(f"written   : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

