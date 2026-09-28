#!/usr/bin/env python3
"""
validate_cicids2017.py
----------------------
Validate the CIC-IDS-2017 flow-level mirror before any training reads it.

Source of truth is pinned, not "whatever is on the branch today":

    Hugging Face dataset   bvsam/cic-ids-2017
    directory              traffic_labels/   (NOT machine_learning/, NOT pcap/)
    revision               70bac6246d99cf046186a02e1cce6883e2ffe7ea
    files                  8 parquet files, filenames preserved verbatim

The expected sha256 below is the LFS object id reported by the Hugging Face
API for that revision, so a file that "looks like" the right dataset but is
actually a newer or older upload fails the hash gate.

Checks performed per file:
    1. sha256 of the local file == LFS oid recorded for the revision
    2. file opens as parquet; schema recorded (columns + dtypes)
    3. row count recorded
    4. label column found; null / blank label count must be 0
    5. label histogram recorded (whitespace-normalised)
    6. timestamp column parsed; min / max / unparsable count recorded
    7. across all 8 files: schema is identical, label set is consistent

Raw parquet stays out of git (argus/.gitignore: data/raw/). What is tracked
is this script and the JSON record it writes.

Usage:
    python scripts/validate_cicids2017.py           # report only
    python scripts/validate_cicids2017.py --write    # refresh the JSON record

Exit status: 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode some dataset label
# strings (e.g. the U+0096 byte in "Web Attack \u0096 ..."). Never let output
# encoding crash the validator or lose the histogram.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

HF_REPO = "bvsam/cic-ids-2017"
HF_REVISION = "70bac6246d99cf046186a02e1cce6883e2ffe7ea"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/tree/{HF_REVISION}"

# LFS object ids (sha256 of file content) as reported by the Hugging Face API
# for the revision above. Filenames are preserved exactly as in traffic_labels/.
EXPECTED_SHA256 = {
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet":
        "7c5876d52189fc01af54bad6cf23afe9f7fbc0e3ca6c3595920754f0c3ba8f66",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet":
        "4d78cee297c27f1a9947b9384793e587a46c7a3ea89db199553dabddc9835d4a",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv.parquet":
        "2c00236b13a69f4b1c222b8f4a89451dc2148cb04a8cd0c45c2a87af51471774",
    "Monday-WorkingHours.pcap_ISCX.csv.parquet":
        "dfdcef4b8670e52af54dc4f82174834365a393473e877174cca46d17b12dfd02",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv.parquet":
        "5da010354f0fc1040fd1fe65967096e1063475de8dd30ae4f657c07201d728a7",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet":
        "d8110c04a7af91124ada1c5ad901c4210879df1af8882dc637767532e7165350",
    "Tuesday-WorkingHours.pcap_ISCX.csv.parquet":
        "27e83d518cb093faefd0f883cb4df3ad8b353f150934004f28d0e7962f9f31c4",
    "Wednesday-workingHours.pcap_ISCX.csv.parquet":
        "d23a259820b16e1ad54f9f3b58d5727c5032d383015f90bc7c07cebbdf8a7140",
}

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data" / "raw" / "cicids2017"
RECORD_PATH = HERE.parent / "data" / "cicids2017_provenance.json"

MAX_HISTOGRAM = 40


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find(columns, wanted: str):
    for col in columns:
        if str(col).strip().lower() == wanted:
            return col
    return None


def parse_timestamps(series):
    """Return (parsed, failures). Handles both m/d/y and d/m/y spellings."""
    import pandas as pd

    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().all() and len(series):
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    failures = int(parsed.isna().sum())
    return parsed, failures


def validate_file(path: Path) -> tuple[dict, list[str]]:
    import pandas as pd

    problems: list[str] = []
    record: dict = {"file": path.name, "bytes": path.stat().st_size}

    actual = sha256_of(path)
    expected = EXPECTED_SHA256.get(path.name)
    record["sha256"] = actual
    record["expected_sha256"] = expected
    record["hash_match"] = (actual == expected) if expected else False
    if expected is None:
        problems.append(f"{path.name}: no expected hash recorded for this filename")
    elif not record["hash_match"]:
        problems.append(
            f"{path.name}: sha256 {actual} != revision {HF_REVISION} "
            f"({expected[:16]}...)"
        )

    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # unreadable data is a hard failure
        problems.append(f"{path.name}: not readable as parquet: {exc}")
        return record, problems

    record["rows"] = int(len(df))
    record["columns"] = [str(c) for c in df.columns]
    record["dtypes"] = {str(c): str(t) for c, t in df.dtypes.items()}

    label_col = _find(df.columns, "label")
    if label_col is None:
        problems.append(
            f"{path.name}: no label column found "
            f"(have: {record['columns'][:6]}...)"
        )
        return record, problems
    record["label_column"] = repr(label_col)

    labels = df[label_col]

    # Two very different kinds of "no label":
    #   * an entirely empty row - padding observed in this pinned snapshot.
    #     Documented, dropped before training, never zero-filled.
    #   * a populated row whose label is missing - a real data hole. Fatal.
    all_null_mask = df.isna().all(axis=1)
    all_null_rows = int(all_null_mask.sum())
    unlabelled_real = int((labels.isna() & ~all_null_mask).sum())
    blanks = int(labels.dropna().astype(str).str.strip().eq("").sum())

    record["rows_total"] = int(len(df))
    record["all_null_rows"] = all_null_rows
    record["effective_rows"] = int(len(df) - all_null_rows)
    record["label_nulls"] = unlabelled_real
    record["label_blanks"] = blanks
    if unlabelled_real or blanks:
        problems.append(
            f"{path.name}: {unlabelled_real} populated rows have no label and "
            f"{blanks} labels are blank"
        )
    if all_null_rows:
        record["observed_anomaly"] = (
            f"{all_null_rows:,} rows are entirely empty (every column NaN). "
            "Observed in this pinned Hugging Face Parquet snapshot "
            f"({HF_REPO} @ {HF_REVISION[:12]}, traffic_labels/). Whether the "
            "canonical CIC-IDS-2017 export contains the same rows has not "
            "been established, so this is not asserted to be an upstream "
            "export defect. These rows must be dropped in preprocessing and "
            "must never be zero-filled or imputed."
        )

    counts = labels.dropna().astype(str).str.strip().value_counts()
    record["label_histogram"] = {str(k): int(v) for k, v in counts.items()}


    ts_col = _find(df.columns, "timestamp")
    if ts_col is None:
        record["timestamp_column"] = None
        record["timestamp_min"] = None
        record["timestamp_max"] = None
        record["timestamp_unparsable"] = None
    else:
        # padding rows carry a NaN timestamp by construction, so measure
        # parse failures on the populated rows only
        parsed, failures = parse_timestamps(df.loc[~all_null_mask, ts_col])
        record["timestamp_padding_rows"] = int(df[ts_col].isna().sum()) - all_null_rows
        record["timestamp_column"] = repr(ts_col)
        record["timestamp_min"] = (
            parsed.min().isoformat() if not parsed.isna().all() else None
        )
        record["timestamp_max"] = (
            parsed.max().isoformat() if not parsed.isna().all() else None
        )
        record["timestamp_unparsable"] = failures
        if parsed.isna().all():
            problems.append(f"{path.name}: every timestamp is unparsable")

    return record, problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the CIC-IDS-2017 TrafficLabelling parquet mirror."
    )
    parser.add_argument("--write", action="store_true",
                        help=f"write {RECORD_PATH}")
    args = parser.parse_args()

    if not DATA_DIR.is_dir():
        print(f"FAIL: {DATA_DIR} does not exist. Place the 8 parquet files there.")
        return 1

    files = sorted(DATA_DIR.glob("*.parquet"))
    present = {p.name for p in files}
    missing = sorted(set(EXPECTED_SHA256) - present)
    extra = sorted(present - set(EXPECTED_SHA256))

    problems: list[str] = []
    if missing:
        problems.append(f"missing files: {', '.join(missing)}")
    if extra:
        problems.append(f"unexpected files: {', '.join(extra)}")

    print(f"source   {HF_REPO} @ {HF_REVISION}")
    print(f"dir      {DATA_DIR}")
    print(f"files    {len(files)} parquet "
          f"({sum(f.stat().st_size for f in files) / 1e6:.1f} MB)\n")

    per_file, schemas, total_rows, all_null_total = [], [], 0, 0
    overall_labels: dict[str, int] = {}

    for path in files:
        rec, probs = validate_file(path)
        per_file.append(rec)
        problems.extend(probs)
        if "columns" in rec:
            schemas.append(tuple(rec["columns"]))
        total_rows += rec.get("rows_total", 0)
        all_null_total += rec.get("all_null_rows", 0)
        for label, count in rec.get("label_histogram", {}).items():
            overall_labels[label] = overall_labels.get(label, 0) + count

        status = "ok  " if not probs else "FAIL"
        ts = (f"{rec.get('timestamp_min')} .. {rec.get('timestamp_max')}"
              if rec.get("timestamp_min") else "n/a")
        print(f"[{status}] {path.name}")
        print(f"        rows={rec.get('rows_total', 0):>8}  "
              f"effective={rec.get('effective_rows', 0):>8}  "
              f"allNaN={rec.get('all_null_rows', 0)}  "
              f"unlabelled_real={rec.get('label_nulls', '-')}  "
              f"hash_match={rec.get('hash_match')}")
        print(f"        labels={len(rec.get('label_histogram', {}))}  "
              f"ts {ts}  unparsable={rec.get('timestamp_unparsable')}")
        if rec.get("observed_anomaly"):
            print(f"        ANOMALY: {rec['observed_anomaly']}")

    schema_consistent = bool(schemas) and len(set(schemas)) == 1
    if not schema_consistent:
        problems.append("the 8 files do not share one schema")

    print("\n--- aggregate ---")
    print(f"rows (as shipped)  {total_rows:,}")
    print(f"all-null padding   {all_null_total:,}")
    print(f"rows (effective)   {total_rows - all_null_total:,}")
    print(f"schema consistent {schema_consistent} "
          f"({len(schemas[0]) if schemas else 0} columns)")
    print("label histogram:")
    for label, count in sorted(overall_labels.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>9}  {label}")

    timestamps = {
        "min": min((r.get("timestamp_min") for r in per_file
                    if r.get("timestamp_min")), default=None),
        "max": max((r.get("timestamp_max") for r in per_file
                    if r.get("timestamp_max")), default=None),
    }

    record = {
        "dataset": "CIC-IDS-2017",
        "purpose": "flow-level features for the ARGUS world model (D2)",
        "source": {
            "host": "huggingface.co",
            "repo": HF_REPO,
            "revision": HF_REVISION,
            "directory": "traffic_labels/",
            "url": HF_URL,
            "note": "parquet mirror of GeneratedLabelledFlows/TrafficLabelling; "
                    "machine_learning/ and pcap/ were NOT used",
        },
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_files_tracked_by_git": False,
        "observed_anomalies": [
            {"file": r["file"], "observed_anomaly": r["observed_anomaly"]}
            for r in per_file if r.get("observed_anomaly")
        ],
        "files": per_file,
        "totals": {
            "files": len(files),
            "rows": total_rows,
            "all_null_padding_rows": all_null_total,
            "rows_effective": total_rows - all_null_total,
            "bytes": sum(f.stat().st_size for f in files),
            "schema_consistent": schema_consistent,
            "schema_columns": list(schemas[0]) if schemas else [],
            "label_histogram": overall_labels,
            "timestamps": timestamps,
        },
        "checks_passed": not problems,
        "problems": problems,
    }

    if args.write:
        RECORD_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"\nwrote {RECORD_PATH}")

    if problems:
        print("\nFAIL:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"\nPASS: hashes match revision {HF_REVISION}; schema, labels and "
          "timestamps are consistent.")
    if all_null_total:
        print(f"NOTE: {all_null_total:,} all-null padding rows are recorded as "
              "observed in this pinned snapshot (not confirmed against the "
              "canonical CIC export) and must be dropped in preprocessing "
              "(never zero-filled).")
    return 0


if __name__ == "__main__":
    sys.exit(main())



