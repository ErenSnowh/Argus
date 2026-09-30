#!/usr/bin/env python3
"""
prepare_cicids2017.py
---------------------
WP2 flow-tier pipeline: pinned CIC-IDS-2017 flow tables -> host/time-binned sequences
+ a manifest a later stage is allowed to trust (plan D1 / D2, agent-prompt WP2).

    python scripts/prepare_cicids2017.py --check
    python scripts/prepare_cicids2017.py
    python scripts/prepare_cicids2017.py --out data/processed --bin-seconds 30
    python scripts/prepare_cicids2017.py --max-rows 200000 --check   # smoke run

What it does, in order:

  1. GATES on the pinned snapshot. Every file listed in
     `data/cicids2017_provenance.json` must exist with the recorded byte size and
     sha256. A wrong variant, a truncated download or a re-export fails here, before
     any number is computed.
  2. DROPS (never zero-fills) the all-null padding rows the pinned snapshot carries in
     Thursday-Morning-WebAttacks (288,602 rows measured - see the provenance record's
     `observed_anomalies`), plus exact duplicate rows. Both counts go in the manifest.
     30 feature columns are then taken from the real CICFlowMeter tables; the 8
     PACKET_LEVEL_COLUMNS are NOT synthesised and NOT zero-filled - they are simply not
     part of the flow-tier schema (they arrive through the packet tier, joined on the
     identical (src_ip, bin_start) key).
  3. BINS every flow into one row per (src_ip, 60 s wall-clock bin) via
     `ml/world_model/binning.build_host_time_bins` - the D1 definition, shared with the
     packet tier's keying so the two can be joined without re-binning.
  4. Builds sequences with `make_sequences(W=10, horizons=(1,2,4))`: 10 contiguous
     minutes of one host's history, targets at t+k and "any non-BENIGN in (t, t+k]".
  5. CHECKS the raw label histogram against the published CIC-IDS-2017 family counts at
     +/-3 % (exact for Infiltration 36 / SQLi 21 / Heartbleed 11, where 3 % is less than
     one flow), and the per-file row counts against the provenance record at +/-3 %.
     Any violation makes `--check` fail and NO parquet is written.

`--check` prints the manifest (per-day rows, label histogram, hosts, bins, sequences per
horizon) and writes nothing. Without `--check` it writes
`<out>/cicids2017_bins_60s.parquet` + `<out>/manifest.json`; both stay gitignored
(argus/.gitignore: data/processed/). A `--max-rows` run is a truncated smoke run: it is
marked `truncated: true`, its anchors are recorded but NOT enforced, and its parquet is
named `.maxrowsN.parquet` so it can never be mistaken for the real artifact.

No training, no model code, no PCAP, no README metrics are touched by this script.

WARNING for the next stage. Infiltration (36 flows) and Heartbleed (11) are kept out of
the stage targets, so the produced parquet carries `-1` in `y_stage_*` for the bins those
families dominate. That is a MARK, not a mask: `MultiTaskLoss` builds a default
`nn.CrossEntropyLoss()` (model.py:324, used at :342 and :368) whose `ignore_index` is
-100, so `-1` is NOT ignored there and would be read as a class index. Do not point
training at this artifact until WP3 has added and tested masking in both stage losses.
The manifest records the obligation under `stage_target_masking`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Windows consoles default to cp1252; the snapshot's labels contain U+0096, so output
# encoding must never crash the pipeline or lose a histogram bin.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.world_model.binning import (  # noqa: E402
    FLOW_TIER_FEATURE_LIST,
    STAGE_EXCLUDED_FAMILIES,
    STAGE_TARGET_IGNORE,
    build_host_time_bins,
    make_sequences,
    normalise_cicids2017_labels,
)

PROVENANCE_PATH = ROOT / "data" / "cicids2017_provenance.json"
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "cicids2017"
DEFAULT_OUT_DIR = ROOT / "data" / "processed"

# D1 sequence geometry. Fixed here rather than exposed as flags: they are the definition
# of the task ("10 minutes of history, next stage in 1/2/4 minutes"), not a knob.
W_SEQUENCE = 10
HORIZONS = (1, 2, 4)
STRIDE = 1

# Published CIC-IDS-2017 family counts (the numbers the plan quotes), compared against
# the RAW family histogram before any stage merging. `exact` marks the three families
# whose published count is so small that +/-3 % is under one flow.
LABEL_ANCHOR_TOLERANCE_PCT = 3.0
LABEL_ANCHORS = {
    "BENIGN": (2_270_000, False),
    "DoS Hulk": (231_000, False),
    "PortScan": (159_000, False),
    "DDoS": (128_000, False),
    "DoS GoldenEye": (10_300, False),
    "FTP-Patator": (7_900, False),
    "SSH-Patator": (5_900, False),
    "DoS slowloris": (5_800, False),
    "DoS Slowhttptest": (5_500, False),
    "Bot": (2_000, False),
    "Web Attack - Brute Force": (1_500, False),
    "Web Attack - XSS": (650, False),
    "Infiltration": (36, True),
    "Web Attack - Sql Injection": (21, True),
    "Heartbleed": (11, True),
}

# Raw CICFlowMeter column -> canonical name, limited to the columns this pipeline uses.
# The target names are the ones dataset_loader._CICIDS2017_COLUMN_MAP already uses
# (dataset_loader.py:565-624) so a future --bins loader and this pipeline agree on the
# 39-column `feature_list`. "Fwd Header Length.1" is the duplicate column CICFlowMeter
# exports; the first "Fwd Header Length" is the one used.
CICIDS2017_COLUMN_MAP = {
    "Source IP": "src_ip",
    "Destination IP": "dst_ip",
    "Destination Port": "dst_port",
    "Timestamp": "timestamp",
    "Label": "label",
    "Flow Duration": "flow_duration_ms",
    "Total Fwd Packets": "total_fwd_packets",
    "Total Backward Packets": "total_bwd_packets",
    "Total Length of Fwd Packets": "total_fwd_bytes",
    "Total Length of Bwd Packets": "total_bwd_bytes",
    "Fwd Packet Length Mean": "fwd_packet_len_mean",
    "Fwd Packet Length Std": "fwd_packet_len_std",
    "Bwd Packet Length Mean": "bwd_packet_len_mean",
    "Bwd Packet Length Std": "bwd_packet_len_std",
    "Flow Bytes/s": "flow_bytes_per_sec",
    "Flow Packets/s": "flow_packets_per_sec",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "SYN Flag Count": "syn_flag_count",
    "ACK Flag Count": "ack_flag_count",
    "RST Flag Count": "rst_flag_count",
    "PSH Flag Count": "psh_flag_count",
    "FIN Flag Count": "fin_flag_count",
    "URG Flag Count": "urg_flag_count",
    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "avg_packet_size",
    "Fwd Header Length": "fwd_header_len",
    "Bwd Header Length": "bwd_header_len",
}

# Exact-duplicate detection key. CICFlowMeter's "Flow ID" is the per-flow identity
# (5-tuple + start), so it joins the duplicate key even though it is not a feature: two
# rows that agree on every consumed column AND the flow id are one flow exported twice.
# Measured over the pinned snapshot this finds 204 rows, against 203 for a comparison
# over all 85 raw columns (one Wednesday row differs only in a column this pipeline
# never reads), and it does not touch the exact-count anchor families.
DUPLICATE_KEY_COLUMN = "Flow ID"


# ---------------------------------------------------------------------------
# Snapshot gate and per-file cleaning
# ---------------------------------------------------------------------------


def file_digests(path: Path) -> tuple[str, str]:
    """(md5, sha256) of one file, in a single pass (306 MB for the whole snapshot)."""
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def git_sha() -> str:
    """HEAD of the checkout the artifact was produced from ('unknown' outside git)."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
                                capture_output=True, text=True, timeout=15)
    except Exception:
        return "unknown"
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else "unknown"


# The files whose bytes produced the artifact. binning.py and prepare_cicids2017.py build
# it; the test file is listed too so a reviewer can see the whole WP2 change at once.
PIPELINE_SOURCE_FILES = (
    "ml/world_model/binning.py",
    "scripts/prepare_cicids2017.py",
    "tests/test_prepare_cicids2017.py",
)
CODE_PRODUCING_FILES = PIPELINE_SOURCE_FILES[:2]
# git runs from the repo root while these names are argus-relative; this is the one
# directory name that joins them.
REPO_DIR_NAME = ROOT.name


def _run_git(arguments: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    """Run git from the repo root and return the CompletedProcess (caller checks rc).

    Return-code checking is the caller's job on purpose: every consumer of git here
    decides what an UNKNOWN state means, and the only safe default for provenance is
    "unknown / not reproducible", never "clean".
    """
    return subprocess.run(["git", *arguments], cwd=ROOT.parent, capture_output=True,
                          text=True, timeout=timeout)


class PorcelainParseError(ValueError):
    """A `git status --porcelain` line that cannot be parsed.

    Raised instead of skipped, because skipping removes that file's status token, and
    `describe_git_status` treats a tracked file with NO token as clean. A silently dropped
    line would therefore turn a modified file into a "committed and reproducible" claim -
    the exact failure this module exists to prevent.
    """


def parse_porcelain_statuses(stdout: str, strip_prefix: str | None = None,
                             strict: bool = True) -> dict[str, str]:
    """Parse `git status --porcelain` output into {path: two-column status}.

    Porcelain v1 emits TWO status columns, a space, then the path: `XY path`, where X is
    the index status and Y the worktree status. That is why `line.partition(" ")` is wrong:
    an unstaged modification is the line " M path", whose first character is a space, so
    partitioning on the first space loses the path entirely and the file then looks absent -
    which a naive caller reads as "clean". Both columns are taken by index here, and a
    rename ("R  old -> new") is recorded under its destination path.

    `strip_prefix` removes a leading directory from the KEY, which is how the keys are
    normalised to the argus-relative names `describe_git_status` looks up. This must be
    done at the single place the keys are produced: an earlier version stripped the prefix
    only when a `scope` was passed, while `code_provenance` called `git_status()` with no
    scope, so the returned keys kept their "argus/" prefix, matched nothing, and a tracked
    modified file was reported clean.

    `strict=True` (the only mode `git_status` uses) raises PorcelainParseError on a
    malformed non-empty line instead of dropping it.
    """
    statuses: dict[str, str] = {}
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        if len(line) < 4 or line[2] != " ":
            if strict:
                raise PorcelainParseError(
                    f"line {line_number} is not a valid porcelain status "
                    f"(expected 'XY path'): {line!r}"
                )
            continue
        index_status, worktree_status = line[0], line[1]
        path = line[3:].strip()
        if " -> " in path:  # rename: record the destination
            path = path.split(" -> ", 1)[1].strip()
        path = path.replace("\\", "/")
        if len(path) > 1 and path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        if not path:
            if strict:
                raise PorcelainParseError(
                    f"line {line_number} has no path: {line!r}"
                )
            continue
        if strip_prefix and path.startswith(f"{strip_prefix}/"):
            path = path[len(strip_prefix) + 1:]
        statuses[path] = f"{index_status}{worktree_status}"
    return statuses


# A tracked file is clean only when BOTH porcelain columns are blank.
CLEAN_PORCELAIN_STATUS = "  "


def git_status(paths=PIPELINE_SOURCE_FILES,
               scope: str | None = None) -> tuple[dict[str, str], str | None]:
    """(statuses keyed by ARGUS-relative path, error). `error` is None on success.

    The returned keys are always argus-relative, which is what `describe_git_status` looks
    up. `scope` only widens the pathspec sent to git; it must never be what normalises the
    keys, because `code_provenance` calls this with scope=None.

    Any failure - git could not run, a non-zero return code, or unparseable output - is
    returned as an error so `code_provenance` fails closed instead of reading "no output"
    as "nothing modified".
    """
    command = ["status", "--porcelain", "--untracked-files=all"]
    if scope:
        command.append(scope)
    command.extend(f"{REPO_DIR_NAME}/{name}" for name in paths)
    try:
        result = _run_git(command)
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, f"git status could not be run: {exc}"
    if result.returncode != 0:
        return {}, f"git status exited {result.returncode}: {result.stderr.strip()}"
    try:
        return parse_porcelain_statuses(
            result.stdout, strip_prefix=REPO_DIR_NAME, strict=True), None
    except PorcelainParseError as exc:
        return {}, f"git status output could not be parsed: {exc}"


def git_tracked(paths=PIPELINE_SOURCE_FILES) -> tuple[set[str], str | None]:
    """(argus-relative names git tracks, error). `error` is None on success."""
    try:
        result = _run_git(["ls-files", "--",
                           *[f"{REPO_DIR_NAME}/{name}" for name in paths]])
    except (OSError, subprocess.SubprocessError) as exc:
        return set(), f"git ls-files could not be run: {exc}"
    if result.returncode != 0:
        return set(), f"git ls-files exited {result.returncode}: {result.stderr.strip()}"
    tracked = set()
    for line in result.stdout.splitlines():
        name = line.strip().replace("\\", "/")
        if name.startswith(f"{REPO_DIR_NAME}/"):
            tracked.add(name[len(REPO_DIR_NAME) + 1:])
    return tracked, None


def git_is_dirty(scope: str | None = None) -> tuple[bool | None, str | None]:
    """(dirty?, error). `dirty` is None - not False - when it cannot be determined."""
    command = ["status", "--porcelain"]
    if scope:
        command.extend(["--", scope])
    try:
        result = _run_git(command)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"git status could not be run: {exc}"
    if result.returncode != 0:
        return None, f"git status exited {result.returncode}: {result.stderr.strip()}"
    return bool(result.stdout.strip()), None


def describe_git_status(name: str, tracked: bool, statuses: dict[str, str]) -> str:
    """Human-readable, fail-closed state for one pipeline file.

    `UNKNOWN` is returned whenever the evidence is missing or ambiguous - an untracked file
    with no porcelain line, or a tracked file with a line that does not parse. It is never
    upgraded to `clean`, because `clean` is a claim about the file's bytes and the only
    way to earn it is a tracked file with both porcelain columns blank.
    """
    if not (ROOT / name).exists():
        return "absent"
    token = statuses.get(name)
    if token is None:
        return "clean" if tracked else "untracked"
    if not tracked:
        return "untracked" if token == "??" else f"untracked-porcelain:{token!r}"
    if token == CLEAN_PORCELAIN_STATUS:
        return "clean"
    if token == "??":
        # porcelain says untracked, ls-files says tracked: the two disagree, so we do not
        # guess
        return "UNKNOWN (ls-files says tracked, porcelain says untracked)"
    return f"modified:{token}"


def code_provenance() -> dict:
    """Identify the code that produced this manifest, not just the commit it sits on.

    `git_sha` on its own is misleading while the pipeline is uncommitted: it names the
    BASE commit, which does not contain these files at all. So the manifest also records
    whether the tree is dirty, the porcelain status of each WP2 file, and their sha256 -
    the hashes are what actually reproduce the artifact.

    FAIL CLOSED. Every git call returns its error, and any error or ambiguous state makes
    `reproducible_from_git_sha` false. An unknown state is never reported as clean, and the
    note is conditional: once the code IS committed and clean, it says the commit is
    sufficient instead of always claiming git_sha is a base commit.
    """
    statuses, status_error = git_status()
    tracked, tracked_error = git_tracked()
    scoped, scoped_error = git_is_dirty(scope=REPO_DIR_NAME)
    dirty, dirty_error = git_is_dirty()
    errors = [error for error in (status_error, tracked_error, scoped_error, dirty_error)
              if error]

    files = {}
    for name in PIPELINE_SOURCE_FILES:
        path = ROOT / name
        files[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
            "git_status": describe_git_status(name, name in tracked, statuses),
            "tracked": name in tracked,
        }

    committed = (
        not errors
        and all(files[name]["git_status"] == "clean" and files[name]["tracked"]
                for name in CODE_PRODUCING_FILES)
    )
    if errors:
        note = (
            "GIT PROVENANCE INCOMPLETE - the artifact is NOT reproducible from git_sha "
            "alone and this manifest says so. Errors: " + " | ".join(errors)
        )
    elif committed:
        note = (
            "The WP2 code that produced this manifest is committed and clean, so git_sha "
            "is sufficient to identify it; the per-file sha256 values above are recorded "
            "as a cross-check."
        )
    else:
        note = (
            "git_sha is the BASE commit only. The sha256 values above identify the code "
            "that actually produced this manifest. While pipeline_code_committed is false "
            "the artifact is NOT reproducible from git_sha alone - commit the WP2 files "
            "and regenerate, or reproduce from the recorded hashes."
        )
    return {
        "git_sha": git_sha(),
        "git_dirty": dirty,
        "git_dirty_under_argus": scoped,
        "git_errors": errors,
        "pipeline_files": files,
        "pipeline_code_committed": committed,
        "reproducible_from_git_sha": committed,
        "note": note,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prepare_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def load_provenance(path: Path | None = None) -> dict:
    """Load the pinned-snapshot record (default: argus/data/cicids2017_provenance.json)."""
    path = PROVENANCE_PATH if path is None else path
    if not path.exists():
        raise SystemExit(
            f"provenance record not found: {path}\n"
            "WP2 gates on the pinned snapshot recorded there (written by "
            "scripts/validate_cicids2017.py --write). No substitute dataset is used."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def verify_snapshot(raw_dir: Path, provenance: dict) -> dict[str, tuple[str, str]]:
    """Byte-size + sha256 gate over every file in the provenance record.

    Returns {filename: (md5, sha256)}. Any mismatch exits before a single number is
    computed: a wrong variant, a truncated download or a re-export must not produce a
    manifest that later stages trust.
    """
    digests: dict[str, tuple[str, str]] = {}
    problems: list[str] = []
    for entry in provenance["files"]:
        name = entry["file"]
        path = raw_dir / name
        if not path.exists():
            problems.append(f"{name}: missing at {path}")
            continue
        size = path.stat().st_size
        if size != entry["bytes"]:
            problems.append(f"{name}: {size} bytes on disk, pinned snapshot has "
                            f"{entry['bytes']}")
            continue
        md5, sha = file_digests(path)
        if sha != entry["expected_sha256"]:
            problems.append(f"{name}: sha256 {sha} != pinned {entry['expected_sha256']}")
            continue
        digests[name] = (md5, sha)
    if problems:
        raise SystemExit("snapshot gate failed - nothing written:\n  - "
                         + "\n  - ".join(problems))
    return digests


def _values_equal(left, right) -> bool:
    """Value equality that treats two NaNs as equal (so a collision is real, not noise)."""
    try:
        if pd.isna(left) and pd.isna(right):
            return True
    except (TypeError, ValueError):
        pass
    return bool(left == right)


def clean_flow_table(path: Path, max_rows: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Read one pinned flow table, drop padding/duplicates, rename to canonical columns.

    Returns (frame, report). The report is what the manifest's per-file block is built
    from, so every dropped row is accounted for:

      * `padding_rows_dropped` - rows where EVERY column is NaN. This is how the pinned
        snapshot stores the Thursday-Morning-WebAttacks padding (288,602 rows); they are
        all-NaN, so binning them would either raise on the missing timestamp or invent a
        state at epoch 0. Dropped, counted, never zero-filled or imputed.
      * `duplicates_dropped` - rows removed because an EARLIER row in the same file has
        the same value in every column this pipeline consumes plus `Flow ID`. That is NOT
        the same as "exact duplicate", so both counts are reported:
          - `duplicates_exact`: the removed row is identical to its predecessor in ALL 85
            exported columns (203 rows over the pinned snapshot);
          - `duplicates_consumed_key_collisions`: the removed row agrees on every consumed
            column + Flow ID but differs in at least one column this pipeline never reads
            (1 row, in Wednesday). `duplicates_collision_details` names those columns.
        ACCURATE IMPACT STATEMENT, taken from a measurement on this snapshot rather than
        from an argument (`measure_collision_impact` re-attaches the row, re-bins both
        variants with the real code and diffs the 39 model columns). Dropping the
        Wednesday collision changed 23 of the 39 model inputs in exactly one bin:
          - `n_flows` -1, and with it the two derived ratios `packets_per_flow` and
            `connection_repetition` (n_flows / n_edges, where n_edges is unchanged);
          - `cross_subnet_edges` -1 (a sum of a per-flow boolean);
          - EVERY one of the 17 mean-aggregated columns moved, because removing a copy
            from the averaging set shifts the average even though the value itself is
            identical. The deltas are tiny but real (e.g. flow_duration_ms
            65583994.511 -> 65590009.184 microseconds) and an earlier draft of this
            docstring wrongly claimed the means were unchanged;
          - the sum-aggregated columns move ONLY where the duplicated value is non-zero:
            `total_fwd_packets` and `ack_flag_count` moved; the other eight sums are 0 in
            that bin and are therefore unchanged;
          - unchanged: `flow_iat_max` (the kept copy has the same value), and every
            nunique-based column - `unique_dst_ports_per_src`, `src_fanout`,
            `unique_dst_hosts`, `unique_src_hosts`, `dst_fanin`, `new_host_edges`,
            `new_dst_ports` - since an identical copy adds no new distinct key.
        `label_flows` also moved by one (the dropped row is a DoS Hulk flow, and
        label_flows counts non-BENIGN flows), so the cleaned histogram for that family is
        one lower and the anchor comparison observes one fewer flow. The anchor RESULT is
        unchanged - DoS Hulk observes 231,071 against an expected 231,000, +0.031 %, far
        inside +/-3 % - but the observed number is not the pre-dedup number, and the
        manifest now reports both the deltas and the pre/post label counts.
        The POLICY is still to drop the collision: the bin genuinely records the same flow
        twice. What was wrong before is the claim that the drop could not change anything.
      * `unmapped_labels` - raw labels outside the pinned vocabulary, collected over the
        whole file so the failure names all of them. Non-empty makes the run fail.

    Every column is read (not just the consumed ones) because the exact-duplicate count
    is only meaningful against the full exported row; that is what separates a true
    duplicate from a consumed-key collision.
    """
    try:
        full = pd.read_parquet(path)
    except Exception as exc:  # a wrong variant: missing keys or feature columns
        raise SystemExit(
            f"{path.name}: cannot read the pinned CICFlowMeter columns ({exc})"
        )
    missing = [column for column in CICIDS2017_COLUMN_MAP if column not in full.columns]
    if missing:
        raise SystemExit(f"{path.name}: pinned table is missing column(s): "
                         + ", ".join(sorted(missing)))

    rows_raw = int(len(full))
    if max_rows is not None:
        full = full.head(int(max_rows))

    padding = full.isna().all(axis=1)
    padding_dropped = int(padding.sum())
    full = full.loc[~padding]

    consumed_columns = list(CICIDS2017_COLUMN_MAP) + [DUPLICATE_KEY_COLUMN]
    consumed_mask = full[consumed_columns].duplicated()
    exact_mask = full.duplicated()
    duplicates_dropped = int(consumed_mask.sum())
    duplicates_exact = int((consumed_mask & exact_mask).sum())
    collision_positions = [position for position in full.index[consumed_mask & ~exact_mask]]

    frame = full.loc[~consumed_mask, list(CICIDS2017_COLUMN_MAP)]
    frame = frame.rename(columns=CICIDS2017_COLUMN_MAP)

    collision_details: list[dict] = []
    if collision_positions:
        # An explicit row-id column, because merge() resets the index: comparing the
        # merged frame's position against the original label silently compares the wrong
        # rows (it reported "kept row 0" and ~48 differing columns for a pair that
        # differs in one unconsumed field).
        work = full.reset_index(drop=False).rename(columns={"index": "__row_id"})
        keys = work.loc[work["__row_id"].isin(collision_positions),
                        consumed_columns].drop_duplicates()
        neighbourhood = work.merge(keys, on=consumed_columns, how="inner", sort=False)
        for later in collision_positions:
            key = work.loc[work["__row_id"] == later, consumed_columns]
            same_key = neighbourhood.merge(key, on=consumed_columns, how="inner")
            earlier = sorted(int(value) for value in same_key["__row_id"]
                            if int(value) != later)
            if not earlier:
                continue
            kept = full.loc[earlier[0]]
            dropped = full.loc[later]
            differing = [column for column in full.columns
                         if not _values_equal(kept[column], dropped[column])]
            collision_details.append({
                "row_index": int(later),
                "kept_row_index": earlier[0],
                "raw_label": str(dropped["Label"]),
                "differing_columns": differing,
                "differing_columns_are_consumed": bool(
                    set(differing) & set(CICIDS2017_COLUMN_MAP)
                ),
            })

    duplicate_families = full.loc[consumed_mask, "Label"].astype(str).value_counts().to_dict()
    # The consumed values of every collision row, so the caller can re-attach them and
    # MEASURE the before/after on the real binning code instead of asserting an impact.
    # Never reaches the manifest: build_manifest copies named fields only.
    collision_rows = (full.loc[collision_positions, list(CICIDS2017_COLUMN_MAP)]
                      .rename(columns=CICIDS2017_COLUMN_MAP).reset_index(drop=True))

    families, unmapped = normalise_cicids2017_labels(frame["label"].tolist())
    report = {
        "name": path.name,
        "rows_raw": rows_raw,
        "padding_rows_dropped": padding_dropped,
        "duplicates_dropped": duplicates_dropped,
        "duplicates_exact": duplicates_exact,
        "duplicates_consumed_key_collisions": duplicates_dropped - duplicates_exact,
        "duplicates_collision_details": collision_details,
        "duplicates_by_raw_label": {str(k): int(v) for k, v in duplicate_families.items()},
        "rows_after_clean": int(len(frame)),
        "unmapped_labels": unmapped,
        "label_histogram": {
            str(k): int(v)
            for k, v in pd.Series(families).value_counts().items()
        },
        "_collision_rows": collision_rows,
    }
    return frame, report


def measure_collision_impact(kept_frame: pd.DataFrame, collision_rows: pd.DataFrame,
                             bin_seconds: int) -> dict:
    """Measure, on the real binning code, what dropping the collision rows changed.

    The rows are re-attached and both variants are passed through the same
    `build_host_time_bins`, then the 39 model columns are diffed per (host, bin). This
    exists because the honest answer was previously asserted rather than measured, and it
    was asserted wrongly: the two rows are identical in every CONSUMED column (they differ
    only in a column this pipeline never reads, which is what makes them a collision rather
    than a true duplicate), so removing one copy moves every sum-aggregated counter in that
    bin by one copy of its value, shifts every mean, and also moves the count-derived
    columns. Reporting the measured deltas replaces the argument.
    """
    if collision_rows is None or len(collision_rows) == 0:
        return {"measured": False, "reason": "no consumed-key collision rows"}
    before = build_host_time_bins(
        pd.concat([kept_frame, collision_rows], ignore_index=True), bin_seconds=bin_seconds)
    after = build_host_time_bins(kept_frame, bin_seconds=bin_seconds)
    key = ["src_ip", "bin_index"]
    merged = before.merge(after, on=key, suffixes=("_before", "_after"))
    changed: dict[str, int] = {}
    for column in FLOW_TIER_FEATURE_LIST:
        # the merged frame carries the suffixed names, not the bare ones
        if f"{column}_before" not in merged.columns:
            continue
        left = merged[f"{column}_before"]
        right = merged[f"{column}_after"]
        differs = [
            index for index in merged.index
            if not _values_equal(left[index], right[index])
        ]
        if differs:
            sample = differs[0]
            changed[column] = {
                "bins_changed": len(differs),
                "example_before": _jsonable(left[sample]),
                "example_after": _jsonable(right[sample]),
            }
    label_changed = [
        index for index in merged.index
        if merged[f"label_before"][index] != merged[f"label_after"][index]
    ]
    return {
        "measured": True,
        "method": "collision rows re-attached, both variants re-binned, 39 model columns "
                  "diffed per (src_ip, bin_index)",
        "model_inputs_changed": changed,
        "model_inputs_changed_count": len(changed),
        "model_inputs_unchanged_count": len(FLOW_TIER_FEATURE_LIST) - len(changed),
        "bin_labels_changed": len(label_changed),
        "label_flows_changed": int(
            (merged["label_flows_before"] != merged["label_flows_after"]).sum()
        ) if "label_flows_before" in merged.columns else 0,
    }


def _jsonable(value):
    """A JSON-serialisable form of one cell (numpy scalars and NaN included)."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


# ---------------------------------------------------------------------------
# Anchor checks
# ---------------------------------------------------------------------------


def measure_nan_exposure(bins, sequences) -> dict:
    """Post-aggregation missingness: NaN cells per feature, bins affected, X affected.

    Published in the manifest so the WP3 handoff is durable rather than living only in a
    review conversation. Two stages matter and they are very different numbers:
      * the BIN table, where an all-NaN sum group (or a mean over no known value) stays NaN
        because `_aggregate_bins` uses min_count=1 - "not computable" is visible in the
        artifact instead of masquerading as 0;
      * `sequences.X`, which carries those values forward verbatim (`make_sequences` does
        `group[features].to_numpy(dtype=np.float32)` with no fill), so a bin with a NaN
        poisons its whole window unless WP3 handles it.
    """
    features = list(FLOW_TIER_FEATURE_LIST)
    by_feature = {
        column: int(bins[column].isna().sum()) for column in features
    }
    by_feature = {column: count for column, count in by_feature.items() if count}
    bins_with_nan = bins[features].isna().any(axis=1)
    X = sequences.X
    if X.size:
        sequence_nan = int(np.isnan(X).sum())
        sequences_with_nan = int(np.isnan(X).any(axis=(1, 2)).sum())
    else:
        sequence_nan = 0
        sequences_with_nan = 0
    return {
        "measured": True,
        "bins_total": int(len(bins)),
        "bin_nan_cells_total": int(sum(by_feature.values())),
        "bin_nan_cells_by_feature": by_feature,
        "bins_with_at_least_one_nan": int(bins_with_nan.sum()),
        "bins_with_nan_fraction": (
            round(float(bins_with_nan.mean()), 8) if len(bins) else 0.0
        ),
        "sequences_total": int(sequences.n_sequences),
        "sequence_x_nan_cells": sequence_nan,
        "sequences_with_at_least_one_nan": sequences_with_nan,
        "policy": "NaN is preserved end to end; nothing is zero-filled. SUM columns use "
                  "min_count=1 so an all-NaN group stays NaN rather than becoming 0.",
        "nan_policy": {
            "required": "impute_from_training_split_statistics",
            "imputation_fit_on": "the TRAINING split only, per P-A/P-B chronology; a test "
                                 "bin must never contribute an imputation statistic",
            "finiteness_assertion": "np.isfinite(X).all() AFTER imputation, on every "
                                    "batch - not before, because the artifact "
                                    "deliberately contains NaN",
            "accepted_alternative": "drop the affected sequences and record the count in "
                                    "the checkpoint provenance; requires a deliberate "
                                    "decision, not a fallback",
            "rejected": "nanmean/nanstd alone. NaN-safe statistics do NOT make a "
                        "Transformer's input or loss finite: the NaN still enters the "
                        "attention product and the output projection, and a single NaN "
                        "propagates to every timestep it touches. Filling train_mean / "
                        "train_std is necessary but not sufficient.",
        },
    }


def check_label_anchors(observed: dict[str, int]) -> list[dict]:
    """Raw family histogram vs the published CIC counts.

    Families the plan quotes as exact (Infiltration 36, SQLi 21, Heartbleed 11) are
    compared exactly: +/-3 % of 11 flows is less than one flow, so a percentage
    tolerance would be meaningless there.
    """
    anchors: list[dict] = []
    for family, (expected, exact) in LABEL_ANCHORS.items():
        seen = int(observed.get(family, 0))
        deviation = ((seen - expected) / expected * 100.0) if expected else float("inf")
        within = (seen == expected) if exact else (abs(deviation) <= LABEL_ANCHOR_TOLERANCE_PCT)
        anchors.append({
            "family": family,
            "expected": expected,
            "observed": seen,
            "deviation_pct": round(deviation, 3),
            "tolerance_pct": 0.0 if exact else LABEL_ANCHOR_TOLERANCE_PCT,
            "exact": bool(exact),
            "within_tolerance": bool(within),
        })
    return anchors


def check_file_rows(provenance: dict, reports: list[dict]) -> list[dict]:
    """Per-file row counts vs the provenance record, at the same +/-3 %."""
    observed_by_name = {report["name"]: int(report["rows_raw"]) for report in reports}
    checks: list[dict] = []
    for entry in provenance["files"]:
        name = entry["file"]
        expected = int(entry["rows"])
        seen = observed_by_name.get(name, 0)
        deviation = ((seen - expected) / expected * 100.0) if expected else float("inf")
        checks.append({
            "name": name,
            "expected_rows": expected,
            "observed_rows": seen,
            "deviation_pct": round(deviation, 3),
            "tolerance_pct": LABEL_ANCHOR_TOLERANCE_PCT,
            "within_tolerance": bool(abs(deviation) <= LABEL_ANCHOR_TOLERANCE_PCT),
        })
    return checks


def summarise_stage_histogram(bins) -> dict[str, int]:
    counts = bins["label"].value_counts()
    return {str(stage): int(count) for stage, count in counts.items()}


def summarise_per_day(bins) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    rows_per_day = {str(day): int(count) for day, count in bins["day"].value_counts().items()}
    per_day: dict[str, dict[str, int]] = {}
    for day, group in bins.groupby("day", sort=True):
        per_day[str(day)] = {
            str(stage): int(count) for stage, count in group["label"].value_counts().items()
        }
    return rows_per_day, per_day


def summarise_horizons(sequences) -> dict[str, dict[str, int]]:
    """Per-horizon target counts, with the two invariants made explicit.

    `valid_windows` is the SAME number for every horizon and is not a per-horizon
    filter: make_sequences only emits a window when the bins for t+1, t+2 and t+4 all
    exist (W + max(horizons) contiguous bins), so there is one set of windows and every
    horizon is scored on all of them. The attack counts must be non-decreasing in k,
    because `attack_within_k` is an OR over more bins - a window that is benign at t+1 can
    still be an attack at t+4. That is checked here and raises if violated, rather than
    being printed as two unrelated numbers that look contradictory.
    """
    windows = sequences.n_sequences
    counts: dict[str, dict[str, int]] = {}
    for horizon in sequences.horizons:
        attack = sequences.y_attack_within_h[horizon]
        positives = int(attack.sum())
        marked = int((sequences.y_stage_h[horizon] == STAGE_TARGET_IGNORE).sum())
        counts[str(horizon)] = {
            "valid_windows": windows,
            "attack_within_positive": positives,
            "attack_within_negative": windows - positives,
            "stage_targets": windows,
            "stage_marked_ignored_not_yet_masked": marked,
        }
    positives = [counts[str(horizon)]["attack_within_positive"]
                 for horizon in sequences.horizons]
    if any(later < earlier for earlier, later in zip(positives, positives[1:])):
        raise ValueError(
            "attack_within counts must be non-decreasing in k, got "
            f"{dict(zip(sequences.horizons, positives))}"
        )
    return counts


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _relative(path: Path) -> str:
    """Path relative to the repo root when possible, so the manifest is portable."""
    try:
        return path.resolve().relative_to(ROOT.parent.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def run_pipeline(raw_dir: Path, provenance: dict, bin_seconds: int,
                 max_rows: int | None) -> tuple[pd.DataFrame, object, list[dict], dict]:
    """Clean every pinned file, bin it, and build the sequences.

    Returns (bins, sequences, per-file reports, extras). No file is written here, so
    `--check` can run the whole thing and still leave the tree untouched.
    """
    bin_frames: list[pd.DataFrame] = []
    reports: list[dict] = []
    unmapped_labels: list[str] = []
    infinite_cells = 0
    missing_cells = 0
    collision_impact: dict = {"measured": False,
                              "reason": "no consumed-key collision rows in this run"}

    for entry in provenance["files"]:
        frame, report = clean_flow_table(raw_dir / entry["file"], max_rows=max_rows)
        unmapped_labels.extend(report["unmapped_labels"])
        collision_rows = report.pop("_collision_rows", None)
        if frame.empty:
            report["bins"] = 0
            reports.append(report)
            continue
        bins = build_host_time_bins(frame, bin_seconds=bin_seconds, entity="src_ip")
        infinite_cells += int(bins.attrs.get("infinite_cells", 0))
        missing_cells += int(bins.attrs.get("missing_cells", 0))
        report["bins"] = int(len(bins))
        report["hosts"] = int(bins["src_ip"].nunique())
        report["bin_seconds_min"] = (int(bins["bin_index"].min()),
                                     int(bins["bin_index"].max()))
        if collision_rows is not None and len(collision_rows):
            # Measure the real before/after of the dedup policy on THIS file.
            impact = measure_collision_impact(frame, collision_rows, bin_seconds)
            impact["file"] = entry["file"]
            collision_impact = impact
        reports.append(report)
        bin_frames.append(bins)

    if unmapped_labels:
        raise SystemExit(
            "unmapped CIC-IDS-2017 label(s) - refusing to relabel them BENIGN:\n  - "
            + "\n  - ".join(sorted(set(unmapped_labels)))
        )
    if not bin_frames:
        raise SystemExit("no bins were produced from the pinned snapshot")

    all_bins = pd.concat(bin_frames, ignore_index=True)
    all_bins = all_bins.sort_values(["src_ip", "bin_index"]).reset_index(drop=True)
    sequences = make_sequences(all_bins, W=W_SEQUENCE, horizons=HORIZONS,
                               stride=STRIDE, bin_seconds=bin_seconds)
    extras = {
        "unmapped_labels": sorted(set(unmapped_labels)),
        "infinite_cells": infinite_cells,
        "missing_cells": missing_cells,
        "collision_impact": collision_impact,
    }
    return all_bins, sequences, reports, extras


def build_manifest(bins, sequences, reports, extras, provenance: dict,
                   digests: dict[str, tuple[str, str]], bin_seconds: int,
                   max_rows: int | None, raw_dir: Path) -> dict:
    """The manifest the later stages are allowed to trust (plan 2.2 field list)."""
    rows_per_day, label_histogram_per_day = summarise_per_day(bins)

    raw_histogram: dict[str, int] = {}
    for report in reports:
        for family, count in report["label_histogram"].items():
            raw_histogram[family] = raw_histogram.get(family, 0) + int(count)

    anchors = check_label_anchors(raw_histogram)
    file_rows = check_file_rows(provenance, reports)

    files: list[dict] = []
    for report, entry in zip(reports, provenance["files"]):
        md5, sha = digests[entry["file"]]
        files.append({
            "name": report["name"],
            "md5": md5,
            "sha256": sha,
            "bytes": int(entry["bytes"]),
            "rows_raw": report["rows_raw"],
            "rows_after_clean": report["rows_after_clean"],
            "rows_provenance": int(entry["rows"]),
            "padding_rows_dropped": report["padding_rows_dropped"],
            "duplicates_dropped": report["duplicates_dropped"],
            "duplicates_exact": report["duplicates_exact"],
            "duplicates_consumed_key_collisions": report["duplicates_consumed_key_collisions"],
            "duplicates_collision_details": report["duplicates_collision_details"],
            "duplicates_by_raw_label": report["duplicates_by_raw_label"],
            "bins": int(report.get("bins", 0)),
            "hosts": int(report.get("hosts", 0)),
            "label_histogram": report["label_histogram"],
        })

    manifest = {
        "dataset": "CIC-IDS-2017",
        "variant": "flow_tier_host_time_bins",
        "schema": "v2 (24 flow + 6 extended + 8 topology + n_flows = 39 columns)",
        "source_dir": _relative(raw_dir),
        "source_provenance": {
            "record": _relative(PROVENANCE_PATH),
            "record_sha256": hashlib.sha256(PROVENANCE_PATH.read_bytes()).hexdigest(),
            "source": provenance.get("source", {}),
            "retrieved_utc": provenance.get("retrieved_utc"),
            "observed_anomalies": provenance.get("observed_anomalies", []),
        },
        "bin_seconds": int(bin_seconds),
        "W": W_SEQUENCE,
        "horizons": list(HORIZONS),
        "stride": STRIDE,
        "truncated": max_rows is not None,
        "truncated_max_rows": max_rows,
        "anchors_enforced": max_rows is None,
        "files": files,
        "rows_raw_total": int(sum(f["rows_raw"] for f in files)),
        "rows_after_clean_total": int(sum(f["rows_after_clean"] for f in files)),
        "padding_rows_dropped": int(sum(f["padding_rows_dropped"] for f in files)),
        "duplicates_removed": int(sum(f["duplicates_dropped"] for f in files)),
        "duplicates_exact_rows": int(sum(f["duplicates_exact"] for f in files)),
        "duplicates_consumed_key_collisions": int(
            sum(f["duplicates_consumed_key_collisions"] for f in files)),
        "duplicates_collision_details": [
            {"file": f["name"], **detail}
            for f in files for detail in f["duplicates_collision_details"]
        ],
        "duplicates_key_note": (
            "rows are dropped when an earlier row matches on every consumed "
            "CICFlowMeter column plus Flow ID. duplicates_exact_rows are identical in ALL "
            "85 exported columns. A consumed-key collision agrees on every consumed column "
            "but differs in at least one column this pipeline never reads (listed per row "
            "in duplicates_collision_details). Dropping it is NOT a no-op. Measured on the "
            "pinned snapshot it moved 23 of the 39 model inputs in one bin: n_flows, both "
            "derived ratios, cross_subnet_edges, the two non-zero sums, and all 17 "
            "mean-aggregated columns (removing a copy shifts the average). Max and every "
            "nunique-based column were unchanged. label_flows moved by one, so the "
            "observed count the anchor check compares is one lower than pre-dedup. The "
            "exact deltas are in duplicates_collision_impact; see clean_flow_table."
        ),
        "unmapped_labels": extras["unmapped_labels"],
        "hosts": int(bins["src_ip"].nunique()),
        "bins": int(len(bins)),
        "rows_per_day": rows_per_day,
        "label_histogram": raw_histogram,
        "label_histogram_provenance": provenance.get("totals", {}).get("label_histogram", {}),
        "label_histogram_per_day": label_histogram_per_day,
        "bin_label_histogram": summarise_stage_histogram(bins),
        "stage_valid_bins": int(bins["stage_valid"].sum()),
        "stage_ignored_bins": int((~bins["stage_valid"]).sum()),
        # Two separate counts, deliberately: an infinity is a value this pipeline
        # destroyed (inf -> NaN); a NaN already in the export is a value it never had.
        # An earlier single `non_finite_feature_cells_dropped` conflated them and used
        # np.isinf, so it silently omitted pre-existing missingness.
        "infinite_feature_cells_replaced_with_nan": extras["infinite_cells"],
        "feature_cells_missing_in_source": extras["missing_cells"],
        "sum_policy": "SUM columns use min_count=1, so an all-NaN group stays NaN and is "
                      "never reported as 0",
        "nan_exposure": measure_nan_exposure(bins, sequences),
        "duplicates_collision_impact": extras["collision_impact"],
        "sequences": sequences.n_sequences,
        "sequences_per_horizon": {
            str(k): int(len(v)) for k, v in sequences.y_stage_h.items()
        },
        "target_counts_per_horizon": summarise_horizons(sequences),
        "horizon_notes": (
            "sequences_per_horizon is the TOTAL number of valid windows and is identical "
            "for every horizon: make_sequences only emits a window when the bins for "
            "t+1, t+2 AND t+4 all exist, so no horizon filters the window set. "
            "target_counts_per_horizon then breaks those same windows down; "
            "attack_within_positive is non-decreasing in k by construction (an OR over "
            "more bins), which summarise_horizons asserts."
        ),
        "stage_target_ignore_index": STAGE_TARGET_IGNORE,
        "stage_target_masking": {
            "status": "required, NOT implemented (WP3)",
            "where": [
                "ml/world_model/model.py:324 (nn.CrossEntropyLoss(), default ignore_index=-100)",
                "ml/world_model/model.py:342 (primary stage loss)",
                "ml/world_model/model.py:368 (horizon stage loss)",
            ],
            "why": "the stage losses do not ignore the -1 markers, so a training run "
                   "started before WP3 reads them as class indices instead of skipping "
                   "them; these samples are excluded from the targets in the DATA only",
            "fix": "pass ignore_index=stage_target_ignore_index, or apply "
                   "BinnedSequences.stage_target_masks(), in both stage losses",
            "wp3_action": "add the masking in model.py AND replace the TODO(WP3) guard "
                          "test in tests/test_prepare_cicids2017.py in the same commit, "
                          "so the suite is never red on a correct WP3",
            "all_ignored_batch": "WP3 must also handle the fully-ignored batch. "
                                 "nn.CrossEntropyLoss(ignore_index=-1) returns NaN when "
                                 "EVERY target in the batch is the ignore marker, because "
                                 "the reduction divides by zero. With 24-25 marked "
                                 "stage targets in 17,086 windows a small batch can be all "
                                 "marker. WP3 must guard it (skip the step, or contribute "
                                 "a zero term) and assert the loss is finite on an "
                                 "all-marker batch. This is independent of the binary "
                                 "attack targets, which carry no ignore marker and must "
                                 "keep being tested.",
        },
        "feature_list": list(FLOW_TIER_FEATURE_LIST),
        "excluded_from_stage_head": sorted(STAGE_EXCLUDED_FAMILIES),
        "excluded_from_stage_head_enforced_by_loss": False,
        "label_anchors": anchors,
        "label_anchor_violations": [
            a["family"] for a in anchors if not a["within_tolerance"]
        ],
        "file_row_checks": file_rows,
        "file_row_violations": [
            c["name"] for c in file_rows if not c["within_tolerance"]
        ],
        "code": code_provenance(),
    }
    return manifest


def print_report(manifest: dict) -> None:
    """Human-readable manifest (per-day rows, label histogram, hosts, sequences)."""
    line = "-" * 78
    print(line)
    print("CIC-IDS-2017 flow tier -> host/time bins   (plan D1, WP2)")
    print(line)
    print(f"  source dir       : {manifest['source_dir']}")
    print(f"  grid             : bin_seconds={manifest['bin_seconds']}  "
          f"W={manifest['W']}  horizons={manifest['horizons']}  "
          f"stride={manifest['stride']}")
    print(f"  rows raw         : {manifest['rows_raw_total']:,}")
    print(f"  rows after clean : {manifest['rows_after_clean_total']:,}  "
          f"(padding dropped {manifest['padding_rows_dropped']:,}, "
          f"duplicate rows dropped {manifest['duplicates_removed']:,})")
    print(f"  duplicates       : {manifest['duplicates_removed']:,} rows = "
          f"{manifest['duplicates_exact_rows']:,} identical in all 85 exported columns + "
          f"{manifest['duplicates_consumed_key_collisions']:,} consumed-key collision(s)")
    for detail in manifest["duplicates_collision_details"]:
        print(f"      collision: {detail['file']} row {detail['row_index']} "
              f"(kept row {detail['kept_row_index']}), label {detail['raw_label']!r}, "
              f"differs only in unconsumed column(s): "
              f"{', '.join(detail['differing_columns'])}")
    print(f"  hosts / bins     : {manifest['hosts']:,} / {manifest['bins']:,}")
    print(f"  sequences        : {manifest['sequences']:,}")
    code = manifest["code"]
    print(f"  code             : {code['git_sha'][:7]} is the BASE commit; pipeline code is "
          f"{'committed + clean' if code['reproducible_from_git_sha'] else 'UNCOMMITTED or DIRTY'}"
          f" (git_dirty={code['git_dirty']}, argus_dirty={code['git_dirty_under_argus']})")
    for name, info in code["pipeline_files"].items():
        print(f"      [{info['git_status'] or 'clean':<2}] "
              f"{(info['sha256'] or '-')[:16]}  {name}")
    if manifest["truncated"]:
        print(f"  TRUNCATED run    : --max-rows {manifest['truncated_max_rows']} "
              "(anchors recorded, NOT enforced)")

    print("\nper-file rows:")
    for entry in manifest["files"]:
        print(f"    {entry['rows_raw']:>9,} -> {entry['rows_after_clean']:>9,} rows "
              f"{entry['bins']:>7,} bins  {entry['name']}")

    print("\nper-day bins (stage labels of the bins):")
    for day in sorted(manifest["label_histogram_per_day"]):
        histogram = manifest["label_histogram_per_day"][day]
        stages = ", ".join(f"{stage}={count:,}" for stage, count
                           in sorted(histogram.items(), key=lambda kv: -kv[1]))
        print(f"    {day}  bins={manifest['rows_per_day'][day]:>7,}  {stages}")

    print("\nraw label histogram (CIC families, before stage merging):")
    for family, count in sorted(manifest["label_histogram"].items(),
                               key=lambda kv: -kv[1]):
        print(f"    {count:>9,}  {family}")

    print(f"\nlabel anchors (+/-{LABEL_ANCHOR_TOLERANCE_PCT:g} %, exact for the three "
          "small families):")
    for anchor in manifest["label_anchors"]:
        flag = "ok  " if anchor["within_tolerance"] else "FAIL"
        print(f"    {flag} {anchor['family']:<26} observed {anchor['observed']:>9,}  "
              f"expected {anchor['expected']:>9,}  dev {anchor['deviation_pct']:+.2f} %"
              + ("  (exact)" if anchor["exact"] else ""))

    print("\nbin stage labels (what the stage head would see):")
    for stage, count in sorted(manifest["bin_label_histogram"].items(),
                               key=lambda kv: -kv[1]):
        print(f"    {count:>9,}  {stage}")
    print(f"    stage_valid=False (binary-only; y_stage_* carries the "
          f"{manifest['stage_target_ignore_index']} marker, which the loss does NOT mask "
          f"yet - WP3): {manifest['stage_ignored_bins']:,} bins of the families "
          f"{', '.join(manifest['excluded_from_stage_head'])}")

    print("\nsequences per horizon (one window set, scored at every horizon: a window is")
    print("emitted only if the t+1, t+2 AND t+4 bins all exist, so the window count is the")
    print("same; attack_within is an OR over more bins, so positives never decrease in k):")
    print("      k   valid windows   attack_within=1   attack_within=0   stage marked (-1)")
    for horizon, row in manifest["target_counts_per_horizon"].items():
        print(f"    {horizon:>4}   {row['valid_windows']:>13,}   "
              f"{row['attack_within_positive']:>16,}   "
              f"{row['attack_within_negative']:>16,}   "
              f"{row['stage_marked_ignored_not_yet_masked']:>16,}")

    print(f"\nfeature_list ({len(manifest['feature_list'])} columns):")
    print("    " + ", ".join(manifest["feature_list"]))

    print("\nmanifest (json):")
    print(json.dumps(manifest, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CIC-IDS-2017 flow tier -> host/time-binned Parquet + manifest "
                    "(plan D1/D2, WP2). Training, PCAPs and README numbers are "
                    "out of scope."
    )
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR),
                        help="directory holding the pinned flow tables "
                             f"(default: {DEFAULT_RAW_DIR})")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR),
                        help=f"output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--bin-seconds", type=int, default=60,
                        help="wall-clock bin width in seconds (default: 60)")
    parser.add_argument("--check", action="store_true",
                        help="print the manifest and exit without writing anything")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="read at most N rows per file (truncated smoke run; "
                             "anchors are recorded but not enforced)")
    args = parser.parse_args(argv)

    if args.bin_seconds <= 0:
        parser.error("--bin-seconds must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out)
    if not raw_dir.is_dir():
        raise SystemExit(
            f"raw dir not found: {raw_dir}\n"
            "The pinned CIC-IDS-2017 flow tables must exist there; this pipeline never "
            "substitutes data/sample_datasets/ for them."
        )

    provenance = load_provenance()
    digests = verify_snapshot(raw_dir, provenance)
    bins, sequences, reports, extras = run_pipeline(
        raw_dir, provenance, args.bin_seconds, args.max_rows)
    manifest = build_manifest(bins, sequences, reports, extras, provenance, digests,
                              args.bin_seconds, args.max_rows, raw_dir)
    print_report(manifest)

    violations = manifest["label_anchor_violations"] + manifest["file_row_violations"]
    if violations:
        if manifest["anchors_enforced"]:
            print("\nFAIL: anchor violation(s) - no parquet and no manifest written:")
            for family in manifest["label_anchor_violations"]:
                print(f"  - label family outside tolerance: {family}")
            for name in manifest["file_row_violations"]:
                print(f"  - per-file row count outside tolerance: {name}")
            return 1
        print("\nWARNING: truncated run (--max-rows): anchors recorded but not enforced.")

    if args.check:
        print("\n--check: nothing written (no parquet, no manifest). "
              "Drop --check to write both.")
        return 0

    suffix = "" if args.max_rows is None else f".maxrows{args.max_rows}"
    parquet_path = out_dir / f"cicids2017_bins_{args.bin_seconds}s{suffix}.parquet"
    manifest_path = out_dir / f"manifest{suffix}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    bins.to_parquet(parquet_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {parquet_path}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
