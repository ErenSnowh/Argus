"""
test_prepare_cicids2017.py
--------------------------
Tests for the WP2 flow-tier pipeline: host/time binning (`ml/world_model/binning.py`)
and the CLI that turns the pinned CIC-IDS-2017 flow tables into binned sequences plus a
manifest (`scripts/prepare_cicids2017.py`).

Every fixture here is built in-process. The pinned snapshot is 306 MB of gitignored
Parquet that a fresh clone does not have, so the suite applies the same rules to small
synthetic tables and only the CLI tests touch a (temporary, synthetic) raw directory.

Run: pytest argus/tests/test_prepare_cicids2017.py -q
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.world_model.binning import (  # noqa: E402
    FLOW_TIER_FEATURE_LIST,
    MAX_COLUMNS,
    MEAN_COLUMNS,
    STAGE_TARGET_IGNORE,
    SUM_COLUMNS,
    TOPOLOGY_FEATURE_COLUMNS,
    UnmappedLabelError,
    build_host_time_bins,
    make_sequences,
    normalise_cicids2017_label,
    normalise_cicids2017_labels,
)
from ml.world_model.features import (  # noqa: E402
    ATTACK_STAGE_INDEX,
    FLOW_FEATURE_COLUMNS,
    PACKET_LEVEL_COLUMNS,
)
from scripts.prepare_cicids2017 import (  # noqa: E402
    CICIDS2017_COLUMN_MAP,
    DUPLICATE_KEY_COLUMN,
    LABEL_ANCHORS,
    PorcelainParseError,
    check_label_anchors,
    clean_flow_table,
    main,
    measure_collision_impact,
    measure_nan_exposure,
    summarise_horizons,
)
import scripts.prepare_cicids2017 as prepare  # noqa: E402

DAY = "2017-07-04"
BASE = pd.Timestamp(f"{DAY}T09:00:00Z")

FEATURE_DEFAULTS = {column: 1.0 for column in (*SUM_COLUMNS, *MEAN_COLUMNS, *MAX_COLUMNS)}


def flows(spec, base: pd.Timestamp = BASE) -> pd.DataFrame:
    """Canonical flow table. `spec` rows are (minute_offset, src, dst, dst_port, label)."""
    rows = []
    for offset, src, dst, dst_port, label in spec:
        row = dict(FEATURE_DEFAULTS)
        row.update({
            "src_ip": src,
            "dst_ip": dst,
            "dst_port": dst_port,
            "timestamp": base + pd.Timedelta(minutes=offset),
            "label": label,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def run_of_bins(host: str, labels, base: pd.Timestamp = BASE, dst: str = "10.0.0.9",
                port: int = 80) -> pd.DataFrame:
    """One consecutive bin per label, one minute apart."""
    return flows([(index, host, dst, port, label) for index, label in enumerate(labels)],
                 base=base)


# ---------------------------------------------------------------------------
# (i) no sequence crosses a day or a host boundary
# ---------------------------------------------------------------------------


def test_no_sequence_crosses_a_day_or_host_boundary():
    # Host A runs straight through midnight (23:45 -> 00:08 on 4/5 July). Its bins are
    # contiguous in absolute time, so a builder that only checked absolute contiguity
    # would stitch 23:50..23:59 with 00:00..00:03 into one "10 minute history" and call
    # it a state transition. Host B is the control: 20 bins inside one day.
    midnight = pd.Timestamp("2017-07-04T23:45:00Z")
    host_a = run_of_bins("10.0.0.1", ["BENIGN"] * 24, base=midnight)
    host_b = run_of_bins("10.0.0.2", ["BENIGN"] * 20)
    bins = build_host_time_bins(pd.concat([host_a, host_b], ignore_index=True),
                               bin_seconds=60)

    assert bins["src_ip"].nunique() == 2
    assert bins.loc[bins["src_ip"] == "10.0.0.1", "day"].value_counts().to_dict() == {
        "2017-07-04": 15, "2017-07-05": 9,
    }

    sequences = make_sequences(bins, W=10, horizons=(1, 2, 4), bin_seconds=60)
    # 14 contiguous bins are needed for W=10 with max horizon 4, so host A yields 2
    # windows on 4 July and none on 5 July (9 bins), and host B yields 7. A builder that
    # ignored the day boundary would see 24 bins for host A and produce 11 windows.
    assert sequences.n_sequences == 9
    meta = sequences.meta
    assert (meta["bin_index_end"] - meta["bin_index_start"] == 9).all()
    assert (meta["bin_index_end"] == meta["bin_index"]).all()
    assert ((meta["t_start"] - meta["window_start"]) == pd.Timedelta(minutes=9)).all()
    assert (meta["window_start"].dt.strftime("%Y-%m-%d") == meta["day"]).all()
    assert (meta["t_start"].dt.strftime("%Y-%m-%d") == meta["day"]).all()
    assert not ((meta["host"] == "10.0.0.1") & (meta["day"] == "2017-07-05")).any()

    # Every sequence's ten bins exist, belong to that one host, and share its day.
    for row in meta.itertuples(index=False):
        window = bins[(bins["src_ip"] == row.host)
                      & bins["bin_index"].between(row.bin_index_start, row.bin_index_end)]
        assert len(window) == 10
        assert set(window["day"]) == {row.day}


def test_a_hole_in_the_run_ends_the_sequence():
    # 30 minutes with the bin at minute 12 deleted: 12 bins before the hole (too few for
    # W=10 + max horizon 4 = 14) and 17 after it, so only the second run yields windows.
    spec = [(minute, "10.0.0.1", "10.0.0.9", 80, "BENIGN") for minute in range(30)]
    spec = [row for row in spec if row[0] != 12]
    bins = build_host_time_bins(flows(spec), bin_seconds=60)
    sequences = make_sequences(bins, W=10, horizons=(1, 2, 4), bin_seconds=60)

    assert len(bins) == 29
    assert sequences.n_sequences == 4          # 17 - 14 + 1, and nothing before the hole
    hole = int(bins["bin_index"].iloc[0]) + 12
    for start, end in zip(sequences.meta["bin_index_start"],
                          sequences.meta["bin_index_end"]):
        assert not (start <= hole <= end)


# ---------------------------------------------------------------------------
# (ii) y_attack_within_h1 == (y_stage_h1 != 0)
# ---------------------------------------------------------------------------


def test_attack_within_h1_equals_stage_h1_nonzero():
    # 20 benign bins, then a 6-bin PortScan burst: long enough that every horizon has
    # bins on both sides of the transition.
    bins = build_host_time_bins(run_of_bins("10.0.0.1", ["BENIGN"] * 20 + ["PortScan"] * 6),
                               bin_seconds=60)
    sequences = make_sequences(bins, W=10, horizons=(1, 2, 4), bin_seconds=60)

    assert sequences.n_sequences == 13
    assert sequences.bin_seconds == 60                # explicit, never inferred
    # k=1: the OR covers exactly the single target bin, so this is an identity.
    assert np.array_equal(sequences.y_attack_within_h[1], (sequences.y_stage_h[1] != 0))
    assert sequences.y_attack_within_h[1].sum() >= 1
    assert sequences.y_stage_h[1].max() == ATTACK_STAGE_INDEX["PortScan"]
    # h=4 is NOT the same quantity and must not be asserted equal to the endpoint stage:
    # an attack at t+1 can be followed by a benign t+4. What must hold is monotonicity -
    # the OR runs over more bins, so h=4 positives dominate h=1.
    assert (sequences.y_attack_within_h[4] >= sequences.y_attack_within_h[1]).all()
    assert (sequences.y_attack_within_h[2] >= sequences.y_attack_within_h[1]).all()
    assert (sequences.y_attack_within_h[4] >= sequences.y_attack_within_h[2]).all()


def test_attack_within_h4_is_not_equivalent_to_the_stage_at_t_plus_4():
    # A one-bin attack: PortScan at t+1 only, benign again at t+2..t+4. The binary
    # "attack within 4 minutes" target is 1, while the stage AT t+4 is BENIGN - so
    # `y_attack_within_h4 == (y_stage_h4 != 0)` is false. An earlier version of this
    # suite asserted that equality on a longer burst, where it happened to hold by
    # accident; this fixture is the case that makes the difference observable.
    labels = ["BENIGN"] * 20 + ["PortScan"] + ["BENIGN"] * 4
    bins = build_host_time_bins(run_of_bins("10.0.0.1", labels), bin_seconds=60)
    sequences = make_sequences(bins, W=10, horizons=(1, 2, 4), bin_seconds=60)

    endpoint_benign = sequences.y_stage_h[4] == ATTACK_STAGE_INDEX["BENIGN"]
    assert endpoint_benign.any(), "fixture must contain a window whose t+4 bin is benign"
    # the windows this fixture is built for: the attack happened inside (t, t+4] but the
    # t+4 bin itself is benign again
    attack_inside_but_endpoint_benign = endpoint_benign & (sequences.y_attack_within_h[4] == 1)
    assert attack_inside_but_endpoint_benign.sum() >= 1
    # for those windows the two quantities DISAGREE, so they are not equivalent
    assert not np.array_equal(sequences.y_attack_within_h[4],
                              (sequences.y_stage_h[4] != 0))
    assert (sequences.y_stage_h[4][attack_inside_but_endpoint_benign] == 0).all()
    assert (sequences.y_attack_within_h[4][attack_inside_but_endpoint_benign] == 1).all()
    # ... and the k=1 identity is unaffected by that difference
    assert np.array_equal(sequences.y_attack_within_h[1], (sequences.y_stage_h[1] != 0))
    assert (sequences.y_attack_within_h[4] >= sequences.y_attack_within_h[1]).all()


def test_attack_within_h1_identity_survives_the_stage_exclusions():
    # A Heartbleed burst: y_stage_* carries the -1 marker (STAGE_TARGET_IGNORE) while the
    # bin stays an attack, and -1 != 0 keeps the identity true. The marker is NOT masked
    # by the current losses - that gap is asserted in the test at the end of this file.
    bins = build_host_time_bins(
        run_of_bins("10.0.0.1", ["BENIGN"] * 18 + ["Heartbleed"] * 2), bin_seconds=60)
    assert bins["label"].iloc[-1] == "WebAttack"                    # mapped stage kept
    assert not bool(bins["stage_valid"].iloc[-1])                   # marked, not masked
    assert bins["label_flows"].iloc[-1] == 1                        # ... and it IS malicious

    sequences = make_sequences(bins, W=10, horizons=(1, 2, 4), bin_seconds=60)
    assert np.array_equal(sequences.y_attack_within_h[1], (sequences.y_stage_h[1] != 0))
    ignored = np.flatnonzero(sequences.y_stage_h[4] == STAGE_TARGET_IGNORE)
    assert ignored.size >= 1
    assert sequences.y_attack_within_h[4][ignored].min() == 1
    assert ATTACK_STAGE_INDEX["LateralMovement"] not in set(sequences.y_stage_h[4].tolist())


# ---------------------------------------------------------------------------
# (iii) unmapped labels raise
# ---------------------------------------------------------------------------


def test_unmapped_labels_raise():
    with pytest.raises(UnmappedLabelError):
        normalise_cicids2017_label("Not an attack")
    with pytest.raises(UnmappedLabelError):
        normalise_cicids2017_label(None)

    # The historical bug this guards: every Web-Attack row in the pinned snapshot is
    # spelled with U+0096, which dataset_loader's map lacks, so those rows used to fall
    # through `.get(x, "BENIGN")` and become benign traffic.
    assert normalise_cicids2017_label("Web Attack \u0096 XSS") == "Web Attack - XSS"

    table = flows([(0, "10.0.0.1", "10.0.0.9", 80, "Not an attack")])
    with pytest.raises(UnmappedLabelError) as excinfo:
        build_host_time_bins(table, bin_seconds=60)
    assert "Not an attack" in str(excinfo.value)

    families, unmapped = normalise_cicids2017_labels(["BENIGN", "Nope", "Nope"])
    assert families[0] == "BENIGN"
    assert unmapped == ["Nope"]                 # distinct offenders, no exception raised


# ---------------------------------------------------------------------------
# D1 bin semantics
# ---------------------------------------------------------------------------


def test_bin_key_label_and_aggregation_follow_d1():
    spec = [
        (0, "10.0.0.1", "10.0.0.9", 80, "BENIGN"),
        (0, "10.0.0.1", "10.0.0.7", 443, "BENIGN"),
        (0, "10.0.0.1", "10.0.0.9", 22, "PortScan"),
        (1, "10.0.0.1", "10.0.0.9", 80, "BENIGN"),
    ]
    bins = build_host_time_bins(flows(spec), bin_seconds=60)
    assert list(bins["bin_index"]) == [int(BASE.timestamp()) // 60,
                                       int(BASE.timestamp()) // 60 + 1]

    first = bins.iloc[0]
    assert first["n_flows"] == 3
    assert first["unique_dst_ports_per_src"] == 3      # nunique(Destination Port)
    assert first["src_fanout"] == 2 and first["unique_dst_hosts"] == 2
    assert first["new_host_edges"] == 2 and first["new_dst_ports"] == 3
    assert first["cross_subnet_edges"] == 0
    assert first["connection_repetition"] == 1.0       # 3 flows, 3 distinct edges
    assert first["packets_per_flow"] == pytest.approx(
        (first["total_fwd_packets"] + first["total_bwd_packets"]) / 3)
    # most severe flow in the bin decides the label; BENIGN only if all are BENIGN
    assert first["label"] == "PortScan"
    assert first["label_index"] == ATTACK_STAGE_INDEX["PortScan"]
    assert first["label_flows"] == 1
    assert first["bin_start"] == BASE and first["day"] == DAY

    second = bins.iloc[1]                              # history is remembered per host+day
    assert second["new_host_edges"] == 0
    assert second["new_dst_ports"] == 0
    assert second["label"] == "BENIGN" and second["label_flows"] == 0


# ---------------------------------------------------------------------------
# The pipeline drops padding and duplicates - it never zero-fills
# ---------------------------------------------------------------------------

RAW_DEFAULTS = {
    raw: 1.0 for raw in CICIDS2017_COLUMN_MAP
    if raw not in ("Source IP", "Destination IP", "Destination Port", "Timestamp", "Label")
}


def raw_rows(spec, base: pd.Timestamp = BASE) -> pd.DataFrame:
    """Rows in the raw CICFlowMeter spelling, as the Parquet snapshot stores them."""
    rows = []
    for index, (offset, label) in enumerate(spec):
        row = dict(RAW_DEFAULTS)
        row.update({
            "Source IP": "10.0.0.1",
            "Destination IP": "10.0.0.9",
            "Destination Port": 80,
            "Timestamp": base + pd.Timedelta(minutes=offset),
            "Label": label,
            DUPLICATE_KEY_COLUMN: f"flow-{index}",      # unique per flow
        })
        rows.append(row)
    return pd.DataFrame(rows)


def test_padding_rows_are_dropped_and_duplicates_are_counted(tmp_path):
    real = raw_rows([(0, "BENIGN"), (1, "BENIGN"), (2, "FTP-Patator")])
    duplicated = real.iloc[[0]].copy()                   # same Flow ID -> same flow twice
    padding = pd.DataFrame({column: [None] * 5 for column in real.columns})
    name = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet"
    path = tmp_path / name
    pd.concat([real, duplicated, padding], ignore_index=True).to_parquet(path, index=False)

    frame, report = clean_flow_table(path)
    assert report["rows_raw"] == 9
    assert report["padding_rows_dropped"] == 5           # all-NaN rows, never filled
    assert report["duplicates_dropped"] == 1
    assert report["rows_after_clean"] == 3
    assert report["unmapped_labels"] == []
    assert report["label_histogram"] == {"BENIGN": 2, "FTP-Patator": 1}
    assert DUPLICATE_KEY_COLUMN not in frame.columns
    assert frame["timestamp"].isna().sum() == 0

    bins = build_host_time_bins(frame, bin_seconds=60)
    assert int(bins["n_flows"].sum()) == 3               # padding contributed nothing


# ---------------------------------------------------------------------------
# Git provenance: porcelain parsing and fail-closed behaviour
# ---------------------------------------------------------------------------


def test_porcelain_parser_handles_every_status_shape():
    # Porcelain v1 is "XY path". The two shapes a partition(" ") parser gets wrong are
    # " M path" (leading space: unstaged modification) and "R  old -> new".
    stdout = (
        " M argus/ml/world_model/binning.py\n"
        "M  argus/scripts/prepare_cicids2017.py\n"
        "MM argus/tests/test_prepare_cicids2017.py\n"
        "?? argus/data/processed/x.parquet\n"
        "R  argus/old_name.py -> argus/new_name.py\n"
        "A  argus/added.py\n"
    )
    parsed = prepare.parse_porcelain_statuses(stdout)

    assert parsed["argus/ml/world_model/binning.py"] == " M"
    assert parsed["argus/scripts/prepare_cicids2017.py"] == "M "
    assert parsed["argus/tests/test_prepare_cicids2017.py"] == "MM"
    assert parsed["argus/data/processed/x.parquet"] == "??"
    assert parsed["argus/new_name.py"] == "R "          # rename recorded under new path
    assert "argus/old_name.py" not in parsed
    assert parsed["argus/added.py"] == "A "
    assert all(path for path in parsed)                # nothing with an empty path


def test_porcelain_parser_strips_scope_and_keeps_leading_space_status():
    parsed = prepare.parse_porcelain_statuses(
        " M argus/ml/world_model/binning.py\n", strip_prefix="argus")
    assert parsed == {"ml/world_model/binning.py": " M"}


def test_porcelain_parser_raises_on_malformed_lines_in_strict_mode():
    # A malformed non-empty line must NOT be dropped: dropping it removes that file's
    # status token, and a tracked file with no token is reported "clean" downstream.
    for bad in ("?? \n", "M\n", "XY\n", "M  \n"):
        with pytest.raises(PorcelainParseError):
            prepare.parse_porcelain_statuses(bad, strict=True)
    # a separator that is not a space is malformed too
    with pytest.raises(PorcelainParseError):
        prepare.parse_porcelain_statuses("M?argus/x.py\n", strict=True)
    # blank lines are ignored, not errors
    assert prepare.parse_porcelain_statuses("\n\n  \n", strict=True) == {}
    # non-strict mode still drops them (kept only for callers that explicitly opt out)
    assert prepare.parse_porcelain_statuses("?? \n", strict=False) == {}


def test_git_status_normalises_root_prefixed_keys_for_a_tracked_modified_file(monkeypatch):
    """The blocker this test exists for: root-prefixed keys must not escape the lookup.

    An earlier version stripped the "argus/" prefix only when `scope` was passed, while
    `code_provenance` called `git_status()` with no scope. The keys therefore stayed
    "argus/ml/world_model/binning.py", `describe_git_status` looked up
    "ml/world_model/binning.py", found no token, and - because the file WAS tracked -
    reported it clean, so a MODIFIED tracked file was published as committed and
    reproducible. The test below drives real git-shaped output (root-prefixed paths) through
    the real `git_status()` and `code_provenance()` rather than pre-stripped fixtures.
    """
    root_status = (
        " M argus/ml/world_model/binning.py\n"
        "M  argus/scripts/prepare_cicids2017.py\n"
        "M  argus/tests/test_prepare_cicids2017.py\n"
    )

    def fake_run_git(arguments, timeout=30):
        if arguments[0] == "ls-files":
            stdout = "".join(f"argus/{name}\n" for name in
                             ("ml/world_model/binning.py",
                              "scripts/prepare_cicids2017.py",
                              "tests/test_prepare_cicids2017.py"))
        else:
            stdout = root_status
        return subprocess.CompletedProcess(args=arguments, returncode=0,
                                           stdout=stdout, stderr="")

    monkeypatch.setattr(prepare, "_run_git", fake_run_git)
    monkeypatch.setattr(prepare, "git_sha", lambda: "b" * 40)

    statuses, error = prepare.git_status()
    assert error is None
    # keys are argus-relative, which is what describe_git_status looks up
    assert set(statuses) == {"ml/world_model/binning.py",
                             "scripts/prepare_cicids2017.py",
                             "tests/test_prepare_cicids2017.py"}
    assert statuses["ml/world_model/binning.py"] == " M"

    code = prepare.code_provenance()
    for name in ("ml/world_model/binning.py", "scripts/prepare_cicids2017.py"):
        assert code["pipeline_files"][name]["tracked"] is True
        assert code["pipeline_files"][name]["git_status"].startswith("modified")
        assert code["pipeline_files"][name]["git_status"] != "clean"
    assert code["pipeline_code_committed"] is False
    assert code["reproducible_from_git_sha"] is False
    assert "BASE commit only" in code["note"]


def test_code_provenance_fails_closed_on_unparseable_git_output(monkeypatch):
    # A malformed status line must surface as an error, not as a missing token that reads
    # as "clean" for a tracked file.
    def fake_run_git(arguments, timeout=30):
        if arguments[0] == "ls-files":
            return subprocess.CompletedProcess(
                args=arguments, returncode=0,
                stdout="argus/ml/world_model/binning.py\n"
                       "argus/scripts/prepare_cicids2017.py\n", stderr="")
        return subprocess.CompletedProcess(args=arguments, returncode=0,
                                           stdout="?? \n", stderr="")
    monkeypatch.setattr(prepare, "_run_git", fake_run_git)
    monkeypatch.setattr(prepare, "git_sha", lambda: "c" * 40)

    code = prepare.code_provenance()
    assert code["git_errors"] and "could not be parsed" in code["git_errors"][0]
    assert code["pipeline_code_committed"] is False
    assert code["reproducible_from_git_sha"] is False
    assert "GIT PROVENANCE INCOMPLETE" in code["note"]


@pytest.mark.parametrize("token, tracked, expected_prefix", [
    ("  ", True, "clean"),          # both columns blank -> the only clean case
    ("??", False, "untracked"),
    (" M", True, "modified"),       # unstaged modification
    ("M ", True, "modified"),       # staged modification
    ("MM", True, "modified"),       # both
    ("??", True, "UNKNOWN"),        # ls-files and porcelain disagree -> never "clean"
])
def test_describe_git_status_never_invents_clean(token, tracked, expected_prefix):
    described = prepare.describe_git_status(
        "ml/world_model/binning.py", tracked=tracked, statuses={
            "ml/world_model/binning.py": token})
    assert described.startswith(expected_prefix), described


def test_describe_git_status_reports_clean_only_for_a_tracked_blank_pair():
    # tracked + no porcelain line at all -> clean
    assert prepare.describe_git_status("ml/world_model/binning.py", True, {}) == "clean"
    # untracked + no porcelain line -> untracked, never clean
    assert prepare.describe_git_status(
        "ml/world_model/binning.py", False, {}) == "untracked"


def test_code_provenance_fails_closed_when_git_fails(monkeypatch):
    # A git command that fails must make the manifest say "not reproducible", never
    # "clean". Each helper is stubbed to the (value, error) shape it now returns.
    def failing(*args, **kwargs):
        return ({}, "git status exited 128: fatal: not a git repository")
    def failing_dirty(*args, **kwargs):
        return (None, "git status exited 128: fatal: not a git repository")
    monkeypatch.setattr(prepare, "git_status", failing)
    monkeypatch.setattr(prepare, "git_tracked", failing)
    monkeypatch.setattr(prepare, "git_is_dirty", failing_dirty)
    monkeypatch.setattr(prepare, "git_sha", lambda: "unknown")

    code = prepare.code_provenance()

    assert code["pipeline_code_committed"] is False
    assert code["reproducible_from_git_sha"] is False
    assert code["git_errors"] and "not a git repository" in code["git_errors"][0]
    assert code["git_dirty"] is None            # unknown, not False
    assert code["git_dirty_under_argus"] is None
    assert "GIT PROVENANCE INCOMPLETE" in code["note"]
    assert "NOT reproducible" in code["note"]
    # a file that exists is still hashed, so the manifest keeps a usable fingerprint
    assert len(code["pipeline_files"]["ml/world_model/binning.py"]["sha256"]) == 64


def test_code_provenance_note_is_conditional_on_committed_state(monkeypatch):
    producing = {"ml/world_model/binning.py", "scripts/prepare_cicids2017.py"}
    clean = {name: "  " for name in producing}
    monkeypatch.setattr(prepare, "git_status", lambda *a, **k: (clean, None))
    monkeypatch.setattr(prepare, "git_tracked", lambda *a, **k: (producing, None))
    monkeypatch.setattr(prepare, "git_is_dirty", lambda *a, **k: (False, None))
    monkeypatch.setattr(prepare, "git_sha", lambda: "a" * 40)

    code = prepare.code_provenance()
    assert code["pipeline_code_committed"] is True
    assert code["reproducible_from_git_sha"] is True
    # the "BASE commit only" wording must NOT appear once the code is committed
    assert "BASE commit only" not in code["note"]
    assert "committed and clean" in code["note"]

    # ... and it returns the moment a producing file is untracked
    monkeypatch.setattr(prepare, "git_tracked", lambda *a, **k: (set(), None))
    untracked = prepare.code_provenance()
    assert untracked["pipeline_code_committed"] is False
    assert "BASE commit only" in untracked["note"]


def test_git_helpers_return_error_capable_pairs():
    # This IS a git repository, so all three succeed; the point is that each returns a
    # (value, error) pair rather than a bare value, so a future failure is representable.
    statuses, status_error = prepare.git_status(paths=("ml/world_model/binning.py",))
    tracked, tracked_error = prepare.git_tracked()
    dirty, dirty_error = prepare.git_is_dirty()
    assert status_error is None and isinstance(statuses, dict)
    assert tracked_error is None and isinstance(tracked, set)
    assert dirty_error is None and isinstance(dirty, bool)


# ---------------------------------------------------------------------------
# Sum aggregation must not turn missing data into a zero
# ---------------------------------------------------------------------------


def test_all_nan_sum_group_stays_nan_and_is_counted_separately():
    # pandas' plain sum() of an all-NaN group returns 0.0, which would contradict the
    # stated "never replaced with 0" policy. min_count=1 keeps it NaN.
    spec = [(0, "10.0.0.1", "10.0.0.9", 80, "BENIGN")]
    table = flows(spec)
    table["total_fwd_packets"] = float("nan")          # counter unknown for this flow
    table["flow_duration_ms"] = float("nan")
    table["total_fwd_bytes"] = float("inf")             # an infinity, not a NaN

    bins = build_host_time_bins(table, bin_seconds=60)
    row = bins.iloc[0]

    # an all-NaN sum column stays NaN; it is NOT reported as 0
    assert pd.isna(row["total_fwd_packets"]), "all-NaN sum must stay NaN, not become 0"
    # the infinity was converted to NaN rather than kept as inf
    assert pd.isna(row["total_fwd_bytes"])
    # a NaN mean column stays NaN too, and nothing is silently zeroed
    assert pd.isna(row["flow_duration_ms"])
    # n_flows is a COUNT of rows, not a sum of values: it is still 1
    assert int(row["n_flows"]) == 1
    # the two cell classes are counted separately, so neither is mistaken for the other
    assert bins.attrs["infinite_cells"] == 1
    assert bins.attrs["missing_cells"] == 2       # the two NaNs written above


def test_partial_nan_sum_group_sums_the_values_it_has():
    # min_count=1 must not turn a PARTIAL group into NaN: one known value still sums.
    spec = [(0, "10.0.0.1", "10.0.0.9", 80, "BENIGN"),
            (0, "10.0.0.1", "10.0.0.9", 81, "BENIGN")]
    table = flows(spec)
    table["total_fwd_packets"] = [float("nan"), 4.0]
    bins = build_host_time_bins(table, bin_seconds=60)
    assert bins.iloc[0]["total_fwd_packets"] == pytest.approx(4.0)
    assert int(bins.iloc[0]["n_flows"]) == 2


def test_collision_drop_is_measured_not_asserted_away(tmp_path):
    # Pins what dropping a consumed-key collision actually does to the 39 model inputs,
    # measured through the real binning code.
    #
    # Two earlier claims were WRONG and are corrected here. (1) "the drop changes nothing"
    # - false, the sums and counts move. (2) "only sums move, the means are unchanged" -
    # also false: removing one copy from an averaging set shifts every mean, even though
    # the value is identical. The first version of this test missed (2) only because its
    # fixture gave every flow the same feature value, which makes the means invariant by
    # construction; the fixture below deliberately mixes values, as real bins do.
    real = raw_rows([(0, "BENIGN"), (0, "PortScan")])   # two flows in the SAME bin
    real[DUPLICATE_KEY_COLUMN] = ["flow-0", "flow-1"]
    real.loc[0, "Total Fwd Packets"] = 4.0              # different values, so means move
    real.loc[1, "Total Fwd Packets"] = 10.0
    real.loc[0, "Flow Duration"] = 100.0
    real.loc[1, "Flow Duration"] = 900.0
    collision = real.iloc[[0]].copy()          # same consumed values, same Flow ID
    collision["Fwd Header Length.1"] = 999.0    # ... differing only where we never look
    path = tmp_path / "Tuesday-WorkingHours.pcap_ISCX.csv.parquet"
    pd.concat([real, collision], ignore_index=True).to_parquet(path, index=False)

    frame, report = clean_flow_table(path)
    assert report["duplicates_consumed_key_collisions"] == 1
    impact = measure_collision_impact(
        frame, report["_collision_rows"], bin_seconds=60)

    assert impact["measured"] is True
    changed = impact["model_inputs_changed"]

    # A non-zero sum loses one copy of its value: (4 + 10 + 4) -> (4 + 10) = 18 -> 14.
    assert changed["total_fwd_packets"]["example_before"] == pytest.approx(18.0)
    assert changed["total_fwd_packets"]["example_after"] == pytest.approx(14.0)

    # MEANS MOVE. Before: (100 + 100 + 900)/3; after: (100 + 900)/2. This is the point the
    # earlier draft got wrong, so it is asserted explicitly.
    assert "flow_duration_ms" in changed
    assert changed["flow_duration_ms"]["example_before"] == pytest.approx(1100.0 / 3)
    assert changed["flow_duration_ms"]["example_after"] == pytest.approx(500.0)

    # The count and the derived ratios move.
    assert changed["n_flows"]["example_before"] - changed["n_flows"]["example_after"] == 1
    assert "packets_per_flow" in changed
    assert "connection_repetition" in changed

    # Unchanged: the max (the kept copy holds the same value) and every nunique-based
    # column (an identical copy adds no new distinct key).
    for untouched in ("flow_iat_max", "unique_dst_ports_per_src", "src_fanout",
                      "unique_dst_hosts", "unique_src_hosts", "new_host_edges",
                      "new_dst_ports"):
        assert untouched not in changed, f"{untouched} should be unchanged by one fewer copy"

    # The bin still exists and still has a single stage label; only the attack-flow count
    # inside it changes, because the dropped copy was a BENIGN flow.
    assert impact["bin_labels_changed"] == 0
    assert impact["label_flows_changed"] == 0


def test_dropping_a_malicious_duplicate_lowers_label_flows_but_not_the_bin_label(tmp_path):
    # The BENIGN case above cannot show the label-flow effect, because label_flows counts
    # non-BENIGN flows. The pinned collision is a DoS Hulk duplicate, so this case uses a
    # malicious duplicate: dropping it must take label_flows down by exactly one while the
    # bin's stage label stays the same, because the kept copy is still there.
    real = raw_rows([(0, "BENIGN"), (0, "PortScan"), (0, "DDoS")])
    real[DUPLICATE_KEY_COLUMN] = ["flow-0", "flow-1", "flow-2"]
    collision = real.iloc[[2]].copy()          # duplicate of the DoS Hulk row
    collision["Fwd Header Length.1"] = 999.0   # ... differing only in an unconsumed column
    path = tmp_path / "Wednesday-workingHours.pcap_ISCX.csv.parquet"
    pd.concat([real, collision], ignore_index=True).to_parquet(path, index=False)

    frame, report = clean_flow_table(path)
    assert report["duplicates_consumed_key_collisions"] == 1
    assert report["duplicates_exact"] == 0

    before = build_host_time_bins(
        pd.concat([frame, report["_collision_rows"]], ignore_index=True), bin_seconds=60)
    after = build_host_time_bins(frame, bin_seconds=60)

    # the bin is DDoS either way: the kept DoS Hulk copy keeps the max severity
    assert before.iloc[0]["label"] == "DDoS" == after.iloc[0]["label"]
    # but one fewer non-BENIGN flow survives in it
    assert before.iloc[0]["label_flows"] - after.iloc[0]["label_flows"] == 1
    assert after.iloc[0]["label_flows"] == 2

    impact = measure_collision_impact(frame, report["_collision_rows"], bin_seconds=60)
    assert impact["label_flows_changed"] == 1
    assert impact["bin_labels_changed"] == 0
    # the DDoS counter is one lower, which is what the anchor's cleaned count sees
    assert impact["model_inputs_changed"]["n_flows"]["example_before"] - \
        impact["model_inputs_changed"]["n_flows"]["example_after"] == 1


def test_nan_exposure_is_published_for_bins_and_sequences():
    # The WP3 handoff has to survive in the manifest, not only in a review conversation.
    labels = ["BENIGN"] * 20 + ["BENIGN"] * 4
    table = run_of_bins("10.0.0.1", labels)
    # make the FIRST bin's rate features unknown, and nothing else
    table.loc[0, "flow_bytes_per_sec"] = float("nan")
    table.loc[0, "flow_packets_per_sec"] = float("nan")
    bins = build_host_time_bins(table, bin_seconds=60)
    sequences = make_sequences(bins, W=10, horizons=(1, 2, 4), bin_seconds=60)
    exposure = measure_nan_exposure(bins, sequences)

    assert exposure["measured"] is True
    assert exposure["bin_nan_cells_by_feature"] == {
        "flow_bytes_per_sec": 1, "flow_packets_per_sec": 1}
    assert exposure["bin_nan_cells_total"] == 2
    assert exposure["bins_with_at_least_one_nan"] == 1
    assert exposure["bins_total"] == len(bins)
    # the NaN survives into X (no zero-filling anywhere), so WP3 has to handle it
    assert exposure["sequence_x_nan_cells"] == 2
    assert exposure["sequences_with_at_least_one_nan"] == 1
    assert exposure["sequences_total"] == sequences.n_sequences
    assert "min_count=1" in exposure["policy"]
    # The handoff must be ONE coherent policy. An earlier version offered "use NaN-safe
    # statistics" as an option, which is incoherent on its own: nanmean/nanstd do not stop a
    # NaN from entering attention and the output projection. The finiteness assertion is
    # also explicitly AFTER imputation, because the artifact deliberately contains NaN.
    policy = exposure["nan_policy"]
    assert policy["required"] == "impute_from_training_split_statistics"
    assert "TRAINING split only" in policy["imputation_fit_on"]
    assert "AFTER imputation" in policy["finiteness_assertion"]
    assert "do NOT make" in policy["rejected"]
    assert "nan_policy" in exposure and "wp3_action" not in exposure


# ---------------------------------------------------------------------------
# Anchor gate and the 39-column schema
# ---------------------------------------------------------------------------


def test_duplicate_report_separates_exact_duplicates_from_key_collisions(tmp_path):
    # Two different removals that both look like "duplicates" on the consumed key:
    #   a) a true duplicate, identical in every exported column;
    #   b) a consumed-key collision, same consumed columns AND same Flow ID, differing
    #      only in a column this pipeline never reads.
    real = raw_rows([(0, "BENIGN"), (1, "BENIGN")])
    real[DUPLICATE_KEY_COLUMN] = ["flow-0", "flow-1"]
    exact_duplicate = real.iloc[[0]].copy()
    collision = real.iloc[[0]].copy()
    collision["Fwd Header Length.1"] = 999.0      # unconsumed duplicate-export column
    path = tmp_path / "Tuesday-WorkingHours.pcap_ISCX.csv.parquet"
    pd.concat([real, exact_duplicate, collision], ignore_index=True).to_parquet(
        path, index=False)

    frame, report = clean_flow_table(path)

    assert report["duplicates_dropped"] == 2
    assert report["duplicates_exact"] == 1
    assert report["duplicates_consumed_key_collisions"] == 1
    detail = report["duplicates_collision_details"]
    assert len(detail) == 1
    assert detail[0]["raw_label"] == "BENIGN"
    assert detail[0]["differing_columns"] == ["Fwd Header Length.1"]
    assert detail[0]["differing_columns_are_consumed"] is False
    assert detail[0]["kept_row_index"] < detail[0]["row_index"]
    assert len(frame) == 2


def test_label_anchors_are_exact_for_the_three_small_families():
    observed = {family: expected for family, (expected, _) in LABEL_ANCHORS.items()}
    anchors = {anchor["family"]: anchor for anchor in check_label_anchors(observed)}
    assert all(anchor["within_tolerance"] for anchor in anchors.values())

    # +/-3 % of 11 flows is less than one flow, so those three are compared exactly.
    observed["Heartbleed"] = 12
    anchors = {anchor["family"]: anchor for anchor in check_label_anchors(observed)}
    assert anchors["Heartbleed"]["exact"] is True
    assert anchors["Heartbleed"]["within_tolerance"] is False
    assert [a["family"] for a in check_label_anchors(observed)
            if not a["within_tolerance"]] == ["Heartbleed"]

    # A large family is checked at the tolerance: 2 % passes, 4 % fails.
    observed["PortScan"] = int(159_000 * 1.02)
    assert {a["family"]: a for a in check_label_anchors(observed)}["PortScan"]["within_tolerance"]
    observed["PortScan"] = int(159_000 * 1.04)
    assert not {a["family"]: a
                for a in check_label_anchors(observed)}["PortScan"]["within_tolerance"]


def test_flow_tier_schema_is_39_columns_without_packet_level_features():
    assert len(FLOW_TIER_FEATURE_LIST) == 39
    assert FLOW_TIER_FEATURE_LIST[:24] == FLOW_FEATURE_COLUMNS
    assert FLOW_TIER_FEATURE_LIST[-1] == "n_flows"
    for column in PACKET_LEVEL_COLUMNS:
        assert column not in FLOW_TIER_FEATURE_LIST      # never zero-filled into v2
    for column in TOPOLOGY_FEATURE_COLUMNS:
        assert column in FLOW_TIER_FEATURE_LIST

    aggregated = set(SUM_COLUMNS) | set(MEAN_COLUMNS) | set(MAX_COLUMNS)
    assert len(aggregated) == 28
    assert aggregated <= set(FLOW_TIER_FEATURE_LIST)
    assert aggregated <= set(CICIDS2017_COLUMN_MAP.values())
    derived = (set(FLOW_TIER_FEATURE_LIST) - aggregated
               - set(TOPOLOGY_FEATURE_COLUMNS) - {"n_flows"})
    assert derived == {"unique_dst_ports_per_src", "packets_per_flow"}


# ---------------------------------------------------------------------------
# CLI: --check writes nothing, a clean run writes parquet + manifest, and an anchor
# violation stops the write
# ---------------------------------------------------------------------------


def synthetic_snapshot(tmp_path, minutes: int = 26):
    """A minimal stand-in for the pinned snapshot: one Parquet + its provenance record."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    name = "Tuesday-WorkingHours.pcap_ISCX.csv.parquet"
    spec = [(minute, "BENIGN") for minute in range(20)]
    spec += [(20 + index, "FTP-Patator") for index in range(minutes - 20)]
    path = raw_dir / name
    table = raw_rows(spec)
    table.to_parquet(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "files": [{"file": name, "bytes": path.stat().st_size, "sha256": sha,
                   "expected_sha256": sha, "rows": len(table)}],
        "totals": {"label_histogram": {"BENIGN": 20, "FTP-Patator": minutes - 20}},
        "source": {"host": "synthetic-test-fixture"},
    }
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps(record), encoding="utf-8")
    return raw_dir, prov


def test_cli_check_prints_the_manifest_and_writes_nothing(tmp_path, monkeypatch, capsys):
    raw_dir, prov = synthetic_snapshot(tmp_path)
    monkeypatch.setattr(prepare, "PROVENANCE_PATH", prov)
    out_dir = tmp_path / "processed"

    code = main(["--raw-dir", str(raw_dir), "--out", str(out_dir),
                 "--bin-seconds", "60", "--check", "--max-rows", "500"])

    assert code == 0
    assert not out_dir.exists() or not list(out_dir.iterdir())
    printed = capsys.readouterr().out
    assert "hosts / bins" in printed
    assert "feature_list (39 columns)" in printed
    assert "--check: nothing written" in printed
    manifest = json.JSONDecoder().raw_decode(printed.split("manifest (json):", 1)[1].lstrip())[0]
    assert manifest["bin_seconds"] == 60
    assert manifest["W"] == 10 and manifest["horizons"] == [1, 2, 4]
    assert manifest["unmapped_labels"] == []
    assert manifest["duplicates_removed"] == 0
    assert manifest["padding_rows_dropped"] == 0
    assert manifest["bins"] == 26 and manifest["hosts"] == 1
    assert manifest["sequences_per_horizon"] == {"1": 13, "2": 13, "4": 13}
    # --max-rows marks the run truncated and records the anchors without enforcing them
    assert manifest["truncated"] is True and manifest["anchors_enforced"] is False
    assert manifest["label_anchor_violations"]            # they really cannot match here
    assert manifest["file_row_checks"][0]["within_tolerance"] is True


def test_cli_writes_parquet_and_manifest(tmp_path, monkeypatch, capsys):
    raw_dir, prov = synthetic_snapshot(tmp_path)
    monkeypatch.setattr(prepare, "PROVENANCE_PATH", prov)
    out_dir = tmp_path / "processed"

    code = main(["--raw-dir", str(raw_dir), "--out", str(out_dir), "--max-rows", "500"])

    assert code == 0
    capsys.readouterr()
    parquet_path = out_dir / "cicids2017_bins_60s.maxrows500.parquet"
    manifest_path = out_dir / "manifest.maxrows500.json"
    assert parquet_path.exists() and manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["feature_list"] == FLOW_TIER_FEATURE_LIST
    assert manifest["bins"] == 26
    assert manifest["label_histogram"] == {"BENIGN": 20, "FTP-Patator": 6}
    assert manifest["bin_label_histogram"] == {"BENIGN": 20, "BruteForce": 6}
    assert manifest["rows_after_clean_total"] == 26
    assert manifest["duplicates_exact_rows"] == 0
    assert manifest["duplicates_consumed_key_collisions"] == 0
    assert manifest["duplicates_collision_details"] == []
    # The two WP3 obligations must both be carried in the artifact: the NaN policy and the
    # fully-ignored stage batch. Dropping either would leave WP3 to rediscover it.
    nan_policy = manifest["nan_exposure"]["nan_policy"]
    assert nan_policy["required"] == "impute_from_training_split_statistics"
    assert "AFTER imputation" in nan_policy["finiteness_assertion"]
    masking = manifest["stage_target_masking"]
    assert masking["status"] == "required, NOT implemented (WP3)"
    assert "returns NaN when" in masking["all_ignored_batch"]
    assert "ignore_index=-1" in masking["all_ignored_batch"]
    counts = manifest["target_counts_per_horizon"]
    assert list(counts) == ["1", "2", "4"]
    assert all(row["valid_windows"] == 13 for row in counts.values())
    assert all(row["attack_within_positive"] + row["attack_within_negative"] == 13
               for row in counts.values())
    code = manifest["code"]
    # provenance must identify the CODE, not just the base commit, and must do so the
    # same way whether or not the pipeline files happen to be committed right now
    assert len(code["git_sha"]) == 40
    assert code["reproducible_from_git_sha"] == code["pipeline_code_committed"]
    for name in ("ml/world_model/binning.py", "scripts/prepare_cicids2017.py"):
        assert len(code["pipeline_files"][name]["sha256"]) == 64
        assert "git_status" in code["pipeline_files"][name]
    assert manifest["source_provenance"]["record"].endswith("provenance.json")
    assert len(manifest["code"]["prepare_script_sha256"]) == 64

    bins = pd.read_parquet(parquet_path)
    assert list(bins.columns) == (
        ["src_ip", "bin_start", "day", "bin_index"] + FLOW_TIER_FEATURE_LIST
        + ["label", "label_index", "label_flows", "stage_valid"]
    )
    assert len(bins) == 26
    assert int(bins["n_flows"].sum()) == 26
    assert set(bins["label"]) == {"BENIGN", "BruteForce"}


def test_cli_fails_on_an_anchor_violation_without_writing(tmp_path, monkeypatch, capsys):
    # No --max-rows: the run is a real one, so the anchors are enforced. 20 BENIGN rows
    # cannot match the published BENIGN count, so the pipeline must refuse to write a
    # manifest a later stage would trust.
    raw_dir, prov = synthetic_snapshot(tmp_path)
    monkeypatch.setattr(prepare, "PROVENANCE_PATH", prov)
    out_dir = tmp_path / "processed"

    code = main(["--raw-dir", str(raw_dir), "--out", str(out_dir)])

    assert code == 1
    printed = capsys.readouterr().out
    assert "FAIL: anchor violation(s)" in printed
    assert "label family outside tolerance: BENIGN" in printed
    assert not out_dir.exists() or not list(out_dir.iterdir())


def test_cli_refuses_a_snapshot_whose_hash_does_not_match(tmp_path, monkeypatch):
    raw_dir, prov = synthetic_snapshot(tmp_path)
    record = json.loads(prov.read_text(encoding="utf-8"))
    record["files"][0]["expected_sha256"] = "0" * 64
    prov.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(prepare, "PROVENANCE_PATH", prov)

    with pytest.raises(SystemExit) as excinfo:
        main(["--raw-dir", str(raw_dir), "--out", str(tmp_path / "out"), "--check"])

    assert "snapshot gate failed" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Per-horizon reporting: one window set, every horizon scored on all of it
# ---------------------------------------------------------------------------


def test_target_counts_per_horizon_report_total_windows_and_never_decrease():
    # 12 benign bins then 8 PortScan bins. Every horizon is scored on the SAME windows
    # (a window needs the t+1, t+2 and t+4 bins to exist), and attack_within is an OR over
    # more bins, so the positives must be non-decreasing in k - which is what made the
    # old printout look like k=4 had more windows than k=2.
    bins = build_host_time_bins(
        run_of_bins("10.0.0.1", ["BENIGN"] * 12 + ["PortScan"] * 8), bin_seconds=60)
    sequences = make_sequences(bins, W=10, horizons=(1, 2, 4), bin_seconds=60)
    counts = summarise_horizons(sequences)

    assert list(counts) == ["1", "2", "4"]
    for row in counts.values():
        assert row["valid_windows"] == sequences.n_sequences
        assert row["stage_targets"] == sequences.n_sequences
        assert (row["attack_within_positive"] + row["attack_within_negative"]
                == sequences.n_sequences)
    positives = [counts[key]["attack_within_positive"] for key in ("1", "2", "4")]
    assert positives == sorted(positives)
    assert positives[0] < positives[-1]          # the burst is longer than one bin


# ---------------------------------------------------------------------------
# The D3 stage exclusion is a MARK in the data. Nothing masks it in the loss yet.
# ---------------------------------------------------------------------------


def test_stage_targets_are_marked_not_yet_masked():
    """Pins the gap so the docstrings cannot drift back into overclaiming.

    WP2 encodes STAGE_TARGET_IGNORE in `y_stage_*`; the stage losses do not ignore it
    yet. This test is written against model.py's SOURCE (so it runs without torch) and
    fails on purpose once WP3 adds `ignore_index` to the stage losses.

    TODO(WP3): in the SAME commit that adds the masking, delete or rewrite this test
    (assert the mask IS honoured instead) and update the `stage_target_masking` block in
    scripts/prepare_cicids2017.py plus the docstrings in ml/world_model/binning.py. Do
    not merge WP3 with this test still asserting the gap - a red suite is not a guard.
    Grep for TODO(WP3) to find the other transition guards.
    """
    model_source = (ROOT / "ml" / "world_model" / "model.py").read_text(encoding="utf-8")
    assert "nn.CrossEntropyLoss()" in model_source, "expected the default CE loss"
    assert "ignore_index" not in model_source, (
        "model.py now masks stage targets - update binning.py, "
        "prepare_cicids2017.py and this test together"
    )
    # -1 must stay a marker that is NOT silently ignored by a default loss ...
    assert STAGE_TARGET_IGNORE == -1
    assert STAGE_TARGET_IGNORE != -100
    # ... and it must keep y_attack_within_h1 == (y_stage_h1 != 0) true.
    bins = build_host_time_bins(
        run_of_bins("10.0.0.1", ["BENIGN"] * 18 + ["Heartbleed"] * 2), bin_seconds=60)
    sequences = make_sequences(bins, W=10, horizons=(1, 2, 4), bin_seconds=60)
    marked = sequences.y_stage_h[4] == STAGE_TARGET_IGNORE
    assert marked.sum() >= 1
    assert sequences.stage_ignore_index == STAGE_TARGET_IGNORE
    # the mask WP3 will consume selects exactly those rows, and nothing else
    assert np.array_equal(sequences.stage_target_masks()["y_stage_h4"], ~marked)
    assert np.array_equal(sequences.y_attack_within_h[1], (sequences.y_stage_h[1] != 0))
    # a marked stage target is still an attack in the binary target at that horizon
    assert sequences.y_attack_within_h[4][marked].min() == 1


def test_default_loss_would_not_ignore_the_stage_marker():
    """The behaviour half of the gap above; needs torch, so it skips without it."""
    torch = pytest.importorskip("torch")

    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])   # class 0, confident
    marker = torch.tensor([STAGE_TARGET_IGNORE])

    # A loss configured the WP3 way skips the sample entirely -> zero contribution.
    masked = float(torch.nn.CrossEntropyLoss(ignore_index=STAGE_TARGET_IGNORE)(logits, marker))
    assert masked == pytest.approx(0.0, abs=1e-6)

    # The loss as it stands today does NOT skip it: torch either rejects the out-of-range
    # class index or scores it as a class. Both are "not ignored", which is the gap.
    try:
        default_value = float(torch.nn.CrossEntropyLoss()(logits, marker))
    except (RuntimeError, IndexError, AssertionError):
        default_value = None
    assert default_value is None or default_value > 0.0
