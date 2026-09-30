# Review bundle - D1/D2/D3 + label-map, packet-tier extractor

Branch `review/d1-d2-d3-packet-tier`, in order: `b36148c` (code), `a61ed27`
(bundle), `e107158` (round-1 fixes), `505b0b2` (hash note), `f812408` (round 2),
`21b6fd1` (merge of `origin/main`), `496821c` + `1bebbd0` + `f267565` +
`158c0c2` (round 3), `c4d81d4` (bundle refreshed for rounds 2 and 3), `c72de6e` + `907c15e` +
this commit (round 4: plan card E, the bundle itself, then the probe guard and
the re-smoke becoming a test).
A commit cannot list its own hash, so the
authority for "what is on the branch" is
`git log --oneline main..HEAD` - review the branch tip. Base `main` =
`c53652f`, with `origin/main` `eda53c4` merged in; that merge touches none of the
files under review. Prepared 2026-09-28, extended 2026-09-29.

**Review rounds 1, 2, 3 and 4 are folded in.** Round 1's findings are described
item by item in **item 8**, round 2's in **item 9**, round 3's in **item 10**,
round 4's below (**item 11**);
the `.diff` artifacts were regenerated against `main`, so they already contain
every fix. WP2/training not started; no PCAPs downloaded; `main` untouched -
merge only on approval.

## Contents

| File | What it is |
|---|---|
| `plan-world-model-core.diff` | the complete plan diff: D1, D2, D3 (stage vocabulary, tau/lead time, packet-tier arm), 2.2 manifest anchors, label map, risk 6 - 8 hunks |
| `extractor-pcap_bins-and-pcap_stream_bins.diff` | exact diff of the extractor (2 new files) |
| `tests-test_pcap_bins.diff` | exact diff of the tests (1 new file) |
| `consumers-features-and-dashboard.diff` | exact diff of the two demo-path consumers (`features.py`, `dashboard/app.py`) that disclose what `retransmission_count` really is (round 3) |
| `README.md` | this file - answers to all review questions |

Regenerate with:

```
# two dots against main: picks up every commit on this branch plus any
# uncommitted fix on top of it (three dots would compare commits only):
git diff main -- docs/plan/world-model-core.md
git diff main -- argus/ml/world_model/pcap_bins.py argus/scripts/pcap_stream_bins.py
git diff main -- argus/tests/test_pcap_bins.py
git diff main -- argus/ml/world_model/features.py argus/dashboard/app.py
```

(`--output=<file>` rather than `>`, so the bytes stay git's own.)

## 1. Changed plan sections (all 8 hunks in `plan-world-model-core.diff`)

| Hunk | Section | Change |
|---|---|---|
| `@@ -30,11 +30,11 @@` | **D1** Decision + Rejected alternatives | bin key pinned as `bin_start = floor(unix_seconds / 60) * 60` (UTC), key `(src_ip, bin_start)`; the 8 packet columns are now "**not part of the flow-tier schema (v2) and are never zero-filled into it**; they arrive, if at all, through the packet tier of the amended D2"; rejected-alt (b) rewritten to name the gated 52.43 GB set and the NaN-with-coverage-flag rule; **round 4** - `unique_dst_ports_per_src` leaves the `sum` list (a per-bin nunique of `Destination Port` cannot be summed: the line now says nunique over the bin's flows, computed from `Destination Port`, and freezes the name), and the timestamp claim is **measured, not assumed** - Monday is second-precision, Tuesday-Friday minute-precision (`cicids2017_provenance.json`) - so "the floor is lossless" is replaced by the quantization-aware feasible-interval rule on the PCAP side |
| `@@ -44,11 +44,37 @@` | **D2** | four additions (+ the round-1, round-2 and round-3 paragraphs and the round-3 re-smoke): (i) *acquisition status* - 8 Parquet files from the pinned HF mirror `bvsam/cic-ids-2017 @ 70bac624`, validated by `scripts/validate_cicids2017.py`, 288,602 all-null padding rows observed-not-asserted, dropped never zero-filled; (ii) *flow vs. packet features - conflict resolution*: two-tier model (see item 6); (iii) *PCAP source decision*: options (a) Thursday-only / (b) stream-and-cut <= 2 GB / (c) baseline-only, with the three gates that hold under any option (streaming fixed-60 s extractor, measured clock offset, measured coverage; until measured on real CIC PCAPs they are "not yet measured"); (iv) *extractor implementation status*: what is implemented and green today; (v) round-2 amendment - bounded reordering and the feasible-interval offset; (vi) round-3 amendment - fail-closed late packets, the majority-window interval, `dup_range_count`; (vii) the round-3 re-smoke and the round-1 line it supersedes |
| `@@ -57,6 +83,11 @@` | **D3 - stage vocabulary** (new) | the stage head is trained and scored on the **6 observed stages** (BENIGN, PortScan, WebAttack, BruteForce, Botnet, DDoS); Exfiltration does not occur in CIC-IDS-2017; Infiltration (36 flows) and Heartbleed (11) enter the **binary target only** with their support listed - so no stage class is trained on 11 or 36 samples |
| `@@ -66,6 +97,19 @@` | **D3 - tau and lead time** (new) | tau is chosen on a **validation slice that never touches test bins** (P-A: last 15 % by time of each day's training portion; P-B: one training day held out in rotation), the benign-FPR printed beside lead time is the **test-set** FPR at that tau, lead time is reported with its onset count n (expected 12-15: attacker 172.16.0.1 plus bot-infected 192.168.10.x hosts), and the **per-victim (dst-host) view is rejected** - a second state definition and a second benchmark, and at n = 12-15 a per-victim cell is a median over one onset - with the victim side still reconstructable from `dst_fanin` / `unique_dst_hosts` / `src_fanout` and the `dst_ip` on every onset row |
| `@@ -80,6 +124,8 @@` | **D3** | *packet-tier arm*: only if a PCAP source is approved, one extra arm restricted to PCAP-covered bins, labelled `packet_tier`, provenance carries `pcap_file`, `clock_offset_sec`, `bins_flow_total`, `bins_covered`, `attack_bins_covered`; uncovered bins excluded (NaN, never zero-filled); one Thursday PCAP = at most a Thursday-scoped `pilot` row; flow-tier arm stays the headline |
| `@@ -157,8 +203,35 @@` | **2.2 manifest + label-map preamble** (new) | the manifest additionally carries `label_anchors[{family, expected, observed, deviation_pct, within_tolerance}]` and `label_anchor_violations` (must be empty), and `prepare_cicids2017.py --check` fails on any family outside **±3 %** (exact for the 36 / 21 / 11-flow families, where ±3 % is under one flow) - a guard against a wrong column or a silently truncated file, never a substitute for the pinned revision + SHA-256 + byte size; checked against this pinned snapshot **every anchor passes today**, largest deviation Bot 1,966 vs 2.0 k = -1.7 %. Then: **two rows of the label map are superseded** by D3's stage vocabulary (Infiltration -> LateralMovement would be a 7th class the data cannot train; Heartbleed -> WebAttack would put 11 flows in a stage class) - WP2 keeps their rationale for provenance and the binary label and excludes both families from `y_stage_*` |
| `@@ -172,7 +245,7 @@` | **label map** | the mojibake row is no longer hypothetical: measured over all 8 files, every Web-Attack row spells all three families `Web Attack \u0096 ...` (U+0096) - none use ASCII hyphen or U+FFFD; `_CICIDS2017_LABEL_MAP` lacks that spelling, so today every such row falls through `.get(x, "BENIGN")` at `dataset_loader.py:691` (V12, confirmed live); new entries map Brute Force -> BruteForce, XSS/Sql Injection -> WebAttack; unknown labels now **raise** with `unmapped_labels: []` required in the manifest |
| `@@ -291,12 +364,14 @@` | **Risks** | Risk 6 "PS 1 asks for packet-level features" is **not** declared resolved - it now reads "**Mitigation designed** (two-tier model); **resolved only when** a PCAP source is approved and the `packet_tier` arm has run", and keeps the pilot-vs-coverage and 52.43 GB-separate-approval statements. Round 3 called this "Resolved in D2"; that was ahead of the evidence, since no PCAP has been measured - see item 6 |

## 2. Exact code and test diffs

* `extractor-pcap_bins-and-pcap_stream_bins.diff` - both files are **new** (`new file mode 100644`):
  * `argus/ml/world_model/pcap_bins.py`: `bin_start_for`, `BackwardsTimestampError`, `_OnlineStats` (O(1) Welford count/mean/variance), `_BinAccumulator` (per-direction merged sequence ranges, `dup_range_count`, `SEQ_RANGES_CAP`, and a 1-byte keep-alive / window probe that is skipped as a duplicate and counted in `keep_alive_probes_skipped` - round 4), `_range_covered` / `_range_insert`, `extract_pcap_bins` (streams `PcapReader`; exactly one record per `(src_ip, bin_start)`; closes bins on a watermark scan run at bin boundaries; `on_late="fail"` raises, `on_late="drop"` counts the loss per bin; retained memory is `bin_seconds + reorder_grace_sec` and closed bins are freed), `_is_closed`, `_PrecisionCounter` / `infer_timestamp_quantization` (each source's step *determined* from its own values), `feasible_offset_interval`, `_majority_consensus_window` (largest set of deltas one offset can explain, two pointers, O(n)), `measure_clock_offset` (protocol-aware, per-connection-instance matching; `observed_delta` reported separately from a guarded `apply` decision), `measure_bin_coverage` (unique host-minute bins, null on an empty denominator), `join_packet_features`, `unsupported_or_null`.
  * `argus/scripts/pcap_stream_bins.py`: the CLI (`--pcap`, `--flows` - many files *and* directories at once, `--bin-seconds`, `--out`, `--apply-offset`, `--on-late {fail,drop}`, `--reorder-grace-sec`) plus the flow-table layer `_find_col`, `_cell_str`, `_norm_proto`, `load_flow_rows`, `resolve_flow_paths`, `load_flow_tables`. Two things refuse rather than proceed, and both **exit 1 with no output file at all**: a refused `--apply-offset` (needs `--flows` *and* an eligible measurement) and a late packet under the default `--on-late fail`.
* `tests-test_pcap_bins.diff` - `argus/tests/test_pcap_bins.py` (**new**), **66 tests** (count verified
with `pytest tests/test_pcap_bins.py --collect-only -q | tail -1`), grouped in the file by section header - the section is the useful grouping, since a test's "round" says less than what it pins: fixed bins keyed like D1; RST never a retransmission and the demo path's own disclosure (round 3 adds `test_demo_path_is_a_sequence_regression_proxy_and_says_so`, which runs one capture through both estimators: 1 for the demo proxy, 0 for the streaming detector); no zero-fill; offset measured vs `not_measured`; coverage percentages; the `PcapReader`-never-`rdpcap` guard; the pandas `us`/`ns` epoch regression; **late packets fail closed** (`test_late_packet_fails_closed_and_names_both_remedies`, `test_late_packet_is_dropped_and_counted_only_when_requested`, `test_reorder_grace_keeps_a_recent_bin_open_without_failing`, `test_invalid_late_policy_and_grace_are_rejected`); **reordering is memory-bounded** (`test_default_late_policy_fails_and_grace_is_bounded`, `test_late_burst_for_one_bin_is_counted_once_per_bin`, `test_hours_late_packet_never_keeps_an_old_bin_alive`, `test_retained_bins_do_not_grow_with_capture_length`); watermark scans scale with bins; **timestamp precision determined, never assumed** (incl. `test_feasible_interval_is_the_intersection_of_per_pair_bounds`); **the duplicate-range detector and the semantics it reports** (incl. `test_one_byte_keep_alive_is_not_a_duplicate`, `test_in_window_reordering_is_not_a_sequence_regression`, `test_out_of_order_range_memory_is_capped`, `test_packet_tier_has_no_training_or_benchmark_consumer_yet`); **feasible-interval offset and guarded apply** (incl. `test_outlier_deltas_are_counted_not_averaged_and_capped`, `test_inconsistent_pair_tolerance_is_inclusive_at_the_bound`, `test_one_contradictory_pair_is_excluded_not_averaged_into_the_interval`, `test_two_offset_populations_are_refused_as_inconsistent`, `test_minute_precision_is_pinnable_only_by_dense_sampling`,
`test_sample_portscan_planted_offset_is_refused_at_second_precision`); flow-table null / multi-file loading; and six CLI end-to-end tests (`test_cli_applies_measured_offset_so_flow_bins_join`, `test_cli_refuses_apply_offset_and_writes_no_output`, `test_cli_all_benign_flow_table_reports_null_attack_coverage`, `test_cli_late_packet_exits_1_until_drop_is_requested`, `test_cli_widening_the_grace_absorbs_the_reordering`, `test_cli_rejects_bad_arguments`).
* `consumers-features-and-dashboard.diff` - the two demo-path consumers that describe `retransmission_count` (`ml/world_model/features.py` docstring + inline heuristic comment; `dashboard/app.py`'s `approximations` entry), so the UI cannot imply the streaming extractor's semantics.

Verification runs: 2026-09-29 round 3 -> `pytest argus/tests/test_pcap_bins.py` **64 passed**, full suite **119 passed, 4 skipped**; 2026-09-30 round 4 (probe guard + fixture re-smoke test) -> **66 passed**, full suite **121 passed, 4 skipped** (skips, `pytest -rs`: 3 x PyTorch not installed, 1 x antivirus blocked the EICAR fixture).

## 3. Timestamp-to-epoch conversion code

`argus/scripts/pcap_stream_bins.py:124-212` (`load_flow_rows`) - round-1 version,
the null accounting is the addition, the epoch maths is unchanged apart from the
single-pass index map:

```python
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
```

Notes:

* `dayfirst=True` matches CIC's `dd/mm/YYYY` format; `errors="coerce"` is
  counted, never swallowed. `timestamp_null` (cells that were already null),
  `timestamp_unparsable` (cells that failed to parse - obtained by subtracting
  the nulls, so padding is never reported as garbage) and `timestamp_all_null`
  (the whole column null: the pinned snapshot's 288,602 padding rows) all land
  in the output provenance, per file and totalled.
* **The `"nan"` string bug**: the old `str(r[col] or "")` turned a float NaN
  into the literal string `"nan"` (NaN is truthy), which then polluted join
  keys and labels. Every cell now goes through `_cell_str`
  (`pcap_stream_bins.py:98-112`), which maps `None`/NaN/NA/empty/`"nan"`/
  `"none"`/`"nat"` to `None`, and a row without a real IP or port is dropped
  and counted in `rows_dropped_missing_identity` instead of being joined on a
  fabricated key. Pinned by
  `test_load_flow_rows_null_label_and_missing_identity_never_become_nan`,
  `test_load_flow_rows_all_null_timestamp_column_is_padding_not_garbage` and
  `test_load_flow_rows_separates_unparsable_from_null_timestamps`.
* Epochs are computed once, vectorised, and mapped back **by index**
  (`epoch_by_idx`) in a single pass - no positional `zip` that could drift if a
  row were skipped mid-loop.
* Naive timestamps are taken as-is (no timezone shift); if a flow clock is
  timezone-labelled the *measured* offset exposes it as a large `|median|`
  (hours), which is reported with a `timezone_scale` flag and never silently
  corrected.
* **The pandas 3 regression**: pandas 3 parses to `datetime64[us]`. Calling
  `.astype("int64")` directly on a `us` array yields **microseconds**, so
  dividing by `1e9` produced epoch values 1000x too small (≈1.78e6 instead of
  ≈1.78e9) - the original bug. The fix forces `to_numpy(dtype="datetime64[ns]")`
  first (the numpy unit cast does the scaling correctly), then
  `astype("int64") / 1e9` is seconds. Guarded by
  `test_load_flow_rows_epoch_not_corrupted_by_pandas_unit`.

## 4. Why the round-1 median read +1.220 s for a +2.000 s plant, and what is applied now

**Sign convention.** `measure_clock_offset` (`pcap_bins.py:918-1226`) computes,
for every protocol-aware directional 5-tuple present in both sources - paired
**per connection instance**, not per tuple - the delta

```
offset = flow_ts - first_packet_ts      (one delta per matched pair)
```

A **positive** offset means the flow-table clock reads *later* than the PCAP
clock (flow timestamps are shifted into the future relative to packets). The
PCAP is what gets corrected: `--apply-offset` adds `apply.correction_sec` (the
midpoint of the feasible interval, and only when the interval is narrow enough
to be applied) to packet timestamps before binning, aligning the PCAP onto the
flow-table clock. The
plan defines the sign the same way (D2: "flow timestamp minus first packet
time").

**Why not exactly +2.000.** The synthetic flow table writes timestamps as
`%d/%m/%Y %H:%M:%S` - **second precision, fractional part discarded** - while
the PCAP keeps full sub-second precision. The fixture planted

```
flow_ts = floor(pcap_t + 2.0) = floor(pcap_t) + 2        (whole seconds)
```

so each matched tuple contributes

```
delta = flow_ts - pcap_t = 2.0 - frac(pcap_t)      where frac = pcap_t - floor(pcap_t)
```

and the *measured* statistic is `2.0 - median(frac)`, not `2.0`. Measured
`frac` distribution over the 200 TCP 5-tuples of `data/sample_portscan.pcap`:

```
frac: min=0.7510  p25=0.7640  median=0.7802  p75=0.7918  max=0.8031
predicted measured median = 2 - 0.7802 = 1.219773
predicted iqr            = p75 - p25    = 0.027819
predicted range          = [2-max, 2-min] = [1.1969, 1.2490]
```

**Live re-verification (2026-09-29, review round 3 - same fixture construction,
plus a microsecond-precision twin of the same 200 rows):**

```
second-precision table  (q_flow=1s, q_packet=0.001s)
  offset    : measured n=200 consistent=200 inconsistent=0
              delta=[+1.197,+1.249]s  median(biased)=+1.220s
  interval  : [+1.247972, +2.196924) feasible width=0.948952s
  apply     : REFUSED -> exit 1, no output file (0.948952s over the 0.1s bound)
  coverage  : 1/1 flow bins (100.0%), attack 1/1 (100.0%)

microsecond table       (q_flow=0.001s, q_packet=0.001s)
  offset    : measured n=200 consistent=200 inconsistent=0
              delta=[+2.000,+2.000]s  median(biased)=+2.000s
  interval  : [+1.999000, +2.001000) feasible width=0.002000s
  apply     : applied +2.000000s (worst-case error 0.001000s)
  coverage  : 1/1 flow bins (100.0%), attack 1/1 (100.0%)
```

**What rounds 2 and 3 changed about this number.** Under the round-1 rule
("apply the median while the IQR looks small") this fixture *applied* +1.220 s,
and that correction was 0.78 s away from the planted truth - the IQR was small
only because the fixture's `frac` values happen to cluster, which is exactly the
kind of false confidence the median cannot defend. The round-2 rule asks the
question the median cannot answer: which offsets are consistent with *every*
matched pair given each clock's quantization? Here the answer is a 0.948952 s
interval that contains both +2.0 s (the truth) and +1.220 s (the biased median)
and refuses to choose, so `--apply-offset` exits 1 and writes nothing. The same
capture against a microsecond-precision flow table - where the flow clock can
actually express the sub-second - pins the interval to 0.002 s, and its midpoint
+2.000000 s is applied with the 0.001 s worst-case error the two clocks jointly
allow. Nothing was tuned to make that happen: one binary, two tables, different
evidence, different decisions - and the refusal is the honest one.

Two magnitude signatures the measurement reports instead of hiding: the pandas
`[us]` epoch regression, which would drive a delta to about -1.78e9 s (item 3,
pinned by `test_load_flow_rows_epoch_not_corrupted_by_pandas_unit`), and a
whole-hour magnitude, which is flagged `timezone_scale` and reported, never
silently corrected.

## 5. The streaming 60 s host-keyed path: a standalone extractor, not consumed by any training or benchmark code (pinned by `test_packet_tier_has_no_training_or_benchmark_consumer_yet`)

It is the **D2-designated extraction entry point**, and a *standalone* one -
extraction and measurement end to end, with no training or benchmark consumer:
that boundary is a test, not a comment (`test_packet_tier_has_no_training_or_benchmark_consumer_yet`
walks the tree and fails if any file but the CLI and these tests imports
`pcap_bins`), and honest boundaries about what does not exist yet:

**Wired:**

1. `argus/scripts/pcap_stream_bins.py` imports *only* from
   `ml.world_model.pcap_bins` (`bin_start_for`, `extract_pcap_bins`,
   `join_packet_features`, `measure_bin_coverage`, `measure_clock_offset`) -
   there is no second binning implementation.
2. The bin key is the identical D1 formula on both sides of the join:
   * flow side - `load_flow_tables` / `load_flow_rows` parse each table's
     `Timestamp` into unix seconds, then `main()` computes
     `flow_keys = [(r["src"], bin_start_for(r["ts"], args.bin_seconds))]`
     (`pcap_stream_bins.py:381-382`);
   * PCAP side - `extract_pcap_bins` keys every packet
     `key = (ip.src, bin_start_for(ts, bin_seconds))` (`pcap_bins.py:593`);
     `bin_start_for = floor(ts / bin_seconds) * bin_seconds`.
   * D1's own plan text now states the same formula (`world-model-core.md`
     D1 Decision amendment), so the contract is written down, not implied.
3. The join is a **left join on flow bins**: `join_packet_features(flow_keys,
   records)` returns one row per flow bin; uncovered bins carry
   `packet_features: null` + `packet_features_covered: false`
   (`test_uncovered_flow_bins_are_null_never_zero`). Coverage percentages
   (overall + attack) are computed by `measure_bin_coverage` and written into
   the output JSON - the exact fields the D3 `packet_tier` arm's provenance
   requires.
4. Cross-references point the *demo* path at this path, so no demo artifact can
   be mistaken for the pipeline's:
   `features.py:510-513` (the demo extractor docstring says the D1-keyed
   extractor *is* `ml/world_model/pcap_bins.py`, CLI `scripts/pcap_stream_bins.py`);
   `dashboard/app.py:693-695` (API disclosure names the same CLI); the plan
   D2/D3 amendments name the same files.
5. Guardrails travel with the path: `PcapReader` (never `rdpcap`) is enforced
   by `test_extractor_streams_with_pcapreader_never_rdpcap`; a packet that
   targets an already-closed bin fails closed - the run exits 1 with no output
   file at all, and only `--on-late drop` accepts the loss as a *counted* gap
   (round 3 removed the `--on-backwards merge` opt-in this line used to
   describe); `--apply-offset` refuses to run without `--flows` **and** refuses
   an ineligible measurement (exit 1, stderr reason, no output file), so a
   correction is applied only when it was *measured*, outlier-free and
   unambiguous.

**Not yet wired (by design, gated):**

* `argus/scripts/prepare_cicids2017.py` does **not exist** - WP2 has not
  started. The plan's D2 names it as the future producer of
  `cicids2017_bins_60s.parquet`; when written, its binning must use the D1
  formula above so `pcap_stream_bins.py --flows <parquet>` joins directly.
* No real CIC PCAP has been run through the CLI: `clock_offset` and
  `coverage` **on real CIC PCAPs remain "not yet measured"** - stated in the
  plan (D2 extractor status amendment) and unchanged by this review bundle.
* The demo path (`features.extract_features_from_pcap`) is deliberately *not*
  the pipeline path: fixed-slice, not wall-clock bins; both docstrings say so.

## 6. D1's flow-only schema vs D2's packet tier - the reconciliation

They are reconciled by the **two-tier model** (plan D2 "conflict resolution"
amendment), not by contradiction:

| | Flow tier (D1) | Packet tier (D2) |
|---|---|---|
| Columns | 39 = 24 flow + 6 extended + 8 topology + `n_flows` | the 8 `PACKET_LEVEL_COLUMNS` |
| Source | flow table only, PCAP-free | PCAP via `pcap_bins.py` |
| Key | `(src_ip, floor(unix/60)*60)` | **identical** `(src_ip, bin_start)` |
| Missing data | n/a | `null` + `packet_features_covered=false` - never zero-filled, never imputed |
| Role | **headline** - every headline D3 number | pre-registered **ablation** on covered bins only, coverage % in results JSON |
| Servable | yes - schema v2 unchanged | no - reported separately |

So D1's decision ("packet columns are **not part of** the flow-tier schema")
and D2's decision ("packet columns arrive through the packet tier, joined as
NaN + coverage flag") are two halves of one rule: *the servable model never
carries packet features; packet features exist only as a PCAP-joined,
coverage-scoped analysis arm*. Risk 6 records that design, not a resolution: it
reads "Mitigation designed (two-tier model); resolved only when a PCAP source is
approved and the `packet_tier` arm has run", because until a real CIC PCAP has
been measured the packet tier exists only as a design - the same fact the
Status section states from the code side ("Real CIC PCAP clock offset and bin
coverage: not yet measured").

## 7. Can one Thursday PCAP support the planned multi-day evaluation?

**No - and the plan forbids claiming otherwise.** Three statements pin it:

1. **D3 packet-tier arm**: "One Thursday PCAP yields at most a
   Thursday-scoped `pilot` row - never a multi-day packet-level claim - and
   the flow-tier arm remains the headline under every option."
2. **D2 scope honesty**: a Thursday-only PCAP (8.30 GB) "is *not*
   packet-level coverage for the five-day benchmark. Full coverage is 5 files
   / 52.43 GB ... which exceeds the free disk space and is a separate
   approval. No doc, metric or slide may claim 'packet-level results on
   CIC-IDS-2017' without it."
3. **Risk 6**: "A Thursday-only PCAP is a pilot, not benchmark coverage; full
   five-day PCAP coverage (52.43 GB) is a separate approval."

Concretely: P-A trains/test-splits **by day**; a single Thursday capture
supplies packet bins for one day only, so any packet-tier arm over a multi-day
split would have uncovered (NaN) bins on the other four days and would be
excluded from the arm by the no-zero-fill rule - leaving n=1 day. The
multi-day evaluation therefore runs on the **flow tier** (headline, complete);
the packet tier on one Thursday PCAP is a Thursday `pilot` row validating the
pipeline, nothing more.

## 8. Review round 1 - what each finding changed

| Finding | Fix | Pinned by |
|---|---|---|
| Backwards timestamps could pass silently (bins closed while the capture moved back) | watermark closing with `on_backwards="fail"` as the default, raising `BackwardsTimestampError` naming the closed key and the opt-in; `on_backwards="merge"` folds the packet into the retained accumulator and counts it in `backwards_packets_merged`, bounded by `reorder_grace_sec`; one record per key holds either way | `test_backwards_packet_fails_closed_by_default`, `test_backwards_merge_keeps_and_counts_the_packet`, `test_reorder_grace_keeps_a_recent_bin_open_without_failing`, `test_invalid_backwards_policy_and_grace_are_rejected` |
| Per-packet list accumulation - unaffordable on a multi-GB capture | `_OnlineStats` (Welford count/mean/population-variance) for TTL / TCP window / payload, plus only the per-direction sequence-end map; the close scan runs on bin-boundary crossings and is counted in `stats["watermark_scans"]` | `test_watermark_scans_scale_with_bins_not_packets` (600 packets -> at most 11 scans) |
| `measure_clock_offset` conflated "observed" with "applied" and could apply one median derived from too few or contradictory matches | split `observed_delta` (median, p25/p75, min/max, pairs, quantization, deltas) from an explicit `apply` block (`eligible`, `reason`, `correction_sec`, `flags`); application refused below 3 matched instances or IQR > 1 s (quantization-aware), `timezone_scale` flagged when abs(median) >= 3600 s; the CLI may use only `apply.correction_sec` and **exits 1** on a refusal | `test_offset_applied_only_when_measured_and_unambiguous`, `test_offset_apply_refused_when_too_few_matches`, `test_offset_apply_refused_on_ambiguous_dispersion`, `test_offset_flags_timezone_scale_without_correcting_it`, `test_cli_refuses_apply_offset_and_writes_no_output` |
| TCP and UDP collided on one key, and a reused tuple collapsed to its earliest flow | key is `(proto, src, dst, sport, dport)`; a reused tuple is split into connection instances (idle gap >= 120 s, or a fresh SYN >= 30 s after the instance start) and flow starts are paired i-to-i with connection starts, unpaired instances counted | `test_offset_matching_is_protocol_aware`, `test_offset_pairs_per_connection_instance_not_earliest` |
| `--flows` accepted exactly one file, so a multi-day table set needed repeated runs and lost per-file provenance | `resolve_flow_paths` + `load_flow_tables`: several files and/or directories in one run, deterministic parquet-then-csv sorted order, per-file stats (`files`) plus totals; missing paths and empty directories fail closed | `test_load_flow_tables_concatenates_files_and_directories`, `test_resolve_flow_paths_rejects_missing_or_empty_inputs` |
| NaN cells reached the join as the string `"nan"`; null timestamp padding was indistinguishable from garbage; rows could join on an empty key | `_cell_str` maps nulls to `None` (never `"nan"`), rows without a real IP/port are dropped and counted, `timestamp_null` / `timestamp_unparsable` / `timestamp_all_null` are separated, `Protocol` is normalised to `tcp`/`udp` | `test_load_flow_rows_null_label_and_missing_identity_never_become_nan`, `test_load_flow_rows_all_null_timestamp_column_is_padding_not_garbage`, `test_load_flow_rows_separates_unparsable_from_null_timestamps`, `test_load_flow_rows_normalises_protocol_codes` |
| Coverage denominators counted repeated rows, and an all-benign table printed attack coverage 0.0% | both sides de-duplicated to unique `(src_ip, bin_start)` host-minute bins first; a zero denominator yields `null` in the JSON and `"n/a"` on screen, never 0.0 | `test_bin_coverage_percentages`, `test_cli_all_benign_flow_table_reports_null_attack_coverage` |
| CLI behaviour existed only as a hand-run | five end-to-end tests drive `main(argv)` and assert on exit codes, the written JSON, and stdout/stderr - including the applied-vs-unapplied offset changing the join | `test_cli_applies_measured_offset_so_flow_bins_join`, `test_cli_refuses_apply_offset_and_writes_no_output`, `test_cli_all_benign_flow_table_reports_null_attack_coverage`, `test_cli_backwards_packet_exits_1_until_merge_is_requested`, `test_cli_rejects_bad_arguments` |

How to read this table now: it records what round 1 changed, and some of its
names have since been revised (the findings themselves still stand):

* `on_backwards="fail"/"merge"` became `on_late="fail"/"drop"`, and
  `backwards_packets_merged` no longer exists; `test_backwards_*`,
  `test_cli_backwards_*` and `test_invalid_backwards_*` are now
  `test_late_packet_*`, `test_cli_late_packet_*` and
  `test_invalid_late_policy_*` (round 3, item 10).
* The "apply the median while IQR <= 1 s" rule in the offset row was replaced by
  the feasible interval (round 2, item 9), so
  `test_offset_applied_only_when_measured_and_unambiguous` and
  `test_offset_apply_refused_on_ambiguous_dispersion` are now
  `test_offset_interval_contains_theta_and_width_decides_application` and
  `test_two_offset_populations_are_refused_as_inconsistent`.
* Everything else cited above - `test_offset_apply_refused_when_too_few_matches`,
  `test_offset_matching_is_protocol_aware`,
  `test_offset_pairs_per_connection_instance_not_earliest`,
  `test_cli_refuses_apply_offset_and_writes_no_output`, the coverage and
  flow-loading tests - still exists under the same name.

What did *not* change: the D1 bin key, the two-tier model (item 6), the
no-zero-fill rule, the Thursday-pilot limits (item 7), and the D2 gates - real
CIC PCAP offset and coverage are still **not yet measured**.

## 9. Review round 2 - what each finding changed

| Finding | Fix | Pinned by |
|---|---|---|
| The retransmission proxy counted honest reordering ("seq below the highest end seen" - seq 1000, 1020, 1010 scored 1 retransmission) and needed unbounded history to be right | a bin keeps one **merged sequence-range set per directional 5-tuple** and counts only a segment whose **whole byte range was already observed**; RST and pure ACKs never counted; memory bounded at `SEQ_RANGES_CAP = 64` merged ranges with overflow dropped and counted in `seq_range_sets_capped` | `test_in_window_reordering_is_not_a_sequence_regression`, `test_duplicate_byte_range_is_a_sequence_regression`, `test_partial_overlap_that_adds_new_bytes_is_not_counted`, `test_pure_acks_are_never_counted` (that is round 3's `test_keep_alives_are_never_counted`, renamed in round 4 because a keep-alive is not a pure ACK), `test_out_of_order_range_memory_is_capped` |
| A 0 s default grace made every slightly-out-of-order capture an error, while a generous grace was an unbounded backlog | `reorder_grace_sec` defaults to **1 s** and is hard-capped at `MAX_REORDER_GRACE_SEC = 300`; `stats.peak_retained_bins` and `stats.watermark_scans` report both bounds instead of asserting them | `test_default_late_policy_fails_and_grace_is_bounded`, `test_invalid_late_policy_and_grace_are_rejected`, `test_retained_bins_do_not_grow_with_capture_length`, `test_watermark_scans_scale_with_bins_not_packets` |
| Applying the **median** of quantized deltas is biased by up to one quantization step - this very fixture had +1.220 s applied against a +2.0 s truth, and the old IQR guard passed it | per-pair constraint `[delta - q_packet, delta + q_flow)` intersected into the **feasible interval** (`lo = max - q_packet`, `hi = min + q_flow`); apply the **midpoint only when the interval is non-empty and at most `OFFSET_MAX_INTERVAL_WIDTH_SEC = 0.1 s` wide**, reporting `max_error_sec` = half its width; the median is demoted to `median_sec_biased_under_quantization` and is never applied | `test_feasible_interval_is_the_intersection_of_per_pair_bounds`, `test_offset_interval_contains_theta_and_width_decides_application`, `test_minute_precision_is_pinnable_only_by_dense_sampling`, `test_applied_correction_keeps_packet_bins_on_the_flow_bins` |
| Timestamp precision was assumed where it should be measured | `_PrecisionCounter` determines each source's step (0.001 / 1 / 60 s grids, >= 20 values, counted over values rather than min/max); a *declared* step can never make the estimate more confident than the values allow; an undetermined step blocks application instead of defaulting to 1 s | `test_precision_reads_subsecond_second_and_minute_grids`, `test_precision_stays_unknown_from_too_few_values`, `test_precision_counts_values_not_min_max`, `test_declared_precision_never_beats_the_measured_values`, `test_offset_refused_when_precision_cannot_be_determined` |

## 10. Review round 3 - what each finding changed

| Finding | Fix | Pinned by |
|---|---|---|
| A handful of outlier pairs could move - or widen, or empty - the very interval that decides what gets applied, because that interval was the intersection over *all* pairs | the interval is built from the **majority consensus window** only: the largest set of deltas one constant offset could explain (one `q_flow + q_packet` window, two pointers, O(n)). Excluded pairs are counted in `observed_delta.n_inconsistent`, never averaged in, and more than `OFFSET_MAX_INCONSISTENT_FRACTION = 5%` of them is a refusal in its own right (`flag: inconsistent_pairs`) | `test_outlier_deltas_are_counted_not_averaged_and_capped`, `test_inconsistent_pair_tolerance_is_inclusive_at_the_bound`, `test_one_contradictory_pair_is_excluded_not_averaged_into_the_interval`, `test_two_offset_populations_are_refused_as_inconsistent` |
| `on_backwards="merge"` gave a closed bin a second lifetime: the record had already been emitted, memory was retained for a packet that might never come, and "how much reordering is acceptable" silently decided how much **memory** the run held | `on_late={fail,drop}`. A late packet - one arriving more than `bin_seconds + grace` after its bin started - **fails closed** by default: `BackwardsTimestampError` names the ts, the bin key, the watermark and both remedies, and the CLI exits 1 **before writing any output file**, so no partial bin set can look like a result. `on_late="drop"` is the only way to accept the loss, and it counts it in `backwards_packets_late_dropped` plus `bins_with_dropped_packets`. Reordering inside the grace is not late at all (that bin is still open), and with `merge` gone every accumulator in the map is open by definition and freed the moment it closes | `test_late_packet_fails_closed_and_names_both_remedies`, `test_late_packet_is_dropped_and_counted_only_when_requested`, `test_late_burst_for_one_bin_is_counted_once_per_bin`, `test_hours_late_packet_never_keeps_an_old_bin_alive`, `test_cli_late_packet_exits_1_until_drop_is_requested`, `test_cli_widening_the_grace_absorbs_the_reordering` |
| `seq_regression_count` named the counter after the rule that had just been removed, and the demo path described a high-water-mark heuristic in retransmission language | renamed `dup_range_count` - named for what it counts; the emitted column stays `retransmission_count` because `PACKET_LEVEL_COLUMNS` freezes that name, and `FEATURE_SEMANTICS` still carries the public wording. The demo path now states, in `features.py` and in the dashboard's own `approximations`, that it is a **sequence-regression proxy (demo path only; counts reordered segments too)** | `test_demo_path_is_a_sequence_regression_proxy_and_says_so` (one capture: 1 through the demo path, 0 through the streaming detector), `test_column_name_is_kept_and_its_semantics_are_reported` |
| A test bent around the operating system: `str(path.relative_to(ROOT))` renders `scripts\pcap_stream_bins.py` on Windows and `scripts/pcap_stream_bins.py` on POSIX, so the assertion had to accept both spellings | `relative_to(ROOT).as_posix()` - one form, one comparison | `test_packet_tier_has_no_training_or_benchmark_consumer_yet` |

What rounds 2 and 3 did *not* change: the D1 bin key, the two-tier model (item 6),
the no-zero-fill rule, the Thursday-pilot limits (item 7), the frozen column
names, and the D2 gates - real CIC PCAP offset and coverage are still **not yet
measured**.

## 11. Review round 4 - what each finding changed

| Finding | Fix | Where |
|---|---|---|
| D1 aggregated `unique_dst_ports_per_src` with `sum`, but a per-bin nunique of `Destination Port` cannot be summed | the aggregation sentence now says nunique over the bin's flows, computed from `Destination Port`, never summed; the frozen name stays | plan D1, hunk `@@ -30,11 +30,11 @@` |
| D1 asserted minute-precision timestamps as if that applied to every day - true Tuesday-Friday, **false for Monday** - and built "the floor is lossless" on that claim | precision is **measured, not asserted**: `validate_cicids2017.py::measure_timestamp_precision` derives each file's step from its own values and writes `quantization_sec` + `basis` into `cicids2017_provenance.json` (Monday **1.0 s**, Tuesday-Friday **60.0 s**); the plan states both and swaps the lossless-floor claim for the quantization-aware feasible-interval rule on the PCAP side | plan D1; `argus/scripts/validate_cicids2017.py`; `argus/data/cicids2017_provenance.json` |
| `retrieved_utc` was `datetime.now()`, so refreshing the provenance record claimed a new retrieval | pinned constant, commented as a recorded fact | same validator |
| D3 never said what the stage head may be trained on, while the label map sent Infiltration -> LateralMovement and Heartbleed -> WebAttack (a 7th class; an 11-sample class) | **Stage vocabulary**: 6 observed stages; the two small families enter the **binary** target only with their support; the two label-map rows are declared superseded, their rationale kept for provenance | plan D3 + label-map preamble |
| tau had no defined selection slice, the FPR beside lead time was ambiguous, lead time had no onset count, and a per-victim view was implied | **tau and lead time**: tau from a validation slice that never touches test bins, test-set FPR at that tau labelled as such, onset count n (12-15) reported; **per-victim (dst-host) view rejected** with reasons and a reconstruction path from the results JSON | plan D3 |
| the manifest had no independent sanity check on the dataset it describes | `label_anchors` + `label_anchor_violations: []` checked at **±3 %** by `prepare_cicids2017.py --check` (exact for the 36/21/11 families), measured against this snapshot - all pass, worst Bot -1.7 %; explicitly a guard, never a substitute for revision+SHA+size pinning | plan 2.2 |
| Risk 6 was written as **resolved** while nothing had been measured on a PCAP | "**Mitigation designed** (two-tier model); **resolved only when** a PCAP source is approved and the `packet_tier` arm has run" | plan Risks; item 6 above |
| item 4 was titled as if the +1.220 s were a mystery about the planted offset, and argued from "measured == predicted" and a `(P-1, P]` tolerance band - the wrong criterion, since that band passes the very number round 1 wrongly applied | retitled to what happened (the round-1 **median** read +1.220 s for a +2.000 s truth) and **both paragraphs deleted**; the sign/unit-bug facts the deleted paragraph carried are kept, in the place they belong | item 4 above |
| item 5 asked "is the streaming path *wired into the pipeline*?", which over-sold it | retitled to what a test actually pins: **a standalone extractor, not consumed by any training or benchmark code** (`test_packet_tier_has_no_training_or_benchmark_consumer_yet`), and the "consumer" bullet now says plainly that the cross-references point the *demo* path here | item 5 above |
| a 1-byte keep-alive / window probe sits in already-covered sequence space (`seq = SND.NXT - 1`), so the duplicate-range rule counted it: an idle connection that merely keeps itself alive read as a sequence regression | `_observe_tcp` skips a covered segment that occupies exactly one sequence number and ends at the edge of the observed run, and counts the skip in `keep_alive_probes_skipped` -> `stats.tcp_keep_alive_probes_skipped`, because an exemption that is not reported is a silent one. `FEATURE_SEMANTICS` and `APPROXIMATIONS` now name probes alongside RST and pure ACKs, and the disclosed cost is stated where the rule lives: a genuine one-byte retransmission exactly at that edge is indistinguishable from a probe | `test_one_byte_keep_alive_is_not_a_duplicate` (20 probes -> 0 counted, 20 skipped, all 21 packets still counted; and the exemption stays narrow - a 1-byte duplicate away from the edge is still counted) |
| `test_keep_alives_are_never_counted` pinned **zero-length ACKs**, which are not keep-alives - a keep-alive carries a sequence number | renamed `test_pure_acks_are_never_counted`, pointing at the new probe test for the case it never covered | `test_pure_acks_are_never_counted`, `test_one_byte_keep_alive_is_not_a_duplicate` |
| `measure_clock_offset`'s docstring illustrated the median bias with an invented +7.0 s read for a +7.34 s offset, while the number this bundle actually measured was on the tip of the same file | the example is now the measurement: +1.220 s read for a +2.000 s plant over 200 consistent pairs, the 0.780 s shortfall equal to the deltas' mean sub-step - the value round 1 applied | docstring only (item 4 keeps the reasoning) |
| the status block's re-smoke numbers existed only as prose - nothing failed if the fixture measurement drifted | the claim is now a committed test: it streams `data/sample_portscan.pcap` with `with PcapReader`, builds one row per TCP directional 5-tuple planting +2.0 s at second precision (`floor(t + 2.0)`) and at millisecond precision (`floor((t + 2.0) * 1000) / 1000`), and pins `n_matched_pairs == 200`, `n_inconsistent == 0`, median 1.219773, width 0.948952, `lo <= 2.0 < hi`, the refusal (`eligible is False`, `"wide"` in the reason) - so the refusal is pinned to the width guard - and the ms twin applying within 1 ms; skips loudly if the capture is absent | `test_sample_portscan_planted_offset_is_refused_at_second_precision` (2026-09-30) |
| every `file:line` reference and hunk header in this bundle had drifted as the files grew | re-derived from this tip: `pcap_bins.py:593` / `:918-1226`, `pcap_stream_bins.py:98-112` / `:124-212` / `:381-382`, `features.py:510-513`, `dashboard/app.py:693-695`, and all 8 hunk headers in item 1; the `.diff` artifacts regenerated with `git diff --output=` | throughout |

## Status of this bundle

* 66/66 `test_pcap_bins.py`; full suite **121 passed, 4 skipped**
  (2026-09-30, round 4; `pytest -rs` skips: 3 x PyTorch not installed,
  1 x antivirus blocked the EICAR fixture).
* CLI re-smoked end-to-end on `data/sample_portscan.pcap` against two synthetic
  flow tables built from the capture itself, both planting the same +2.0 s
  offset. The **second-precision** (CIC text format) table measures
  `median(biased) = +1.220 s` over n=200 consistent pairs and a
  **0.948952 s feasible interval**, so `--apply-offset` **exits 1 and writes no
  output file** - item 4 explains why that is the correct answer rather than a
  threshold to widen; the same measurement without `--apply-offset` is written
  with the refusal recorded in `clock_offset.apply`, coverage 1/1 bins and
  1/1 attack bins. Two twin figures exist because the row builders differ; each is
  labeled in place: (CLI smoke, µs-rounded rows: +2.000000 / 0.0009998)
  pins the interval to 0.002 s, applies, and reports coverage 1/1 and
  attack 1/1 with all 8 packet columns populated; (test, ms-floored rows:
  1.999501 / 0.000501) applies within 1 ms of the same +2.000 s plant.
  Both figures, and the second-precision refusal above, are reproduced by
  `test_sample_portscan_planted_offset_is_refused_at_second_precision`
  (pairs are correction_sec / max_error_sec).
* Late packets re-smoked on the same capture with one packet 7200 s early: the
  default run exits 1 naming ts, bin key, watermark and both remedies and
  writes nothing; `--on-late drop` exits 0 with 1 dropped packet against exactly
  the bin it would have filled (`bins_with_dropped_packets`), 201 packets
  considered, `peak_retained_bins: 1`.
* **Real CIC PCAP clock offset and bin coverage: not yet measured.** The D2
  gates (streaming extractor, measured offset, measured coverage) are
  unchanged; PCAP source option (a)/(b)/(c) is still unapproved; no PCAP
  download has happened.
* Review gate: rounds 1-4 are commits on
  `review/d1-d2-d3-packet-tier` (the branch is on `origin`); `origin/main` is
  merged in and touches none of the files under review; nothing merges into
  `main` without approval.
