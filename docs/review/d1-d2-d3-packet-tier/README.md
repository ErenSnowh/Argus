# Review bundle - D1/D2/D3 + label-map, packet-tier extractor

Branch `review/d1-d2-d3-packet-tier` (code commit `b36148c`, bundle commit
`a61ed27`, round-1 fixes `e107158`) against `main` (`c53652f`). Prepared
2026-09-28.

**Review round 1 is folded in.** The findings of the first review round are
fixed in the same three files and are described item by item in **item 8**;
the `.diff` artifacts here were regenerated against `main`, so they
already contain those fixes. WP2/training not started; no PCAPs downloaded;
`main` untouched - merge only on approval.

## Contents

| File | What it is |
|---|---|
| `plan-world-model-core.diff` | the complete plan diff: D1, D2, D3, label map, risk 6 |
| `extractor-pcap_bins-and-pcap_stream_bins.diff` | exact diff of the extractor (2 new files) |
| `tests-test_pcap_bins.diff` | exact diff of the tests (1 new file) |
| `README.md` | this file - answers to all review questions |

Regenerate with:

```
# two dots against main: picks up every commit on this branch plus any
# uncommitted fix on top of it (three dots would compare commits only):
git diff main -- docs/plan/world-model-core.md
git diff main -- argus/ml/world_model/pcap_bins.py argus/scripts/pcap_stream_bins.py
git diff main -- argus/tests/test_pcap_bins.py
```

## 1. Changed plan sections (all 5 hunks in `plan-world-model-core.diff`)

| Hunk | Section | Change |
|---|---|---|
| `@@ -30,11 +30,11 @@` | **D1** Decision + Rejected alternatives | bin key pinned as `bin_start = floor(unix_seconds / 60) * 60` (UTC), key `(src_ip, bin_start)`; the 8 packet columns are now "**not part of the flow-tier schema (v2) and are never zero-filled into it**; they arrive, if at all, through the packet tier of the amended D2"; rejected-alt (b) rewritten to name the gated 52.43 GB set and the NaN-with-coverage-flag rule |
| `@@ -44,7 +44,22 @@` | **D2** | four additions (+ 1 round-1 paragraph): (i) *acquisition status* - 8 Parquet files from the pinned HF mirror `bvsam/cic-ids-2017 @ 70bac624`, validated by `scripts/validate_cicids2017.py`, 288,602 all-null padding rows observed-not-asserted, dropped never zero-filled; (ii) *flow vs. packet features - conflict resolution*: two-tier model (see item 6); (iii) *PCAP source decision*: options (a) Thursday-only / (b) stream-and-cut <= 2 GB / (c) baseline-only, with the three gates that hold under any option (streaming fixed-60 s extractor, measured clock offset, measured coverage; until measured on real CIC PCAPs they are "not yet measured"); (iv) *extractor implementation status*: what is implemented and green today |
| `@@ -80,6 +95,8 @@` | **D3** | *packet-tier arm*: only if a PCAP source is approved, one extra arm restricted to PCAP-covered bins, labelled `packet_tier`, provenance carries `pcap_file`, `clock_offset_sec`, `bins_flow_total`, `bins_covered`, `attack_bins_covered`; uncovered bins excluded (NaN, never zero-filled); one Thursday PCAP = at most a Thursday-scoped `pilot` row; flow-tier arm stays the headline |
| `@@ -172,7 +189,7 @@` | **label map** | the mojibake row is no longer hypothetical: measured over all 8 files, every Web-Attack row spells all three families `Web Attack \u0096 ...` (U+0096) - none use ASCII hyphen or U+FFFD; `_CICIDS2017_LABEL_MAP` lacks that spelling, so today every such row falls through `.get(x, "BENIGN")` at `dataset_loader.py:691` (V12, confirmed live); new entries map Brute Force -> BruteForce, XSS/Sql Injection -> WebAttack; unknown labels now **raise** with `unmapped_labels: []` required in the manifest |
| `@@ -291,12 +308,13 @@` | **Risks** | item 6 "PS 1 asks for packet-level features" rewritten from open risk to **Resolved in D2 (two-tier model)** - see item 6 below |

## 2. Exact code and test diffs

* `extractor-pcap_bins-and-pcap_stream_bins.diff` - both files are **new** (`new file mode 100644`):
  * `argus/ml/world_model/pcap_bins.py` (750 lines): `bin_start_for`, `BackwardsTimestampError`, `_OnlineStats` (O(1) Welford count/mean/variance), `_BinAccumulator`, `extract_pcap_bins` (streams `PcapReader`; exactly one record per `(src_ip, bin_start)`; closes bins on a watermark scan run at bin boundaries; `on_backwards="fail"` raises, `on_backwards="merge"` folds late packets into retained accumulators bounded by `reorder_grace_sec`), `_is_closed`, `measure_clock_offset` (protocol-aware, per-connection-instance matching; `observed_delta` reported separately from a guarded `apply` decision), `measure_bin_coverage` (unique host-minute bins, null on an empty denominator), `join_packet_features`, `unsupported_or_null`.
  * `argus/scripts/pcap_stream_bins.py` (408 lines): the CLI (`--pcap`, `--flows` - many files *and* directories at once, `--bin-seconds`, `--out`, `--apply-offset`, `--on-backwards`, `--reorder-grace-sec`) plus the flow-table layer `_find_col`, `_cell_str`, `_norm_proto`, `load_flow_rows`, `resolve_flow_paths`, `load_flow_tables`. `--apply-offset` requires `--flows` **and** an eligible measurement: a refusal prints the reason to stderr and exits 1 without writing any output file.
* `tests-test_pcap_bins.diff` - `argus/tests/test_pcap_bins.py` (**new**), 33 tests. First round (10): `test_fixed_60s_bins_keyed_by_src`, `test_rst_not_counted_true_retransmission_counted`, `test_demo_extractor_no_longer_counts_rst_as_retrans`, `test_uncovered_flow_bins_are_null_never_zero`, `test_tcp_window_null_when_no_tcp_packets`, `test_clock_offset_measured_on_matched_5tuple`, `test_clock_offset_not_measured_when_no_match`, `test_bin_coverage_percentages`, `test_extractor_streams_with_pcapreader_never_rdpcap`, `test_load_flow_rows_epoch_not_corrupted_by_pandas_unit`. Round 1 (23), grouped in the file by section: backwards/merge + watermark - `test_backwards_packet_fails_closed_by_default`, `test_backwards_merge_keeps_and_counts_the_packet`, `test_reorder_grace_keeps_a_recent_bin_open_without_failing`, `test_invalid_backwards_policy_and_grace_are_rejected`, `test_unsupported_or_null_names_only_null_statistics`, `test_watermark_scans_scale_with_bins_not_packets`; offset observation vs guarded apply - `test_offset_applied_only_when_measured_and_unambiguous`, `test_offset_apply_refused_when_too_few_matches`, `test_offset_apply_refused_on_ambiguous_dispersion`, `test_offset_flags_timezone_scale_without_correcting_it`, `test_offset_matching_is_protocol_aware`, `test_offset_pairs_per_connection_instance_not_earliest`; flow-table null / multi-file loading - `test_load_flow_rows_null_label_and_missing_identity_never_become_nan`, `test_load_flow_rows_all_null_timestamp_column_is_padding_not_garbage`, `test_load_flow_rows_separates_unparsable_from_null_timestamps`, `test_load_flow_rows_normalises_protocol_codes`, `test_load_flow_tables_concatenates_files_and_directories`, `test_resolve_flow_paths_rejects_missing_or_empty_inputs`; CLI end-to-end - `test_cli_applies_measured_offset_so_flow_bins_join`, `test_cli_refuses_apply_offset_and_writes_no_output`, `test_cli_all_benign_flow_table_reports_null_attack_coverage`, `test_cli_backwards_packet_exits_1_until_merge_is_requested`, `test_cli_rejects_bad_arguments`.

Verification run on 2026-09-28 (round 1): `pytest argus/tests/test_pcap_bins.py` -> **33 passed**; full suite -> **88 passed, 4 skipped**.

## 3. Timestamp-to-epoch conversion code

`argus/scripts/pcap_stream_bins.py:96-173` (`load_flow_rows`) - round-1 version,
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
  (`pcap_stream_bins.py:70-83`), which maps `None`/NaN/NA/empty/`"nan"`/
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

## 4. Why a "+2 s planted" fixture measured +1.220 s

**Sign convention.** `measure_clock_offset` (`pcap_bins.py:455-662`) computes,
for every protocol-aware directional 5-tuple present in both sources - paired
**per connection instance**, not per tuple - the delta

```
offset = flow_ts - first_packet_ts      (median over matched pairs)
```

A **positive** offset means the flow-table clock reads *later* than the PCAP
clock (flow timestamps are shifted into the future relative to packets). The
PCAP is what gets corrected: `--apply-offset` adds `apply.correction_sec` (the
median, and only when the guard declares it eligible) to packet timestamps
before binning, aligning the PCAP onto the flow-table clock. The
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

**Live re-verification (2026-09-28, rebuilt fixture, exact same construction):**

```
offset    : measured median=+1.220s iqr=0.028s n=200
coverage  : 1/1 flow bins (100.0%), attack 1/1 (100.0%)
median=1.219773 iqr=0.027672 n=200
predicted measured median = 1.219773
```

Measured == predicted to 6 decimals, so the number is fully explained: the
0.780 s gap is the **sub-second component of the PCAP clock that second-precision
flow timestamps cannot represent**, not a clock-offset estimation error.

**Expected tolerance.** For a plant of `+P` seconds written second-precision,
any correct measurement must satisfy `measured ∈ (P-1, P]` (frac lies in
`[0,1)`), specifically `[P - max_frac, P - min_frac]` for a given capture:
here `[1.197, 1.249]`, IQR bounded by the frac span (0.052). The CLI's own
report shows `p25/p75/min/max/dispersion_iqr_sec` so reviewers can check the
band. Outside `(-inf, P]` would indicate a sign or unit bug - the pandas
`[us]` bug drove the measured delta to about -1.78e9 s (flow epoch 1000x too
small minus real PCAP seconds; flagged by the regression test); a whole
hours-sized offset would indicate a timezone-labelled flow clock (reported,
never silently corrected).

## 5. Is the streaming 60 s host-keyed path actually wired into the pipeline?

It is wired as the **D2-designated extraction entry point**, with honest
boundaries about what does not exist yet:

**Wired:**

1. `argus/scripts/pcap_stream_bins.py` imports *only* from
   `ml.world_model.pcap_bins` (`bin_start_for`, `extract_pcap_bins`,
   `join_packet_features`, `measure_bin_coverage`, `measure_clock_offset`) -
   there is no second binning implementation.
2. The bin key is the identical D1 formula on both sides of the join:
   * flow side - `load_flow_tables` / `load_flow_rows` parse each table's
     `Timestamp` into unix seconds, then `main()` computes
     `flow_keys = [(r["src"], bin_start_for(r["ts"], args.bin_seconds))]`
     (`pcap_stream_bins.py:309-310`);
   * PCAP side - `extract_pcap_bins` keys every packet
     `key = (ip.src, bin_start_for(ts, bin_seconds))` (`pcap_bins.py:380`);
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
4. Cross-references put every consumer on this path:
   `features.py:511-513` (the demo extractor docstring says the D1-keyed
   extractor *is* `ml/world_model/pcap_bins.py`, CLI `scripts/pcap_stream_bins.py`);
   `dashboard/app.py:690-692` (API disclosure names the same CLI); the plan
   D2/D3 amendments name the same files.
5. Guardrails travel with the path: `PcapReader` (never `rdpcap`) is enforced
   by `test_extractor_streams_with_pcapreader_never_rdpcap`; a packet that
   targets an already-closed bin aborts the run unless `--on-backwards merge`
   is explicitly requested; `--apply-offset` refuses to run without `--flows`
   **and** refuses an ineligible measurement (exit 1, stderr reason, no output
   file), so a correction is applied only when it was *measured* and
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
coverage-scoped analysis arm*. Risk 6 was rewritten from "open risk" to
"Resolved in D2 (two-tier model)" to record this.

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

What did *not* change: the D1 bin key, the two-tier model (item 6), the
no-zero-fill rule, the Thursday-pilot limits (item 7), and the D2 gates - real
CIC PCAP offset and coverage are still **not yet measured**.

## Status of this bundle

* 33/33 `test_pcap_bins.py`; full suite **88 passed, 4 skipped** (2026-09-28,
  round 1).
* CLI re-smoked end-to-end on `data/sample_portscan.pcap` against a synthetic
  **two-file** flow table: offset measured +1.220 s / IQR 0.028 s / n=200
  (fully explained in item 4; flag `second_precision_flow_timestamps`),
  `--apply-offset` applied +1.220 s, coverage 100% / attack 100%. The same
  command with a one-row flow table exits 1 with the refusal reason and writes
  no output file.
* **Real CIC PCAP clock offset and bin coverage: not yet measured.** The D2
  gates (streaming extractor, measured offset, measured coverage) are
  unchanged; PCAP source option (a)/(b)/(c) is still unapproved; no PCAP
  download has happened.
* Review gate: round-1 fixes are working-tree changes on
  `review/d1-d2-d3-packet-tier` (the branch is on `origin`); nothing merges
  into `main` without approval.
