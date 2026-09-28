# Review bundle - D1/D2/D3 + label-map, packet-tier extractor

Branch `review/d1-d2-d3-packet-tier` (commit `b36148c`) against `main` (`c53652f`).
Prepared 2026-09-28. Nothing pushed; WP2/training not started; no PCAPs downloaded.

## Contents

| File | What it is |
|---|---|
| `plan-world-model-core.diff` | the complete plan diff: D1, D2, D3, label map, risk 6 |
| `extractor-pcap_bins-and-pcap_stream_bins.diff` | exact diff of the extractor (2 new files) |
| `tests-test_pcap_bins.diff` | exact diff of the tests (1 new file) |
| `README.md` | this file - answers to all review questions |

Regenerate with:

```
git diff main...HEAD -- docs/plan/world-model-core.md
git diff main...HEAD -- argus/ml/world_model/pcap_bins.py argus/scripts/pcap_stream_bins.py
git diff main...HEAD -- argus/tests/test_pcap_bins.py
```

## 1. Changed plan sections (all 5 hunks in `plan-world-model-core.diff`)

| Hunk | Section | Change |
|---|---|---|
| `@@ -30,11 +30,11 @@` | **D1** Decision + Rejected alternatives | bin key pinned as `bin_start = floor(unix_seconds / 60) * 60` (UTC), key `(src_ip, bin_start)`; the 8 packet columns are now "**not part of the flow-tier schema (v2) and are never zero-filled into it**; they arrive, if at all, through the packet tier of the amended D2"; rejected-alt (b) rewritten to name the gated 52.43 GB set and the NaN-with-coverage-flag rule |
| `@@ -44,7 +44,20 @@` | **D2** | four additions: (i) *acquisition status* - 8 Parquet files from the pinned HF mirror `bvsam/cic-ids-2017 @ 70bac624`, validated by `scripts/validate_cicids2017.py`, 288,602 all-null padding rows observed-not-asserted, dropped never zero-filled; (ii) *flow vs. packet features - conflict resolution*: two-tier model (see item 6); (iii) *PCAP source decision*: options (a) Thursday-only / (b) stream-and-cut <= 2 GB / (c) baseline-only, with the three gates that hold under any option (streaming fixed-60 s extractor, measured clock offset, measured coverage; until measured on real CIC PCAPs they are "not yet measured"); (iv) *extractor implementation status*: what is implemented and green today |
| `@@ -80,6 +93,8 @@` | **D3** | *packet-tier arm*: only if a PCAP source is approved, one extra arm restricted to PCAP-covered bins, labelled `packet_tier`, provenance carries `pcap_file`, `clock_offset_sec`, `bins_flow_total`, `bins_covered`, `attack_bins_covered`; uncovered bins excluded (NaN, never zero-filled); one Thursday PCAP = at most a Thursday-scoped `pilot` row; flow-tier arm stays the headline |
| `@@ -172,7 +187,7 @@` | **label map** | the mojibake row is no longer hypothetical: measured over all 8 files, every Web-Attack row spells all three families `Web Attack \u0096 ...` (U+0096) - none use ASCII hyphen or U+FFFD; `_CICIDS2017_LABEL_MAP` lacks that spelling, so today every such row falls through `.get(x, "BENIGN")` at `dataset_loader.py:691` (V12, confirmed live); new entries map Brute Force -> BruteForce, XSS/Sql Injection -> WebAttack; unknown labels now **raise** with `unmapped_labels: []` required in the manifest |
| `@@ -291,12 +306,13 @@` | **Risks** | item 6 "PS 1 asks for packet-level features" rewritten from open risk to **Resolved in D2 (two-tier model)** - see item 6 below |

## 2. Exact code and test diffs

* `extractor-pcap_bins-and-pcap_stream_bins.diff` - both files are **new** (`new file mode 100644`):
  * `argus/ml/world_model/pcap_bins.py` (405 lines): `bin_start_for`, `_BinAccumulator`, `extract_pcap_bins` (streams `PcapReader`, flushes closed bins, stops flushing on non-monotonic time), `measure_clock_offset`, `measure_bin_coverage`, `join_packet_features`.
  * `argus/scripts/pcap_stream_bins.py` (216 lines): the CLI (`--pcap`, `--flows`, `--bin-seconds`, `--out`, `--apply-offset`; `--apply-offset` requires `--flows` so an offset can only be applied when it was measured).
* `tests-test_pcap_bins.diff` - `argus/tests/test_pcap_bins.py` (**new**), 10 tests:
  `test_fixed_60s_bins_keyed_by_src`, `test_rst_not_counted_true_retransmission_counted`, `test_uncovered_flow_bins_are_null_never_zero`, `test_tcp_window_null_when_no_tcp_packets`, `test_clock_offset_measured_on_matched_5tuple`, `test_clock_offset_not_measured_when_no_match`, `test_bin_coverage_percentages`, `test_extractor_streams_with_pcapreader_never_rdpcap`, `test_demo_extractor_no_longer_counts_rst_as_retrans`, `test_load_flow_rows_epoch_not_corrupted_by_pandas_unit`.

Verification run on 2026-09-28: `pytest argus/tests/test_pcap_bins.py` -> **10 passed**; full suite -> **65 passed, 4 skipped**.

## 3. Timestamp-to-epoch conversion code

`argus/scripts/pcap_stream_bins.py:79-104` (`load_flow_rows`), verbatim:

```python
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
```

Notes:

* `dayfirst=True` matches CIC's `dd/mm/YYYY` format; `errors="coerce"` counts
  unparsable timestamps into `timestamp_unparsable` in the output provenance
  instead of failing silently or crashing.
* Naive timestamps are taken as-is (no timezone shift); if a flow clock is
  timezone-labelled the *measured* offset exposes it as a large `|median|`
  (hours), which is reported, never silently corrected.
* **The pandas 3 regression**: pandas 3 parses to `datetime64[us]`. Calling
  `.astype("int64")` directly on a `us` array yields **microseconds**, so
  dividing by `1e9` produced epoch values 1000x too small (≈1.78e6 instead of
  ≈1.78e9) - the original bug. The fix forces `to_numpy(dtype="datetime64[ns]")`
  first (the numpy unit cast does the scaling correctly), then
  `astype("int64") / 1e9` is seconds. Guarded by
  `test_load_flow_rows_epoch_not_corrupted_by_pandas_unit`.

## 4. Why a "+2 s planted" fixture measured +1.220 s

**Sign convention.** `measure_clock_offset` (`pcap_bins.py:267-341`) computes,
for every directional 5-tuple present in both sources, the delta

```
offset = flow_ts - first_packet_ts      (median over matched tuples)
```

A **positive** offset means the flow-table clock reads *later* than the PCAP
clock (flow timestamps are shifted into the future relative to packets). The
PCAP is what gets corrected: `--apply-offset` adds `median_sec` to packet
timestamps before binning, aligning the PCAP onto the flow-table clock. The
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
   * flow side - `load_flow_rows` parses the flow table's `Timestamp`, then
     `main()` computes `flow_keys = [(r["src"], bin_start_for(r["ts"], bin_seconds))]`
     (`pcap_stream_bins.py:151-155`);
   * PCAP side - `extract_pcap_bins` keys every packet
     `key = (ip.src, bin_start_for(ts, bin_seconds))` (`pcap_bins.py:232`);
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
   by `test_extractor_streams_with_pcapreader_never_rdpcap`; `--apply-offset`
   refuses to run without `--flows`, so an offset can only be applied when it
   was *measured*.

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

## Status of this bundle

* 10/10 `test_pcap_bins.py`; full suite **65 passed, 4 skipped** (2026-09-28).
* CLI re-smoked end-to-end on `data/sample_portscan.pcap`: offset
  +1.220 s / IQR 0.028 s / n=200 (fully explained in item 4), coverage 100% /
  attack 100%, `--apply-offset` path exercised.
* **Real CIC PCAP clock offset and bin coverage: not yet measured.** The D2
  gates (streaming extractor, measured offset, measured coverage) are
  unchanged; PCAP source option (a)/(b)/(c) is still unapproved; no PCAP
  download has happened.
* Review gate: this branch is not pushed; merge/push only on approval.