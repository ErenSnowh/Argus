# ARGUS - World-Model Core: Phase 1 Plan

| | |
|---|---|
| Problem statement | SIH 2026, PS 26153 (NTRO / NCIIPC) |
| Baseline commit | `bcc62be` (`main`) |
| Date | 2026-09-28 |
| Governing rule | No unprovable numbers. Every metric in a README, docstring, dashboard string or slide must trace to a committed results JSON with a provenance block (dataset, files, rows, split protocol, git SHA, UTC timestamp). If it cannot be measured, write "not yet measured". |
| Landed so far | WP0 (fresh-clone green, truthful health flags, artifact fallback) and WP1.5 (no fabricated forecast for unparseable uploads) |

## 2.1 Decisions

### Evidence base (re-verified against the tree on 2026-09-28)

| # | Defect | Evidence (file:line) |
|---|--------|----------------------|
| V1 | The engine that runs on a fresh clone is a hand-typed table, argmaxed at every step | `ml/world_model/predictor.py:205-226` (`TRANSITION_MATRIX`, `STAGE_INFILTRATION_BASE`), dispatch `:306-308`, `_predict_empirical` `:415-443` |
| V2 | "Sequences" are windows over consecutive CSV rows, i.e. flows from every host interleaved | `ml/world_model/dataset_loader.py:299-312` |
| V3 | `load_cicids2017` drops `Source IP`, `Destination IP` and `Timestamp`, so a state is anchored to neither host nor time; `train_test_split_by_day` is unreachable for 2017 | `dataset_loader.py:648-704` (returns `WORLD_MODEL_FEATURES + ["label"]`), `:353` |
| V4 | 8 packet-level features, absent from every CICFlowMeter CSV, are silently zero-filled | `dataset_loader.py:696-698`; schema `ml/world_model/features.py:71-80,97-102` |
| V5 | Every accuracy figure is a synthetic artifact: independent Gaussian blobs with separated means, 70% of sequences from 8 fixed scripts | `features.py:150-375` (`unique_dst_ports_per_src`: PortScan 180+/-50 vs BENIGN 2+/-1) |
| V6 | Horizon head `h300` (offset 10 with `seq_len` 10) is never trained but is still reported | `ml/world_model/model.py:286`, silent skip `:331-333` |
| V7 | Attention hooks never fire; explainability silently degrades to "largest z-score" | `model.py:200-209`; `predictor.py:142-162` |
| V8 | Causal mask is only moved to the device when `x.is_cuda` | `model.py:235-237` |
| V9 | The benchmark scores stage-at-t-given-x_t (detection), so the World Model gets no forecasting advantage; the lead-time metric rewards alerting at step 0 of every sequence | `ml/world_model/benchmark.py:471-487`, `:115-191`, `:152-163` |
| V10 | Unprovable numbers in docs and UI | 38 features (`README.md:60,112`) vs `NUM_FEATURES` = 46 (`features.py:97-102,139`); 612,079 params / 99.975% / AUC 0.983 (`README.md:160,176-179`); "Bayesian Dirichlet smoothing" (`README.md:141`) and "attention-weighted" (`README.md:36`), neither of which exists; "37 unit tests" (`README.md:412,491`) vs "22 tests" (`argus/README.md:113,145`); embedded-API-key claims (`argus/README.md:69,80,119,151`, `deploy/README.md:3`, `deploy/Dockerfile:11`) contradict `config.py:9`; unreproducible CSE-CIC-2018 claim (`dashboard/static/index.html:1086`); hard-coded UI numbers (`index.html:262,512,517,933,1155`) |
| V11 | `torch` (and `shap`, `requests`) are undeclared, so Render and every judge run only the heuristic path | `argus/requirements.txt` |
| V12 | `load_cicids2017` maps unknown labels to BENIGN through `.get(x, "BENIGN")`, so a mojibake Web-Attack label is filed as benign | `dataset_loader.py:691` |
| V13 | Uncommitted `/api/upload` answered an unparseable file with a hard-coded CRITICAL forecast (`0.954`, `+67.5s`) built from a synthetic sample, and read a non-existent key `total_packets` (the analyzer returns `packet_count`) so a literal `15.0` was always used | fixed in WP1.5; underlying key mismatch at `mcp_server/pcap_forensics.py:98` |

### D1 - A state S_t is one source host aggregated over one 60 s bin

**Decision.** Entity = source host (`Source IP`); bin = a fixed 60 s wall-clock window; S_t = the aggregate of every flow that host emitted inside the bin. Bin edges are fixed to the wall clock: `bin_start = floor(unix_seconds / 60) × 60` (UTC) and the bin key is `(src_ip, bin_start)`; flow timestamps in this snapshot are second-precision, so the floor is lossless. The packet tier (amended D2) keys its bins with the identical formula, so flow and PCAP bins join without re-binning. Per-bin features: the 24 `FLOW_FEATURE_COLUMNS` (`features.py:33-58`) and 6 `EXTENDED_FLOW_COLUMNS` (`:61-68`), aggregated with `sum` for counters (`*_packets`, `*_bytes`, `*_flag_count`, `unique_dst_ports_per_src`) and `mean` + `std` for durations, lengths, IATs, rates and ratios; the 8 `TOPOLOGY_FEATURE_COLUMNS` (`:85-94`) computed from the bin's real `dst_ip`, `dst_port` and `timestamp` columns (`src_fanout`, `dst_fanin`, `unique_dst_hosts`, `unique_src_hosts`, and `new_host_edges` / `cross_subnet_edges` / `new_dst_ports` measured against every prior bin of that host, `connection_repetition`); plus `n_flows`. The 8 `PACKET_LEVEL_COLUMNS` (`:71-80`) are **not part of the flow-tier schema (v2) and are never zero-filled into it**; they arrive, if at all, through the packet tier of the amended D2, joined on the identical `(src_ip, bin_start)` key as **NaN with a coverage flag wherever the PCAP does not cover the bin**. Schema v2 = 24 + 6 + 8 topology + `n_flows` = **39 columns**, stored in the checkpoint as an ordered `feature_list`.

**Rationale.** `P(S_{t+1} | S_t)` only means something if S_t is the state of a persisting entity. V2 windows interleaved flows from every host on the network, which is a per-flow classifier in disguise and cannot express "this host is advancing through the kill chain". Host-plus-time binning makes the prediction target concrete ("the attacker's next stage, for this host, in k minutes"), turns `attack_within_k` into a genuine temporal target, and is the only formulation under which `train_test_split_by_day` (`dataset_loader.py:353`) becomes usable at all. 60 s because the CIC-IDS-2017 attacks are minutes-scale (patator bursts about 2 min, DoS Hulk about 1 h) while CICFlowMeter expires a flow after 120 s idle / 600 s active, so one bin holds a coherent burst; 60 s also yields at least two bins per short attack, the minimum for a k-step rollout to say anything. 30 s is carried as a reported sensitivity check, not as the headline.

**Rejected alternatives.** (a) Keep 46 columns and zero-fill the 8 packet-level ones - ships 8 constant inputs and makes every real-data result uninterpretable. (b) Derive the packet-level columns from raw packets and fill the gaps - full coverage needs the 5-file, 52.43 GB PCAP set (separately gated), and any gap-filled cell would be the same uninterpretable constant as (a); uncovered bins stay NaN with a coverage flag instead (amended D2). (c) Entity = the 5-tuple flow - each "sequence" would be a single connection and no attacker progression exists inside it. (d) 30 s bins as the headline - noisier aggregates and many more empty bins.

**Label per bin.** `label_bin = max(ATTACK_STAGE_INDEX[f] for f in flows in bin)`, using the severity order already defined at `features.py:117-126`; BENIGN only when every flow in the bin is BENIGN. Rejected: majority vote (a scan bin with mixed traffic would be labelled BENIGN at exactly the moment the attacker advances); first-non-benign (order-dependent, arbitrary tie-break).

**Sequence and targets.** W = 10 consecutive bins of the same host, i.e. 10 minutes of history; stride 1 by default and at least 5 for the budgeted training run. Targets for k in {1, 2, 4} minutes: `y_stage_h{k}` = stage at t+k, `y_attack_within_h{k}` = 1 if any bin in (t, t+k] is non-BENIGN, plus `y_stage_now` = stage at t. The horizon heads are renamed `k1`, `k2`, `k4`: the seconds-based names (`h30/h60/h120/h300`) imply a step duration the pipeline never defines, and `h300` is dead code (V6). `bin_seconds` becomes the single place a step duration is defined.

### D2 - CIC-IDS-2017 first, human-supplied, never substituted

**Decision.** Primary dataset = **CIC-IDS-2017, `GeneratedLabelledFlows/TrafficLabelling/`**, whose CSVs carry `Flow ID`, `Source IP`, `Destination IP` and `Timestamp`. The `MachineLearningCVE/` variant has no IP or timestamp columns and is useless for D1 - and it is exactly the variant `load_cicids2017` accepts silently today (V3). A new `argus/scripts/prepare_cicids2017.py` reads `argus/data/raw/cicids2017/` (gitignored since WP0), validates the expected file names and headers, cleans, bins and writes `argus/data/processed/cicids2017_bins_60s.parquet` plus `manifest.json`.

**Amended 2026-09-28 after acquisition (for review).** The raw files are no longer hypothetical: 8 Parquet files (305.9 MB) were obtained as the pinned Hugging Face mirror of `TrafficLabelling/` (`bvsam/cic-ids-2017` @ `70bac6246d99cf04`; the UNB direct download is dead - redirect loop) and hash-validated against that revision's LFS oids by `scripts/validate_cicids2017.py`, recorded in `argus/data/cicids2017_provenance.json`. In this snapshot 288,602 rows of `Thursday-WorkingHours-Morning-WebAttacks` are all-null padding **observed in this pinned snapshot, not asserted as a defect in the canonical CIC export**; effective rows total 2,830,743, matching the published dataset size. They are dropped in preprocessing, never zero-filled or imputed.

**Amended 2026-09-28 (for review): flow vs. packet features - conflict resolution.** PS §1 demands flow- *and* packet-level features; D1's flow-tier schema removes the 8 `PACKET_LEVEL_COLUMNS` because no CICFlowMeter export contains them (V4); the demo extractor and this plan point at PCAPs. Resolution - a two-tier feature model:

1. **Flow tier (headline).** The 39 columns of D1, built from the flow table alone - unchanged, PCAP-free. Every headline D3 number stands on this tier.
2. **Packet tier (extension).** The same 8 columns, computed from PCAPs by the streaming extractor `ml/world_model/pcap_bins.py` (CLI `scripts/pcap_stream_bins.py`) in the *same* fixed 60 s bins keyed `(src_ip, bin_start)` (D1), joined as **NaN + coverage flag on uncovered bins - never zero-filled, never imputed**, and reported only as a pre-registered ablation on the covered bins, with coverage percentages in the results JSON.

Scope honesty: a Thursday-only PCAP (`Thursday-WorkingHours.pcap`, 8.30 GB) would be a **pilot** that validates the pipeline on one day - it is *not* packet-level coverage for the five-day benchmark. Full coverage is 5 files / 52.43 GB (Mon 10.82, Tue 11.05, Wed 13.42, Thu 8.30, Fri 8.84 GB), which exceeds the free disk space and is a separate approval. No doc, metric or slide may claim "packet-level results on CIC-IDS-2017" without it.

**PCAP source decision (needed before any large download).** Options: **(a)** Thursday-only full file - smallest day, one-day pilot covering both Thursday flow files; **(b)** stream-and-cut a bounded time window (<= 2 GB) instead of the full file; **(c)** baseline-only - no PCAP now, packet tier deferred, D1/D3 ship flow-tier only. Whichever is approved, the gates are the same: the extractor must be the streaming fixed-60 s-bin one (`PcapReader`, never `rdpcap`, never ten equal-duration global windows, RST never counted as retransmission - the demo path's RST bug is fixed too); the flow-vs-PCAP **clock offset must be measured** (median over matched 5-tuples of flow timestamp minus first packet time, dispersion reported) against known flow starts, never assumed; and flow/PCAP bin coverage (all bins and attack bins) must be measured and written to the provenance block. Until those measurements exist on real CIC PCAPs they are reported as "not yet measured".

**Amended 2026-09-28 (for review): extractor implementation status.** `ml/world_model/pcap_bins.py` (CLI `scripts/pcap_stream_bins.py`) is implemented and green under `tests/test_pcap_bins.py` (fixed-bin keying to the second, RST never counted as retransmission in both the new extractor and the demo path, no zero-fill of uncovered bins, null-not-zero TCP window, clock-offset measured and `not_measured` statuses, coverage percentages, `PcapReader`-never-`rdpcap` guard, and a pandas timestamp-unit regression test). The CLI is smoke-tested end-to-end on `data/sample_portscan.pcap` with a synthetic flow table carrying a planted offset: clock offset measured at median +1.220 s over 200 matched directional 5-tuples (IQR 0.028 s), coverage 100% all bins / 100% attack bins, all 8 packet columns populated. Clock offset and bin coverage **on real CIC PCAPs remain "not yet measured"** - the gates above are unchanged.

**Amended 2026-09-28 (review round 1): extractor hardening.** The first review round's findings on that extractor are fixed and each is pinned by a test (`tests/test_pcap_bins.py`, now 33 tests): backwards timestamps can no longer pass silently - `extract_pcap_bins` closes bins on a watermark scan that runs **per bin boundary** (with O(1) Welford accumulators, no per-packet TTL/window lists), the default `on_backwards="fail"` raises `BackwardsTimestampError` on a packet that targets an already-closed bin, and bounded reordering is opt-in (`on_backwards="merge"` plus `reorder_grace_sec`, counted in `backwards_packets_merged`); the clock offset separates the raw `observed_delta` from an explicit `apply` decision and matching is **protocol-aware and per connection instance** (`(proto, src, dst, sport, dport)`, new instance on an idle gap >= 120 s or a fresh SYN >= 30 s later, never collapsed to the earliest flow of a tuple), with application refused - never silently applied as 0 - below 3 matched instances or above a quantization-aware dispersion bound (IQR > 1 s), and a `timezone_scale` flag when `|median| >= 3600 s`; `--apply-offset` may use only `apply.correction_sec` and **exits 1** on a refusal without writing output; `--flows` now accepts several files *and* directories (deterministic parquet-then-csv sorted order) so a multi-day table set joins in one run; flow-table null handling is explicit - NaN cells never become the string `"nan"`, rows without a real IP/port are dropped and counted, an all-null `Timestamp` column is reported as `timestamp_null` with `timestamp_all_null: true` (the pinned snapshot's padding) and is never conflated with `timestamp_unparsable`; and coverage denominators count **unique** `(src_ip, bin_start)` host-minute bins, with a zero denominator (all-benign table) reported as `null` / `"n/a"` instead of 0%. Re-smoked on `data/sample_portscan.pcap` against a synthetic two-file flow table: measured median +1.220 s, IQR 0.028 s, n=200, flag `second_precision_flow_timestamps`, applied +1.220 s, coverage 1/1 bins and 1/1 attack bins; the same command with a one-row flow table exits 1 with the refusal reason and writes no file. Clock offset and bin coverage **on real CIC PCAPs remain "not yet measured"** - the gates above are unchanged.

Pitfalls the script asserts rather than assumes: `Infinity` and NaN in `Flow Bytes/s` and `Flow Packets/s`; the mojibake `Web Attack \u0096 ...` (U+0096) labels - in this snapshot all three Web-Attack families of the Thursday-morning file spell them that way; `d/m/Y H:M` timestamps written with a 12-hour clock and no AM/PM marker in some files; exact-duplicate rows; and whitespace-padded headers, for which the existing map already needs both `" Label"` and `"Label"` (`dataset_loader.py:622-623`).

If the raw files cannot be downloaded in-session, **the human supplies them and I wait**. `data/sample_datasets/*` are 30-row fixtures produced by `scripts/generate_sample_datasets.py`; they will not stand in for real telemetry, and no number derived from them will reach the README. The second dataset is CTU-13 (`.binetflow`, per-scenario botnet timelines) and only after CIC-IDS-2017 has produced a number.
### D3 - Two evaluation protocols, three baselines, an honest lead time

**P-A (in-distribution forecast).** Per day, chronological split at the 70 % time
mark: train on the earlier bins, test on the later bins, and no window may
straddle the split point.

**P-B (out-of-distribution / unseen attack).** Leave-one-day-out over
Tuesday, Wednesday, Thursday and Friday. Monday is all-benign and is used for
training only.

**Baselines.** Logistic regression on x_t only (the PS-mandated baseline),
random forest on x_t only, and logistic regression on the lagged window
`x_{t-W+1..t}` concatenated, which is the fair temporal baseline. If the
Transformer cannot beat the lagged LR, the README says so.

**Metrics** per k in {1, 2, 4}: macro-F1, per-stage F1 together with its support
count, benign FPR, ROC-AUC and PR-AUC for `attack_within_h{k}`, and ECE with 10
bins.

**Lead time, redefined.** For each ground-truth onset (the first non-BENIGN bin
of a host after at least 5 BENIGN bins), lead = onset time minus the time of the
first *sustained* alert (`P(attack_within_h1) > tau` on 2 consecutive bins)
inside the preceding 10 bins. Report the median lead and the early-warning rate
**at the tau that yields benign-FPR <= 1 %**, and print that FPR next to it.
Any alert outside the 10-bin window is a false positive, not lead time. This
replaces `compute_forecast_lead_time` (`benchmark.py:115-191`), which credits the
earliest predicted attack step and therefore maximises "lead" for a model that
alerts on step 0 of every sequence (V9).

**Rejected.** Reporting lead time without its FPR (the failure mode we are
fixing); a single random 80/20 split (leaks future days into training); k=1
only (a k-step rollout must be demonstrated at more than one horizon).

**Amended 2026-09-28 (for review): packet-tier arm.** If and only if a PCAP source is approved (D2), add one arm: the same P-A/P-B protocols restricted to the bins covered by the PCAP join, labelled `packet_tier` in the results JSON, whose provenance carries `pcap_file`, measured `clock_offset_sec`, `bins_flow_total`, `bins_covered` and `attack_bins_covered`. Uncovered bins are excluded from this arm (NaN, never zero-filled). One Thursday PCAP yields at most a Thursday-scoped `pilot` row - never a multi-day packet-level claim - and the flow-tier arm remains the headline under every option.

### D4 - Explainability: attention plus Integrated Gradients, or an explicit "none"

**Decision.** The primary method is a subclass of `nn.TransformerEncoderLayer`
that calls `self_attn(..., need_weights=True, average_attn_weights=False)` and
stores the per-head weights, producing a `(layers, heads, W, W)` timestep
attribution. That alone answers "which timestep", not "which feature", and the
PS asks for feature-level explanation, so it is combined with Integrated
Gradients over the raw `(W, F)` input - about 20 lines against the `torch` we are
already adding, with no new dependency and no Captum - to produce a `(W, F)`
attribution map the dashboard can render. Every entry of `driving_features`
gains `"method": "attention+ig" | "ig" | "none"`. Today
`AttentionExplainer.explain_step()` never receives attention (V7) and falls back
to `_fallback_explain()`, a "largest z-scored value" magnitude proxy
(`predictor.py:142-162`). On the heuristic path the header must read
"Heuristic kill-chain prior (no trained model loaded)" and the method must be
`"none"` - never a magnitude proxy presented as attribution.

**Rejected.** (a) Attention alone - it does not localise features, so it does not
answer the PS question. (b) SHAP over the sequence model - `shap` is an
undeclared dependency (V11) and a sampling explainer over a Transformer has no
CPU time budget. (c) Keep the z-score fallback under a better name - it is still
not attribution.

### D5 - Demo integrity: name the engine, ship the numbers

**Decision.** Every forecast dict carries `"engine": "neural" | "heuristic"` and
`"model_provenance": {...}`; the dashboard renders a visible badge; the
heuristic explanation header names itself. The demo checkpoint
(`argus/ml/pretrained/world_model.pt`; 612,079 float32 parameters is 2.45 MB,
re-measured after D1 changes the input width), its metrics JSON, the benchmark
JSON and `PROVENANCE.md` are committed in the **new, non-gitignored**
`argus/ml/pretrained/`, whose contract is documented in
`argus/ml/pretrained/README.md`. `ml/artifacts/` stays gitignored for local runs
and loaders fall back `artifacts/` then `pretrained/`. A CPU `torch` wheel goes
into `requirements.txt` and the Dockerfile so Render and a judge's clone run the
neural path. `argus demo` will **not** auto-train: silently starting a 30-60
minute CPU training run on the first request is a worse lie than an honest
badge, so it prints the exact command instead.

**Rejected.** (a) Auto-train on first request - unpredictable latency, and a
background-trained model is indistinguishable from the heuristic while it runs.
(b) Keep the checkpoint out of git and require `argus train` - a judge then sees
a heuristic 99 % CRITICAL forecast and concludes the world model is fake, which
would be the correct conclusion. (c) Say nothing and let the UI imply the model
ran - the current state.

## 2.2 Data-pipeline spec

New module `argus/ml/world_model/binning.py`:

    SEVERITY = ATTACK_STAGE_INDEX                      # features.py:117-126

    def normalise_cicids2017_label(raw: str) -> str    # mojibake/whitespace -> canonical key
    def build_host_time_bins(df, bin_seconds=60, entity="src_ip") -> pd.DataFrame
    def make_sequences(bins, W=10, horizons=(1,2,4), stride=1) -> BinnedSequences

`BinnedSequences` is a dataclass with `X (N, W, F) float32`, `y_stage_now (N,)`,
`y_stage_h{k} (N,)`, `y_attack_within_h{k} (N,)`, a `meta` DataFrame
(`host`, `day`, `bin_index`, `t_start`, `n_flows`) and the schema fields
`feature_list`, `bin_seconds`, `W`, `horizons`.

`prepare_cicids2017.py` CLI: `--raw-dir argus/data/raw/cicids2017 --out
argus/data/processed --bin-seconds 60 --check --max-rows N`. `--check` prints the
manifest and exits without training anything.

Parquet schema, one row per `(src_ip, bin_start)`: `src_ip` str, `bin_start`
timestamp[ns] UTC, `day` str, `bin_index` int32, `n_flows` int32, the 30
aggregated flow/extended features, the 8 topology features, `label` str,
`label_index` int8, `label_flows` int32 (non-BENIGN flows in the bin). The 39
model columns are exactly `feature_list`, in that order.

`manifest.json`: `dataset`, `variant`, `source_dir`, `files[{name, md5, rows_raw,
rows_after_clean}]`, `rows_raw_total`, `rows_after_clean_total`,
`duplicates_removed`, `unmapped_labels` (must be empty), `bin_seconds`, `hosts`,
`bins`, `label_histogram_per_day`, `sequences_per_horizon`, `feature_list`, and
`code{git_sha, utc_timestamp, prepare_script_sha256}`.

Exact CIC-IDS-2017 to 8-stage map, starting from `dataset_loader.py:626-645`:

| Raw label | Stage | Keep / change, and why |
|---|---|---|
| BENIGN | BENIGN | keep |
| Bot | Botnet | keep (C2, T1071) |
| DDoS | DDoS | keep (Impact) |
| DoS Hulk / GoldenEye / slowloris / Slowhttptest | DDoS | **keep with an explicit caveat** - our vocabulary has no DoS stage, so the four DoS families are merged into the Impact slot and the results table states "DoS+DDoS merged" |
| FTP-Patator / SSH-Patator | BruteForce | keep (T1110) |
| Heartbleed | WebAttack | keep - T1190 exploit of a public-facing service is Initial Access in our vocabulary; 11 flows total, so per-stage metrics are reported as "support 11, not interpretable" |
| Infiltration | LateralMovement | keep - the dataset's Infiltration scenarios are dropbox/lateral; 36 flows, same caveat |
| PortScan | PortScan | keep (T1046) |
| Web Attack - Brute Force | BruteForce | **keep, and documented** - the technique is credential access, so BruteForce rather than WebAttack; consequently no report may describe that detection as "WebAttack detected" |
| Web Attack - XSS | WebAttack | keep |
| Web Attack - Sql Injection | WebAttack | keep; 21 flows, same caveat |
| mojibake `Web Attack \u0096 ...` (U+0096 - what this pinned snapshot actually contains: measured over all 8 files, every Web-Attack row spells all three families that way; none use ASCII hyphen or U+FFFD) | BruteForce for the Brute Force family, WebAttack for XSS / Sql Injection (same targets as the hyphen rows) | **new entries required** - `_CICIDS2017_LABEL_MAP` (`dataset_loader.py:639-644`) has ASCII-hyphen and en-dash (U+2013) spellings but not U+0096, so on this snapshot every Web-Attack row falls through `.get(x, "BENIGN")` at `dataset_loader.py:691` (V12, confirmed live against the snapshot); U+FFFD and hyphen variants stay as defensive entries |
| any other unknown label | **raise** | **change** - both `load_cicids2017` and the new script fail loudly, listing the distinct unmapped labels, and the manifest requires `unmapped_labels: []` |

`load_cicids2017` change: add `"Source IP" -> src_ip`, `"Destination IP" ->
dst_ip`, `"Destination Port" -> dst_port`, `"Flow ID" -> flow_id` and
`"Timestamp" -> timestamp` to `_CICIDS2017_COLUMN_MAP`, and add
`keep_keys: bool = False`; when True it returns the key columns alongside the
features so `binning.py` can consume them. The default keeps the existing
signature, so `tests/test_core.py` (the 8-dataset load test) keeps passing.

Quarantine: the zero-fill loop (`dataset_loader.py:696-698`) is deleted from every
real-data path, and `generate_state_sequences` (`:265-326`) is renamed
`legacy_synthetic_state_sequences` and documented as synthetic-only so that no
real dataset can reach it. `scripts/generate_sample_datasets.py` gains a
docstring and the dashboard "Dataset Inspector" copy is changed to "synthetic
format fixtures for smoke tests - not dataset excerpts".

## 2.3 Model / training / benchmark spec

**model.py.** (1) `HORIZON_OFFSETS` is keyed `k1/k2/k4` with offsets taken from
`make_sequences(horizons=...)`, and the `offset >= seq_len` skip (`:331-333`)
becomes an `assert` so no head can be silently untrained. (2) The causal mask is
moved using `x.device` instead of `is_cuda` (`:235-237`). (3) A
`StateTransitionEncoderLayer(nn.TransformerEncoderLayer)` captures per-head
weights; because PyTorch may take a fused path under `eval()`/`no_grad`, the
captured tensor is asserted non-empty and the method degrades to `"none"`
otherwise. (4) The checkpoint payload becomes `{schema_version, feature_list,
n_features, bin_seconds, W, horizons, train_mean, train_std, class_weights,
dataset_manifest_sha256, git_sha, utc_timestamp, config, model_state_dict}`.

**train.py.** Accepts `--dataset cicids2017 --bins <parquet>` (the current
`choices` at `:390` omit 2017 entirely although the loader exists), plus
`--horizons`, `--bin-seconds`, `--class-weight balanced|focal` and
`--cpu-budget-minutes M`, which stops at the measured budget and reports how
many epochs actually fit instead of assuming a fixed epoch count does. The
metrics dict (`:355-371`) gains a `provenance` block. Class imbalance is handled
by weighted cross-entropy by default, with focal loss behind the flag.

**benchmark.py.** `run_benchmark` is rewritten to D3 and gains `--protocol
{P-A,P-B}` and `--bins`. `compute_forecast_lead_time` is renamed
`legacy_compute_forecast_lead_time` and left uncalled;
`compute_onset_lead_time(probs, meta, tau_policy="fpr<=0.01")` replaces it. The
heuristic arm (`:488-547`) is kept as a reported baseline and must be labelled
`"engine": "heuristic"` in the JSON. Results JSON:

    {"schema_version": 2,
     "provenance": {"dataset", "variant", "files", "rows", "sequences",
                    "split_protocol", "bin_seconds", "W", "horizons",
                    "git_sha", "utc_timestamp", "host", "device"},
     "protocols": {"P-A": {"k1": {...}, "k2": {...}, "k4": {...}}, "P-B": {...}},
     "lead_time": {"tau": ..., "benign_fpr_at_tau": ..., "median_lead_sec": ...,
                   "early_warning_rate": ...}}

validated against a committed JSON-schema at
`argus/tests/schemas/benchmark_results.schema.json`.

**One source of truth.** `/api/benchmark` serves exactly this file (the WP0
fallback chain), the label at `index.html:1086` is rendered from it as
`{dataset} | {n rows} flows | {split_protocol} | {git_sha[:7]}` or
"No benchmark results loaded", and `scripts/render_readme_metrics.py` rewrites
the README table between `<!-- metrics:begin -->` and `<!-- metrics:end -->`. If
a number is not in the JSON, it does not appear in the README or the UI.

## 2.4 Work packages

Skeleton = WP0 to WP8 as specified, one conventional commit each, with the full
verification block pasted in every commit body.

| WP | Est. | Acceptance test | Commit |
|----|------|-----------------|--------|
| WP0 fresh-clone green | 2 h | `pytest argus/tests -q` gives 0 failed with and without torch; `/api/benchmark` falls back to `ml/pretrained/` and skips with a reason if neither exists; health serves `rf_model_trained`, `world_model_trained`, `torch_available` | **done** |
| WP1 demo integrity | 3 h | with no checkpoint, `POST /api/forecast` returns `"engine": "heuristic"` and the UI shows the badge; the `index.html:1086` label is JSON-driven; no "Dirichlet" or "attention-weighted" claims remain | `feat(world-model): name the forecast engine and its provenance` |
| WP1.5 upload truthfulness | 2 h | a corrupt CSV and a corrupt PCAP each produce an explicit 400 or packet evidence, never a CRITICAL forecast; the `+67.5s` toast is gone | **done** |
| WP2 real-data pipeline | 1-2 d | `prepare_cicids2017.py --check` prints the manifest; tests assert no sequence crosses a day or host boundary, `y_attack_within_h1 == (y_stage_h1 != 0)`, and unmapped labels raise | `feat(data): host/time-binned CIC-IDS-2017 pipeline + manifest` |
| WP3 model fixes | 1 d | `train --dataset synthetic --epochs 2` and `--dataset cicids2017 --epochs 1 --max-sequences 2000` both finish on CPU and write a checkpoint that `predictor.py` loads | `fix(world-model): real horizon heads, device-safe mask, provenance checkpoints` |
| WP4 honest explainability | 1 d | captured attention has shape `(layers, heads, W, W)` with rows summing to 1 +/- 1e-4; attribution for a scan ranks `unique_dst_ports` or `src_fanout` in the top 3, and the test states which dataset it used | `feat(world-model): per-head attention + integrated gradients attribution` |
| WP5 forecast benchmark | 1-2 d | the JSON validates against the committed schema; the report shows LR-x_t, LR-lagged, RF and the World Model for k in {1,2,4} under P-A and P-B; every README number is script-generated | `feat(benchmark): K-step forecast protocols, baselines, calibrated lead time` |
| WP6 ship checkpoint and numbers | 1 d | fresh clone, install, uvicorn, then `/api/forecast` returns `"engine": "neural"`, `/api/benchmark` returns the committed real results, and `docker build` succeeds | `chore(release): ship the CIC-IDS-2017 checkpoint, metrics and provenance` |
| WP7 documentation truth pass | 0.5 d | the WP7 grep returns nothing | `docs: make every claim traceable to a result artifact` |
| WP8 CTU-13 (candidate to cut) | 1 d | one extra P-A row through the same `binning.py`, no new features | `feat(data): CTU-13 P-A row` |

Flags: **(1)** WP0 and WP1.5 were committed together because both rewrite
`dashboard/app.py` and `tests/test_core.py`; splitting them would need hunk
surgery that risks a broken intermediate commit. **(2)** WP8 is the first thing
to cut if the schedule slips past 20 Oct: a second dataset adds less to the
graded claim than a clean P-B protocol does. **(3)** WP3's `assert` that every
horizon head receives targets must not be softened to make the CPU budget pass;
`--cpu-budget-minutes` is the release valve instead.

## 2.5 Risks and open questions

1. **Dataset availability (highest).** CIC-IDS-2017 sits behind a manual UNB
   download form and cannot be fetched from this session. WP2 does not start
   until the CSVs exist at `argus/data/raw/cicids2017/`, and `data/sample_datasets/`
   will not be substituted.
2. **Class imbalance.** Monday alone is roughly 530 k benign flows of roughly
   2.8 M; the entire Infiltration class is 36 flows and Heartbleed 11, and far
   fewer *bins*. Macro-F1 over 8 classes will be brutal, and a per-bin label can
   read BENIGN while an attack is present in the same minute from another host.
   Plan: always report per-stage F1 with its support, use weighted CE, and write
   "support too small to interpret" rather than quoting an F1 of 1.0 from 11 rows.
3. **The Transformer may not beat lagged-LR.** This is a real possible outcome
   and the honest negative result is worth more than a fabricated positive one.
   The README will say, in these words: "On CIC-IDS-2017 under protocol P-A at
   k=1 the world model reaches macro-F1 X against Y for LR on x_t and Z for LR on
   the lagged window; it does/does not beat the PS baseline." Next things to try,
   in order: longer W, stride equal to the horizon, topology ablations, a GRU
   sequence baseline as a cheaper competitor, and only then more capacity.
4. **CPU budget.** Minutes per epoch are unknown until measured on the machine
   available (612 k parameters, W=10, 39 features). Mitigations are in place:
   `--max-sequences`, stride >= horizon, `--cpu-budget-minutes`, float32, batch
   32. The measured figure goes into `PROVENANCE.md` as a measurement, never as
   an estimate.
5. **Timestamp integrity.** If the 12-hour clock and missing AM/PM markers cannot
   be resolved deterministically, P-A's chronological split for those days is not
   trustworthy. Mitigation: assert per-day monotonicity and a within-day span
   below 24 h; on failure, fall back to CICFlowMeter row order and mark those
   days "row-order split (timestamp ambiguous)" in the manifest and the results
   JSON.
6. **PS §1 asks for packet-level features.** Resolved in D2 (two-tier model):
   the servable schema stays flow-tier; packet-tier features come from PCAPs
   joined as NaN-with-coverage on the same 60 s bins and are reported only as
   a scoped ablation with measured clock offset and coverage. A Thursday-only
   PCAP is a pilot, not benchmark coverage; full five-day PCAP coverage
   (52.43 GB) is a separate approval. WP1.5's upload response still never
   invents a feature vector.
7. **New dependencies.** Parquet needs `pyarrow`; the CPU torch wheel is around
   200 MB. Both go into `requirements.txt`, and the Dockerfile must pin the CPU
   index or the image balloons. `shap` stays out.
8. **Two vocabularies in one repo.** The RF classifier's labels
   (`ml/flow_features.py`) and the world-model stages (`features.py:105-114`)
   happen to be the same 8 names today and nothing enforces it. WP3 must assert
   the two lists are equal at training time.
9. **Status vocabulary.** After WP0 `/api/health` is unambiguous, but
   `/api/world-model/status` still returns `model_trained`
   (`predictor.py:588-606`) and `index.html:1568` still reads that field. WP1
   must add `world_model_trained` and `engine` there.
10. **Pickle on a shipped file.** `torch.load(..., weights_only=False)` at
    `predictor.py:247` and `benchmark.py:432` executes pickle on a file that
    will now be committed. WP3 switches to `weights_only=True` with a
    plain-dict checkpoint.