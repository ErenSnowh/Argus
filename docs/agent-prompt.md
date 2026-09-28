# ARGUS — Agent Brief: Make the World-Model Core Real (Plan → Act)

> Paste everything below this line into your coding agent. If your agent has
> separate PLAN and ACT modes: run Phase 1 in PLAN mode, stop for approval,
> then run Phase 2 in ACT mode. If it has one mode, it must still stop after
> Phase 1 and wait for an explicit "approved" before writing code.

---

## 0. Your role and the single rule that overrides everything else

You are the lead engineer on **ARGUS** (`ErenSnowh/Argus`, working dir `argus/`),
a submission for **Smart India Hackathon 2026, Problem Statement 26153** (NTRO /
NCIIPC): *"AI-based Network Attack Forecasting from Network Traffic Data using
World Models."* The PS is graded on ONE hard thing:

> Learn the state-transition dynamics **P(S_{t+1} | S_t)** of a network from
> **real traffic telemetry**, run a **K-step forward rollout** to forecast the
> attacker's **next MITRE ATT&CK stage(s)**, **explain** each forecast per
> feature, and **beat a logistic-regression baseline** on a fair, leak-free
> evaluation.

Today that core is a facade over otherwise good scaffolding (agents, MCP,
guardrails, statutory reports, dashboard). Your job is to make the core real
and make every claim in the repo provable from the repo.

**Overriding rule — no unprovable numbers.** Every metric, dataset name, row
count, or "verified result" that appears in any README, docstring, dashboard
string, or slide-facing text MUST trace to a committed results JSON that was
produced by a committed command, with a provenance block (dataset, files,
rows, split protocol, git SHA, UTC timestamp). If you cannot produce a number,
write "not yet measured" — never invent, extrapolate, or leave a stale number.

**Feature freeze.** Do not add new agents, MCP tools, knowledge bases, report
types, dashboard tabs, datasets, or "innovations." Touch `agents/`,
`mcp_server/*_report.py`, `mcp_server/knowledge_base.py`,
`ml/world_model/counterfactual.py`, and dashboard tabs only where a work
package below explicitly requires it. Anything you think is missing goes in
a `docs/plan/backlog.md` file, not in code.

---

## 1. Ground truth about the repo (verified 2026-09-28 — trust this, re-verify only if a line has moved)

### Environment & tests
- Fresh clone + `pip install -r argus/requirements.txt` + `pytest argus/tests -q` →
  **1 failed, 48 passed, 3 skipped.** `test_dashboard_benchmark_endpoint`
  (`tests/test_core.py:426`) calls `/api/benchmark`, which 404s because it reads
  `ml/artifacts/benchmark_results.json` — a **gitignored** artifact that only exists
  on the author's machine (`dashboard/app.py:345-351`).
- `torch` is NOT in `requirements.txt` or the Dockerfile (`world_model` extra only).
  The Render/Docker image and any judge's fresh clone therefore **never** run the
  Transformer.

### The "World Model" that runs on a fresh clone is a hard-coded table
- `ml/world_model/predictor.py:205-227` — `TRANSITION_MATRIX` and
  `STAGE_INFILTRATION_BASE` are hand-typed dicts. `_predict_empirical()` (l.415+)
  takes `max(transitions, key=transitions.get)` at every step → a **deterministic
  script**. Every port-scan-like input yields the identical path
  `WebAttack→BruteForce→LateralMovement→Botnet→Exfiltration`, 99%, CRITICAL.
- `forecast_infiltration()` (l.620-642) returns no field saying which engine
  produced the result. The dashboard, the MCP tool, the CLI and the reports all
  present heuristic output as a model forecast. `/api/world-model/status` says
  `model_trained: false` while `/api/health` says `model_trained: true` (that one
  is the RF, `dashboard/app.py:62-64`).
- README's "Empirical Markov ... with Bayesian Dirichlet smoothing" — there is no
  smoothing, no estimation from data, nothing Bayesian.

### The 99.975% accuracy is a synthetic artifact
- `ml/world_model/features.py:150-375` — every class is an independent Gaussian
  blob with wildly separated means (PortScan `unique_dst_ports_per_src` 180±50 vs
  BENIGN 2±1; DDoS `syn_flag_count` 400±100 vs 1±0.8). 70 % of training sequences
  are one of 8 fixed scripts (`ATTACK_SCENARIOS`, l.377-409). Any model scores ~100 %.
- The "8 public dataset samples" in `data/sample_datasets/` are 30-row fixtures
  fabricated by `scripts/generate_sample_datasets.py`. **No real data has ever gone
  through the pipeline.**
- README's RF "93 %" vs WM "99.975 %" table compares models trained on two
  different synthetic generators (`ml/flow_features.py` vs `ml/world_model/features.py`).

### The real-data path cannot work as designed
- `dataset_loader.py:load_cicids2017` (l.648-704) drops `Source IP`,
  `Destination IP`, `Timestamp` (none are in `_CICIDS2017_COLUMN_MAP`; the function
  returns `combined[WORLD_MODEL_FEATURES + ["label"]]`). The 2018 loader keeps only
  `timestamp`.
- `generate_state_sequences()` (l.265-326) windows over **consecutive CSV rows** —
  flows from every host on the network interleaved. That is not a state
  trajectory of anything. `train_test_split_by_day()` exists (l.353) but is
  unreachable for 2017 because the timestamp is dropped.
- 16 of the 46 features (8 packet-level + 8 topology) are **zero-filled** on real
  data (`for col in WORLD_MODEL_FEATURES: if col not in combined.columns: combined[col] = 0.0`)
  while being the most discriminative columns in the synthetic generator → a
  synthetic-trained model cannot transfer, and a real-trained model ignores a
  third of its inputs.
- `train.py:390` `--dataset` choices are `synthetic, cicids2018, ctu13` — no
  `cicids2017`, even though the loader exists.

### The model has dead code paths
- `model.py:_register_attention_hooks / _attention_hook` (l.200-209): PyTorch's
  `nn.TransformerEncoderLayer._sa_block` calls `self_attn(..., need_weights=False)`,
  so the hook receives `(out, None)` and **never** appends. In eval/no-grad the
  fused fast path may bypass the module hook entirely. Net effect:
  `AttentionExplainer.explain_step()` always falls to `_fallback_explain()` =
  "largest |z-scored value|" (`predictor.py:142-162`). The README's "attention-
  weighted feature attribution" is not happening. Verify in 3 lines before fixing.
- `WorldModelLoss.HORIZON_OFFSETS["h300"] = 10` with default `seq_len = 10` →
  `offset >= seq_len` → the `h300` head is **never trained** (l.286, l.332) but is still
  reported in `horizon_forecasts`.
- Horizon heads are named in seconds (h30/h60/h120/h300) but a "step" has no
  defined duration anywhere in the data pipeline.
- Causal mask is only moved to device `if x.is_cuda` (l.235-237) — breaks on MPS.

### The benchmark measures detection, not forecasting
- `benchmark.py:run_benchmark` evaluates every model on **stage at t given x_t**.
  The World Model gets no forecasting advantage to demonstrate; the comparison is
  meaningless for the PS.
- `compute_forecast_lead_time()` (l.115-191) credits the *earliest* predicted
  attack step — a model that alerts on step 0 of every sequence gets maximum
  "lead time." The metric rewards false positives and has no FP term.

### Documentation / UI drift (all must be fixed or removed)
- `README.md` says 38 features; `WORLD_MODEL_FEATURES` has **46** (24+6+8+8).
- `README.md` says "37 unit tests / 36 passed, 1 skipped"; `argus/README.md` says
  "22 tests"; reality is 52 collected.
- `argus/README.md` still says the Gemini key "ships embedded in `config.py`";
  `config.py:_DEFAULT_API_KEY = ""` (correctly removed on 2026-06-26). Fix the doc,
  never re-add a key.
- `dashboard/static/index.html:1050`: *"Empirical 3-Way Comparative Evaluation on
  Multi-Day CSE-CIC-2018 Dataset (Zero Temporal Leakage)"* — nothing in the repo
  can reproduce this. Must become a dynamic label read from the results JSON.
- README "Verified Benchmark Results" table, "Training Configuration" table
  (612,079 params, 57.9 s, 24 epochs), and "Infiltration AUC 0.983" are all
  synthetic-only and must be replaced by real numbers or "not yet measured."

### Timeline you are working against
- SIH 2026: idea submission closes 30 Sep 2026; evaluation 10 Sep–30 Oct; results
  first week of Nov; mentoring 10–30 Nov; 36-hour Grand Finale in Dec 2026.
  Work packages are ordered so that the highest-value evidence (a real-data
  K-step benchmark) lands first.

---

## 2. PHASE 1 — PLAN (stop after this and wait for approval)

Produce **one** document, `docs/plan/world-model-core.md`, and nothing else
(no code changes in this phase). Read the files cited above before writing;
quote line numbers. The document has exactly these sections:

### 2.1 Decisions (each: the decision, 2–4 sentences of rationale, rejected alternatives)

**D1 — What is a state S_t?** Propose and justify. Default proposal you must
argue for or against:
- Entity = **source host (IP)**; bin = **fixed 60 s window** (30 s as a
  sensitivity check); S_t = aggregate of all flows from that host in the bin.
- Per-bin features = the existing 24 flow features aggregated (sum for counts/
  bytes, mean+std for rates/lengths/IATs), plus **real** topology features
  computed from the bin (`src_fanout` = distinct dst IPs, `unique_dst_ports`,
  `new_host_edges` vs. all prior bins of that host, `cross_subnet_edges`,
  `connection_repetition`), plus `n_flows`. Packet-level features (TTL/window/
  fragments/retransmissions) are unavailable in CICFlowMeter CSVs: decide
  whether to drop them from the schema for the real pipeline or compute them
  only for the PCAP path — do not zero-fill silently.
- Label per bin = most severe non-BENIGN stage among the bin's flows, else
  BENIGN (document the severity order = `ATTACK_STAGE_INDEX`).
- Sequence = W consecutive bins of one host (W = 10 → 10 min of history);
  targets = stage at t+k and `attack_within_k` = any non-BENIGN in (t, t+k],
  for k ∈ {1, 2, 4} (rename the horizon heads to k-based names).

**D2 — Which real dataset first, and how it gets onto the machine.**
Default: **CIC-IDS-2017, `GeneratedLabelledFlows/TrafficLabelling/` CSVs**
(these carry `Flow ID, Source IP, Destination IP, Timestamp`; the
`MachineLearningCVE/` variant does NOT and is useless for D1). Write a
`scripts/prepare_cicids2017.py` that expects the CSVs at
`argus/data/raw/cicids2017/` (gitignored), validates the expected file names
and column headers, handles the known pitfalls (`Flow Bytes/s` and
`Flow Packets/s` Infinity/NaN, the mojibake `Web Attack \ufffd ...` labels in
the Thursday-morning file, `d/m/Y H:M` timestamps with a 12-hour clock and no
AM/PM marker in some files, exact-duplicate rows), and emits a single
Parquet of binned host sequences plus a `manifest.json` (source files, MD5s,
row counts before/after cleaning, label histogram per day). Second dataset
= CTU-13 (`.binetflow`, per-scenario botnet timelines) only after CIC-IDS-2017
produces a number. State clearly in the plan that the human must supply the
raw files if they cannot be downloaded in-session; you will not fabricate a
stand-in.

**D3 — Evaluation protocol.** Two protocols, both required:
- **P-A (in-distribution forecast):** per day, chronological split at the 70 %
  time mark (train on earlier bins, test on later bins, no window may straddle
  the split point).
- **P-B (out-of-distribution / unseen attack):** leave-one-day-out over
  Tue/Wed/Thu/Fri (Monday is all-benign; use it only in training).
- Baselines: **LR on x_t only** (the PS-mandated baseline), **RF on x_t only**,
  and one **LR on x_{t−W+1..t} concatenated** (a fair temporal baseline; if the
  Transformer cannot beat this, say so).
- Metrics per horizon k: macro-F1, per-stage F1, benign-FPR, ROC-AUC and PR-AUC
  for `attack_within_k`, plus reliability (ECE, 10 bins).
- **Lead time, redefined:** for each ground-truth onset (first non-BENIGN bin of
  a host after ≥ 5 BENIGN bins), lead = onset_time − time of the first
  *sustained* alert (P(attack_within_k) > τ for 2 consecutive bins) within the
  preceding 10 bins; report median lead and early-warning rate **at the τ that
  gives benign-FPR ≤ 1 %**, and print the FPR next to it. Alerts outside the
  10-bin window are false positives, not lead time.

**D4 — Explainability.** Pick one primary method and justify:
(a) subclass `nn.TransformerEncoderLayer` to call `self_attn(..., need_weights=True,
average_attn_weights=False)` and store per-head weights (timestep attribution),
combined with (b) gradient × input or Integrated Gradients over the
`(W × F)` input for per-feature attribution (Captum if you add a dep, else a
20-line manual IG). Attention alone answers "which timestep," not "which
feature" — the PS asks for feature-level explanation. Whatever you pick must
produce a `(W, F)` attribution map that the dashboard can render and that
degrades to an explicit `"method": "none"` — never a silent magnitude proxy
presented as attribution.

**D5 — Demo integrity.** Every forecast dict gets
`"engine": "neural" | "heuristic"` and `"model_provenance": {...}`; the
dashboard shows a visible badge; the heuristic path's explanation header
says "Heuristic kill-chain prior (no trained model loaded)". A small trained
checkpoint (≈2.5 MB fp32 at 612 k params — confirm) + its metrics/benchmark
JSON + `PROVENANCE.md` ship in a **new, non-ignored** directory
`argus/ml/pretrained/`; `ml/artifacts/` stays gitignored for local runs;
loaders fall back `artifacts/ → pretrained/`. `torch` (CPU wheel) goes into
`requirements.txt` and the Dockerfile so Render and judges run the neural
path. Decide whether `argus demo` auto-trains when neither exists.

### 2.2 Data-pipeline spec
Function signatures, module locations, Parquet schema, manifest schema, and
the exact CIC-IDS-2017 → 8-stage label map (start from
`_CICIDS2017_LABEL_MAP`, l.626-645; keep or change each mapping with one line
of rationale — e.g. is "Web Attack – Brute Force" BruteForce or WebAttack?).

### 2.3 Model / training / benchmark spec
What changes in `model.py`, `train.py`, `benchmark.py`, `predictor.py`;
new results-JSON schema with the provenance block; how the README table,
`/api/benchmark`, and `index.html:1050` all read from that one JSON.

### 2.4 Work packages
Use the WP list in Phase 2 verbatim as the skeleton; add estimates, the exact
acceptance test for each, and the order of commits. Flag anything you think
should be cut or re-ordered and why.

### 2.5 Risks & open questions
Include at minimum: dataset availability, class imbalance (Monday/benign
dominance, tiny Infiltration/Heartbleed counts), the possibility that the
Transformer does not beat lagged-LR, CPU training time budget, and what you
will write in the README if a result is bad.

**Then stop. Print the plan path and a 10-line summary. Do not begin Phase 2
until the human replies "approved" (optionally with edits).**

---

## 3. PHASE 2 — ACT (only after approval; execute in this order)

Work on the current branch. One commit per WP (conventional commits). After
every WP run the **full** verification block (§4) and paste its output in the
commit body or PR description. If a WP's acceptance test fails, fix it before
moving on — do not skip ahead.

### WP0 — Green on a fresh clone (≤ 2 h)
- Make `test_dashboard_benchmark_endpoint` pass without local artifacts:
  `/api/benchmark` falls back to `ml/pretrained/benchmark_results.json`; test
  skips with a clear reason if neither exists (temporary until WP6 ships the file).
- Rename `/api/health.model_trained` → `rf_model_trained`, add
  `world_model_trained`; update anything that reads it.
- Add `argus/data/raw/` and `argus/data/processed/` to `.gitignore`.
- **Acceptance:** `pytest argus/tests -q` → 0 failed on a fresh venv with and
  without torch installed.

### WP1 — Demo integrity (≤ 3 h)
- `engine` + `model_provenance` fields in `forecast_infiltration()`,
  `forecast_infiltration_counterfactual()`, the MCP tools, the CLI output, and
  the CERT-In/NCIIPC report text where a forecast is quoted.
- Heuristic path: rename `_predict_empirical` → `_predict_heuristic_prior`,
  header text as in D5, and break the determinism (either sample from the
  transition row with a fixed seed derived from the input, or return the
  full next-stage distribution instead of an argmax — say which in the docstring).
- Dashboard: engine badge on the forecast panel; `index.html:1050` label
  becomes `{dataset} · {n_rows} flows · {split_protocol} · {git_sha[:7]}` from
  the results JSON, or "No benchmark results loaded".
- Remove the fabricated "Bayesian Dirichlet" and "attention-weighted" wording
  wherever it appears until WP4 makes it true.
- **Acceptance:** with no checkpoint present, `curl -X POST /api/forecast`
  returns `"engine": "heuristic"` and the UI shows it; a screenshot goes in the PR.

### WP2 — Real data pipeline (CIC-IDS-2017) (1–2 days)
- `scripts/prepare_cicids2017.py` per D2 → `data/processed/cicids2017_bins_60s.parquet`
  + `manifest.json`.
- `ml/world_model/binning.py`: `build_host_time_bins(df, bin_seconds, entity="src_ip")`,
  `make_sequences(bins, W, horizons=(1,2,4))` returning `X (N,W,F)`, `y_stage_now`,
  `y_stage_h{k}`, `y_attack_within_h{k}`, `meta (host, day, t_start)`.
- `load_cicids2017` keeps `src_ip, dst_ip, timestamp`; `train.py` and
  `benchmark.py` accept `--dataset cicids2017 --bins <parquet>`.
- Delete or clearly quarantine the zero-fill loop; schema for the real pipeline
  per D1.
- Update `scripts/generate_sample_datasets.py` docstring and the dashboard
  "Dataset Inspector" copy so the fixtures are labelled **"synthetic format
  fixtures for smoke tests — not dataset excerpts."**
- **Acceptance:** `python scripts/prepare_cicids2017.py --check` prints the
  manifest (per-day rows, label histogram, hosts, sequences per horizon) and a
  unit test asserts no sequence straddles a day or a host boundary and that
  `y_attack_within_h1` equals `y_stage_h1 != 0`.

### WP3 — Model fixes (≤ 1 day)
- Horizon heads renamed `k1/k2/k4`, offsets tied to `make_sequences` horizons;
  loss asserts every head receives targets (no silently skipped head).
- Device handling via `x.device`, not `is_cuda`.
- Checkpoint saves: schema version, feature list, bin size, W, horizons,
  normalisation stats, dataset manifest hash, git SHA.
- Class-imbalance handling (weighted CE or focal — document the choice) and
  a `--cpu-budget-minutes` flag so a demo-size training fits in the time you
  measure on the machine you have.
- **Acceptance:** `python -m ml.world_model.train --dataset synthetic --epochs 2`
  and `--dataset cicids2017 --epochs 1 --max-sequences 2000` both complete on
  CPU and write a checkpoint that `predictor.py` loads.

### WP4 — Honest explainability (≤ 1 day)
- Implement D4. `driving_features` entries gain `"method": "attention+ig" | "ig" | "none"`.
- Dashboard attention/attribution map renders the `(W, F)` matrix; if
  `method == "none"` it says so instead of drawing magnitudes.
- **Acceptance:** a unit test asserts captured attention has shape
  `(layers, heads, W, W)` and rows sum to 1 ± 1e-4; a second test asserts
  attributions for a scan sequence rank `unique_dst_ports` or `src_fanout` in
  the top 3 (on real data if available, else on synthetic, and the test says which).

### WP5 — Forecast benchmark (1–2 days)
- Rewrite `benchmark.py` per D3: protocols P-A and P-B, three baselines,
  per-horizon metrics, redefined lead time with its FPR, ECE. Output
  `benchmark_results.json` with the provenance block; `generate_benchmark_report`
  renders it; `argus benchmark --dataset cicids2017` runs it end-to-end.
- Remove `compute_forecast_lead_time` in its current form (or keep it behind
  a `legacy_` name that nothing calls).
- **Acceptance:** the JSON validates against a committed JSON-schema; the
  report shows LR-x_t, LR-lagged, RF, WM side by side for k ∈ {1,2,4} under
  P-A and P-B; every number in the README table is copied from this file by a
  script (`scripts/render_readme_metrics.py`), not by hand.

### WP6 — Ship the checkpoint + real numbers (≤ 1 day, after WP5 has run on real data)
- Train the demo checkpoint on CIC-IDS-2017 within the CPU budget; commit
  `ml/pretrained/{world_model.pt, world_model_metrics.json, benchmark_results.json, PROVENANCE.md}`.
- `torch` CPU wheel in `requirements.txt` (pin an index URL if needed) and in
  the Dockerfile; Dockerfile also copies `ml/pretrained/`.
- Un-skip the WP0 test.
- **Acceptance:** fresh clone → `pip install -r requirements.txt` →
  `uvicorn dashboard.app:app` → `/api/forecast` returns `"engine": "neural"`,
  `/api/benchmark` returns the committed real-data results, `/api/world-model/status`
  shows the provenance. `docker build` succeeds and the container does the same.

### WP7 — Documentation truth pass (≤ half day)
- `README.md`: 46 features (or the post-D1 count), real test counts from
  `pytest --collect-only -q | tail -1`, delete the synthetic "Verified Benchmark
  Results" and "Training Configuration" tables, insert the WP5-rendered table
  with its provenance line, describe the heuristic prior honestly (one
  paragraph, as a fallback, with its limits).
- `argus/README.md`: remove every "API key embedded" sentence; fix the test
  count; point to `docs/plan/world-model-core.md`.
- `docs/research/existing-solutions-and-usps.md` "Current ARGUS state" paragraph
  updated to what is now true.
- Add `docs/EVALUATION.md`: how to reproduce every number in ≤ 10 commands.
- **Acceptance:** `grep -rn "99.97\|0.983\|612,079\|38-Feature\|38 features\|Dirichlet\|embedded" README.md argus/README.md argus/dashboard/static/index.html` returns nothing.

### WP8 (only if time remains) — CTU-13 as the second dataset via the same
`binning.py`, run P-A only, add a row to the results table. No new features.

---

## 4. Verification block (run after every WP; paste output)

```bash
cd argus
python -m pytest tests -q --no-header -p no:cacheprovider
python -c "from ml.world_model.features import WORLD_MODEL_FEATURES as F; print('features', len(F))"
python - <<'EOF'
import json, subprocess, urllib.request
p = subprocess.Popen(["python","-m","uvicorn","dashboard.app:app","--port","8123"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import time; time.sleep(5)
try:
    for path, data in [("/api/health", None), ("/api/world-model/status", None),
                       ("/api/forecast", b'{"k_steps":4}'), ("/api/benchmark", None)]:
        req = urllib.request.Request("http://127.0.0.1:8123"+path, data=data, headers={"content-type":"application/json"})
        try:
            body = json.loads(urllib.request.urlopen(req).read())
            keep = {k: body[k] for k in body if k in ("status","rf_model_trained","world_model_trained","model_trained","engine","predicted_stages","dataset","split_protocol","provenance")}
            print(path, "->", json.dumps(keep)[:400])
        except Exception as e:
            print(path, "->", e)
finally:
    p.terminate()
EOF
grep -rn "99.97\|0.983\|612,079\|38-Feature\|38 features\|Dirichlet\|embedded" ../README.md README.md dashboard/static/index.html || echo "no stale claims"
git status --short
```

---

## 5. Definition of done

1. A judge who runs `git clone && pip install -r requirements.txt && uvicorn dashboard.app:app`
   sees a **neural** forecast produced by a checkpoint trained on **CIC-IDS-2017
   host/time-binned sequences**, with a visible provenance line.
2. `argus benchmark --dataset cicids2017` reproduces every number in the README
   on their machine, including the LR baseline the PS mandates, for k ∈ {1,2,4},
   under both protocols, with the redefined lead time shown next to its FPR.
3. The explanation panel shows a real `(W, F)` attribution with its method named.
4. `pytest` is green on a fresh clone with and without torch.
5. No sentence in the repo claims something the repo cannot demonstrate.
6. If the Transformer does **not** beat the lagged-LR baseline, the README says
   so plainly and the plan's §2.5 explains what you would try next. An honest
   negative result with a real pipeline beats a fabricated positive one — the
   evaluators are NTRO/NCIIPC engineers and they will run the code.

---

## 6. Addendum — review notes after WP0 landed (2026-09-28)

Amendments agreed during review of the first pass; they override the text above where they conflict.

1. **Run WP5 (forecast benchmark) before WP4 (explainability).** The real-data number decides whether there is a story; attribution maps do not.
2. **PCAP uploads forecast, they do not degrade to evidence-only.** `ml/world_model/features.py:extract_features_from_pcap()` already builds a real `(W, 46)` window matrix; pass it as `context_window` and declare its approximations (`fwd/bwd split 50/50`, `down_up_ratio` fixed 0.5, `retransmissions ≈ RST`, fixed 20 B headers) in a `feature_extraction` block. Evidence-only is the fallback only when a capture yields no IP packets. After D1, the PCAP path must re-bin to the same 60 s bins used in training instead of slicing the capture into W equal parts.
3. **Heuristic prior rule order.** `predictor.py:_infer_initial_stage` tests `pps > 2000 → DDoS` before `unique_ports > 15 → PortScan`, so a single-source port scan sliced into millisecond windows is labelled DDoS. Test PortScan first and require `dst_fanin > 1` for DDoS. Spend no more than 15 minutes on it; D1 is the real fix.
4. **`engine` is emitted by `forecast_infiltration()` itself** (`"neural"` only when a checkpoint actually loaded), and a single `resolve_world_model_checkpoint()` (artifacts → pretrained) is used by the predictor, `get_model_info`, and `/api/health`. No duplicated path constants.
5. **CSV uploads accept CICFlowMeter headers** via the existing column maps before the required-column check; report `feature_coverage` with the absent columns enumerated. A genuine export lacks `unique_dst_ports_per_src` and `packets_per_flow` (derived, not exported).
6. **Approval checklist for the Phase 1 plan** (D1 / D3 / label map) — the plan is not approved unless it states all of the following:
   - CIC-IDS-2017 has one attacker host (172.16.0.1) plus a few bot-infected 192.168.10.x hosts → roughly 12–15 attack onsets in total; lead time is reported with *n*, and a per-victim (dst-host) view is either added or explicitly rejected.
   - One bin duration (60 s) is used everywhere: training bins, horizon names (`k1/k2/k4`), and the PCAP path.
   - Features unavailable from CSV are **dropped** from the real-data schema, not zero-filled.
   - The dataset has **no Exfiltration**, 36 Infiltration flows and 11 Heartbleed flows: the stage head is trained and scored on the **6 observed stages** (BENIGN, PortScan, WebAttack, BruteForce, Botnet, DDoS); Infiltration/Heartbleed stay in the binary target only.
   - Manifest sanity anchors (MachineLearningCVE counts, TrafficLabelling within a few %): BENIGN ≈ 2.27 M, DoS Hulk ≈ 231 k, PortScan ≈ 159 k, DDoS ≈ 128 k, GoldenEye ≈ 10.3 k, FTP-Patator ≈ 7.9 k, SSH-Patator ≈ 5.9 k, slowloris ≈ 5.8 k, Slowhttptest ≈ 5.5 k, Bot ≈ 2.0 k, Web BF ≈ 1.5 k, XSS ≈ 650, Infiltration 36, SQLi 21, Heartbleed 11.
   - Protocol P-B (leave-one-day-out) is scored on the binary `attack_within_k` target only, because each stage is day-specific.
   - Lead time is always printed next to the benign-FPR at which it was measured (τ chosen for FPR ≤ 1 %), and the lagged-LR baseline is present.
