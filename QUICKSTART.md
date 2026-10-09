# ARGUS — Quick Start Guide
### SIH 2026 · Problem Statement 26153 · NTRO / NCIIPC

Get from zero to a live forecast in under 5 minutes.

---

## Prerequisites

- Python 3.10+
- Optional (for neural engine): PyTorch 2.0+ — `pip install torch`
- Optional (for multi-agent swarm): `GOOGLE_API_KEY` in `argus/.env`

---

## 1. Install

```bash
cd argus
pip install -e ".[world_model]"
```

> Without `[world_model]` the RF classifier and heuristic Markov engine still
> work — only the PyTorch Transformer is optional.

---

## 2. Train the models

```bash
# RF flow classifier (< 10 seconds, synthetic data)
python scripts/train_model.py

# World Model — synthetic validation (< 1 minute, no GPU needed)
python -m ml.world_model.train --validate-supervision --sequences 500 --seq-len 10 --epochs 50

# CIC-IDS-2017 data prep: 60-s host/time bins + manifest (data stays gitignored)
python scripts/prepare_cicids2017.py
python scripts/prepare_cicids2017.py --check     # prints the manifest, writes nothing

# NOTE — training on the binned sequences is the remaining WP6 wiring:
# train.py's --dataset loaders read raw CSVs (they fail on the bins parquet),
# while the bin -> sequence path lives in ml/world_model/binning.py
# (make_sequences). Track: docs/plan/world-model-core.md.
```

After training, copy the checkpoint so it ships with the repo:

```bash
cp ml/artifacts/world_model.pt ml/pretrained/world_model.pt
```

The dashboard's `ENGINE:` badge will then show **NEURAL** instead of HEURISTIC.

---

## 3. Start the dashboard

```bash
uvicorn dashboard.app:app --port 8000 --reload
```

Open **http://localhost:8000** — no internet connection required.

---

## 4. Jury demo flow

### 4a. Upload a CSV telemetry file

1. On the **Forecast & Radar** tab, click **"Choose File"**
2. Select `data/sample_datasets/cicids2017/sample_cicids2017.csv`
3. Click **"Run Simulation"**
4. The K-step probability fan chart updates in real time
5. MITRE ATT&CK stages appear for T+0 → T+300s
6. Driving features panel shows SHAP or attention attribution

### 4b. Upload a PCAP file

1. Same flow with `data/sample_portscan.pcap`
2. The backend extracts 46 features via Scapy and runs the forecast

### 4c. Counterfactual SOC decision

1. Switch to the **Counterfactual Sandbox** tab
2. Select action: `BLOCK_SRC`, `BLOCK_DST_PORT`, or `THROTTLE`
3. Click **"Simulate Mitigation"**
4. See baseline vs mitigated risk timelines side by side
5. Read the recommended action and risk delta

### 4d. Generate a CERT-In / NCIIPC compliance report

1. Switch to the **Statutory** tab
2. Select CII sector (e.g., "Power & Energy")
3. Click **"Generate NCIIPC Report"**
4. Report includes Section 70A language, 6-hour SLA attestation, and incident timeline

### 4e. Run the benchmark

```bash
cd argus
python -m ml.world_model.benchmark --dataset synthetic --sequences 500
```

View results at **http://localhost:8000** → **Benchmark Observatory** tab.

---

## 5. Run the test suite

```bash
cd argus
pytest tests/ -v --tb=short
```

All tests should pass. The key test files:

| File | What it covers |
|------|---------------|
| `tests/test_core.py` | RF classifier, heuristic predictor, MCP tools |
| `tests/test_pcap_bins.py` | PCAP feature extraction, CIC-IDS-2017 binning |
| `tests/test_prepare_cicids2017.py` | Full CIC-IDS-2017 data pipeline |

---

## 6. Health check

```bash
curl http://localhost:8000/api/health
```

Expected response (after training, on a machine with PyTorch):

```json
{
  "status": "ok",
  "rf_model_trained": true,
  "world_model_trained": true,
  "torch_available": true,
  "world_model_source": "world_model.pt",
  "offline": false
}
```

Without PyTorch or a checkpoint: `"world_model_trained": false` and
`"world_model_source": null` — the heuristic Markov prior runs and the
dashboard's ENGINE badge shows HEURISTIC.

---

## 7. What makes ARGUS different

| Feature | Zeek / Suricata / Snort | ARGUS |
|---------|------------------------|-------|
| Threat detection | Per-packet / per-flow rules | Temporal World Model P(S_{t+1}\|S_t) |
| Prediction horizon | Reactive (T=0) | K-step forward simulation (T+30s → T+300s) |
| Kill-chain mapping | Partial, rule-based | 8-stage MITRE ATT&CK, per-step prediction |
| XAI | None | SHAP (neural) + Attention weights |
| SOC decision support | Alert queue only | Counterfactual "what-if" simulation |
| Compliance | None | Auto CERT-In + NCIIPC Section 70A reports |
| Air-gap capable | Yes | Yes — fully offline, no internet required |

---

## 8. Key file locations

| Path | Description |
|------|-------------|
| `ml/world_model/model.py` | Temporal Transformer architecture |
| `ml/world_model/predictor.py` | K-step rollout + SHAP + graph signals |
| `ml/world_model/graph_signals.py` | Lateral movement graph anomaly detection |
| `ml/world_model/features.py` | 46-feature schema |
| `ml/world_model/binning.py` | CIC-IDS-2017 60s window pipeline |
| `ml/world_model/counterfactual.py` | Action-conditioned risk simulation |
| `ml/world_model/benchmark.py` | 3-way LR vs RF vs World Model comparison |
| `dashboard/app.py` | FastAPI backend (20+ endpoints) |
| `dashboard/static/index.html` | Offline HTML5 dashboard |
| `agents/orchestrator.py` | Google ADK multi-agent swarm |
| `mcp_server/server.py` | FastMCP tool server (19 tools) |
| `security/guardrails.py` | Hash-chained audit log + PII redaction |
| `ml/pretrained/benchmark_results.json` | Committed benchmark results (synthetic split, with provenance) |

---

## 9. Environment variables

Copy `argus/.env.example` → `argus/.env` and fill in:

```
GOOGLE_API_KEY=<your Gemini API key>     # Required for multi-agent swarm
VIRUSTOTAL_API_KEY=                       # Optional: live IOC enrichment
```

The system runs fully without these keys — only the Google ADK swarm tab
requires a Gemini key. All other features (forecast, counterfactual, reports,
benchmark) work offline.
