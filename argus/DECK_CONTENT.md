# ARGUS — SIH 2026 Deck Content & Exact Numbers

> **Problem Statement 26153** · *AI-based Network Attack Forecasting from Network Traffic Data using World Models* · Client: **NTRO / NCIIPC**
> Source of every number below: `ml/artifacts/world_model_metrics.json` and `ml/pretrained/benchmark_results.json` (committed). **Data = synthetic** (honest label — real CIC-IDS-2017 run is prepared and pending; see last slide).

---

## SLIDE 1 — Title

**ARGUS**
Autonomous Attack **Forecasting** & Multi-Agent SOC Co-Pilot

- Smart India Hackathon 2026 · Problem Statement **26153**
- *Predicts the next kill-chain stage before it happens — not another reactive IDS.*

Speaker note: Lead with the one-line differentiator. ARGUS forecasts; Zeek/Suricata/Snort classify what already happened.

---

## SLIDE 2 — The Problem (PS 26153)

- Critical Information Infrastructure (power, telecom, banking, transport, govt, strategic) is under continuous, staged attack.
- Existing IDS/IPS are **reactive**: they label a packet/flow *after* the malicious payload is on the wire.
- NTRO/NCIIPC need **proactive** warning + **statutory** reporting (CERT-In 6-hour SLA, NCIIPC Section 70A).

**The ask:** read network traffic over time → model its dynamics → forecast the next attack stage → explain it → report it.

---

## SLIDE 3 — Why existing tools fall short

| Tool | What it does | What it can't do |
|------|--------------|------------------|
| Snort / Suricata | Signature match per packet | No forecast; unknown attacks slip through |
| Zeek | Protocol logging + scripting | Descriptive, not predictive |
| Wazuh / SELKS | SIEM correlation of past events | Reactive; no P(next stage) |

All are **per-flow classifiers**. None answer: *"Given the last 10 minutes, what MITRE stage are we in 5 minutes from now?"*

---

## SLIDE 4 — ARGUS core idea: a World Model of the network

ARGUS learns the **state-transition dynamics**:

$$\mathcal{P}(S_{t+1} \mid S_t, \mathbf{x}_t)$$

- A **Temporal Transformer** consumes a window of traffic state and rolls forward **K steps** autoregressively.
- At each step it emits: the next **state**, the **MITRE ATT&CK stage**, and an **infiltration probability**.
- Multi-horizon anchors forecast at **+30s / +60s / +120s / +300s**.

Speaker note: This is the "world model" the problem statement names — borrowed from model-based RL, applied to network defense.

---

## SLIDE 5 — Architecture (exact spec)

**Model:** 4-layer causal Transformer decoder
- `d_model = 128`, **4** attention heads, `dim_ff = 256`
- **619,803** trainable parameters
- 3 prediction heads + 4 horizon-anchor heads:
  - **State head** — regress next 46-dim feature vector (loss weight α = 0.25)
  - **Stage head** — 8-class MITRE stage, cross-entropy with `ignore_index=-1` masking (β = 0.30)
  - **Infiltration head** — binary attack probability (γ = 0.25)
  - **Horizon heads** — +30/+60/+120/+300s anchors (δ = 0.20)

**Input schema:** **46 features** = 24 flow + 6 extended + 8 packet-level + 8 topology

**Pipeline:** CSV / PCAP / 8 dataset loaders → 60-second host-time bins → sequence windows → Transformer → forecast + XAI + graph signals → dashboard + CERT-In/NCIIPC report.

---

## SLIDE 6 — The 8 MITRE kill-chain stages ARGUS forecasts

`BENIGN → Reconnaissance → Initial Access → Credential Access → Lateral Movement → Command & Control → Exfiltration → Impact`

- Per-step stage prediction across the K-step rollout.
- Stage-target masking (`STAGE_TARGET_IGNORE = -1`) keeps ambiguous bins (e.g. slow Infiltration/Heartbleed) from corrupting the stage head — safe to train on real CIC-IDS-2017.

---

## SLIDE 7 — Explainability (XAI) — required by PS

Every forecast carries a per-feature attribution, tagged with its method:

- **SHAP** (`shap.GradientExplainer` on the Transformer) → `xai_method: "shap_gradient"`
- **Attention fallback** (multi-layer attention weights) → `xai_method: "attention_weights"`
- Plus **TreeSHAP** on the RF flow classifier.

**Graph signals** (lightweight, no GNN training): degree-centrality delta, new-edge burst, cross-subnet ratio, composite **lateral-movement signal** — catches lateral movement (T1021) that flow-only features miss.

---

## SLIDE 8 — Results: training (synthetic validation run)

> Honest framing: synthetic data is well-separated **by design** — its purpose is to prove the full neural pipeline learns end-to-end (all heads active, non-zero gradients). It is **not** a performance claim. Real-data numbers are one command away (last slide).

| Metric | Value |
|--------|-------|
| Training data | **Synthetic**, 500 sequences (400 train / 100 test) |
| Sequence length | 10 steps × 30s = 5 min |
| Features | 46 |
| Parameters | 619,803 |
| Epochs | **26** (early stop, patience 8) |
| Best validation loss | **0.4413** |
| **Stage accuracy** | **99.8%** |
| **Infiltration AUC** | **0.975** |
| Training time | **25.1 s** (CPU) |
| Supervision check | PASSED — all 3 heads active |

---

## SLIDE 9 — Results: 3-way benchmark (synthetic)

Split: 400 train / 100 test · infiltration ratio 0.84 train / 0.83 test · **leakage_detected: false**

| Model | Accuracy | F1-macro | FPR | PR-AUC | AUC-ROC |
|-------|----------|----------|-----|--------|---------|
| Logistic Regression | 0.997 | 0.9961 | 0.0029 | 0.9999 | 1.0 |
| Random Forest | 1.000 | 1.0000 | 0.000 | 1.0 | 1.0 |
| **World Model (ARGUS)** | 0.997 | 0.9958 | 0.0029 | 0.9999 | 1.0 |

**On synthetic data all three hit the ceiling** (RF nominally wins by 0.4% F1). The honest takeaway: the classifiers and the world model are *equivalent at classification* here — but only the world model also **forecasts forward** and **predicts the stage sequence**, which the benchmark's lead-time metric measures:

- World Model early-warning lead time: mean **0.7s**, up to **4.3s** on BruteForce onset.
- Early-warning rate on synthetic: 1.1% (1 of 91 attack sequences) — abrupt synthetic attacks give little runway; **real temporal data is where lead time is expected to pay off.**

Speaker note: Do **not** claim ARGUS beats the baselines on accuracy here — it doesn't, and a judge will check the JSON. Claim the *capability gap*: forecasting + stage sequencing + counterfactual, which LR/RF structurally cannot do.

---

## SLIDE 10 — What makes ARGUS genuinely different

1. **Temporal World Model** — P(S₍ₜ₊₁₎ | S₍ₜ₎), not per-flow labels.
2. **K-step forward simulation** with per-step MITRE stage forecast + multi-horizon anchors.
3. **Action-conditioned counterfactuals** — "if I BLOCK_SRC now, what's the new infiltration trajectory?" (`counterfactual.py`: BLOCK_SRC / BLOCK_DST_PORT / THROTTLE).
4. **Statutory compliance generation** — CERT-In (6-hr SLA) + NCIIPC Section 70A across 6 CII sectors.
5. **Multi-agent swarm** — 6 Google-ADK agents over a 19-tool MCP server.
6. **Air-gap capable** — offline deterministic engine, hash-chained tamper-evident audit log, never conflates neural vs heuristic engine.

---

## SLIDE 11 — System scope (what's actually built)

- **8 dataset loaders**: CIC-IDS-2017/2018, UNSW-NB15, CTU-13, CICIoT2023, LANL Auth, DARPA/NSL-KDD, synthetic generator.
- **19 MCP tools** (detection, ATT&CK, CAPEC, IOC, PCAP forensics, playbooks, CERT-In, NCIIPC, world-model forecast + counterfactual, KB/CVE).
- **6 ADK agents**: triage · enrichment · forensics · remediation (propose-only) · compliance · report.
- **FastAPI dashboard**: SSE streaming, CSV/PCAP upload, fan-chart forecast, SHAP/attention panel, counterfactual sandbox.
- **Security**: tool allowlists (RBAC), PII/secret redaction, prompt-injection detection, SHA-256 HMAC hash-chained audit.
- **Tests**: **163 passed, 1 skipped**.

---

## SLIDE 12 — Live demo flow (jury)

1. Open dashboard → health badge shows **ENGINE: NEURAL** (checkpoint loaded).
2. Upload `sample_cicids2017.csv` → **Run Simulation**.
3. Watch the **fan chart**: infiltration probability across +30/+60/+120/+300s with widening confidence.
4. Read the **kill-chain stages** per step + the **SHAP driving features**.
5. Open **Counterfactual Sandbox** → BLOCK_SRC → see the risk-delta trajectory.
6. Generate the **NCIIPC** report for "Power & Energy" → valid Section 70A advisory.

---

## SLIDE 13 — Roadmap / honesty slide

- **Done:** full neural pipeline trained + committed (synthetic checkpoint, WP6); dashboard, XAI, graph signals, compliance, benchmark harness all live.
- **Next (one command — data already binned at `data/processed/cicids2017_bins_60s.parquet`):**
  ```
  python -m ml.world_model.train --dataset cicids2018 \
      --path data/processed/cicids2017_bins_60s.parquet --epochs 100 --seq-len 15
  python -m ml.world_model.benchmark --dataset cicids2018 \
      --path data/processed/cicids2017_bins_60s.parquet
  ```
  → replaces synthetic numbers with real CIC-IDS-2017 PR-AUC lift, FPR on minority classes, and early-warning lead time.

---

### Appendix — numbers to have on hand for Q&A

- Params: 619,803 · Layers: 4 · d_model: 128 · heads: 4 · dim_ff: 256
- Loss weights: state 0.25 / stage 0.30 / infiltration 0.25 / horizon 0.20
- Best val loss 0.4413 · stage acc 99.8% · infiltration AUC 0.975 · 26 epochs · 25.1s CPU
- Benchmark winner on synthetic: RF (F1 1.0); WM F1 0.9958; WM−RF F1 = −0.0042; WM−LR F1 = −0.0003
- WM lead time: mean 0.7s, BruteForce 4.3s; early-warning rate 1.1% (synthetic)
- 46 features = 24 flow + 6 extended + 8 packet + 8 topology
- 8 datasets · 19 MCP tools · 6 agents · 163 tests pass / 1 skip
