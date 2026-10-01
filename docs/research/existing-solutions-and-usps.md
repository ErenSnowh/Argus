# Existing Solutions & USPs — PS 26153: AI-based Network Attack Forecasting (World Models)

> **Problem Statement 26153** (NTRO / NCIIPC, SIH). Learn evolving network state from
> traffic telemetry, forecast attack **progression** before compromise completes, map to
> **MITRE ATT&CK** stages, and explain predictions (attention / SHAP). Fully open-source,
> runs offline. Datasets: CIC-IDS2017/18, UNSW-NB15, CTU-13, CICIoT2023, LANL, DARPA.

This document surveys **what already exists** across research, commercial products, and
open-source tooling, with each entry's **USP** (unique selling point) and its **gap vs. this
PS**. The last two sections synthesize the whitespace and position ARGUS against it.

**How to read the taxonomy.** The PS asks for something narrow and specific: a *learned
generative model of network-state transition dynamics* `P(S_t+1 | S_t)` over **real traffic
features**, used for **K-step forward simulation** of **attack-stage progression**, with
**per-feature explainability**. Almost everything below does *part* of this. The taxonomy is
organized so you can see exactly which part each prior solution solves — and which part
nobody quite ties together.

---

## 0. Quick verdict (the gap in one paragraph)

Existing work clusters into five buckets: **(a) static detection/anomaly** classifiers
(treat each flow independently — exactly what the PS criticizes), **(b) event/alert-sequence
prediction** (Tiresias, DeepCASE — closest in spirit, but predict the *next alert*, not the
future *network state* from raw flow/packet telemetry), **(c) provenance-graph APT detection**
(UNICORN→Kairos→Flash — powerful attack-progression modeling, but on host audit logs, not
network flow/packet features, and mostly *detect* rather than *forecast forward*), **(d) RL
world-model gyms** (CybORG/CAGE, CyberBattleSim — learn transition dynamics + defense policy,
but in *simulation*, not from real traffic telemetry), and **(e) commercial NDR/XDR** (Darktrace,
Vectra — behavioral, but closed/black-box and detection-first, not open forward-simulation).
**No mainstream solution combines all of: real flow+packet telemetry → learned transition
dynamics → K-step forward rollout of attack stage → open-source + SHAP/attention explanation.**
That intersection is the PS's target and ARGUS's differentiation.

---

## 1. Foundational "World Models" (the concept the PS invokes)

These are the origin of the term. None are cyber-specific — they matter because the PS explicitly
frames the task as a *world model* (`P(S_t+1|S_t)` + forward rollout).

| Solution | What it is | USP | Gap vs PS |
|---|---|---|---|
| **World Models** (Ha & Schmidhuber, 2018) | VAE + MDN-RNN that learns a latent "dream" of an environment; agent trains inside the imagined rollout | First to show an agent can learn a compressed generative model of environment dynamics and plan inside it | Game/robotics domain; no cyber, no traffic, no ATT&CK |
| **PlaNet / Dreamer / DreamerV2 / DreamerV3** (Hafner et al., 2019–2023) | Latent-space recurrent state-space models (RSSM) that predict future latent states and rewards; plan by imagination | SOTA sample-efficient model-based RL; the reference architecture for "learn dynamics, roll out K steps" | Not applied to network security; needs an environment/reward, not raw telemetry |
| **MuZero** (Schrittwieser et al., 2020) | Learns a model of dynamics sufficient for planning without knowing the rules | Planning with a *learned* latent transition model | General RL, not cyber |

**Takeaway for ARGUS:** the RSSM / latent-transition idea (encode state → predict next latent →
decode / classify stage) is the blueprint; the novelty is applying it to **flow+packet feature
sequences** with **ATT&CK-stage decoding**.

---

## 2. Academic — sequence-model attack / event prediction (closest prior art)

| Solution | What it is | USP | Gap vs PS |
|---|---|---|---|
| **Tiresias** (Shen et al., CCS 2018) | RNN that predicts the *next security event* from a stream of past events | First large-scale demonstration that security events are predictable with deep sequence models | Predicts next *alert/event*, not future *network state* from flow/packet features; no ATT&CK stage rollout |
| **DeepLog** (Du et al., CCS 2017) | LSTM models normal log sequences; deviation = anomaly; also builds workflow model | Treats logs as a language; detects anomalies in *sequence*, and can diagnose | Log-based anomaly detection, not forward attack-stage forecasting on traffic |
| **DeepCASE** (van Ede et al., IEEE S&P 2022) | Attention-based contextual analysis of security-event *sequences* to cut analyst workload | **Attention gives interpretability** over which prior events drive a verdict — directly matches the PS's explainability ask | Operates on IDS alerts, not raw flow/packet state; semi-supervised triage, not `P(S_t+1|S_t)` rollout |
| **LSTM/GRU/Transformer NIDS** (many, 2017–2024) | Sequence classifiers over flow windows (CIC-IDS/UNSW/CTU) | Capture temporal structure a per-flow classifier misses | Most still output a *label per window* (classification), not a generative next-state distribution or K-step forecast |
| **Nip in the Bud** (Zhu et al., 2024) | Forecasts & interprets *post-exploitation* attacks in real time using CTI reports | Forecasting + interpretation of attacker's next move | Driven by CTI text + host logs, not network flow/packet telemetry |
| **BiTA** (Nayeri & Rezvani, 2026) | BiGRU-Transformer aggregator in a Temporal Graph Network for **proactive alert prediction** | Explicitly *predicts future alerts* on a temporal graph — very on-theme | Alert-level prediction; not flow+packet feature state or SHAP-per-feature |
| **AlertStar** (Nayeri & Rezvani, 2026) | Path-aware alert prediction on hyper-relational knowledge graphs (multi-hop reasoning) | Multi-hop "what alert follows this chain" reasoning | Knowledge-graph over alerts, not raw traffic state transitions |

**Canonical reference:** *Husák, Komárková, Bou-Harb, Čeleda — "Survey of Attack Projection,
Prediction, and Forecasting in Cyber Security," IEEE Communications Surveys & Tutorials, 2019.*
The definitive taxonomy (attack **projection** = next steps, **intention recognition**,
**intrusion prediction**, **network security situation forecasting**). Cite this to frame the
field; it predates the world-model framing and shows the gap the PS is filling.

---

## 3. Academic — GNN & temporal-graph NIDS / prediction

Represent the network as a graph (hosts/flows as nodes/edges) — the PS's "graph state" option.

| Solution | What it is | USP | Gap vs PS |
|---|---|---|---|
| **E-GraphSAGE** (Lo et al., 2021/22) | Edge-featured GraphSAGE for NetFlow-based NIDS | First to bring **edge/flow features + graph topology** into GNN NIDS | Static classification of flows-as-edges; no temporal transition model / forecast |
| **Anomal-E** (Caville et al., 2022) | Self-supervised edge-GNN NIDS (no labels) | Label-free graph anomaly detection | Detection, not forward simulation or stage mapping |
| **PPT-GNN** (Van Langendonck et al., 2024) | *Pre-trained* spatio-temporal GNN, near-real-time, generalizes across networks with little labeled data | Pretraining → **generalization** (a PS requirement) + spatio-temporal | Still detection/prediction of malicious edges; not explicit `P(S_t+1|S_t)` rollout with ATT&CK stages |
| **Multi-stage Attack Detection & Prediction w/ GNN** (Friji et al., 2024) | 3-stage, kill-chain-inspired IDS that tests **feasibility of predicting later-stage attacks** (IoT) | Directly targets *predicting the next kill-chain stage* — very close to PS intent | IoT feasibility study; not a full traffic world model or SHAP explainer |
| **Euler / temporal-GNN lateral-movement** (2022+) | Temporal GNN over authentication graphs (LANL) to flag lateral movement | Learns **temporal** graph evolution for LANL-style auth data | Lateral-movement detection on auth logs; not a general traffic forecaster |
| **Towards a Generalisable Cyber Defence Agent** (Dudman & Bull, 2025) | GNN-based deep-RL defender that generalizes across topology/size without retraining | **Topology-agnostic generalization** — addresses "generalize to unseen" | RL policy in CybORG simulation, not real-traffic state forecasting |

---

## 4. Academic — provenance-graph APT progression & forecasting

The strongest existing line on **attack progression modeling** — but on **host/system audit
logs (provenance graphs)**, not network flow/packet telemetry. Extremely relevant as prior art
and for ATT&CK-stage mapping ideas.

| Solution | Year | USP |
|---|---|---|
| **SLEUTH / HOLMES** | 2017–19 | Real-time APT reconstruction from audit streams; HOLMES maps to **APT kill-chain stages** — the classic "map events to stages" reference |
| **UNICORN** | 2020 | Runtime provenance-based APT detector using streaming graph sketches; models whole-system evolution |
| **threaTrace** | 2021 | Node-level threat tracing via inductive GraphSAGE on provenance graphs |
| **Kairos** | S&P 2024 | Practical whole-system provenance IDS + **investigation**; strong SOTA baseline for attack-story reconstruction |
| **Flash** | S&P 2024 | Provenance-graph representation learning; efficient, high-fidelity APT detection |
| **MAGIC** | USENIX Sec 2024 | Masked graph representation learning for APT — self-supervised, multi-granularity |
| **NODLINK** | 2023 | Online, fine-grained APT detection + investigation |
| **SHIELD** | 2025 | APT detection **+ LLM-generated explanation** — matches the "interpretable decision support" ask |
| **Slot** | 2024 | Provenance-driven APT detection via **graph reinforcement learning** |
| **TBDetector / LogShield** | 2023 | **Transformer/self-attention** over provenance — attention-based interpretability |

**Gap vs PS:** these operate on **endpoint provenance (process/file/syscall)**, not
**NetFlow/PCAP** features, and most **detect/reconstruct** rather than **forward-simulate K
steps**. ARGUS's angle — the same "model the attack as an evolving process" philosophy, but on
**network traffic** with **explicit forward rollout** — is a genuine differentiator.

---

## 5. Academic — classic attack projection (HMM / Bayesian / attack graphs)

Pre-deep-learning "world models" of attack progression. Conceptually aligned with `P(S_t+1|S_t)`.

| Solution | What it is | USP | Gap vs PS |
|---|---|---|---|
| **Hidden Markov Models for attack prediction** (Årnes et al.; Ourston et al.) | HMM over IDS alerts → predict next attack state / risk | Explicit **state-transition** probability model — the pre-DL analog of a world model | Coarse discrete states; hand-built; doesn't ingest rich flow/packet features |
| **Bayesian Attack Graphs** (Poolsappasit et al., 2012; Frigault) | Probabilistic attack graphs → likelihood of reaching a target | Principled **probability of attacker progression** to crown jewels | Requires vuln/topology model; not learned from live traffic |
| **Variable-length Markov models / suffix trees** | Predict next event from observed sequences | Lightweight sequence prediction | Superseded by RNN/Transformer for rich features |

---

## 6. RL / world-model autonomous cyber-defence environments

These **do** learn transition dynamics + optimal defense policy — but in **simulation/emulation**,
not from real captured traffic. The "world model in cyber" work lives mostly here today.

| Solution | What it is | USP | Gap vs PS |
|---|---|---|---|
| **CybORG + CAGE Challenges 1–4** (TTCP) | Cyber Operations Research Gym; the standard RL env for autonomous cyber defense competitions | De-facto benchmark for **learning attacker/defender dynamics**; active research community | Simulated network state, not real NetFlow/PCAP; policy learning, not traffic forecasting |
| **CybORG++ / MiniCAGE** (Emerson et al., 2024) | Faster, enhanced CybORG | Better fidelity/speed for ACD research | Same simulation gap |
| **Microsoft CyberBattleSim** (2021) | OpenAI-Gym-like network attack simulation for RL | Popular, accessible lateral-movement RL sandbox | Abstract simulated topology; not traffic-level |
| **Yawning Titan** (UK DSTL / Alan Turing) | Abstract, fast ACD simulation | Configurable, scalable ACD training | Simulation only |
| **PrimAITE / FARLAND / CyGIL** | ACD training environments (emulated/high-fidelity) | Range from abstract to emulated-network fidelity | Not learning from real captured flow/packet telemetry |

**Takeaway:** ARGUS learning `P(S_t+1|S_t)` from **real datasets (CIC-IDS/CTU-13/PCAP)** rather
than a hand-built simulator is a meaningful distinction from this entire line.

---

## 7. Commercial NDR / XDR / SIEM (behavioral, "attack-story")

The market analog. Powerful, but **closed-source, cloud-tethered, and detection-first** — the PS
explicitly requires **open-source, offline, explainable**, which rules these out as submissions
but makes them the competitive baseline to cite.

| Product | USP | Gap vs PS |
|---|---|---|
| **Darktrace** (Self-Learning AI; DETECT / RESPOND-Antigena / PREVENT / Cyber AI Analyst) | Unsupervised "pattern of life" per device; **PREVENT models attack paths**; autonomous response; AI analyst auto-investigates | Black-box, proprietary; anomaly-first; "prediction" = attack-path modeling, not learned traffic-state rollout |
| **Vectra AI** (Attack Signal Intelligence) | ML that scores **attacker behaviors** and stitches them along the kill chain; strong prioritization | Proprietary; detection + prioritization, not open forward simulation |
| **ExtraHop Reveal(x)** | Wire-data/NDR with ML behavioral detection at scale | Proprietary; detection-first |
| **Cisco Secure Network Analytics** (ex-Stealthwatch) + **Encrypted Traffic Analytics** | **NetFlow-native** behavioral analytics; detects threats in encrypted traffic without decryption | Proprietary; anomaly/behavioral, not stage-forecast |
| **Corelight** | Zeek-powered NDR; rich open data format | Data platform; detection via added analytics |
| **Microsoft Sentinel (Fusion)** + **Defender XDR** | **Fusion ML correlates alerts into multi-stage incidents**; Defender builds the "attack story" and does **automatic attack disruption** | Cloud SIEM/XDR; correlation of alerts, not traffic-level world model |
| **Palo Alto Cortex XDR / XSIAM** | Causality chains; ML analytics across network+endpoint | Proprietary platform |
| **CrowdStrike Falcon** (Threat Graph, Indicators of Attack) | Graph of relationships across trillions of events; **IOAs = behavior/intent-based** detection | Endpoint-centric, proprietary |
| **IBM QRadar**, **Splunk ES** (+ Risk-Based Alerting, MLTK, ATT&CK mapping), **Exabeam** (Smart Timelines), **Securonix** | SIEM/UEBA: behavioral baselining, **ATT&CK-tagged** detections, incident timelines | Correlation/UEBA, not learned forward simulation of network state |

---

## 8. Attack-path prediction / BAS / exposure management

"Prediction" here = **graph reachability / simulated attack**, not learned temporal dynamics.

| Product | USP | Gap vs PS |
|---|---|---|
| **XM Cyber** | Continuously models **attack paths to critical assets**, finds **chokepoints** | Graph reachability over vulns/config, not traffic-state forecasting |
| **Skybox Security / RedSeal** | Network model + attack-graph analysis of exposure | Config/topology modeling, offline analysis |
| **Cymulate / SafeBreach / AttackIQ** (Breach & Attack Simulation) | Continuously **emulate** attacker TTPs to validate defenses | Runs known attacks; doesn't forecast from live telemetry |
| **Mandiant / Randori** | Adversary emulation + external attack surface | Red-team/ASM, not per-flow forecasting |
| **Recorded Future** | Predictive **threat intelligence** (external signals) | Macro threat forecasting, not per-network state |

---

## 9. Open-source tooling (feature extraction, monitoring, emulation)

Useful **components** (some map to ARGUS's own pipeline), not end-to-end forecasters.

| Tool | Role | USP |
|---|---|---|
| **CICFlowMeter / NFStream** | Flow feature extraction from PCAP → CSV (the CIC-IDS feature set) | Reference extractor for exactly the flow features the PS lists |
| **Zeek (Bro)** | Network security monitor; rich connection logs | Programmable, protocol-aware telemetry (feeds Corelight) |
| **Suricata / Snort** | Signature + some anomaly IDS/IPS | Ubiquitous, rule-based detection baseline |
| **Stratosphere IPS** (StratosphereLinuxIPS, CTU) | Behavioral IPS using **Markov-chain models of traffic behavior**; **creators of CTU-13** | Open-source **behavioral state modeling** of traffic — closest OSS spirit to a "world model," and dataset-native |
| **Security Onion / Wazuh / Elastic Security** | Open SIEM/monitoring stacks | Full detection+hunt platforms, ATT&CK dashboards |
| **MITRE CALDERA** | Automated adversary emulation | Generates realistic multi-stage attack telemetry (great for training/eval data) |
| **MITRE ATT&CK Flow / Attack Flow Builder** | Model & visualize sequences of ATT&CK techniques | Standard way to **represent attack progression** — pairs with the PS's stage-mapping deliverable |
| **Malcolm** | Full-packet analysis suite (Zeek+Arkime) | Turnkey PCAP analysis |

---

## 10. Datasets & knowledge bases (the substrate, not solutions)

Listed in the PS; note them as the training/eval and mapping substrate.

- **Traffic datasets:** CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15, CTU-13, CICIoT2023, LANL
  Authentication, DARPA/OpTC. (CTU-13 and CIC sets are labeled with attack timelines → usable
  for the PS's *supervised dynamics learning* on ground-truth state transitions.)
- **Knowledge bases:** MITRE ATT&CK (tactic/technique taxonomy for stage mapping), CAPEC
  (attack patterns), CVE/NVD (vulnerabilities), MITRE Engage/D3FEND, ATT&CK Flow.

---

## 11. Synthesis — the whitespace ARGUS targets

Cross-cutting gaps common to the prior art:

1. **Detection ≠ forecasting.** The vast majority (E-GraphSAGE, Anomal-E, most NIDS, all
   commercial NDR) classify the *present*. The PS wants a **generative next-state model +
   K-step rollout**.
2. **Alerts/logs ≠ raw traffic state.** The best predictors (Tiresias, DeepCASE, BiTA,
   provenance systems) operate on **alerts or host audit logs**, not the **flow+packet feature
   vector** the PS mandates (TCP flag bitmask, IAT stats, TTL variance, window size, scan
   signatures).
3. **Simulation ≠ real telemetry.** The explicit "world model" cyber work (CybORG/CAGE,
   CyberBattleSim) learns dynamics in a **simulator**, not from **CIC-IDS/CTU-13/PCAP**.
4. **Black-box ≠ explainable + open.** Commercial leaders are closed and cloud-tethered; the PS
   requires **open-source, offline, SHAP/attention explanations**.
5. **Flow-only OR packet-only.** Few combine **flow-level aggregate** features with
   **packet-level timing/sequencing** — the PS explicitly requires both (flow catches SYN
   floods; packet catches slow evasive scans).

**ARGUS positioning / USP against this landscape:**

| PS requirement | What prior art does | ARGUS USP |
|---|---|---|
| Learn `P(S_t+1 \| S_t)` from real traffic | RL gyms learn it in simulation; NIDS classify present | Learn transition dynamics from **CIC-IDS/CTU-13/PCAP** flow+packet features (LSTM/Transformer/GNN) |
| K-step forward simulation of infiltration | Attack-path graphs do reachability, not temporal rollout | **Roll out K windows ahead**, output a probability-over-time timeline |
| Map to MITRE ATT&CK stages | HOLMES/Vectra/Splunk tag detections | Forecast the **predicted future stage** (Recon→Initial Access→Lateral→C2→Exfil), not just tag the present |
| Explainability | DeepCASE/TBDetector use attention; SHAP-IDS exists | **SHAP + attention over the exact flags/ports/IAT features** driving each forecast |
| Open-source + offline | Commercial leaders are closed/cloud | Fully OSS, runs offline (already an ARGUS design constraint) |
| Combine flow + packet levels | Most pick one | **Both levels** fused into one state vector |
| Generalize to unseen attacks | PPT-GNN pretraining; Dudman&Bull generalization | Dynamics learning + held-out-attack eval vs. **logistic-regression baseline** (PS-mandated benchmark) |

> **Current ARGUS state (repo):** ARGUS today is a **detection + triage + response co-pilot** —
> a Random-Forest *static* flow classifier ([argus/ml/model.py](../../argus/ml/model.py)),
> rule-based composite threat scoring
> ([argus/ml/threat_score.py](../../argus/ml/threat_score.py)), and rule-based kill-chain
> correlation ([argus/ml/correlation.py](../../argus/ml/correlation.py)), plus MITRE mapping,
> PCAP forensics, guardrails, and a dashboard. On top of that, the **world-model core is now
> built but not yet trained**: the 46-column schema, the Transformer architecture with K-step
> rollout ([argus/ml/world_model/model.py](../../argus/ml/world_model/model.py)), and the
> real CIC-IDS-2017 host/time-binned sequence pipeline with pinned-hash provenance
> ([argus/scripts/prepare_cicids2017.py](../../argus/scripts/prepare_cicids2017.py)) all
> exist and are unit-tested (164 tests collected). What does **not** exist yet: a trained
> checkpoint and therefore any benchmark numbers (they are *not yet measured* — see
> [world-model-core.md](../plan/world-model-core.md) WP3–WP6); until a checkpoint ships,
> forecasts are served by a hand-set heuristic kill-chain prior, clearly labelled as such.
> Completing the learned `P(S_t+1|S_t)` model + evaluation vs. the logistic-regression
> baseline is the remaining whitespace above.

---

## 12. One-screen comparison

| Category | Representative solutions | Learns transition dynamics? | Forecasts forward (K-step)? | On real flow/packet telemetry? | ATT&CK stage forecast? | Explainable? | Open-source? |
|---|---|---|---|---|---|---|---|
| World models (general) | Dreamer, MuZero | ✅ | ✅ | ❌ | ❌ | partial | ✅ |
| Sequence event prediction | Tiresias, DeepCASE, DeepLog | partial | next-event | ❌ (alerts/logs) | partial | ✅ (attn) | ✅ |
| GNN NIDS | E-GraphSAGE, Anomal-E, PPT-GNN | ❌/temporal | ❌ | ✅ | ❌ | partial | ✅ |
| Provenance APT | Kairos, Flash, MAGIC, HOLMES | partial | partial | ❌ (host logs) | ✅ (HOLMES) | some (SHIELD/attn) | ✅ |
| RL cyber gyms | CybORG/CAGE, CyberBattleSim | ✅ | ✅ | ❌ (simulation) | partial | partial | ✅ |
| Commercial NDR/XDR | Darktrace, Vectra, Cisco SNA | ❌ (behavioral) | attack-path only | ✅ | tagged | ❌ (black-box) | ❌ |
| Attack-path/BAS | XM Cyber, SafeBreach | ❌ | reachability/emulation | ❌ | ✅ | partial | ❌ |
| **ARGUS target (PS 26153)** | — | ✅ | ✅ | ✅ (flow+packet) | ✅ | ✅ (SHAP+attn) | ✅ |

---

## 13. Key references (for the architecture doc / slides)

- Husák et al., *Survey of Attack Projection, Prediction, and Forecasting in Cyber Security*, IEEE COMST, 2019.
- Ha & Schmidhuber, *World Models*, 2018; Hafner et al., *Dreamer / DreamerV3*, 2019–2023.
- Shen et al., *Tiresias: Predicting Security Events Through Deep Learning*, CCS 2018.
- van Ede et al., *DeepCASE: Semi-Supervised Contextual Analysis of Security Events*, IEEE S&P 2022.
- Du et al., *DeepLog*, CCS 2017.
- Lo et al., *E-GraphSAGE*, 2021; Caville et al., *Anomal-E*, 2022; Van Langendonck et al., *PPT-GNN*, 2024.
- Friji et al., *Multi-stage Attack Detection and Prediction Using GNNs (IoT feasibility)*, 2024.
- Han et al., *UNICORN*, NDSS 2020; Cheng et al., *Kairos*, S&P 2024; Rehman et al., *Flash*, S&P 2024; Jia et al., *MAGIC*, USENIX Sec 2024; Milajerdi et al., *HOLMES*, S&P 2019.
- Nayeri & Rezvani, *BiTA* / *AlertStar*, 2026.
- TTCP CAGE Challenges / CybORG; Microsoft CyberBattleSim, 2021; DSTL Yawning Titan.
- Stratosphere IPS (CTU) — behavioral Markov models; creators of CTU-13.
- Vendor docs: Darktrace, Vectra AI, Cisco Secure Network Analytics, Microsoft Sentinel Fusion / Defender XDR, XM Cyber.

*Compiled for ARGUS / SIH PS 26153. Verify vendor feature names against current product docs before final submission.*
