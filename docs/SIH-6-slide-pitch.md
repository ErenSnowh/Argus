# SIH 2026 Final Screening — 6-Slide Pitch (Template-Compliant Content)

> **Template rules (from attached template — non-negotiable):**
> 1. **Maximum 6 slides including title slide.**
> 2. **No paragraphs** — only points / diagrams / infographics / pictures.
> 3. Keep headers exactly as the template: `TITLE PAGE`, `IDEA TITLE`, `TECHNICAL APPROACH`, `FEASIBILITY AND VIABILITY`, `IMPACT AND BENEFITS`, `RESEARCH AND REFERENCES`.
> 4. Do not change the "idea details pointers" inside each slide.
> 5. **Delete slide 7 (IMPORTANT INSTRUCTIONS)** before uploading.
> 6. Export as **high-resolution PDF** and upload to the portal (no .pptx / .docx).
>
> **Evaluator truth:** judges skim hundreds of decks in minutes. Every slide below is built as
> **short bullets + one diagram/infographic per slide**. The deep detail lives in your voice,
> the live demo, and the Q&A bank at the end of this file.
>
> **Product used throughout:** ARGUS — *Autonomous Attack Forecasting & Multi-Agent SOC Co-Pilot*,
> SIH Problem Statement **ID 26153**: *AI-based Network Attack Forecasting from Network Traffic Data using World Models* (NTRO / NCIIPC).

---

## SLIDE 1 — TITLE PAGE

**Fill the template pointers exactly (from your SIH portal registration):**

| Template Pointer | What to Write |
|---|---|
| Problem Statement ID | **26153** |
| Problem Statement Title | **AI-based Network Attack Forecasting from Network Traffic Data using World Models** |
| Theme | *Copy the exact theme string from the SIH portal (e.g., Cyber Security / Defence & Strategic Technologies — use whatever the portal shows)* |
| PS Category | **Software** |
| Team ID | *From portal* |
| Team Name (Registered on portal) | *Exact registered team name* |

**Add below the pointers (still Slide 1, one line only — the "hook"):**

> **ARGUS — an offline AI that predicts the next attack stage *before* the hacker lands, and explains why.**

**Small skills strip at the bottom (icons, not text):**
`PyTorch` · `FastAPI` · `Docker` · `MITRE ATT&CK` · `SHAP` · `100% Open Source`

**Design notes**
- Keep SIH logo top-right as in template; team-name bubble top-left.
- 3-second test: a judge must read PS ID + hook in one glance.
- Tools: Canva/Figma for the header; the template layout itself.

**Speaker note (say when you present, not on slide):**
*"We are Team X, solving PS 26153 for NTRO. Every existing security tool rings the alarm after the breach — we built an AI that calls the play before the attacker scores."*

---

## SLIDE 2 — IDEA TITLE (PROPOSED SOLUTION)

**Keep the template pointer:** `❖ Proposed Solution (Describe your Idea/Solution/Prototype)`
**Sub-headline under it (one line):** **ARGUS — Autonomous Attack Forecasting & Multi-Agent SOC Co-Pilot**

### Diagram (center of slide, no paragraphs): horizontal 4-box pipeline

```
┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐
│  ① INPUT     │ → │  ② WORLD MODEL   │ → │  ③ FORECAST      │ → │  ④ OUTPUT          │
│ Flow CSV /   │   │ Learns P(Sₜ₊₁|Sₜ)│   │ K-step rollout:  │   │ MITRE stage +      │
│ PCAP traffic │   │ — how the network│   │ "what attacker    │   │ SHAP reasons +     │
│ telemetry    │   │   evolves over   │   │  does NEXT" +     │   │ offline dashboard  │
│              │   │   time           │   │ P(infiltration)  │   │ + CERT-In report   │
└──────────────┘   └──────────────────┘   └──────────────────┘   └────────────────────┘
```

### Bullets — Detailed explanation (3 pointers of the template)

- **Watches traffic as a time-story, not isolated packets**
- **Learns network "world model": P(Sₜ₊₁ | Sₜ)**
- **Forecasts K steps ahead → attack probability before compromise**
- **Maps forecast to MITRE ATT&CK kill-chain stages**
- **Explains every alert: SHAP + attention weights**
- **Runs 100% offline — no cloud APIs, air-gap ready**

### Bullets — How it addresses the problem

- **Today's IDS alarms *after* entry — avg breach takes 241 days to contain (IBM 2025)**
- **ARGUS warns during recon/lateral movement — *before* data leaves**
- **Guard analogy: flags the latch-checker before the safe breaks**

### Bullets — Innovation & uniqueness

- **1st open-source combo: real flow+packet telemetry → learned state transitions → K-step attack forecast**
- **Dual engine: Temporal Transformer + zero-dependency Markov fallback (edge/air-gap)**
- **Auto-generates CERT-In (6-hr SLA) & NCIIPC CII advisories**

**Design notes**
- One diagram, ≤6 short bullets total. Numbers bolded.
- Tools: Excalidraw / Eraser.io / mermaid for the pipeline; present as PNG.

**Speaker note:**
*"Static tools judge each packet in isolation. ARGUS models the attacker's story — reconnaissance, initial access, lateral movement, exfiltration — as a sequence, fast-forwards it, and tells the defender the next move minutes before it happens."*

---

## SLIDE 3 — TECHNICAL APPROACH

**Keep the template pointers.** Two zones: **Tech Stack cards** (top) + **Methodology flowchart** (bottom) + **working-prototype thumbnail**.

### Technologies to be used (categorized cards — exact frameworks, no buzzwords)

| Layer | Stack |
|---|---|
| **Languages & Core** | Python 3.10+, NumPy, pandas, Scapy/PyShark |
| **ML / AI** | PyTorch — Temporal Transformer (4-layer, d=128, 612K params); scikit-learn Random Forest + TreeSHAP; empirical Markov fallback |
| **Backend & API** | FastAPI, SSE streaming, MCP tool server, CLI |
| **Frontend** | Offline single-page SOC dashboard (HTML/JS), charts via Chart.js/Canvas |
| **Data (8 loaders)** | CIC-IDS-2017/18, UNSW-NB15, CTU-13, CICIoT2023, LANL, DARPA/NSL-KDD, synthetic temporal |
| **Knowledge & Standards** | MITRE ATT&CK v14 (697 techniques), CAPEC, NIST NVD, NetFlow/PCAP/IPFIX |
| **Deploy / Hardware** | Docker (CPU-only, no GPU needed), works air-gapped |

### Methodology & process (one flowchart, bottom half)

```
PCAP / Flow CSV
   → Feature Extraction (38-feature schema, 5–10 s sliding windows)
   → Sequence Builder  ([t₋₂, t₋₁, t₀] → predict t₊₁)   ← no future leakage (temporal split)
   → Dual ML Engine:  Random-Forest classifier  |  Temporal Transformer world model
   → Three heads:  ① next-state (MSE)  ② ATT&CK stage (CE)  ③ infiltration P (BCE)
   → K-step autoregressive rollout → attack-probability timeline
   → Explainability: TreeSHAP + attention weights
   → Dashboard (SOC Radar · Forecaster) + CERT-In / NCIIPC report
```

### Working prototype (thumbnail + 3 proof points)

- **5-tab live dashboard:** Radar · Forecaster · Knowledge Base · Datasets · CII/NCIIPC
- **CLI:** `classify · forecast · benchmark · investigate`
- **37/37 automated tests passing** · 1-command `docker run`

**Design notes**
- Left: 3–4 colored stack cards. Right/bottom: single flowchart (Excalidraw style).
- Add one screenshot of the **World Model Forecaster tab** (K-step curve) as the prototype proof.

**Speaker note:**
*"Packets are grouped into time windows into a 38-feature state vector. The Transformer is trained with a combined loss — state reconstruction, stage classification, infiltration probability — then rolled out autoregressively K steps. If PyTorch isn't available on an air-gapped edge box, the Markov kill-chain engine gives the same interface with zero dependencies."*

---

## SLIDE 4 — FEASIBILITY AND VIABILITY

**Keep the template pointers.** Three columns: **Feasibility** | **Challenges & Risks** | **Strategies** — plus a bottom **Security & Compliance strip**.

### Column 1 — Analysis of feasibility of the idea

- **Technical:** 612K-param model trains in ~58 s on CPU; inference real-time
- **Operational:** drop-in on existing NetFlow/PCAP sensors; Docker = 1 command
- **Economic:** 100% open-source → zero cloud/API/licence cost
- **Data:** 8 public benchmark datasets already bundled & normalized

### Column 2 — Potential challenges and risks

- **False alarms / alert fatigue on never-seen attack variants**
- **Raw PCAP volume (GBs) & class imbalance**
- **Model drift as attackers change TTPs**
- **AI co-pilot itself: prompt-injection & PII leakage risk**
- **Deployment on legacy/air-gapped CII networks**

### Column 3 — Strategies for overcoming these challenges

- **Temporal no-leak splits + 8-dataset eval vs LR/RF baseline (FPR tracked per class)**
- **Flow-aggregation before inference — never feed raw PCAP to the model**
- **Continuous retraining triggers + kill-chain transition monitoring**
- **Guardrails: PII/secret redaction · RBAC · injection scanning · SHA-256 HMAC audit chain · human approval gate**
- **Markov fallback = works with zero ML dependencies on old hardware**

### Bottom strip — SECURITY & COMPLIANCE (one banner row)

> **Air-gapped / no cloud APIs · AES-256 at rest · tamper-evident HMAC audit log · CERT-In 6-hour reporting SLA (Sec 70B) · NCIIPC CII advisory (Sec 70A, 6 sectors)**

**Design notes**
- Use a 3-column risk-matrix table (Notion/Eraser style) with 🔴 risk → 🟢 mitigation arrows.
- Security banner as a full-width colored footer inside the slide body.

**Speaker note:**
*"Feasibility is proven, not promised: the model already trains on a laptop CPU in under a minute, ships as a Docker container, and every risk we list has a mechanism already implemented in code — redaction, RBAC, audit chain, approval gates."*

---

## SLIDE 5 — IMPACT AND BENEFITS

**Keep the template pointers.** Layout: **Big-number metric cards** (top) + **Audience impact** (left) + **Benefits** (right).

### Top row — measurable metrics (big-number infographic cards)

| 99.97% | 93.0% | 0.983 | K = 5 | 8 | 37/37 |
|---|---|---|---|---|---|
| World-model accuracy | RF baseline (proof it beats static AI) | Infiltration AUC | windows of **early warning** before compromise | benchmark datasets | tests green |

### Potential impact on the target audience

- **NTRO / NCIIPC:** pre-breach forecasting for 6 CII sectors — power, BFSI, telecom, transport, govt, defence
- **SOC analysts:** few, ranked, *explained* alerts instead of thousands of noisy late alarms
- **Enterprises / ISPs / hospitals / railways:** affordable, offline, Indian-made defence stack

### Benefits of the solution

- **Social:** protects hospital, power-grid & citizen data; public trust in digital India
- **Economic:** cuts breach cost — global avg **$4.44M/breach**, 241-day containment (IBM 2025); India logged **2.04M CERT-In incidents (2024)** — we shorten detect-to-contain
- **National / Strategic:** **Atmanirbhar** sovereign offline stack — no foreign cloud, no telemetry leaving the network
- **Environmental:** CPU-only inference, no GPU farm, minimal energy footprint
- **Ecosystem:** MIT-licensed open source → Indian SOC/startups can extend it

**Design notes**
- Metric cards style (Infogram / Rows.com). Keep ≤6 numbers — pick the strongest.
- One small India-map or CII-sector icon row for audience.

**Speaker note:**
*"The judges' question is always 'so what?' — here: a forecaster that gives defenders minutes of warning instead of 241 days of cleanup, at zero licence cost, compliant with Indian law, with no data ever leaving the premises."*

---

## SLIDE 6 — RESEARCH AND REFERENCES

**Keep the template pointer:** `Details / Links of the reference and research work`
Two blocks: **Problem Statutory & Standards** and **Research Foundations** (short links only).

### Block A — Problem statement & statutory references

- SIH PS **26153** — AI-based Network Attack Forecasting using World Models (NTRO/NCIIPC)
- CERT-In Cyber Security Directions 2022 — 6-hr incident reporting — https://www.cert-in.org.in
- NCIIPC (Sec 70A, IT Act 2000) — CII protection — https://nciipc.gov.in
- MITRE ATT&CK v14 — https://attack.mitre.org · CAPEC · NIST NVD API 2.0

### Block B — Research foundations (one line each)

- **Field survey:** Husák et al., *Attack Projection/Prediction/Forecasting Survey*, IEEE COMST 2019
- **World Models:** Ha & Schmidhuber 2018 — https://arxiv.org/abs/1803.10122 · DreamerV3, Hafner et al. 2023
- **Security sequence prediction:** Tiresias (CCS'18) · DeepLog (CCS'17) · DeepCASE (S&P'22)
- **Graph & provenance NIDS:** E-GraphSAGE · Anomal-E · Kairos (S&P 2024)
- **Explainability:** SHAP — Lundberg & Lee, https://arxiv.org/abs/1705.07874
- **Datasets:** CIC-IDS-2017/18 (https://www.unb.ca/cic/datasets/ids-2017.html) · UNSW-NB15 · CTU-13 · CICIoT2023 · LANL
- **Threat/economics data:** IBM Cost of a Data Breach 2025 · CERT-In incident reports 2024 · CloudSEK ThreatLandscape 2024

**Design notes**
- Two clean columns, 8–10 lines max, tiny font acceptable here (references are skim-targeted).
- Optionally QR code → your GitHub repo / demo video.

---

# APPENDIX A — Where every requested topic lives

6 slides can't hold everything — this map prevents content loss:

| Your requested topic | Slide | Backup (verbal / demo / Q&A) |
|---|---|---|
| Real-world problem feel & context | 2 (hook + 241-day stat) | Opening line + Q&A |
| Proposed solution | **2** | — |
| System architecture | **2 (pipeline) + 3 (flowchart)** | Full mermaid diagram on laptop; live dashboard |
| Technical approach | **3** | Deep-dive Q&A (loss heads, training config) |
| Security info | **4 (bottom strip)** | Guardrails demo: HMAC verify, redaction test |
| Feasibility | **4** | Docker run live |
| Gap (vs existing solutions) | **2 (uniqueness bullets)** | One-screen comparison table in backup deck |
| Prototype | **3 (thumbnail)** | **Live demo** — strongest asset, show it |
| Risk mitigation | **4 (column 3)** | — |
| Benchmarks & datasets | **5 (metric cards)** + **6 (dataset links)** | `benchmark.py` run + confusion matrix backup |
| Impact / audience / value / strategic benefits | **5** | — |
| Research & references | **6** | — |
| Architecture detail, agent swarm, MCP, Q&A bank | — | Backup slides (NOT in uploaded PDF — allowed for live session only) |

> ⚠️ The 6-slide cap applies to the **PDF uploaded on the portal**. For the live screening session, keep a **separate backup deck** (not uploaded) with: full architecture, benchmark tables, gap-vs-competitors matrix, and Q&A slides.

---

# APPENDIX B — 30-Second Pitch Script (for the live round)

> "Critical infrastructure defenders are blind between the first scan and the final data theft. Tools today only ring the alarm *after* entry — that's why breaches take 241 days to contain. **ARGUS** is an open-source, fully offline AI **world model** for PS 26153: it learns how network state evolves — P of next state given current state — then fast-forwards K steps to predict *which MITRE ATT&CK stage the attacker hits next*, and explains the call with SHAP. On 8 public datasets it scores **99.97% accuracy vs a 93% static baseline**, with infiltration AUC 0.983 — and it auto-drafts CERT-In's 6-hour report. Built for NTRO's air-gapped networks: no cloud, no GPU, one Docker command."

---

# APPENDIX C — Anticipated Judge Questions (Q&A bank)

1. **How is this different from an IDS/antivirus?** → IDS classifies the *present* packet; ARGUS models *temporal state transitions* and forecasts K steps *ahead* — detection vs projection (cite Husák survey).
2. **Why a world model instead of just an LSTM classifier?** → A classifier labels one window; a world model generates the next state distribution, enabling multi-step rollouts and P(infiltration).
3. **Proof it beats simple ML?** → Same temporal test set: LR/RF baseline vs Transformer — 93.0% → 99.97% accuracy, macro-F1 92.99% → 99.97%, infiltration AUC 0.983; benchmark harness is `ml/world_model/benchmark.py`.
4. **How do you avoid data leakage?** → Temporal (chronological) train/test splits, per-feature Z-score fit on train only.
5. **How is it explainable?** → TreeSHAP on the RF per-flow classifier + attention weights over the sequence; dashboard shows top contributing features (ports, flags, IAT).
6. **Runs truly offline?** → Yes — local ATT&CK/CAPEC/CVE caches, no external APIs; Markov fallback even works without PyTorch.
7. **What about adversarial attacks on your own AI?** → Prompt-injection scanning on telemetry, tool RBAC for agents, human approval gate before any action, tamper-evident HMAC audit chain.
8. **Real PCAP is huge — how do you handle it?** → Flow aggregation into 5–10 s windows + 38-feature extraction first; the model never sees raw packets.
9. **False positives on unseen attacks?** → Per-class FPR tracking, low-confidence downgrade logic, ranked alerts instead of binary alarms; drift retraining strategy on slide 4.
10. **Statutory value?** → Auto-generates CERT-In (6-hr SLA, Sec 70B) and NCIIPC advisory (Sec 70A) with forecast trajectory + ATT&CK/CVE enrichment.
11. **Deployment target?** → Docker container on CPU at the sensor/edge; 612K-param model, ~58 s training, real-time inference.
12. **What's still missing (honest gap)?** → Larger real-world validation beyond 8 public datasets; GNN topology layer on the roadmap; continuous online learning.

---

# APPENDIX D — Round-1 Submission Checklist

- ✔ Exactly **6 slides** (template headers unchanged; Slide 7 "Important Instructions" **deleted**)
- ✔ **Zero paragraphs** — every slide = short bullets + diagram/infographic
- ✔ **PS ID 26153** prominent on Slide 1 with hook sentence
- ✔ Architecture shown as **clean block diagram** (Slides 2 & 3)
- ✔ Tech stack lists **exact frameworks** (Python, PyTorch, scikit-learn, FastAPI, Docker, MITRE ATT&CK)
- ✔ Metrics **quantified** (99.97% / 93.0% / 0.983 / K=5 / 612K params / 37 tests)
- ✔ Risks & mitigations paired on Slide 4; security banner present
- ✔ References with **short working links** on Slide 6
- ✔ Exported as **high-resolution PDF** (File → Export → PDF; fonts embedded; open once to verify)
- ✔ File named: `PS26153_<TeamName>.pdf` (portal conventions as instructed)
- ✔ Backup deck + live demo ready for the screening session (not part of the 6-slide PDF)

**Recommended production tools:** PowerPoint/Google Slides (template base) · Excalidraw or Eraser.io (diagrams) · Canva/Figma (metric cards) · Infogram (big-number charts) · GitHub repo + demo video as QR code.

---

# GENERATED DECK (this branch)

| File | Purpose |
|---|---|
| `docs/SIH_PS26153_ARGUS_6slide.pptx` | **Editable deck** — fill `<from portal>` placeholders (Theme, Team ID, Team Name), swap in official SIH logo over the top-right wordmark, then export |
| `docs/SIH_PS26153_ARGUS_6slide.pdf` | **Portal-ready PDF** — verified 6-page render (rename to `PS26153_<TeamName>.pdf` before upload) |
| `tools/build_sih_deck.py` | Regenerates the PPTX (`python3 tools/build_sih_deck.py`, needs `python-pptx`) |
| `tools/build_sih_deck_pdf.py` | Regenerates the PDF + PNG previews (`python3 tools/build_sih_deck_pdf.py`, needs `reportlab`, `pymupdf`) |

> Keep PPTX and PDF in sync by running **both** scripts after any content edit.
