# ARGUS — 5-Minute Submission Video Script

Target runtime: **4:30–5:00**. Record screen + voiceover; webcam intro/outro
optional but recommended (judges respond well to a face + name up front).

Suggested tools: Loom / OBS for screen capture, the ARGUS dashboard
(`uvicorn dashboard.app:app`) for the live demo segment, and **Google
Antigravity** (Google's agentic IDE) for the "Build" segment — open the repo
in Antigravity and narrate over it live to satisfy the video's "Antigravity"
key-concept requirement honestly, rather than just mentioning the name.

---

## 0:00 – 0:25 — Hook + who you are (0:25)

**Visual:** Webcam or title card with your name/role.

**Say:**
> "SOC analysts spend most of their day doing the same five steps for every
> alert: is it real, what's the IP's reputation, what does the packet
> evidence show, what do we do, and who do we tell. I'm Vinayak — I've built
> intrusion detection models and forensics tooling for this exact workflow by
> hand. This is ARGUS: I turned that workflow into an agent swarm."

## 0:25 – 1:10 — Problem statement (0:45)

**Visual:** Simple slide or just talk to camera; maybe a quick stat overlay
("alert fatigue" framing).

**Say:**
> "Alert fatigue is one of the most-cited reasons SOC analysts burn out — not
> because the work is hard, but because most of it is repetitive: classify,
> enrich, investigate, decide, write it up. That loop, done by hand, doesn't
> scale, and it's exactly where breach response gets slow. ARGUS targets that
> loop specifically — not 'replace the analyst,' but compress the mechanical
> 80% so a human's time goes to the judgment calls that actually need a
> human."

## 1:10 – 2:00 — Why agents (0:50)

**Visual:** Show the architecture diagram from `docs/ARCHITECTURE.md` (the
Mermaid render) on screen.

**Say:**
> "A classifier alone gives you a label and a number. It can't decide what
> that label means, go check an IP's reputation, pull packet evidence, and
> write a report — that's five different jobs needing five different tools
> and five different levels of trust. So ARGUS is five narrow agents, not one
> big prompt: triage classifies and maps to MITRE ATT&CK; enrichment and
> forensics run in parallel as independent evidence streams; remediation
> proposes a playbook — and only proposes, it can't execute anything; report
> synthesizes everything with zero tool access of its own."

**Visual cue while talking:** point at each box in the diagram as you name it.

## 2:00 – 2:30 — Architecture & the build (0:30)

**Visual:** Switch to **Antigravity** with the repo open. Show the file tree:
`agents/`, `mcp_server/`, `security/`, `ml/`. Briefly open
`agents/orchestrator.py` and `mcp_server/server.py`.

**Say:**
> "Built with Google's Agent Development Kit — these are real ADK
> `LlmAgent`s composed with `SequentialAgent` and `ParallelAgent`. Every agent
> talks to a custom MCP server I built with the official MCP SDK, exposing
> six tools: flow classification, ATT&CK lookup, IOC enrichment, file-hash
> verification, PCAP forensics, and playbook generation. I verified this
> server against a real MCP client over stdio, not just unit tests — it
> actually speaks the protocol."

(Use this segment to **actually interact with Antigravity** — e.g. ask it to
explain a function, or make a trivial tweak live — so the video genuinely
demonstrates the tool rather than just naming it.)

## 2:30 – 3:45 — Live demo (1:15)

**Visual:** Switch to the running ARGUS dashboard
(`http://localhost:8000` or your deployed Cloud Run URL). Click **Run
Investigation**.

**Say (as it loads):**
> "This is the public demo — no API key needed, it runs the same MCP tool
> layer the real agents use, with a deterministic orchestrator standing in
> for the LLM for instant, zero-cost reproducibility."

**Say (as results render):**
> "Triage agent classified this as PortScan with high confidence — you can
> see the full probability breakdown. That maps to MITRE ATT&CK T1595.001,
> Active Scanning. Enrichment agent checked the source IP against threat
> intel — flagged as a known malicious Tor exit / C2 relay. Forensics agent
> pulled a structured summary straight from the PCAP: 200 packets, 200
> unique destination ports in 16 milliseconds — a textbook scan signature.
> Remediation agent proposed a playbook — and notice every single step says
> 'requires human approval.' ARGUS doesn't have execute permission on any
> firewall or EDR system, by design. And report agent synthesized all of that
> into one incident report."

**Visual:** Scroll to the audit log panel, click **verify chain integrity**.

**Say:**
> "Every one of those tool calls is in this hash-chained audit log — if
> anyone tampers with it after the fact, this verification breaks instantly."

## 3:45 – 4:15 — Agent Skills CLI (0:30)

**Visual:** Terminal. Run `argus attack BruteForce`, then
`argus investigate --random --pcap data/sample_portscan.pcap --ip 185.220.101.45`.

**Say:**
> "Every stage is also a standalone CLI skill — an analyst or a SOAR playbook
> can call exactly the capability they need without spinning up the full
> agent session."

## 4:15 – 4:45 — Security & deployability close (0:30)

**Visual:** Briefly show `security/guardrails.py` (the `TOOL_ALLOWLISTS`
dict) and `deploy/Dockerfile`.

**Say:**
> "Security wasn't bolted on — least-privilege tool access is enforced twice,
> once at the agent layer and once at the MCP server; everything's redacted
> and audit-logged; and remediation never gets execute permission, anywhere.
> It's containerized and deploys to Cloud Run with one command — that's the
> public link in this submission."

## 4:45 – 5:00 — Close (0:15)

**Visual:** Back to webcam / title card.

**Say:**
> "ARGUS is the workflow I already do by hand, rebuilt as an auditable,
> narrow-permission agent swarm. Thanks for watching — code's on GitHub,
> demo's live at the link below."

---

## Shot list / asset checklist before recording

- [ ] Dashboard running locally or on Cloud Run, model trained
      (`python scripts/train_model.py`), sample PCAP generated
      (`python scripts/make_sample_pcap.py`)
- [ ] `docs/ARCHITECTURE.md` diagram open in a Markdown previewer or rendered
      as an image for the "why agents" segment
- [ ] Antigravity open with the repo loaded for the build segment
- [ ] Terminal font size bumped up for readability on a recording
- [ ] One full successful "Run Investigation" click rehearsed so the demo
      timing is predictable
- [ ] Upload to YouTube as **Public** (required) and attach to the Kaggle
      Writeup's Media Gallery along with at least one cover image / screenshot
