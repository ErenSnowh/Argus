# ARGUS — Autonomous Response & Guarded Unified Security
### A multi-agent SOC co-pilot that triages, investigates, and reports network intrusions — so analysts stop drowning in alerts and start making decisions.

**Track:** Agents for Business *(crossover relevance to Agents for Good — protecting shared digital infrastructure)*

---

## The problem

Every time a SOC analyst gets an alert, the same loop runs: is this real? What's
the reputation of the IP/domain/hash involved? What does the underlying packet
evidence actually show? What should we do about it? Who needs to know? Industry
research consistently flags **alert fatigue** as one of the top drivers of SOC
analyst burnout and turnover — and the irony is that most of that loop is
mechanical. The detection itself is often the easy part; the enrichment,
correlation, and write-up is where the hours go.

This is not a hypothetical problem I picked for a hackathon. As an AI/ML
security engineer, I've built an intrusion-detection classifier on the
CICIDS2017 dataset, a forensics-automation toolkit for CTF and incident
response work, and a Cobalt Strike C2 traffic analyzer that uses MITRE ATT&CK
mapping and VirusTotal enrichment. Each of those was a standalone script I ran
by hand, in sequence, on a single incident. ARGUS is the realization that
those tools were always secretly *one pipeline* — they just needed an agent
swarm to run them as one.

**Business case for the track:** a single SOC analyst-hour costs real money,
and a missed or slow-triaged incident costs far more. Compressing triage time
from "however long it takes a human to run five tools by hand" to "one
automated pass that a human reviews and approves" is a direct, measurable cost
reduction — which is exactly the kind of cost-or-revenue-on-the-line problem
the Agents for Business track is asking for.

## Why agents — not just a bigger model

A classifier gives you a label and a confidence number. It cannot decide what
that label *means*, go check whether the source IP has a reputation, pull the
packet evidence, weigh whether to escalate, draft a remediation plan, and
write an incident report — that's five distinct judgment calls, several of
which can run independently and in parallel, each requiring a different tool
and a different amount of trust. That's precisely the shape a multi-agent
system is built for, and precisely the shape a single prompt or single model
handles badly: cram all of that into one agent and you either give it
everything (a huge, unauditable attack surface) or it does a mediocre job of
all five things at once.

So ARGUS is five narrow agents, not one broad one:

1. **`triage_agent`** — classifies the incoming flow with a Random Forest
   detector and maps the result to MITRE ATT&CK. Decides whether to escalate.
2. **`enrichment_agent`** and **`forensics_agent`** — run **in parallel** as
   independent evidence streams. Enrichment doesn't need PCAP results, and
   forensics doesn't need IOC reputation, so there's no reason to force them
   sequential.
3. **`remediation_agent`** — proposes a playbook. It has a `propose_playbook`
   tool and *no* execute tool, anywhere, by design.
4. **`report_agent`** — pure synthesis. Zero tool access, on purpose: a step
   that only reads upstream results and writes prose has no legitimate reason
   to touch the network.

## Architecture

```
Incoming flow / PCAP / IOC
        │
        ▼
  triage_agent  (classify_flow → lookup_mitre_attack)
        │
        ▼
  ParallelAgent ── enrichment_agent (enrich_ioc, verify_file_hash)
        │      └── forensics_agent  (analyze_pcap_summary, verify_file_hash)
        ▼
  remediation_agent  (propose_playbook — propose only, never executes)
        │
        ▼
  report_agent  (pure synthesis, zero tools)
        │
        ▼
  Markdown incident report
```

All four tool-using agents are real `google.adk.agents.llm_agent.LlmAgent`
instances, composed with ADK's native `SequentialAgent` and `ParallelAgent`
workflow primitives — `SequentialAgent([triage, ParallelAgent([enrichment,
forensics]), remediation, report])`. Each agent connects to a custom **MCP
server** (`mcp_server/server.py`, built with the official `mcp` Python SDK's
`FastMCP`) over stdio, exposing six tools: `classify_flow`,
`lookup_mitre_attack`, `enrich_ioc`, `verify_file_hash`,
`analyze_pcap_summary`, and `propose_playbook`. I verified the server against
a real MCP client (not just unit tests of the underlying functions) to confirm
it speaks the actual protocol ADK and other MCP clients expect.

Full diagrams and a security-control table live in the repo's
`docs/ARCHITECTURE.md`.

## Detection layer

The Random Forest classifier is trained on a synthetic, CICIDS2017-shaped
dataset (`ml/flow_features.py`): 24 statistical flow features — duration,
packet/byte counts, inter-arrival-time statistics, TCP flag counts,
destination-port fan-out — across six classes (BENIGN, DDoS, PortScan,
BruteForce, WebAttack, Botnet). I generate this synthetically rather than
downloading the multi-gigabyte real dataset so the whole pipeline runs
end-to-end with zero external dependencies, but I deliberately injected
feature jitter and label noise into the generator: a first pass came out at a
suspicious 100% accuracy, which would have meant the synthetic classes were
too cleanly separable to teach anything real. With realistic noise added, the
tuned model (5-fold grid search over `n_estimators`, `max_depth`,
`min_samples_leaf`) lands at **~94-95% accuracy and macro-F1**, consistent
with the 94.2% I get on the real CICIDS2017 data in my standalone IDS project.
Swapping in the real dataset only requires changing the data source — the
feature schema and the rest of the pipeline don't change.

## MITRE ATT&CK mapping

Rather than calling the live MITRE CTI API, `mcp_server/mitre_mapping.py` is a
small, deterministic, offline table mapping each detection label to a real
ATT&CK Enterprise technique with a plain-language description and recommended
controls — e.g. PortScan → **T1595.001** (Active Scanning), Botnet → **T1071**
(Application Layer Protocol / C2, the same pattern I look for when analyzing
Cobalt Strike beaconing). This keeps the demo reliable offline and gives
judges consistent technique IDs on every run; a production deployment would
point this at the live STIX bundle instead.

## Security features (the part I care about most)

A security tool that isn't secure-by-design is a liability, not a product —
so I treated ARGUS's own security as a first-class requirement, not an
afterthought:

- **Least-privilege tool allowlists, enforced twice.** Every agent role has an
  explicit allowlist (`security/guardrails.TOOL_ALLOWLISTS`) enforced both at
  the ADK `McpToolset` layer (the agent's tool registry physically doesn't
  contain out-of-scope tools) and inside the MCP server itself — defense in
  depth, so a misconfiguration in one layer doesn't silently grant access.
- **Secret/PII redaction** before anything is logged or returned —
  `security/guardrails.redact()` strips AWS keys, JWTs, API tokens, and emails
  from anything that touches the audit trail.
- **Prompt-injection resistance.** Untrusted, externally-sourced text
  (filenames, payload strings, third-party intel responses) is wrapped and
  flagged before it reaches an LLM context, and every agent's instructions
  explicitly tell it to treat tagged content as data, never as commands.
- **Hash-chained, append-only audit log.** Every tool call — across both the
  live agent pipeline and the offline demo — is written to a JSONL log where
  each entry's hash depends on the previous entry's hash. `argus audit verify`
  recomputes the chain and will flag tampering.
- **No autonomous execution, anywhere.** `remediation_agent` can propose a
  playbook; it cannot call a firewall, EDR, or any other system. Every
  proposed action is explicitly marked as requiring human approval. This was
  a deliberate trade-off: an agent that can independently block traffic or
  isolate a host is a bigger blast-radius risk than most of the incidents
  ARGUS is meant to catch.

## Deployability

`deploy/Dockerfile` bakes a trained model and a sample PCAP into the image and
serves the FastAPI dashboard on container start — `docker build` +
`docker run` gets you a working demo with zero post-deploy steps.
`deploy/docker-compose.yml` wraps the same thing for one-command local dev,
and `deploy/README.md` walks through a Cloud Run deployment
(`gcloud builds submit` → `gcloud run deploy`) that produces a public HTTPS
URL with no environment variables required, since the public dashboard runs
the deterministic offline pipeline rather than calling Gemini directly. The
live ADK + ADK Gemini pipeline (`agents/run_live.py`) is meant to run locally
or from your own server with `GOOGLE_API_KEY` injected as a secret — never
committed.

## Agent Skills CLI

Every pipeline stage is also independently invocable as a discrete "skill"
through the `argus` CLI (`cli/argus_cli.py`, installed via
`pip install -e .`): `argus train`, `argus detect`, `argus enrich`,
`argus forensics`, `argus attack`, `argus playbook`, `argus investigate`
(full one-shot pipeline), and `argus audit verify`. This means an analyst, a
cron job, or a SOAR playbook can call exactly the capability it needs without
spinning up the full agent session — the same way a CLI exposes discrete
"skills" on top of a larger system.

## Two run modes, one tool layer

I built ARGUS with two execution paths on purpose:

- **`agents/run_offline_demo.py`** (powering the public dashboard) runs the
  exact same pipeline shape through deterministic Python control flow calling
  the same `mcp_server/*.py` functions — no API key, no cost, no latency
  variance. This is what makes the public demo link usable by anyone,
  instantly.
- **`agents/run_live.py`** runs the real ADK `SequentialAgent`/`ParallelAgent`
  swarm against Gemini, with agents actually deciding what to call, in what
  order, and writing freeform analyst prose.

Both call the *same* underlying tool implementations, so the offline demo
isn't a separate mock system standing in for the real one — it's the
production tool layer with a rule-based orchestrator in place of the
LLM-driven one. Anything a judge sees on the dashboard is representative of
what the live agents do, minus the agents' own freeform reasoning about when
to escalate.

## Results

- Random Forest detector: **~94-95% accuracy, ~95% macro-F1** on a held-out
  20% split of the synthetic (intentionally noisy) dataset.
- 18 automated tests (`pytest -q`), all offline, covering detection, ATT&CK
  mapping, IOC enrichment, PCAP forensics, playbook generation, and every
  security control — including a test that actively tampers with the audit
  log and confirms `verify_chain()` catches it.
- A real MCP server verified against an actual MCP client over stdio (not
  just unit-tested as plain functions), confirming protocol compliance, not
  just internal correctness.
- A working public dashboard (dark-mode, single-page, zero dependencies on
  third-party JS frameworks) that runs a full investigation — detection →
  ATT&CK mapping → enrichment → forensics → playbook → report — in one click,
  with a live, hash-chain-verifiable audit trail underneath it.

## Limitations and future work

The detection model trains on synthetic, statistically-noised data rather
than the full real CICIDS2017 CSVs — the repo is built so swapping in the real
dataset is a one-line data-source change, but I haven't validated against it
directly in this submission. The MITRE ATT&CK mapping is a small, hand-curated
table (five techniques) rather than the full ATT&CK matrix — extending it is
mechanical, but I scoped it to what the six detection classes actually need.
The live IOC enrichment falls back to a local cache without a VirusTotal key;
with one set, it calls the real API. Given more time, the natural next steps
are: real CICIDS2017 validation, a feedback loop where analyst corrections
retrain the detector, and a second remediation tier that *drafts* (but still
never executes) firewall/EDR API calls for a human to approve with one click
rather than copy-pasting from the proposed playbook.

## Closing

ARGUS isn't a demo problem invented for a hackathon — it's the workflow I
already do by hand, rebuilt as a swarm of narrow, auditable agents with
security treated as a feature, not an afterthought. I think that's what a SOC
co-pilot should look like: not a black box that takes action on your network
without asking, but a fast, transparent, narrow-permission team of specialists
that hands a human exactly what they need to make the call.
