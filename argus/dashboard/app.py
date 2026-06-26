"""
app.py
-------
FastAPI backend for the ARGUS dashboard. Serves both the offline deterministic
pipeline (instant, no latency) and the live ADK + Gemini multi-agent pipeline
(real LLM-driven agents calling MCP tools against the Gemini API).

The API key is auto-loaded from config.py — no setup needed. Anyone who
clones this repo can run both modes immediately:

    uvicorn dashboard.app:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402, F401  -- auto-sets GOOGLE_API_KEY

from agents.run_offline_demo import run_offline_incident  # noqa: E402
from ml.flow_features import FEATURE_COLUMNS, generate_dataset  # noqa: E402
from mcp_server.mitre_mapping import TECHNIQUE_MAP  # noqa: E402
from security.guardrails import AuditLogger  # noqa: E402

app = FastAPI(title="ARGUS SOC Co-Pilot API")

METRICS_PATH = PROJECT_ROOT / "ml" / "artifacts" / "metrics.json"
SAMPLE_PCAP = PROJECT_ROOT / "data" / "sample_portscan.pcap"
AUDIT_LOG = AuditLogger(PROJECT_ROOT / "data" / "audit_log.jsonl")

_DEMO_IOCS = ["185.220.101.45", "45.137.21.9", "8.8.8.8", "203.0.113.77"]


class InvestigateRequest(BaseModel):
    flow: dict | None = None
    suspect_ip: str | None = None
    use_sample_pcap: bool = True


@app.get("/api/health")
def health():
    return {"status": "ok", "model_trained": METRICS_PATH.exists()}


@app.get("/api/config")
def get_config():
    """Tell the frontend whether a Gemini API key is available for live mode."""
    return {"has_api_key": config.has_api_key()}


@app.get("/api/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "Model not trained yet -- run `python scripts/train_model.py`.")
    return json.loads(METRICS_PATH.read_text())


@app.get("/api/attack-map")
def attack_map():
    return {label: m.__dict__ for label, m in TECHNIQUE_MAP.items()}


@app.post("/api/investigate")
def investigate(req: InvestigateRequest):
    if req.flow:
        features = req.flow
    else:
        df = generate_dataset(n_per_class=3, seed=random.randint(0, 999999))
        row = df.sample(1).iloc[0]
        features = {col: float(row[col]) for col in FEATURE_COLUMNS}

    suspect_ip = req.suspect_ip or random.choice(_DEMO_IOCS)
    pcap_path = str(SAMPLE_PCAP) if req.use_sample_pcap and SAMPLE_PCAP.exists() else None

    result = run_offline_incident(features, suspect_ip=suspect_ip, pcap_path=pcap_path)
    result["input_features"] = features
    return result


@app.post("/api/investigate/live")
async def investigate_live(req: InvestigateRequest):
    """Run the real ADK + Gemini multi-agent pipeline and stream results
    via Server-Sent Events (SSE). Each event is a JSON object with
    {author, text?, tool_call?}."""

    # Build the incident description from flow features
    if req.flow:
        features = req.flow
    else:
        df = generate_dataset(n_per_class=3, seed=random.randint(0, 999999))
        row = df.sample(1).iloc[0]
        features = {col: float(row[col]) for col in FEATURE_COLUMNS}

    suspect_ip = req.suspect_ip or random.choice(_DEMO_IOCS)
    pcap_path = str(SAMPLE_PCAP) if req.use_sample_pcap and SAMPLE_PCAP.exists() else None

    description = _build_incident_description(features, suspect_ip, pcap_path)

    async def event_stream():
        from agents.run_live import run_live_incident
        try:
            async for event in run_live_incident(description):
                yield f"data: {json.dumps(event)}\n\n"
            yield f"data: {json.dumps({'author': 'system', 'text': '[DONE]'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'author': 'system', 'text': f'Error: {e}'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _build_incident_description(features: dict, suspect_ip: str | None, pcap_path: str | None) -> str:
    """Build a natural-language incident description from structured features."""
    parts = [
        "Investigate this network flow. Here are the statistical features:",
        ", ".join(f"{k}={v}" for k, v in features.items()) + ".",
    ]
    if suspect_ip:
        parts.append(f"Suspect source IP to enrich: {suspect_ip}.")
    if pcap_path:
        parts.append(f"PCAP file available at: {pcap_path}.")
    return " ".join(parts)


@app.get("/api/audit/tail")
def audit_tail(n: int = 20):
    if not AUDIT_LOG.path.exists():
        return {"entries": []}
    lines = AUDIT_LOG.path.read_text().splitlines()[-n:]
    return {"entries": [json.loads(line) for line in lines]}


@app.get("/api/audit/verify")
def audit_verify():
    return {"valid": AUDIT_LOG.verify_chain()}


STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

