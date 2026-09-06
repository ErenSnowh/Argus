"""
app.py
-------
FastAPI backend for the ARGUS SOC dashboard. Serves both the offline
deterministic pipeline (instant, no latency) and the live ADK + Gemini
multi-agent pipeline (real LLM-driven agents calling MCP tools).

Enhanced with:
  - Alert queue with persistent investigation history
  - Dashboard stats (total alerts, severity breakdown, avg threat score)
  - Investigation timeline
  - CERT-In compliance report download

    uvicorn dashboard.app:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
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

# ---------------------------------------------------------------------------
# In-memory alert queue (persists for the lifetime of the server process)
# ---------------------------------------------------------------------------
_alert_history: list[dict] = []  # Stores all investigation results


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

    # Store in alert history
    alert_entry = {
        "alert_id": str(uuid.uuid4())[:8],
        "timestamp": time.time(),
        "suspect_ip": suspect_ip,
        "predicted_label": result["triage"]["predicted_label"],
        "confidence": result["triage"]["confidence"],
        "threat_score": result.get("threat_score", {}).get("score") if result.get("threat_score") else None,
        "threat_severity": result.get("threat_score", {}).get("severity") if result.get("threat_score") else None,
        "escalated": result["escalated"],
        "certin_report_id": result.get("certin_report", {}).get("report_id") if result.get("certin_report") else None,
        "campaign_detected": result.get("correlation", {}).get("campaign_detected", False) if result.get("correlation") else False,
        "forecast_risk": result.get("forecast", {}).get("risk_level") if result.get("forecast") else None,
        "forecast_max_prob": result.get("forecast", {}).get("max_infiltration_prob") if result.get("forecast") else None,
        "full_result": result,
    }
    _alert_history.append(alert_entry)

    return result


# ---------------------------------------------------------------------------
# Alert Queue & Timeline APIs
# ---------------------------------------------------------------------------

@app.get("/api/alerts")
def get_alerts(severity: str | None = None, limit: int = 50, offset: int = 0):
    """Paginated alert history with optional severity filter."""
    filtered = _alert_history

    if severity:
        filtered = [a for a in filtered if a.get("threat_severity") == severity.upper()]

    # Sort by timestamp descending (newest first)
    filtered = sorted(filtered, key=lambda a: a["timestamp"], reverse=True)

    total = len(filtered)
    page = filtered[offset: offset + limit]

    # Strip full_result from list view (too large)
    slim = []
    for alert in page:
        slim.append({k: v for k, v in alert.items() if k != "full_result"})

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "alerts": slim,
    }


@app.get("/api/alerts/{alert_id}")
def get_alert_detail(alert_id: str):
    """Get full investigation result for a specific alert."""
    for alert in _alert_history:
        if alert["alert_id"] == alert_id:
            return alert["full_result"]
    raise HTTPException(404, f"Alert {alert_id} not found")


@app.get("/api/alerts/stats/summary")
def get_alert_stats():
    """Dashboard summary statistics."""
    if not _alert_history:
        return {
            "total_investigations": 0,
            "severity_breakdown": {},
            "avg_threat_score": 0,
            "critical_alerts": 0,
            "campaigns_detected": 0,
            "certin_reports_generated": 0,
            "escalation_rate": 0,
        }

    total = len(_alert_history)
    severity_counts: dict[str, int] = {}
    threat_scores = []
    critical_count = 0
    campaign_count = 0
    certin_count = 0
    escalated_count = 0

    for alert in _alert_history:
        sev = alert.get("threat_severity", "UNKNOWN")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

        if alert.get("threat_score") is not None:
            threat_scores.append(alert["threat_score"])
        if sev == "CRITICAL":
            critical_count += 1
        if alert.get("campaign_detected"):
            campaign_count += 1
        if alert.get("certin_report_id"):
            certin_count += 1
        if alert.get("escalated"):
            escalated_count += 1

    return {
        "total_investigations": total,
        "severity_breakdown": severity_counts,
        "avg_threat_score": round(sum(threat_scores) / len(threat_scores), 1) if threat_scores else 0,
        "critical_alerts": critical_count,
        "campaigns_detected": campaign_count,
        "certin_reports_generated": certin_count,
        "escalation_rate": round(escalated_count / total * 100, 1) if total else 0,
    }


@app.get("/api/timeline")
def get_timeline(limit: int = 20):
    """Chronological investigation timeline."""
    sorted_alerts = sorted(_alert_history, key=lambda a: a["timestamp"], reverse=True)[:limit]
    timeline = []
    for alert in sorted_alerts:
        timeline.append({
            "alert_id": alert["alert_id"],
            "timestamp": alert["timestamp"],
            "predicted_label": alert["predicted_label"],
            "confidence": alert["confidence"],
            "threat_score": alert.get("threat_score"),
            "threat_severity": alert.get("threat_severity"),
            "suspect_ip": alert.get("suspect_ip"),
            "escalated": alert["escalated"],
            "campaign_detected": alert.get("campaign_detected", False),
            "certin_report_id": alert.get("certin_report_id"),
        })
    return {"timeline": timeline}


@app.get("/api/certin/{alert_id}")
def get_certin_report(alert_id: str):
    """Download the CERT-In compliance report for a specific alert."""
    for alert in _alert_history:
        if alert["alert_id"] == alert_id:
            certin = alert.get("full_result", {}).get("certin_report")
            if certin:
                return {"report_markdown": certin.get("report_markdown", "")}
            raise HTTPException(404, "No CERT-In report for this alert (benign detection)")
    raise HTTPException(404, f"Alert {alert_id} not found")


# ---------------------------------------------------------------------------
# Live investigation (SSE streaming)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Audit log endpoints
# ---------------------------------------------------------------------------

@app.get("/api/audit/tail")
def audit_tail(n: int = 20):
    if not AUDIT_LOG.path.exists():
        return {"entries": []}
    lines = AUDIT_LOG.path.read_text().splitlines()[-n:]
    return {"entries": [json.loads(line) for line in lines]}


@app.get("/api/audit/verify")
def audit_verify():
    return {"valid": AUDIT_LOG.verify_chain()}


# ---------------------------------------------------------------------------
# World Model endpoints
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    flow: dict | None = None
    k_steps: int = 5


@app.post("/api/forecast")
def forecast(req: ForecastRequest):
    """Run the World Model's K-step forward simulation."""
    if req.flow:
        features = req.flow
    else:
        df = generate_dataset(n_per_class=3, seed=random.randint(0, 999999))
        row = df.sample(1).iloc[0]
        features = {col: float(row[col]) for col in FEATURE_COLUMNS}

    try:
        from ml.world_model.predictor import forecast_infiltration
        result = forecast_infiltration(features, k_steps=req.k_steps)
        return result
    except (FileNotFoundError, ImportError) as e:
        raise HTTPException(503, f"World model not available: {e}")


@app.get("/api/world-model/status")
def world_model_status():
    """Check if the world model is trained and return metadata."""
    try:
        from ml.world_model.predictor import get_predictor
        predictor = get_predictor()
        return predictor.get_model_info()
    except Exception as e:
        return {"model_trained": False, "error": str(e)}


@app.get("/api/benchmark")
def benchmark_results():
    """Return benchmark comparison results (World Model vs Logistic Regression)."""
    benchmark_path = PROJECT_ROOT / "ml" / "artifacts" / "benchmark_results.json"
    if not benchmark_path.exists():
        raise HTTPException(404, "Benchmark not run yet. Run `argus benchmark` first.")
    return json.loads(benchmark_path.read_text())


# ---------------------------------------------------------------------------
# Knowledge Base endpoints
# ---------------------------------------------------------------------------

@app.get("/api/datasets")
def datasets():
    """List all supported public cybersecurity datasets."""
    from ml.world_model.dataset_loader import list_datasets
    return list_datasets()


class KBEnrichRequest(BaseModel):
    label: str
    technique_id: str | None = None


@app.post("/api/kb/enrich")
def kb_enrich(req: KBEnrichRequest):
    """Enrich a detection with ATT&CK/CAPEC/CVE context."""
    from mcp_server.knowledge_base import enrich_detection_with_kb
    return enrich_detection_with_kb(req.label, req.technique_id)


@app.get("/api/kb/status")
def kb_status():
    """Return knowledge base cache status."""
    from mcp_server.knowledge_base import get_kb_status
    return get_kb_status()


@app.get("/api/kb/technique/{technique_id}")
def kb_technique(technique_id: str):
    """Look up a specific ATT&CK technique."""
    from mcp_server.knowledge_base import lookup_attack_technique
    result = lookup_attack_technique(technique_id)
    if result:
        return result
    raise HTTPException(404, f"Technique {technique_id} not found")


@app.get("/api/kb/cve/{cve_id}")
def kb_cve(cve_id: str):
    """Look up a specific CVE from NVD."""
    from mcp_server.knowledge_base import lookup_cve
    result = lookup_cve(cve_id)
    if result:
        return result
    raise HTTPException(404, f"CVE {cve_id} not found")

# ---------------------------------------------------------------------------
# NCIIPC Critical Information Infrastructure (CII) Reporting
# ---------------------------------------------------------------------------

class NCIIPCRequest(BaseModel):
    flow: dict | None = None
    sector: str = "Strategic & Defense Operations"
    suspect_ip: str | None = None
    affected_system: str | None = None
    k_steps: int = 5
    predicted_label: str | None = None
    confidence: float | None = None


@app.post("/api/nciipc/generate")
def generate_nciipc(req: NCIIPCRequest):
    """Generate an official NCIIPC Incident Report for Critical Information Infrastructure (CII)."""
    from mcp_server.nciipc_report import generate_nciipc_report
    from ml.model import FlowClassifier
    from ml.world_model.predictor import forecast_infiltration
    from mcp_server.mitre_mapping import TECHNIQUE_MAP

    flow = req.flow
    if not flow:
        df = generate_dataset(n_per_class=3, seed=random.randint(0, 999999))
        row = df.sample(1).iloc[0]
        flow = {col: float(row[col]) for col in FEATURE_COLUMNS}

    if req.predicted_label is not None and req.confidence is not None:
        predicted_label = req.predicted_label
        confidence = req.confidence
    else:
        try:
            clf = FlowClassifier()
            pred = clf.predict(flow)
            predicted_label = pred["predicted_label"]
            confidence = pred["confidence"]
        except Exception:
            predicted_label = "PortScan"
            confidence = 0.95

    mapping = TECHNIQUE_MAP.get(predicted_label)
    tech_id = mapping.technique_id if mapping else None
    tech_name = mapping.technique_name if mapping else None

    # World model forecast
    forecast = None
    try:
        forecast = forecast_infiltration(flow, k_steps=req.k_steps)
    except Exception:
        pass

    suspect_ips = [req.suspect_ip] if req.suspect_ip else [random.choice(_DEMO_IOCS)]
    affected_ips = [req.affected_system] if req.affected_system else ["10.240.1.15"]

    rep = generate_nciipc_report(
        predicted_label=predicted_label,
        confidence=confidence,
        cii_sector=req.sector,
        forecast=forecast,
        mitre_technique_id=tech_id,
        mitre_technique_name=tech_name,
        suspect_ips=suspect_ips,
        affected_ips=affected_ips,
    )
    return rep.to_dict()


# ---------------------------------------------------------------------------
# Dataset Sampling endpoint
# ---------------------------------------------------------------------------

@app.get("/api/datasets/sample")
def dataset_sample(source: str = "cicids2018", limit: int = 10):
    """Sample raw/mapped telemetry records from any of the 8 supported datasets."""
    from ml.world_model.dataset_loader import (
        get_sample_path,
        load_cicids2017,
        load_cicids2018,
        load_unsw_nb15,
        load_ctu13,
        load_ciciot2023,
        load_lanl_auth,
        load_darpa,
    )
    from ml.world_model.features import WORLD_MODEL_FEATURES

    limit = min(max(1, limit), 50)
    source_lower = source.strip().lower()

    if source_lower == "synthetic":
        df = generate_dataset(n_per_class=3)
        sample_df = df.head(limit)
        return {
            "source": "synthetic",
            "name": "Synthetic Temporal Data",
            "total_sample_rows": len(sample_df),
            "feature_count": len(WORLD_MODEL_FEATURES),
            "label_distribution": sample_df["label"].value_counts().to_dict(),
            "rows": sample_df.to_dict(orient="records"),
        }

    sample_path = get_sample_path(source_lower)
    if not sample_path or not sample_path.exists():
        raise HTTPException(404, f"No sample fixture found for dataset: {source}")

    loader_map = {
        "cicids2018": load_cicids2018,
        "cicids2017": load_cicids2017,
        "unsw_nb15": load_unsw_nb15,
        "ctu13": load_ctu13,
        "ciciot2023": load_ciciot2023,
        "lanl": load_lanl_auth,
        "darpa": load_darpa,
    }

    loader_func = loader_map.get(source_lower)
    if not loader_func:
        raise HTTPException(400, f"Unsupported dataset source: {source}")

    try:
        loaded_df = loader_func(str(sample_path), max_rows=limit)
        sample_df = loaded_df.head(limit)
        records = json.loads(sample_df.to_json(orient="records"))
        labels_dist = {str(k): int(v) for k, v in sample_df["label"].value_counts().items()}
        return {
            "source": source_lower,
            "sample_path": str(sample_path.name),
            "total_sample_rows": len(sample_df),
            "feature_count": len(WORLD_MODEL_FEATURES),
            "label_distribution": labels_dist,
            "rows": records,
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to load dataset sample: {e}")


# ---------------------------------------------------------------------------
# Static files & index
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
    '<path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" '
    'stroke="#5EEAD4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="#134E48"/>'
    '</svg>'
)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


