# Deploying ARGUS

ARGUS runs with **zero setup** for the offline deterministic pipeline; the live
**Agent (Gemini)** pipeline reads `GOOGLE_API_KEY` from the environment (or a
git-ignored `.env` file) — no key is stored in the repository. A self-healing
model fallback chain and rate-limit retry logic are built directly into the
client wrapper.

This means that whether you run locally, via Docker, or deploy to Google Cloud Run, the **offline deterministic pipeline** works immediately, and the real **Live Agent (Gemini) pipeline** starts as soon as you provide `GOOGLE_API_KEY`.

---

## Deployment Options

### Option A: Local Docker

To build and run the container locally:
```bash
docker build -t argus-soc -f deploy/Dockerfile .
docker run -p 8000:8000 argus-soc
```
Open [http://localhost:8000](http://localhost:8000) in your browser. You can toggle between **Offline Pipeline** and **Live Agent (Gemini)**, and run investigations.

---

### Option B: Docker Compose

For a single-command setup that also demonstrates the MCP service topology:
```bash
docker compose -f deploy/docker-compose.yml up --build
```
This maps the dashboard port to `8000` and configures the environment.

---

### Option C: Google Cloud Run

To deploy to Google Cloud Run:
```bash
# From the repository root
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/argus-soc -f deploy/Dockerfile .

gcloud run deploy argus-soc \
  --image gcr.io/YOUR_PROJECT_ID/argus-soc \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000
```

Cloud Run will print a public HTTPS URL. This is your live public dashboard URL. Visitors can toggle between the instant offline view and the real live agent reasoning stream.

---

## Local Development & Usage

If you prefer running directly on your machine without containerization:

### 1. Installation
Install the package in editable mode:
```bash
pip install -e ".[dev]"
```

### 2. Generate PCAP & Train Model
Prepare the local resources (these are also baked into the Docker image automatically):
```bash
python scripts/make_sample_pcap.py
python scripts/train_model.py
```

### 3. Run the CLI
- **Offline / deterministic investigation** (instant, local):
  ```bash
  python cli/argus_cli.py investigate --random
  ```
- **Live multi-agent investigation** (runs real Gemini agents calling MCP tools):
  ```bash
  python cli/argus_cli.py investigate --random --live
  ```

### 4. Run the Dashboard Backend
Start the local web server:
```bash
uvicorn dashboard.app:app --reload --port 8000
```
Open [http://localhost:8000](http://localhost:8000).

---

## Configuring the API Keys
- **Gemini (required for live mode)**: Set the `GOOGLE_API_KEY` environment variable or add it to a local `.env` file (which is git-ignored).
- **VirusTotal (IOC Enrichment)**: Export `VIRUSTOTAL_API_KEY` to enable live reputation checks (otherwise, the tool falls back to a mock local database).

