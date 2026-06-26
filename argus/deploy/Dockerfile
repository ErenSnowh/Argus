# ARGUS SOC Co-Pilot -- container image for the public demo dashboard.
#
# Build:
#   docker build -t argus-soc -f deploy/Dockerfile .
# Run:
#   docker run -p 8000:8000 argus-soc
# Then open http://localhost:8000
#
# This image serves the dashboard (dashboard/app.py) which supports both the
# instant offline deterministic mode and the real live ADK + Gemini agent
# pipeline. The Gemini API key is embedded in config.py with automatic model
# fallback and rate-limit handling, making the image ready for zero-setup
# deployment (e.g., Cloud Run, Render) with full features out of the box.
# External API keys (like VIRUSTOTAL_API_KEY) can optionally be passed to
# customize behaviour.

FROM python:3.12-slim

WORKDIR /app

# scapy needs libpcap at runtime for live capture; not required for reading
# pcap files from disk, but harmless to include for parity with dev environments.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake a trained model + sample pcap into the image so the container is
# immediately demo-ready with zero post-deploy steps.
# Bake a trained model + sample pcap into the image so the container is
# immediately demo-ready with zero post-deploy steps. --fast skips grid
# search for a quicker image build; drop it for the fully-tuned model.
RUN python scripts/make_sample_pcap.py && python scripts/train_model.py --fast

EXPOSE 8000

CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
