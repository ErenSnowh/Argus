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
# pipeline. The offline mode needs no credentials; live mode reads
# GOOGLE_API_KEY from the environment (pass it with -e or via Render/Cloud
# Run secrets). Model fallback and rate-limit handling are built into the
# client wrapper. External API keys (like VIRUSTOTAL_API_KEY) can optionally
# be passed to customize behaviour.

FROM python:3.12-slim

WORKDIR /app

# scapy needs libpcap at runtime for live capture; not required for reading
# pcap files from disk, but harmless to include for parity with dev environments.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only PyTorch first, from the dedicated index, so the Linux image
# ships the ~200MB CPU wheel instead of the ~2GB CUDA build. This satisfies the
# torch>=2.2 pin in requirements.txt, which pip then won't re-resolve. With torch
# present the container serves the committed neural checkpoint (engine="neural"),
# not the heuristic fallback.
RUN pip install --no-cache-dir "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the RF model + sample pcap into the image so the container is
# immediately demo-ready with zero post-deploy steps. --fast skips grid
# search for a quicker image build; drop it for the fully-tuned model.
# (The neural world-model checkpoint ships via COPY . . above — see
# ml/pretrained/world_model.pt.)
RUN python scripts/make_sample_pcap.py && python scripts/train_model.py --fast

EXPOSE 8000

CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
