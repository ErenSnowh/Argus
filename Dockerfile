# ARGUS SOC Co-Pilot — Docker image for Render (and any OCI-compatible host).
#
# Build:
#   docker build -t argus-soc .
# Run:
#   docker run -p 8000:8000 -e GOOGLE_API_KEY=your_key argus-soc
# Then open http://localhost:8000
#
# This image serves the ARGUS dashboard (argus/dashboard/app.py) which
# supports both the instant offline deterministic mode and the real live
# ADK + Gemini multi-agent pipeline. The container is demo-ready out of
# the box — a trained model and sample PCAP are baked in during build.

FROM python:3.12-slim

WORKDIR /app

# scapy needs libpcap for reading PCAP files
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY argus/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full argus package into /app/argus
COPY argus/ argus/

# Set PYTHONPATH so all internal imports (config, agents, ml, etc.) resolve
ENV PYTHONPATH=/app/argus

# Bake a trained model + sample PCAP into the image for zero-setup demo
RUN python argus/scripts/make_sample_pcap.py && \
    python argus/scripts/train_model.py --fast

EXPOSE 8000

# Launch the FastAPI dashboard from the argus package
CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/argus"]
