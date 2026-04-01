# ── Seed-VC RunPod Serverless Worker ──────────────────────────────
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# ── System dependencies ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Clone Seed-VC ────────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/Plachtaa/seed-vc.git /app/seed-vc

# ── Install Python dependencies (torch already in base image) ────
RUN pip install --no-cache-dir \
    runpod \
    boto3 \
    requests \
    scipy==1.13.1 \
    librosa==0.10.2 \
    "huggingface-hub>=0.28.1" \
    munch==4.0.0 \
    einops==0.8.0 \
    descript-audio-codec==1.0.0 \
    pydub==0.25.1 \
    resemblyzer \
    jiwer==3.0.3 \
    transformers==4.46.3 \
    soundfile==0.12.1 \
    numpy==1.26.4 \
    hydra-core==1.3.2 \
    pyyaml \
    accelerate

# ── NOTE: Model weights will auto-download on first run ──────────
# Seed-VC inference.py handles downloading checkpoints from HuggingFace
# automatically. This avoids build failures due to network issues.
# First cold start will be slower (~2-3 min extra), subsequent runs
# use RunPod's cached image.

# ── Copy handler ─────────────────────────────────────────────────
COPY handler.py /app/handler.py

# ── Entry point ──────────────────────────────────────────────────
CMD ["python", "-u", "/app/handler.py"]
