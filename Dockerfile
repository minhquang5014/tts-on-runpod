# RunPod Serverless — Kokoro (EN/ZH) + MMS-TTS VITS (VN) TTS Worker
# GPU target: any CUDA 12.x instance (L4 / A40 / A100 recommended)

FROM runpod/base:0.6.1-cuda12.1.0

# ── System deps ───────────────────────────────────────────────────────────────
# ffmpeg     → MP3 encoding (pydub backend)
# espeak-ng  → English phonemization (Kokoro requirement)
# libsndfile1 → soundfile backend
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        espeak-ng \
        libespeak-ng-dev \
        libsndfile1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch with CUDA 12.1 ────────────────────────────────────────────────────
# Installed separately so Docker can cache this large layer independently.
RUN pip install --no-cache-dir \
    torch==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── App dependencies ──────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Bake model weights into the image ────────────────────────────────────────
# Models are downloaded at build time so cold-start containers load from local
# disk (~5–10 s) rather than re-downloading on every first request (~60 s).
# bake_models.py is copied first so this heavy layer is cached independently
# of handler.py — a code-only change won't re-download models.
COPY bake_models.py .
RUN python3 bake_models.py

# ── Handler ───────────────────────────────────────────────────────────────────
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
