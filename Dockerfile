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
        libsndfile1 \
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
RUN python - <<'EOF'
import warnings
warnings.filterwarnings("ignore")

# Kokoro EN + ZH
from kokoro import KPipeline
for code, label in (("a", "English"), ("z", "Chinese")):
    try:
        KPipeline(lang_code=code)
        print(f"  Kokoro lang_code={code} ({label}) baked")
    except Exception as exc:
        print(f"  Kokoro lang_code={code} skipped: {exc}")

# MMS-TTS VITS — Vietnamese
from transformers import VitsModel, AutoTokenizer
VitsModel.from_pretrained("facebook/mms-tts-vie")
AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
print("  MMS-TTS Vietnamese baked")

print("all models baked into image")
EOF

# ── Handler ───────────────────────────────────────────────────────────────────
COPY handler.py .

CMD ["python", "-u", "handler.py"]
