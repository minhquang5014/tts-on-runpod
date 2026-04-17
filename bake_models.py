"""
bake_models.py — run once at Docker build time to download model weights.
Non-fatal: failures are logged but the build succeeds so the image is still
produced. Models that failed to bake will download on the first request.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

print("=== bake_models.py start ===", flush=True)

errors = []

# Kokoro EN
print("\n[1/3] Kokoro English (lang_code=a)...", flush=True)
try:
    from kokoro import KPipeline
    KPipeline(lang_code="a")
    print("      OK", flush=True)
except Exception as exc:
    print(f"      FAILED: {exc}", flush=True)
    errors.append(f"Kokoro EN: {exc}")

# Kokoro ZH
print("[2/3] Kokoro Chinese (lang_code=z)...", flush=True)
try:
    from kokoro import KPipeline
    KPipeline(lang_code="z")
    print("      OK", flush=True)
except Exception as exc:
    print(f"      FAILED: {exc}", flush=True)
    errors.append(f"Kokoro ZH: {exc}")

# MMS-TTS VITS Vietnamese
print("[3/3] MMS-TTS VITS Vietnamese (facebook/mms-tts-vie)...", flush=True)
try:
    from transformers import VitsModel, AutoTokenizer
    VitsModel.from_pretrained("facebook/mms-tts-vie")
    AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    print("      OK", flush=True)
except Exception as exc:
    print(f"      FAILED: {exc}", flush=True)
    errors.append(f"MMS-TTS VN: {exc}")

if errors:
    print(f"\n=== WARNING: {len(errors)} model(s) not baked — will download on first cold start ===", flush=True)
    for e in errors:
        print(f"  - {e}", flush=True)
else:
    print("\n=== all models baked successfully ===", flush=True)
