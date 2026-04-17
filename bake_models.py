"""
bake_models.py — run once at Docker build time to download model weights.
Errors are printed verbosely so build logs show exactly what failed.
"""
import sys
import warnings
warnings.filterwarnings("ignore")

print("=== bake_models.py start ===", flush=True)

# Kokoro EN + ZH
print("\n[1/3] Kokoro English (lang_code=a)...", flush=True)
try:
    from kokoro import KPipeline
    KPipeline(lang_code="a")
    print("      OK", flush=True)
except Exception as exc:
    print(f"      FAILED: {exc}", flush=True)
    sys.exit(1)

print("[2/3] Kokoro Chinese (lang_code=z)...", flush=True)
try:
    from kokoro import KPipeline
    KPipeline(lang_code="z")
    print("      OK", flush=True)
except Exception as exc:
    print(f"      FAILED: {exc}", flush=True)
    sys.exit(1)

# MMS-TTS VITS Vietnamese
print("[3/3] MMS-TTS VITS Vietnamese (facebook/mms-tts-vie)...", flush=True)
try:
    from transformers import VitsModel, AutoTokenizer
    VitsModel.from_pretrained("facebook/mms-tts-vie")
    AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    print("      OK", flush=True)
except Exception as exc:
    print(f"      FAILED: {exc}", flush=True)
    sys.exit(1)

print("\n=== all models baked ===", flush=True)
