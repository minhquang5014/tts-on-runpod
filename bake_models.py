"""
bake_models.py — run once at Docker build time to download model weights.
Downloads Kokoro EN, Kokoro ZH, and MMS-TTS VN in parallel to cut build time.
Non-fatal: failures are logged but the build succeeds so the image is still
produced. Models that failed to bake will download on the first cold start.
"""
import concurrent.futures
import warnings
warnings.filterwarnings("ignore")

print("=== bake_models.py start (parallel) ===", flush=True)


def bake_kokoro_en():
    from kokoro import KPipeline
    KPipeline(lang_code="a")
    return "Kokoro EN"


def bake_kokoro_zh():
    from kokoro import KPipeline
    KPipeline(lang_code="z")
    return "Kokoro ZH"


def bake_vits_vi():
    from transformers import VitsModel, AutoTokenizer
    VitsModel.from_pretrained("facebook/mms-tts-vie")
    AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    return "MMS-TTS VN"


tasks = {
    "Kokoro EN":  bake_kokoro_en,
    "Kokoro ZH":  bake_kokoro_zh,
    "MMS-TTS VN": bake_vits_vi,
}

errors = []
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    futures = {pool.submit(fn): name for name, fn in tasks.items()}
    for future in concurrent.futures.as_completed(futures):
        name = futures[future]
        try:
            future.result()
            print(f"  [OK] {name}", flush=True)
        except Exception as exc:
            print(f"  [FAILED] {name}: {exc}", flush=True)
            errors.append(f"{name}: {exc}")

if errors:
    print(f"\n=== WARNING: {len(errors)} model(s) not baked — will download on first cold start ===", flush=True)
    for e in errors:
        print(f"  - {e}", flush=True)
else:
    print("\n=== all models baked successfully ===", flush=True)
