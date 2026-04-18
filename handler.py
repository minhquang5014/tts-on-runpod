#!/usr/bin/env python3
"""
handler.py — RunPod serverless TTS worker

Language routing (per sentence, auto-detected from Unicode):
  English    → Kokoro-82M  KPipeline lang_code='a'  voice='af_heart'
  Chinese    → Kokoro-82M  KPipeline lang_code='z'  voice='zf_xiaobei'
  Vietnamese → MMS-TTS VITS  facebook/mms-tts-vie

Mixed-language text (e.g. "Let's say 'xin chào' — it means hello!") is split
into sentences, each routed to the correct model, then concatenated. All
segments are RMS-normalised to -20 dBFS before concatenation so volume stays
consistent across models and sample rates.

Input  (job["input"]):
  text          str    required   text to synthesize
  length_scale  float  1.0        >1 = slower speech, <1 = faster (mirrors piper-onnx)

Output:
  { "audio": "<base64 mp3>", "contentType": "audio/mpeg" }
  { "error": "..." }   on failure
"""

import base64
import concurrent.futures
import io
import re
import sys
import time

import numpy as np
import runpod

# ── Language detection (mirrors piper-onnx/synthesize.py) ────────────────────

_VIET_RE = re.compile(
    r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]"
)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef]")


def _word_lang(word: str) -> str:
    """Detect language of a single whitespace-free token."""
    if _CJK_RE.search(word):
        return "zh"
    if _VIET_RE.search(word):
        return "vi"
    return "en"


def segment(text: str) -> list[tuple[str, str]]:
    """
    Word-level language segmentation with context smoothing.

    Each word is classified individually (zh / vi / en).  Short undiacritical
    words that would otherwise be misclassified as English (e.g. Vietnamese
    "nay", "di", "la") are smoothed back to "vi" when both their nearest
    non-English neighbours are also "vi".  Consecutive words of the same
    language are then merged into a single span.

    Example:
      "Hôm nay trời đẹp quá, Tôi muốn go to the beach. Go to beach is 去海滩, có đúng không Sora ơi"
      → [("Hôm nay trời đẹp quá, Tôi muốn", "vi"),
         ("go to the beach. Go to beach is",  "en"),
         ("去海滩,",                             "zh"),
         ("có đúng không Sora ơi",             "vi")]
    """
    words = text.strip().split()
    if not words:
        return []

    langs = [_word_lang(w) for w in words]

    # Smooth isolated "en" tokens that are flanked on both sides by "vi".
    # This absorbs undiacritical Vietnamese words (nay, di, la, co, …) and
    # proper nouns used inside Vietnamese speech (e.g. "Sora ơi").
    smoothed = langs[:]
    for i, lang in enumerate(langs):
        if lang != "en":
            continue
        left  = next((langs[j] for j in range(i - 1, -1, -1)       if langs[j] != "en"), None)
        right = next((langs[j] for j in range(i + 1, len(langs))    if langs[j] != "en"), None)
        if left == "vi" and right == "vi":
            smoothed[i] = "vi"

    # Group consecutive same-language words into spans.
    spans: list[tuple[str, str]] = []
    run_words: list[str] = [words[0]]
    run_lang               = smoothed[0]

    for word, lang in zip(words[1:], smoothed[1:]):
        if lang == run_lang:
            run_words.append(word)
        else:
            spans.append((" ".join(run_words), run_lang))
            run_words = [word]
            run_lang  = lang

    spans.append((" ".join(run_words), run_lang))
    return spans


# ── Parallel model loading ────────────────────────────────────────────────────
# Kokoro (CPU) and VITS (GPU) are independent — load both at the same time to
# cut cold-start from ~60s (sequential) to ~30s (parallel).

from kokoro import KPipeline  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer, VitsModel  # noqa: E402

_KOKORO_VOICES = {
    "en": "af_heart",     # American English — warm female voice
    "zh": "zf_xiaobei",   # Mandarin Chinese female
}


def _load_kokoro() -> dict:
    pipelines = {}
    for code, label in (("a", "English"), ("z", "Chinese")):
        try:
            pipelines[code] = KPipeline(lang_code=code)
            print(f"[tts]   Kokoro '{code}' ({label}) ready", flush=True)
        except Exception as exc:
            print(f"[tts]   Kokoro '{code}' failed: {exc}", flush=True)
    return pipelines


def _load_vits() -> tuple:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tts]   device: {device}", flush=True)
    model     = VitsModel.from_pretrained("facebook/mms-tts-vie").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    sr        = model.config.sampling_rate   # 16 000 Hz
    print(f"[tts]   MMS-TTS Vietnamese ready  sr={sr}", flush=True)
    return model, tokenizer, sr, device


print("[tts] loading all models in parallel…", flush=True)
_t0 = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _pool:
    _kokoro_fut = _pool.submit(_load_kokoro)
    _vits_fut   = _pool.submit(_load_vits)
    KOKORO_PIPELINES                      = _kokoro_fut.result()
    VI_MODEL, VI_TOKENIZER, VI_SR, _DEVICE = _vits_fut.result()
print(f"[tts] all models loaded in {time.perf_counter() - _t0:.1f}s — device={_DEVICE}", flush=True)


# ── Amplitude normalisation ───────────────────────────────────────────────────
# Target RMS: -20 dBFS.  Keeps EN/ZH (Kokoro 24 kHz) and VN (VITS 16 kHz)
# at the same perceived loudness regardless of model output level.

_TARGET_RMS = 10 ** (-20 / 20)   # ≈ 0.1  (linear amplitude)


def _normalise(audio: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:          # silence — don't amplify
        return audio
    return audio * (_TARGET_RMS / rms)


# ── Per-model synthesis ───────────────────────────────────────────────────────

def _synth_kokoro(text: str, lang: str, speed: float) -> tuple[np.ndarray, int]:
    code     = "z" if lang == "zh" else "a"
    voice    = _KOKORO_VOICES.get(lang, "af_heart")
    pipeline = KOKORO_PIPELINES.get(code) or KOKORO_PIPELINES.get("a")
    if pipeline is None:
        raise RuntimeError("Kokoro pipeline not available")

    chunks: list[np.ndarray] = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        if audio is not None:
            chunks.append(audio)

    if not chunks:
        raise RuntimeError("Kokoro produced no audio")

    return np.concatenate(chunks), 24_000


def _synth_vits_vi(text: str, speaking_rate: float) -> tuple[np.ndarray, int]:
    VI_MODEL.config.speaking_rate = speaking_rate
    inputs = VI_TOKENIZER(text, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        waveform = VI_MODEL(**inputs).waveform[0].cpu().numpy()
    return waveform, VI_SR


def _synth_segment(text: str, lang: str, speed: float) -> tuple[np.ndarray, int]:
    if lang == "vi":
        audio, sr = _synth_vits_vi(text, speaking_rate=speed)
    elif lang == "zh":
        audio, sr = _synth_kokoro(text, "zh", speed=speed)
    else:
        audio, sr = _synth_kokoro(text, "en", speed=speed)
    return _normalise(audio), sr


# ── Canonical output sample rate ─────────────────────────────────────────────
# Every segment is resampled to this rate before concatenation and MP3 export,
# regardless of which model produced it.  24 kHz matches Kokoro's native rate
# and is high enough quality for speech delivery on any device.

_OUT_SR = 24_000


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    n_dst       = int(round(len(audio) * dst_sr / src_sr))
    src_indices = np.linspace(0, len(audio) - 1, n_dst)
    return np.interp(src_indices, np.arange(len(audio)), audio)


# ── PCM → base64 MP3 ─────────────────────────────────────────────────────────

def _to_mp3_b64(audio: np.ndarray, sample_rate: int) -> str:
    from pydub import AudioSegment

    pcm = (np.clip(audio, -1.0, 1.0) * 32_767).astype(np.int16)
    seg = AudioSegment(
        data=pcm.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )
    buf = io.BytesIO()
    seg.export(buf, format="mp3", bitrate="128k")
    return base64.b64encode(buf.getvalue()).decode()


# ── RunPod handler ────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    inp          = job.get("input", {})
    text         = inp.get("text", "").strip()
    length_scale = float(inp.get("length_scale", 1.0))

    if not text:
        return {"error": "text is required"}
    if len(text) > 5_000:
        return {"error": "text must be ≤5000 characters"}

    speed    = 1.0 / max(length_scale, 0.25)   # >1 = faster for both engines
    segments = segment(text)

    print(
        f"[tts] synthesize  segments={len(segments)}  "
        f"length_scale={length_scale}  chars={len(text)}",
        flush=True,
    )
    for s, lang in segments:
        print(f"[tts]   [{lang}] {s[:70]}", flush=True)

    try:
        # Synthesize each segment and resample everything to _OUT_SR (24 kHz).
        parts: list[np.ndarray] = []

        for s, lang in segments:
            audio, sr = _synth_segment(s, lang, speed)
            parts.append(_resample(audio, sr, _OUT_SR))

        combined  = np.concatenate(parts) if len(parts) > 1 else parts[0]
        audio_b64 = _to_mp3_b64(combined, _OUT_SR)

        print(f"[tts] done  b64_len={len(audio_b64)}", flush=True)
        return {"audio": audio_b64, "contentType": "audio/mpeg"}

    except Exception as exc:
        print(f"[tts] error: {exc}", flush=True, file=sys.stderr)
        return {"error": str(exc)}


runpod.serverless.start({"handler": handler})
