#!/usr/bin/env python3
"""
test_runpod.py — Interactive latency test against the RunPod serverless TTS endpoint.

Sends text to the RunPod /runsync API, receives base64 MP3 audio, and plays
it locally via ffplay. Prints a timing breakdown for each request.

Usage:
    python test_runpod.py
    python test_runpod.py --scale 1.3    # speech speed (1.0 = normal, 1.2 = slower)

Env vars (set in .env or shell):
    RUNPOD_API_KEY      — your RunPod API key
    RUNPOD_ENDPOINT_ID  — serverless endpoint ID (from RunPod dashboard)
"""

import argparse
import base64
import io
import os
import platform
import subprocess
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Load .env ─────────────────────────────────────────────────────────────────

def _load_env(path: str) -> None:
    try:
        for line in open(path, encoding="utf-8"):
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            eq = t.index("=") if "=" in t else -1
            if eq < 0:
                continue
            key = t[:eq].strip()
            val = t[eq + 1:].strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = val
    except FileNotFoundError:
        pass

_load_env(os.path.join(os.path.dirname(__file__), ".env"))

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_API_KEY     = os.environ.get("RUNPOD_API_KEY",     "")
DEFAULT_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
DEFAULT_SCALE       = 1.2
REQUEST_TIMEOUT     = 120   # RunPod sync jobs can take up to 90 s on cold start

RUNSYNC_URL = "https://api.runpod.io/v2/{endpoint_id}/runsync"


# ── Audio playback ────────────────────────────────────────────────────────────

def play_mp3_bytes(data: bytes) -> None:
    system = platform.system()

    if system == "Darwin":
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            subprocess.run(["afplay", tmp_path], check=False)
        finally:
            os.remove(tmp_path)
        return

    # Windows / Linux — pipe to ffplay
    cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
           "-f", "mp3", "-i", "pipe:0"]
    try:
        subprocess.run(cmd, input=data, check=False)
    except FileNotFoundError:
        out_path = os.path.join(os.path.dirname(__file__), "output.mp3")
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"[test] ffplay not found — saved to {out_path}")
        if system == "Windows":
            os.startfile(out_path)


# ── RunPod request ────────────────────────────────────────────────────────────

def request_tts(
    endpoint_id: str,
    api_key: str,
    text: str,
    length_scale: float,
) -> tuple[bytes, dict]:
    """
    POST to RunPod /runsync.
    Returns (mp3_bytes, timing_dict).
    """
    url     = RUNSYNC_URL.format(endpoint_id=endpoint_id)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {"input": {"text": text, "length_scale": length_scale}}

    t0 = time.perf_counter()

    resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)

    t_response = time.perf_counter()
    resp.raise_for_status()

    body = resp.json()
    t_parsed = time.perf_counter()

    # RunPod wraps the handler output in {"status": "COMPLETED", "output": {...}}
    status = body.get("status", "UNKNOWN")
    if status != "COMPLETED":
        error = body.get("error") or body.get("output", {}).get("error") or status
        raise RuntimeError(f"RunPod job status={status}: {error}")

    output = body.get("output", {})
    if "error" in output:
        raise RuntimeError(f"Handler error: {output['error']}")

    audio_b64: str = output.get("audio", "")
    if not audio_b64:
        raise RuntimeError("No audio in RunPod response")

    mp3_bytes = base64.b64decode(audio_b64)
    t_done    = time.perf_counter()

    execution_ms = body.get("executionTime", 0)  # reported by RunPod (handler time only)

    timing = {
        "request_ms":   (t_response - t0)       * 1000,
        "parse_ms":     (t_done - t_response)    * 1000,
        "total_ms":     (t_done - t0)            * 1000,
        "execution_ms": execution_ms,            # GPU inference time inside worker
        "size_kb":      len(mp3_bytes) / 1024,
        "status":       resp.status_code,
        "runpod_id":    body.get("id", ""),
    }
    return mp3_bytes, timing


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RunPod TTS latency tester")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT_ID,
                        help="RunPod endpoint ID (or set RUNPOD_ENDPOINT_ID)")
    parser.add_argument("--key",      default=DEFAULT_API_KEY,
                        help="RunPod API key (or set RUNPOD_API_KEY)")
    parser.add_argument("--scale",    default=DEFAULT_SCALE, type=float,
                        help="length_scale (1.0=normal, 1.2=slower, 0.8=faster)")
    args = parser.parse_args()

    endpoint_id = args.endpoint.strip()
    api_key     = args.key.strip()
    scale       = args.scale

    if not endpoint_id:
        print("ERROR: provide endpoint ID via --endpoint or RUNPOD_ENDPOINT_ID env var.")
        sys.exit(1)
    if not api_key:
        print("ERROR: provide API key via --key or RUNPOD_API_KEY env var.")
        sys.exit(1)

    print(f"\n-- RunPod TTS Latency Test --")
    print(f"   endpoint : {endpoint_id}")
    print(f"   auth     : {'set' if api_key else 'MISSING'}")
    print(f"   scale    : {scale}  (>1 = slower)")
    print(f"   url      : {RUNSYNC_URL.format(endpoint_id=endpoint_id)}")
    print("   Type 'q' or Ctrl-C to quit.\n")

    request_num = 0
    while True:
        try:
            text = input("Text → ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        if not text or text.lower() == "q":
            break

        request_num += 1
        print(f"[#{request_num}] sending…")

        try:
            mp3_bytes, t = request_tts(endpoint_id, api_key, text, scale)
        except requests.exceptions.Timeout:
            print(f"[#{request_num}] TIMEOUT after {REQUEST_TIMEOUT}s\n")
            continue
        except requests.exceptions.HTTPError as exc:
            print(f"[#{request_num}] HTTP {exc.response.status_code}: "
                  f"{exc.response.text[:300]}\n")
            continue
        except Exception as exc:
            print(f"[#{request_num}] ERROR: {exc}\n")
            continue

        exec_note = (f"  gpu_inference={t['execution_ms']}ms" if t['execution_ms'] else "")
        print(
            f"[#{request_num}] OK  {t['size_kb']:.0f} KB  |  "
            f"total={t['total_ms']:.0f}ms  "
            f"(request={t['request_ms']:.0f}ms  decode={t['parse_ms']:.0f}ms)"
            f"{exec_note}  |  job={t['runpod_id'][:12]}…"
        )
        print(f"[#{request_num}] playing…\n")

        play_mp3_bytes(mp3_bytes)


if __name__ == "__main__":
    main()
