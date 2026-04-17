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
import os
import platform
import re
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
REQUEST_TIMEOUT     = 300   # cold start: container boot + 3 model loads ≈ 90–120s

RUN_URL    = "https://api.runpod.ai/v2/{endpoint_id}/run"
STATUS_URL = "https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"


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
    POST to RunPod /run (async), then poll /status until COMPLETED.
    Returns (mp3_bytes, timing_dict).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {"input": {"text": text, "length_scale": length_scale}}

    t0 = time.perf_counter()

    # Submit job
    run_resp = requests.post(
        RUN_URL.format(endpoint_id=endpoint_id),
        json=payload, headers=headers, timeout=30,
    )
    run_resp.raise_for_status()
    job_id = run_resp.json().get("id")
    if not job_id:
        raise RuntimeError(f"No job ID in response: {run_resp.text}")

    print(f"     job={job_id}  polling", end="", flush=True)

    # Poll until done
    status_url = STATUS_URL.format(endpoint_id=endpoint_id, job_id=job_id)
    body = {}
    while True:
        time.sleep(2)
        print(".", end="", flush=True)
        sr = requests.get(status_url, headers=headers, timeout=30)
        sr.raise_for_status()
        body = sr.json()
        status = body.get("status", "")
        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        if (time.perf_counter() - t0) > REQUEST_TIMEOUT:
            raise RuntimeError(f"Timed out after {REQUEST_TIMEOUT}s")

    print()  # newline after dots
    t_done = time.perf_counter()

    if body.get("status") != "COMPLETED":
        raise RuntimeError(f"Job {body.get('status')}: {body.get('error', '')}")

    output = body.get("output", {})
    if "error" in output:
        raise RuntimeError(f"Handler error: {output['error']}")

    audio_b64: str = output.get("audio", "")
    if not audio_b64:
        raise RuntimeError("No audio in RunPod response")

    mp3_bytes    = base64.b64decode(audio_b64)
    execution_ms = body.get("executionTime", 0)

    timing = {
        "total_ms":     (t_done - t0) * 1000,
        "execution_ms": execution_ms,
        "size_kb":      len(mp3_bytes) / 1024,
        "runpod_id":    job_id,
    }
    return mp3_bytes, timing


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RunPod TTS latency tester")
    parser.add_argument("--scale", default=DEFAULT_SCALE, type=float,
                        help="length_scale (1.0=normal, 1.2=slower, 0.8=faster)")
    args = parser.parse_args()

    # Endpoint and key come exclusively from .env / shell environment.
    raw = DEFAULT_ENDPOINT_ID.strip()
    m   = re.search(r"/v2/([^/]+)", raw)
    endpoint_id = m.group(1) if m else raw
    api_key     = DEFAULT_API_KEY.strip()
    scale       = args.scale

    if not endpoint_id:
        print("ERROR: RUNPOD_ENDPOINT_ID not set in .env")
        sys.exit(1)
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set in .env")
        sys.exit(1)

    print(f"\n-- RunPod TTS Latency Test --")
    print(f"   endpoint : {endpoint_id}")
    print(f"   auth     : {'set' if api_key else 'MISSING'}")
    print(f"   scale    : {scale}  (>1 = slower)")
    print(f"   url      : {RUN_URL.format(endpoint_id=endpoint_id)}")

    # Health check — shows endpoint status and worker counts before first request.
    health_url = f"https://api.runpod.ai/v2/{endpoint_id}/health"
    try:
        h = requests.get(health_url,
                         headers={"Authorization": f"Bearer {api_key}"},
                         timeout=10)
        print(f"   health   : HTTP {h.status_code} — {h.text[:200]}")
    except Exception as exc:
        print(f"   health   : ERROR — {exc}")

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
            f"total={t['total_ms']:.0f}ms"
            f"{exec_note}  |  job={t['runpod_id'][:12]}…"
        )
        print(f"[#{request_num}] playing…\n")

        play_mp3_bytes(mp3_bytes)


if __name__ == "__main__":
    main()
