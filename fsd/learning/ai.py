"""Gemini (vision) and Groq (fast text) clients. Both are optional.

Keys come from .env (GEMINI_API_KEY, GROQ_API_KEY). Model names change often, so
if you don't set GEMINI_MODEL / GROQ_MODEL the client asks the API which models
exist and picks the newest suitable one.

Nothing here ever steers the car. Gemini/Groq only:
  * check the camera angle (small, clamped nudges)
  * diagnose a disengage and propose parameter changes
  * suggest tuning tweaks
Every proposed change must pass replay tests before it is applied.
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time

import cv2
import numpy as np
import requests

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
GROQ_URL = "https://api.groq.com/openai/v1"


def _jpeg_b64(img: np.ndarray, width: int = 768, q: int = 80) -> str:
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {text[:200]}")
    return json.loads(m.group())


class RateGate:
    """Simple per-client limiter so free tiers aren't burned through."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.calls: list[float] = []
        self.lock = threading.Lock()

    def ok(self) -> bool:
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < 60]
            if len(self.calls) >= self.per_minute:
                return False
            self.calls.append(now)
            return True


class Gemini:
    def __init__(self, key: str, model: str | None = None, per_minute: int = 8):
        self.key = key
        self.model = model if model and os.environ.get("GEMINI_MODEL") else None
        self.gate = RateGate(per_minute)
        self.error = ""
        self.ok = bool(key)

    def _pick_model(self) -> str:
        if self.model:
            return self.model
        try:
            r = requests.get(f"{GEMINI_URL}/models", params={"key": self.key, "pageSize": 200}, timeout=15)
            r.raise_for_status()
            names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
                     if "generateContent" in m.get("supportedGenerationMethods", [])]
            flash = [n for n in names if "flash" in n and "image" not in n and "tts" not in n
                     and "audio" not in n and "live" not in n and "lite" not in n]

            def ver(n: str) -> tuple:
                m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
                return (float(m.group(1)) if m else 0.0, "preview" not in n and "exp" not in n, -len(n))
            pick = sorted(flash, key=ver)[-1] if flash else "gemini-2.5-flash"
        except Exception:  # noqa: BLE001
            pick = "gemini-2.5-flash"
        self.model = pick
        return pick

    def ask(self, prompt: str, images: list[np.ndarray] | None = None, timeout: float = 60) -> dict:
        if not self.ok:
            raise RuntimeError("No GEMINI_API_KEY in .env")
        if not self.gate.ok():
            raise RuntimeError("Gemini rate limit (local) - try again in a minute")
        parts: list[dict] = [{"text": prompt}]
        for im in images or []:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": _jpeg_b64(im)}})
        body = {"contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}}
        model = self._pick_model()
        r = requests.post(f"{GEMINI_URL}/models/{model}:generateContent", params={"key": self.key},
                          json=body, timeout=timeout)
        if r.status_code == 404:
            self.model = None
            os.environ.pop("GEMINI_MODEL", None)
            model = self._pick_model()
            r = requests.post(f"{GEMINI_URL}/models/{model}:generateContent", params={"key": self.key},
                              json=body, timeout=timeout)
        if r.status_code != 200:
            self.error = f"Gemini {r.status_code}: {r.text[:200]}"
            raise RuntimeError(self.error)
        data = r.json()
        text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        return _parse_json(text)


class Groq:
    def __init__(self, key: str, model: str | None = None, per_minute: int = 20):
        self.key = key
        self.model = model if model and os.environ.get("GROQ_MODEL") else None
        self.gate = RateGate(per_minute)
        self.ok = bool(key)
        self.error = ""

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}

    def _pick_model(self) -> str:
        if self.model:
            return self.model
        try:
            r = requests.get(f"{GROQ_URL}/models", headers=self._headers(), timeout=15)
            r.raise_for_status()
            ids = [m["id"] for m in r.json().get("data", []) if m.get("active", True)]
            prefs = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-70b-versatile"]
            pick = next((p for p in prefs if p in ids), None)
            if pick is None:
                big = [i for i in ids if re.search(r"(70b|120b|large)", i) and "guard" not in i and "whisper" not in i]
                pick = big[0] if big else (ids[0] if ids else "llama-3.3-70b-versatile")
        except Exception:  # noqa: BLE001
            pick = "llama-3.3-70b-versatile"
        self.model = pick
        return pick

    def ask(self, system: str, user: str, timeout: float = 30) -> dict:
        if not self.ok:
            raise RuntimeError("No GROQ_API_KEY in .env")
        if not self.gate.ok():
            raise RuntimeError("Groq rate limit (local)")
        body = {"model": self._pick_model(), "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        r = requests.post(f"{GROQ_URL}/chat/completions", headers=self._headers(), json=body, timeout=timeout)
        if r.status_code != 200:
            self.error = f"Groq {r.status_code}: {r.text[:200]}"
            raise RuntimeError(self.error)
        return _parse_json(r.json()["choices"][0]["message"]["content"])


CAMERA_PROMPT = """You are checking the camera of a Roblox driving game (ER:LC) for a self-driving system.
The camera should be behind the car, looking straight down the road, with the horizon about 40% from the top,
the road and our lane clearly visible, and traffic lights ahead visible.
Return JSON only: {"road_visible": bool, "lane_visible": bool, "signals_visible": bool,
"yaw_deg": number (-15..15, + = rotate view right), "pitch_deg": number (-10..10, + = tilt view down),
"zoom_steps": integer (-3..3, + = zoom in), "note": short string}.
Use 0 for anything that is already fine."""
