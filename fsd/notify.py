"""Discord webhook logging.

Set DISCORD_WEBHOOK in .env. Everything FSD logs goes there:
  * every log line (batched every few seconds so Discord doesn't rate-limit us)
  * engage / disengage / takeovers
  * Gemini's review of each disengage clip, WITH the frames it looked at
  * every settings change: what changed, old -> new, which test it passed or failed
  * errors and crashes (with the traceback)

Sending happens on a background thread and never slows down or breaks driving.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

import requests

BLUE, GREEN, RED, ORANGE, GREY, PURPLE = 0x3884FF, 0x3CD26E, 0xEB4040, 0xF5A623, 0x7A7F8C, 0x9B59FF


class Discord:
    def __init__(self, url: str | None = None):
        self.url = (url if url is not None else os.environ.get("DISCORD_WEBHOOK", "")).strip()
        self.ok = self.url.startswith("https://discord.com/api/webhooks/") or \
            self.url.startswith("https://discordapp.com/api/webhooks/")
        self.q: queue.Queue = queue.Queue(maxsize=500)
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if self.ok:
            threading.Thread(target=self._run, name="discord", daemon=True).start()

    # ------------------------------------------------------------------ public
    def log(self, line: str) -> None:
        """Plain log line; batched."""
        if not self.ok:
            return
        with self._lock:
            self._lines.append(line)

    def embed(self, title: str, description: str = "", color: int = BLUE, fields: list[tuple] | None = None,
              files: list[Path] | None = None, footer: str = "") -> None:
        if not self.ok:
            return
        e = {"title": title[:256], "description": description[:4000], "color": color,
             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"}
        if fields:
            e["fields"] = [{"name": str(n)[:256], "value": (str(v) or "-")[:1024], "inline": bool(inl)}
                           for n, v, inl in fields[:25]]
        if footer:
            e["footer"] = {"text": footer[:2000]}
        if files and Path(files[0]).suffix.lower() in (".jpg", ".jpeg", ".png"):
            e["image"] = {"url": f"attachment://{Path(files[0]).name}"}
        try:
            self.q.put_nowait(("embed", e, [Path(f) for f in (files or [])][:10]))
        except queue.Full:
            pass

    def stop(self) -> None:
        self._flush_lines()
        self._stop.set()

    # ------------------------------------------------------------------ worker
    def _flush_lines(self) -> None:
        with self._lock:
            lines, self._lines = self._lines, []
        if not lines:
            return
        chunk = ""
        for ln in lines:
            ln = ln[:1800]
            if len(chunk) + len(ln) + 1 > 1900:
                self._send({"content": f"```\n{chunk}```"})
                chunk = ""
            chunk += ln + "\n"
        if chunk:
            self._send({"content": f"```\n{chunk}```"})

    def _run(self) -> None:
        last_flush = time.time()
        while not self._stop.is_set():
            try:
                kind, payload, files = self.q.get(timeout=0.5)
                self._send({"embeds": [payload]}, files)
            except queue.Empty:
                pass
            except Exception:  # noqa: BLE001 - logging must never crash anything
                pass
            if time.time() - last_flush > 3.0:
                last_flush = time.time()
                try:
                    self._flush_lines()
                except Exception:  # noqa: BLE001
                    pass

    def _send(self, body: dict, files: list[Path] | None = None) -> None:
        body = {"username": "ER:LC FSD", "allowed_mentions": {"parse": []}, **body}
        for _ in range(4):
            try:
                if files:
                    handles = []
                    multipart = {}
                    for i, f in enumerate(files):
                        if f.exists():
                            h = open(f, "rb")
                            handles.append(h)
                            mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                                    ".zip": "application/zip", ".txt": "text/plain",
                                    ".csv": "text/csv"}.get(f.suffix.lower(), "application/octet-stream")
                            multipart[f"files[{i}]"] = (f.name, h, mime)
                    try:
                        r = requests.post(self.url, data={"payload_json": json.dumps(body)}, files=multipart,
                                          timeout=20)
                    finally:
                        for h in handles:
                            h.close()
                else:
                    r = requests.post(self.url, json=body, timeout=10)
            except requests.RequestException:
                time.sleep(2)
                continue
            if r.status_code == 429:                       # rate limited: wait as told
                try:
                    time.sleep(float(r.json().get("retry_after", 1.0)) + 0.1)
                except ValueError:
                    time.sleep(1.5)
                continue
            if r.status_code in (401, 404):               # webhook deleted / wrong
                self.ok = False
            return


_instance: Discord | None = None


def discord() -> Discord:
    """One shared webhook sender for the whole app."""
    global _instance
    if _instance is None:
        _instance = Discord()
    return _instance
