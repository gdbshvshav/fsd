"""Always-on 30 second recorder + the training-data recorder.

Ring buffer: ~10 frames/s (640x360 JPEG) plus a telemetry row for every
control tick. On a disengage it keeps recording 3 more seconds (so your
correction is captured), then saves a scenario folder:

    data/scenarios/2026-11-25_14-03-22/
        frames/0000.jpg ...
        log.jsonl        one line per control tick
        meta.json        reason, your note, params version, diagnosis, tags
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCEN_DIR = ROOT / "data" / "scenarios"
TRAIN_DIR = ROOT / "data" / "train" / "raw"


class Recorder:
    def __init__(self, seconds: float = 30.0, fps: float = 10.0, after_s: float = 3.0):
        self.seconds = seconds
        self.frame_period = 1.0 / fps
        self.after_s = after_s
        self.frames: deque[tuple[float, bytes]] = deque()
        self.rows: deque[dict] = deque()
        self._last_frame_t = 0.0
        self._save_at: float | None = None
        self._save_meta: dict | None = None
        self._lock = threading.Lock()
        self.last_saved: Path | None = None
        self.on_saved = None

    def add(self, t: float, frame: np.ndarray | None, row: dict) -> None:
        with self._lock:
            row = dict(row)
            row["t"] = t
            self.rows.append(row)
            if frame is not None and t - self._last_frame_t >= self.frame_period:
                small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self.frames.append((t, buf.tobytes()))
                    self._last_frame_t = t
            horizon = t - self.seconds - self.after_s
            while self.frames and self.frames[0][0] < horizon:
                self.frames.popleft()
            while self.rows and self.rows[0]["t"] < horizon:
                self.rows.popleft()
            due = self._save_at is not None and t >= self._save_at
        if due:
            self._flush()

    def trigger(self, t: float, reason: str, meta: dict) -> None:
        """Call on disengage. Saves after `after_s` more seconds of recording."""
        with self._lock:
            if self._save_at is None:
                self._save_at = t + self.after_s
                self._save_meta = {"reason": reason, "disengage_t": t, **meta}

    def _flush(self) -> None:
        with self._lock:
            frames = list(self.frames)
            rows = list(self.rows)
            meta = self._save_meta or {}
            self._save_at = None
            self._save_meta = None
        threading.Thread(target=self._write, args=(frames, rows, meta), daemon=True).start()

    def _write(self, frames, rows, meta) -> None:
        name = time.strftime("%Y-%m-%d_%H-%M-%S")
        d = SCEN_DIR / name
        (d / "frames").mkdir(parents=True, exist_ok=True)
        t0 = frames[0][0] if frames else (rows[0]["t"] if rows else 0.0)
        index = []
        for i, (t, data) in enumerate(frames):
            (d / "frames" / f"{i:04d}.jpg").write_bytes(data)
            index.append(round(t - t0, 3))
        with open(d / "log.jsonl", "w") as f:
            for r in rows:
                r = dict(r)
                r["t"] = round(r["t"] - t0, 3)
                f.write(json.dumps(r) + "\n")
        meta = dict(meta)
        meta["disengage_t"] = round(meta.get("disengage_t", t0) - t0, 3)
        meta["frame_times"] = index
        meta.setdefault("note", "")
        meta.setdefault("tags", [])
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        self.last_saved = d
        if self.on_saved:
            self.on_saved(d)


def load_scenario(d: Path) -> tuple[list[tuple[float, Path]], list[dict], dict]:
    meta = json.loads((d / "meta.json").read_text())
    rows = [json.loads(line) for line in (d / "log.jsonl").read_text().splitlines() if line.strip()]
    times = meta.get("frame_times", [])
    frames = sorted((d / "frames").glob("*.jpg"))
    pairs = [(times[i] if i < len(times) else i * 0.1, f) for i, f in enumerate(frames)]
    return pairs, rows, meta


def update_meta(d: Path, **changes) -> None:
    f = d / "meta.json"
    meta = json.loads(f.read_text())
    meta.update(changes)
    f.write_text(json.dumps(meta, indent=2))


def list_scenarios() -> list[Path]:
    if not SCEN_DIR.exists():
        return []
    return sorted([p for p in SCEN_DIR.iterdir() if (p / "meta.json").exists()])


class TrainingRecorder:
    """F2: saves full-size frames every 0.5 s while you drive, for training the detector."""

    def __init__(self, period: float = 0.5):
        self.on = False
        self.period = period
        self.last = 0.0
        self.count = 0
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    def toggle(self) -> bool:
        self.on = not self.on
        return self.on

    def add(self, t: float, frame: np.ndarray, speed_kph: float) -> None:
        if not self.on or frame is None or t - self.last < self.period or speed_kph < 3:
            return
        self.last = t
        name = TRAIN_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{int(t * 1000) % 1000:03d}.jpg"
        threading.Thread(target=cv2.imwrite, args=(str(name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]),
                         daemon=True).start()
        self.count += 1
