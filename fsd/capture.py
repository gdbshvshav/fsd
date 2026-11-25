"""Screen capture on its own thread. Always hands out the newest frame (BGR).

Uses dxcam (DXGI desktop duplication, fast on Windows) and falls back to mss.
A file/folder source is supported for replays and tests.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np


class Capture:
    def __init__(self, monitor: int = 0, fps: int = 60):
        self.monitor = monitor
        self.fps = fps
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.backend = "none"
        self.error = ""

    def start(self) -> "Capture":
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> tuple[np.ndarray | None, float, int]:
        with self._lock:
            return self._frame, self._stamp, self._seq

    def _publish(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._stamp = time.perf_counter()
            self._seq += 1

    def _run(self) -> None:
        try:
            self._run_dxcam()
            return
        except Exception as e:  # noqa: BLE001 - fall back on any dxcam problem
            self.error = f"dxcam unavailable ({e}); using mss"
        try:
            self._run_mss()
        except Exception as e:  # noqa: BLE001
            self.error = f"capture failed: {e}"

    def _run_dxcam(self) -> None:
        import dxcam  # type: ignore

        cam = dxcam.create(output_idx=self.monitor, output_color="BGR")
        if cam is None:
            raise RuntimeError("dxcam.create returned None")
        cam.start(target_fps=self.fps, video_mode=True)
        self.backend = "dxcam"
        try:
            while not self._stop.is_set():
                f = cam.get_latest_frame()
                if f is not None:
                    self._publish(np.ascontiguousarray(f))
        finally:
            cam.stop()

    def _run_mss(self) -> None:
        import mss  # type: ignore

        self.backend = "mss"
        period = 1.0 / self.fps
        with mss.mss() as sct:
            mon = sct.monitors[self.monitor + 1] if len(sct.monitors) > self.monitor + 1 else sct.monitors[1]
            while not self._stop.is_set():
                t0 = time.perf_counter()
                shot = np.asarray(sct.grab(mon))[:, :, :3]
                self._publish(np.ascontiguousarray(shot))
                dt = time.perf_counter() - t0
                if dt < period:
                    time.sleep(period - dt)


class FileSource:
    """Plays images from a folder (or one image) as if they were the screen."""

    def __init__(self, path: str | Path, fps: float = 10.0, loop: bool = True):
        p = Path(path)
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.files = sorted(f for f in (p.iterdir() if p.is_dir() else [p]) if f.suffix.lower() in exts)
        self.fps = fps
        self.loop = loop
        self.i = 0
        self._t0 = time.perf_counter()
        self.backend = "file"
        self.error = "" if self.files else f"no images in {p}"
        self._cache: dict[int, np.ndarray] = {}

    def start(self) -> "FileSource":
        return self

    def stop(self) -> None:
        pass

    def latest(self):
        if not self.files:
            return None, 0.0, 0
        tick = int((time.perf_counter() - self._t0) * self.fps)
        idx = tick % len(self.files) if self.loop else min(tick, len(self.files) - 1)
        if idx not in self._cache:
            img = cv2.imread(str(self.files[idx]))
            if img is not None and img.shape[1] != 1920:
                img = cv2.resize(img, (1920, int(img.shape[0] * 1920 / img.shape[1])))
            self._cache = {idx: img}
        return self._cache[idx], time.perf_counter(), tick + 1
