"""Reads numbers off the screen: the speedometer and speed-limit signs.

Runs on its own thread so OCR never slows down steering. Uses RapidOCR
(ONNX, free, offline).

Speedometer: on the first drive it searches the bottom of the screen for a
number next to "MPH"/"KM/H"/"KPH" and remembers where it is, so it works for
gas and electric cars and both units.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque

import numpy as np

KPH_PER_MPH = 1.609344


class OcrEngine:
    def __init__(self):
        self.ok = False
        self.error = ""
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
            self.eng = RapidOCR()
            self.ok = True
        except Exception as e:  # noqa: BLE001
            self.error = f"OCR unavailable: {e}"

    def read(self, img: np.ndarray) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """[(text, (x1,y1,x2,y2), score)]"""
        if not self.ok or img is None or img.size == 0:
            return []
        try:
            res, _ = self.eng(img)
        except Exception:  # noqa: BLE001
            return []
        out = []
        for box, text, score in res or []:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            out.append((str(text), (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))), float(score)))
        return out


def find_speedo(ocr: OcrEngine, frame: np.ndarray) -> tuple[list[float], str] | None:
    """Locate the speedometer. Returns (roi fractions, unit) or None."""
    H, W = frame.shape[:2]
    y0 = int(H * 0.55)
    items = ocr.read(frame[y0:])
    units = [(t, b) for t, b, s in items if re.search(r"(mph|km/?h|kph)", t.lower())]
    nums = [(t, b) for t, b, s in items if re.fullmatch(r"\d{1,3}", t.strip())]
    if not units or not nums:
        return None
    best = None
    for ut, ub in units:
        ucx, ucy = (ub[0] + ub[2]) / 2, (ub[1] + ub[3]) / 2
        for nt, nb in nums:
            ncx, ncy = (nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2
            d = np.hypot(ucx - ncx, ucy - ncy)
            if best is None or d < best[0]:
                best = (d, nb, ub, ut)
    if best is None or best[0] > H * 0.15:
        return None
    _, nb, ub, ut = best
    x1 = min(nb[0], ub[0]) - 10
    y1 = min(nb[1], ub[1]) - 10 + y0
    x2 = max(nb[2], ub[2]) + 10
    y2 = max(nb[3], ub[3]) + 10 + y0
    unit = "mph" if "mph" in ut.lower() else "kph"
    return [max(0, x1) / W, max(0, y1) / H, min(W, x2) / W, min(H, y2) / H], unit


class OcrWorker:
    """Background thread: speedometer ~7x/s, speed-limit signs when candidates appear."""

    def __init__(self, calibration, on_calibrated=None):
        self.cal = calibration
        self.on_calibrated = on_calibrated
        self.engine = OcrEngine()
        self._frame: np.ndarray | None = None
        self._signs: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.speed_kph = 0.0
        self.speed_valid = False
        self.speed_time = 0.0
        self.unit = calibration.speed_unit if calibration.speed_unit != "auto" else "kph"
        self.limit_kph: float | None = None
        self.limit_time = 0.0
        self._recent: deque[float] = deque(maxlen=3)
        self._pending: float | None = None
        self._last_find = 0.0
        self.status = "starting"

    def start(self) -> "OcrWorker":
        threading.Thread(target=self._run, name="ocr", daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def submit(self, frame: np.ndarray, sign_crops: list[np.ndarray] | None = None) -> None:
        with self._lock:
            self._frame = frame
            if sign_crops:
                self._signs = sign_crops[:3]

    def _run(self) -> None:
        if not self.engine.ok:
            self.status = self.engine.error
            return
        while not self._stop.is_set():
            t0 = time.perf_counter()
            with self._lock:
                frame, signs = self._frame, self._signs
                self._signs = []
            if frame is not None:
                self._read_speed(frame)
            for crop in signs:
                self._read_sign(crop)
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, 0.14 - dt))

    def _read_speed(self, frame: np.ndarray) -> None:
        H, W = frame.shape[:2]
        roi = self.cal.speedo_roi
        if roi is None:
            if time.perf_counter() - self._last_find < 1.0:
                return
            self._last_find = time.perf_counter()
            found = find_speedo(self.engine, frame)
            if found is None:
                self.status = "looking for speedometer (start driving)"
                return
            self.cal.speedo_roi, unit = found
            if self.cal.speed_unit == "auto":
                self.unit = unit
            self.status = f"speedometer found ({self.unit})"
            if self.on_calibrated:
                self.on_calibrated()
            roi = self.cal.speedo_roi
        x1, y1, x2, y2 = int(roi[0] * W), int(roi[1] * H), int(roi[2] * W), int(roi[3] * H)
        items = self.engine.read(frame[y1:y2, x1:x2])
        val = None
        for text, _, score in items:
            low = text.lower()
            if self.cal.speed_unit == "auto":
                if "mph" in low:
                    self.unit = "mph"
                elif "km" in low or "kph" in low:
                    self.unit = "kph"
            m = re.search(r"\d{1,3}", text)
            if m and score > 0.5:
                val = float(m.group())
        if val is None:
            if time.perf_counter() - self.speed_time > 1.0:
                self.speed_valid = False
            return
        kph = val * KPH_PER_MPH if self.unit == "mph" else val
        # A single big jump is usually a misread: only accept it if the next read agrees.
        if self.speed_valid and abs(kph - self.speed_kph) > 45:
            if self._pending is None or abs(self._pending - kph) > 10:
                self._pending = kph
                return
            self._recent.clear()
        self._pending = None
        self._recent.append(kph)
        self.speed_kph = float(np.median(self._recent))
        self.speed_valid = True
        self.speed_time = time.perf_counter()

    def _read_sign(self, crop: np.ndarray) -> None:
        from .signs import parse_limit
        items = self.engine.read(crop)
        texts = [t for t, _, s in items if s > 0.5]
        joined = " ".join(texts).upper()
        if texts and ("SPEED" in joined or "LIMIT" in joined or len(texts) <= 2):
            v = parse_limit(texts)
            if v is not None:
                self.limit_kph = v * KPH_PER_MPH if self.cal.sign_unit == "mph" else float(v)
                self.limit_time = time.perf_counter()
