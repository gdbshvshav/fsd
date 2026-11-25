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
    """RapidOCR (fast, ONNX). Falls back to EasyOCR (uses PyTorch) if RapidOCR
    can't be installed, e.g. on a brand-new Python version."""

    def __init__(self):
        self.ok = False
        self.error = ""
        self.kind = ""
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
            self.eng = RapidOCR()
            self.ok, self.kind = True, "rapid"
            return
        except Exception as e:  # noqa: BLE001
            self.error = f"OCR unavailable: {e}"
        try:
            import easyocr  # type: ignore
            import torch
            self.eng = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
            self.ok, self.kind, self.error = True, "easy", ""
        except Exception as e:  # noqa: BLE001
            self.error += f"; EasyOCR unavailable: {e}"

    def _run(self, img: np.ndarray) -> list:
        if self.kind == "rapid":
            res, _ = self.eng(img)
            return res or []
        return [(box, text, score) for box, text, score in self.eng.readtext(img)]

    def read(self, img: np.ndarray) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """[(text, (x1,y1,x2,y2), score)]"""
        if not self.ok or img is None or img.size == 0:
            return []
        try:
            res = self._run(img)
        except Exception:  # noqa: BLE001
            return []
        out = []
        for box, text, score in res:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            out.append((str(text), (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))), float(score)))
        return out


SPEED_RE = re.compile(r"^\s*(\d{1,3})\s*(km\s*/?\s*h|kph|mph)\b", re.I)


def find_speedo(ocr: OcrEngine, frame: np.ndarray) -> tuple[list[float], str] | None:
    """Locate the speedometer. Returns (roi fractions, unit) or None.

    ER:LC draws it as one piece of text like "0 KM/H" or "45 MPH" in a box at the
    bottom-centre; some cars split the number and the unit, so both are handled.
    """
    H, W = frame.shape[:2]
    y0 = int(H * 0.55)
    items = [(t, b, s) for t, b, s in ocr.read(frame[y0:]) if "not read" not in t.lower()]
    cands = []
    for t, b, s in items:                          # "0 KM/H" in one token
        m = SPEED_RE.match(t.replace("KMH", "KM/H"))
        if m:
            cands.append((b, "mph" if "mph" in m.group(2).lower() else "kph"))
    if not cands:                                  # number and unit as separate tokens
        units = [(t, b) for t, b, s in items if re.fullmatch(r"\s*(km\s*/?\s*h|kph|mph)\s*", t, re.I)]
        nums = [(t, b) for t, b, s in items if re.fullmatch(r"\s*\d{1,3}\s*", t)]
        for ut, ub in units:
            near = [(abs((ub[0] + ub[2]) / 2 - (nb[0] + nb[2]) / 2) + abs((ub[1] + ub[3]) / 2 - (nb[1] + nb[3]) / 2), nb)
                    for _, nb in nums]
            if near:
                d, nb = min(near)
                if d < H * 0.12:
                    cands.append(((min(nb[0], ub[0]), min(nb[1], ub[1]), max(nb[2], ub[2]), max(nb[3], ub[3])),
                                  "mph" if "mph" in ut.lower() else "kph"))
    if not cands:
        return None
    # the real speedometer is the one nearest the bottom-centre
    b, unit = min(cands, key=lambda c: abs((c[0][0] + c[0][2]) / 2 - W / 2))
    pad_x, pad_y = (b[2] - b[0]) * 0.6 + 10, (b[3] - b[1]) * 0.5 + 6
    x1, y1 = b[0] - pad_x, b[1] - pad_y + y0
    x2, y2 = b[2] + pad_x, b[3] + pad_y + y0
    return [max(0.0, x1 / W), max(0.0, y1 / H), min(1.0, x2 / W), min(1.0, y2 / H)], unit


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
        self._misses = 0
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
            if "not read" in low:
                continue
            if self.cal.speed_unit == "auto":
                if "mph" in low:
                    self.unit = "mph"
                elif "km" in low or "kph" in low:
                    self.unit = "kph"
            m = SPEED_RE.match(text) or re.fullmatch(r"\s*(\d{1,3})\s*", text)
            if m and score > 0.45:
                val = float(m.group(1))
        if val is None:
            if time.perf_counter() - self.speed_time > 1.0:
                self.speed_valid = False
            self._misses += 1
            if self._misses > 40:               # ~6 s of no reading: the HUD moved, search again
                self.cal.speedo_roi = None
                self._misses = 0
            return
        self._misses = 0
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
