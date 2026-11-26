"""Night, dusk, fog: brighten the picture for the lane / road finders.

ER:LC has a day-night cycle and weather. At night the asphalt is near-black and the
lane paint dim. The paint finder's thresholds are tuned for daylight, so the lines just
vanish and FSD would ask for a takeover every night in-game.

The fix is a person's eyes adapting. We measure how bright the road just past our car
is (median of a patch, ignoring paint) and scale the whole picture so that the road
always looks like daytime road (~target grey). Hue stays the same (yellow stays
yellow), because every channel gets the same gain. A small lift with a gamma curve helps
the darkest parts most, the way headlight-lit road looks.

Only the lane, road and obstacle finders get the brightened picture. Traffic light
lamps are found on the original (at night they are the brightest things on screen
anyway, and brightening would make street lamps look like signals).
"""
from __future__ import annotations

import cv2
import numpy as np


class AutoExposure:
    def __init__(self, target: float = 105.0, max_gain: float = 3.5):
        self.target = target
        self.max_gain = max_gain
        self.gain = 1.0
        self.level = None          # measured road brightness (EMA)
        self._lut = None
        self._lut_gain = None

    @property
    def night(self) -> bool:
        return self.gain > 1.6

    def measure(self, frame: np.ndarray, roi: tuple[int, int, int, int]) -> float | None:
        x1, y1, x2, y2 = roi
        H, W = frame.shape[:2]
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x2 - x1 < 8 or y2 - y1 < 4:
            return None
        patch = frame[y1:y2, x1:x2]
        g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        vals = g.ravel()
        # ignore paint and lamps (the brightest 15%) so a lane line doesn't fool it
        cut = np.percentile(vals, 85)
        road = vals[vals <= cut]
        if road.size < 20:
            return None
        return float(np.median(road))

    def update(self, frame: np.ndarray, roi) -> float:
        m = self.measure(frame, roi)
        if m is None:
            return self.gain
        self.level = m if self.level is None else 0.9 * self.level + 0.1 * m
        want = float(np.clip(self.target / max(self.level, 4.0), 1.0, self.max_gain))
        # only brighten (daylight pictures are left alone); adapt smoothly like an eye
        self.gain += (want - self.gain) * 0.15
        if self.gain < 1.08:
            self.gain = 1.0
        return self.gain

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.gain <= 1.0:
            return frame
        g = round(self.gain, 2)
        if self._lut is None or self._lut_gain != g:
            x = np.arange(256, dtype=np.float32) / 255.0
            # gain plus a gentle gamma lift for the darkest tones, clipped
            y = np.clip(x * g * (1.0 + 0.25 * (1.0 - x)), 0, 1)
            self._lut = (y * 255).astype(np.uint8)
            self._lut_gain = g
        return cv2.LUT(frame, self._lut)


def road_roi(geo, hidden_m: float, W: int, H: int) -> tuple[int, int, int, int]:
    """Screen box of the road ~2-10 m past our own car, lane wide."""
    try:
        d0 = max(4.0, hidden_m + 2.0)
        pts = np.array([[d0, -1.5], [d0, 1.5], [d0 + 8.0, -1.5], [d0 + 8.0, 1.5]], np.float32)
        s = geo.vehicle_to_screen(pts)
        x1, x2 = int(s[:, 0].min()), int(s[:, 0].max())
        y1, y2 = int(s[:, 1].min()), int(s[:, 1].max())
        return x1, y1, x2, y2
    except Exception:  # noqa: BLE001
        return int(W * 0.4), int(H * 0.7), int(W * 0.6), int(H * 0.9)
