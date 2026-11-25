"""Keeps the game camera at a good angle: straight down the road, horizon at ~40%.

Local correction uses the lane vanishing point (many times a second, tiny
nudges). Gemini can also look at a frame every few seconds and suggest a
correction (see learning/ai.py); suggestions are clamped and applied gradually.
"""
from __future__ import annotations

import threading
import time

from ..winput import CameraMover, roblox_focused


class CameraAim:
    MOUSE_PER_PX = 0.35        # mouse counts per screen pixel of error (iterative, so rough is fine)

    def __init__(self):
        self.mover = CameraMover()
        self.last = 0.0
        self.pending_ai: tuple[float, float, int] | None = None
        self.status = ""
        self._lock = threading.Lock()
        # Mouse direction vs camera direction differs between setups; learn it:
        # if a nudge made the error worse, flip that axis.
        self.sign_x = 1.0
        self.sign_y = 1.0
        self._last_err: tuple[float, float] | None = None
        self._last_cmd = (0.0, 0.0)

    def suggest(self, yaw_deg: float, pitch_deg: float, zoom: int) -> None:
        with self._lock:
            self.pending_ai = (max(-15, min(15, yaw_deg)), max(-10, min(10, pitch_deg)), max(-3, min(3, zoom)))

    def update(self, w, p, engaged: bool, frame_w: int, frame_h: int, force: bool = False) -> None:
        now = time.perf_counter()
        if not engaged and not force:
            return
        if now - self.last < (0.15 if force else 0.6):
            return
        if not roblox_focused():
            return
        self.last = now
        dx = dy = 0.0
        geo_w, geo_h = frame_w, frame_h
        if w.vanish_x is not None and abs(w.yaw_rate) < 0.08 and w.lanes.conf > 0.5:
            err_x = w.vanish_x - geo_w / 2
            if abs(err_x) > geo_w * 0.03:
                dx = err_x * self.MOUSE_PER_PX * p["camera_yaw_gain"]
            err_y = w.horizon_y - geo_h * p["camera_target_horizon"]
            if abs(err_y) > geo_h * 0.04:
                dy = err_y * self.MOUSE_PER_PX * p["camera_pitch_gain"]
            if self._last_err is not None:
                (lx, ly), (cx, cy) = self._last_err, self._last_cmd
                if cx and abs(err_x) > abs(lx) * 1.15 and abs(err_x) > geo_w * 0.03:
                    self.sign_x *= -1
                if cy and abs(err_y) > abs(ly) * 1.15 and abs(err_y) > geo_h * 0.04:
                    self.sign_y *= -1
            self._last_err = (err_x, err_y)
            dx *= self.sign_x
            dy *= self.sign_y
            self._last_cmd = (dx, dy)
        with self._lock:
            ai, self.pending_ai = self.pending_ai, None
        zoom = 0
        if ai is not None:
            yaw, pitch, zoom = ai
            dx += yaw * 8.0          # ~8 mouse counts per degree at default sensitivity
            dy += pitch * 8.0
        if dx or dy:
            threading.Thread(target=self.mover.rotate, args=(dx, dy), daemon=True).start()
            self.status = f"camera nudged ({dx:+.0f}, {dy:+.0f})"
        if zoom:
            threading.Thread(target=self.mover.zoom, args=(zoom,), daemon=True).start()
