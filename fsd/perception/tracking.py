"""Object tracking, ego motion (yaw rate) and emergency-light detection."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ..world import Obj


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Tracker:
    """Keeps IDs on objects across frames and estimates their closing speed.

    Objects with a ground position (cars, obstacles, people) are matched by position
    (gated by how far they can plausibly move), others (signs, lights) by box overlap.
    Closing speed vx (m/s, negative = getting closer) is the slope of a least-squares
    line through the last ~0.8 s of positions - much steadier than frame-to-frame
    differences, which made a parked car look like it was driving away.
    """

    def __init__(self, max_age: float = 0.5, window_s: float = 0.8):
        self.tracks: dict[int, dict] = {}
        self.next_id = 1
        self.max_age = max_age
        self.window_s = window_s

    @staticmethod
    def _positional(ob: Obj) -> bool:
        return ob.x < 900 and ob.kind in ("car", "person", "cone", "emergency", "obstacle")

    def _match(self, ob: Obj, used: set) -> int | None:
        best, best_s = None, 0.0
        for tid, tr in self.tracks.items():
            if tid in used or tr["kind"] != ob.kind:
                continue
            if self._positional(ob) and tr["x"] < 900:
                gx = 3.0 + 0.15 * ob.x
                dx, dy = abs(ob.x - tr["x"]), abs(ob.y - tr["y"])
                if dx > gx or dy > 1.8:
                    continue
                sc = 1.0 / (1.0 + dx / gx + dy / 1.8)
            else:
                sc = iou(ob.box, tr["box"])
                if sc <= 0.25:
                    continue
            if sc > best_s:
                best, best_s = tid, sc
        return best

    def update(self, objs: list[Obj], t: float) -> list[Obj]:
        used = set()
        for ob in sorted(objs, key=lambda o: o.x):
            best = self._match(ob, used)
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {"kind": ob.kind, "box": ob.box, "x": ob.x, "y": ob.y, "t": t, "vx": 0.0,
                                     "n": 0, "hist": deque(maxlen=60)}
            tr = self.tracks[best]
            if ob.x < 900:
                tr["hist"].append((t, ob.x))
                while tr["hist"] and t - tr["hist"][0][0] > self.window_s:
                    tr["hist"].popleft()
                if len(tr["hist"]) >= 3:
                    ts = np.array([h[0] for h in tr["hist"]])
                    xs = np.array([h[1] for h in tr["hist"]])
                    if ts[-1] - ts[0] > 0.15:
                        slope = float(np.polyfit(ts - ts[0], xs, 1)[0])
                        tr["vx"] = float(np.clip(slope, -60, 60))
            tr.update(box=ob.box, x=ob.x, y=ob.y, t=t)
            tr["n"] += 1
            ob.track_id = best
            ob.vx = tr["vx"]
            ob.extra["track_n"] = tr["n"]
            used.add(best)
        for tid in [k for k, v in self.tracks.items() if t - v["t"] > self.max_age]:
            del self.tracks[tid]
        return objs


class EgoMotion:
    """Yaw rate from how far the horizon band slides sideways between frames."""

    def __init__(self, focal_px: float):
        self.focal = focal_px
        self.prev: np.ndarray | None = None
        self.prev_t = 0.0
        self.yaw_rate = 0.0
        self._win = None

    def update(self, frame: np.ndarray, horizon_y: float, t: float) -> float:
        H, W = frame.shape[:2]
        y0 = int(max(0, horizon_y - H * 0.12))
        y1 = int(min(H, horizon_y + H * 0.03))
        band = cv2.cvtColor(frame[y0:y1, int(W * 0.05):int(W * 0.95)], cv2.COLOR_BGR2GRAY)
        scale = 0.25
        band = cv2.resize(band, None, fx=scale, fy=scale).astype(np.float32)
        if self._win is None or self._win.shape != band.shape:
            self._win = cv2.createHanningWindow(band.shape[::-1], cv2.CV_32F)
        if self.prev is not None and self.prev.shape == band.shape:
            dt = t - self.prev_t
            if 0.005 < dt < 0.3:
                (dx, _), resp = cv2.phaseCorrelate(self.prev, band, self._win)
                if resp > 0.05:
                    # scene moves LEFT on screen when we turn RIGHT
                    rate = -(dx / scale) / self.focal / dt
                    self.yaw_rate = 0.6 * self.yaw_rate + 0.4 * float(np.clip(rate, -3, 3))
        self.prev, self.prev_t = band, t
        return self.yaw_rate


class EmergencyDetector:
    """Police/fire/EMS lights: bright red AND blue that alternate within ~1 s."""

    def __init__(self):
        self.hist: deque[tuple[float, float, float]] = deque(maxlen=90)

    def update(self, frame: np.ndarray, cars: list[Obj], t: float) -> bool:
        H, W = frame.shape[:2]
        red = blue = 0.0
        regions = [c.box for c in cars if c.x < 80] or []
        for (x1, y1, x2, y2) in regions:
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            bright = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] > 140)
            red += float((bright & ((hsv[:, :, 0] < 8) | (hsv[:, :, 0] > 170))).mean())
            blue += float((bright & (hsv[:, :, 0] > 100) & (hsv[:, :, 0] < 130)).mean())
        self.hist.append((t, red, blue))
        recent = [(r, b) for tt, r, b in self.hist if t - tt < 1.2]
        if len(recent) < 6:
            return False
        rs = np.array([r for r, _ in recent])
        bs = np.array([b for _, b in recent])
        return bool(rs.max() > 0.004 and bs.max() > 0.004 and (rs.std() > 0.002 or bs.std() > 0.002))
