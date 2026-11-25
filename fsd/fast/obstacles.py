"""Something in my lane? Found from "that isn't road" - no AI, every frame.

The patch of road right in front of your bumper is road by definition. Its
colour is learned continuously (separately for sunlit and shadowed road, since
Roblox shadows tint the road blue). Walking up your lane, row by row, we measure
how much of the lane is NOT road-coloured and NOT paint. A car, a person, a cone
or a barrier fills a block of rows like that and also has edges (bodywork,
tyres, tail lights) - a change of road surface is smooth and doesn't.

To avoid phantom braking, a block must persist for a few frames before it's
reported, and it's reported as kind "obstacle": the planner slows for it and
only stops for it when it's close or when the YOLO detector (if available)
agrees.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ..world import Lanes, Obj


class RoadModel:
    """Running colour model of the road surface (Lab), sun and shadow."""

    def __init__(self):
        self.sun = None
        self.shade = None

    def update(self, lab: np.ndarray, shadow: np.ndarray) -> None:
        for name, sel in (("sun", ~shadow), ("shade", shadow)):
            px = lab[sel]
            if len(px) < 40:
                continue
            mean = px.mean(axis=0)
            std = px.std(axis=0) + np.array([6.0, 3.0, 3.0])
            cur = getattr(self, name)
            if cur is None:
                setattr(self, name, (mean, std))
            else:
                m0, s0 = cur
                setattr(self, name, (0.9 * m0 + 0.1 * mean, 0.9 * s0 + 0.1 * std))

    def is_road(self, lab: np.ndarray) -> np.ndarray:
        out = np.zeros(lab.shape[:2], bool)
        for cur in (self.sun, self.shade):
            if cur is None:
                continue
            m, s = cur
            d = np.abs(lab.astype(np.float32) - m) / (3.0 * s)
            out |= (d < 1.0).all(axis=-1)
        return out


class ObstacleFinder:
    def __init__(self, persist: int = 3):
        self.road = RoadModel()
        self.hist: deque = deque(maxlen=6)
        self.persist = persist
        self.debug_rows: list = []

    def find(self, frame: np.ndarray, lanes: Lanes, rows, paint_wide: np.ndarray,
             ego_mask: np.ndarray | None, lane_width: float = 3.7) -> list[Obj]:
        """rows: ScreenLaneFinder (for its per-row distance tables)."""
        H, W = frame.shape[:2]
        if lanes.conf <= 0.15:
            self.hist.append(None)
            return []
        a, b, c = lanes.center
        hw = max(1.2, min(lanes.width_m, 5.0) / 2 - 0.3)
        # rows from the bumper outward, ~every 0.5 m
        ys = []
        last_d = -1.0
        for y in range(H - 1, int(rows.geo.horizon_y) + 3, -1):
            if not rows.row_valid[y]:
                continue
            d = float(rows.row_dist[y])
            if d < 0.8:
                continue
            if d > 60:
                break
            if d - last_d >= 0.5:
                ys.append((y, d))
                last_d = d
        if len(ys) < 5:
            return []
        # learn the road colour from 2-9 m ahead in our lane (just past the car's roof on screen)
        ys_arr = np.array([y for y, _ in ys])
        ds = np.array([d for _, d in ys])
        near = ys_arr[(ds >= 2.0) & (ds <= 9.0)]
        if near.size:
            ya, yb = int(near.min()), int(near.max()) + 1
            dmid = float(ds[(ds >= 2.0) & (ds <= 9.0)].mean())
            cy = a * dmid * dmid + b * dmid + c
            r = int((ya + yb) // 2)
            u0 = int(rows.row_c0[r] + (cy - hw * 0.7) * rows.row_ppm[r])
            u1 = int(rows.row_c0[r] + (cy + hw * 0.7) * rows.row_ppm[r])
            u0, u1 = max(0, u0), min(W, u1)
            if u1 - u0 > 4:
                patch = frame[ya:yb, u0:u1]
                pm = paint_wide[ya:yb, u0:u1] == 0
                if ego_mask is not None:
                    pm &= ego_mask[ya:yb, u0:u1] == 0
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                shadow = (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 2] < 125)
                lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
                self.road.update(lab[pm], shadow[pm])
        if self.road.sun is None and self.road.shade is None:
            return []
        # walk up the lane
        blocked = []
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for y, d in ys:
            cy = a * d * d + b * d + c
            u0 = int(rows.row_c0[y] + (cy - hw) * rows.row_ppm[y])
            u1 = int(rows.row_c0[y] + (cy + hw) * rows.row_ppm[y])
            u0, u1 = max(0, u0), min(W, u1)
            if u1 - u0 < 4:
                blocked.append((d, 0.0, 0.0, u0, u1, y))
                continue
            seg = frame[y:y + 1, u0:u1]
            lab = cv2.cvtColor(seg, cv2.COLOR_BGR2LAB)
            road = self.road.is_road(lab)[0]
            ok = paint_wide[y, u0:u1] == 0
            if ego_mask is not None:
                ok &= ego_mask[y, u0:u1] == 0
            frac = float((~road & ok).sum()) / max(1, ok.sum()) if ok.sum() > 3 else 0.0
            edges = float(np.abs(np.diff(grey[y, u0:u1].astype(np.int16))).mean()) if u1 - u0 > 2 else 0.0
            blocked.append((d, frac, edges, u0, u1, y))
        self.debug_rows = blocked
        # first block of >= 3 consecutive rows mostly not-road, with some edges
        found = None
        run = 0
        for i, (d, frac, edges, u0, u1, y) in enumerate(blocked):
            if frac > 0.45 and edges > 2.0:
                run += 1
                if run >= 3:
                    found = blocked[i - 2]
                    break
            else:
                run = 0
        self.hist.append(found[0] if found else None)
        recent = [h for h in self.hist if h is not None]
        if found is None or len(recent) < self.persist:
            return []
        if max(recent[-self.persist:]) - min(recent[-self.persist:]) > 6:
            return []                        # jumping around: noise
        d, frac, edges, u0, u1, y = found
        cy = a * d * d + b * d + c
        # screen box: from the found row up to ~1.6 m high
        top = int(max(rows.geo.horizon_y, y - 1.6 * rows.row_ppm[y]))
        box = (int(u0), top, int(u1), int(y))
        return [Obj("obstacle", box, float(min(0.9, 0.4 + frac * 0.5)), x=float(d), y=float(cy),
                    extra={"half_width": hw, "source": "free-space"})]
