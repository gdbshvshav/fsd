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

    def is_road(self, lab: np.ndarray, hsv: np.ndarray | None = None) -> np.ndarray:
        out = np.zeros(lab.shape[:2], bool)
        if hsv is not None:
            # Roblox shadows are a dark, saturated-ish blue: whatever they fall on is
            # almost always road here, and they must never look like a car
            out |= (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 2] < 125) & (hsv[:, :, 1] > 60)
        for cur in (self.sun, self.shade):
            if cur is None:
                continue
            m, s = cur
            lo = np.clip(np.floor(m - 3.0 * s), 0, 255).astype(np.uint8)
            hi = np.clip(np.ceil(m + 3.0 * s), 0, 255).astype(np.uint8)
            out |= cv2.inRange(lab, lo, hi) > 0          # same test, done by OpenCV (fast)
        return out


class ObstacleFinder:
    def __init__(self, persist: int = 3):
        self.road = RoadModel()
        self.hist: deque = deque(maxlen=6)
        self.persist = persist
        self.debug_rows: list = []
        self.clear_to = 0.0          # lane positively seen clear up to this distance
        self.hidden_to = 0.0         # the first metres hidden behind our own car on screen
        self._seen_any = False
        self._chain = True

    def find(self, frame: np.ndarray, lanes: Lanes, rows, paint_wide: np.ndarray,
             ego_mask: np.ndarray | None, lane_width: float = 3.7, edges=None) -> list[Obj]:
        """rows: ScreenLaneFinder (for its per-row distance tables)."""
        H, W = frame.shape[:2]
        self.clear_to = 0.0
        self.hidden_to = 0.0
        self._seen_any = False
        self._chain = True
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
            reach = max(50.0, lanes.x_max + 20.0) if abs(lanes.curvature) < 0.004 else max(14.0, lanes.x_max + 4.0)
            if d > 60 or d > reach:
                # past where we see our lane well: the lane is extrapolated a little further so a
                # stopped car 30-50 m ahead (which itself HIDES the lines behind it) is found in time
                break
            if d > lanes.x_max + 4.0:
                # ...but only while the extrapolated lane is still ON the road: where it runs off
                # the road surface (a bend we can't see yet) the "obstacle" would be the verge
                if edges is not None and getattr(edges, "ok", False) and d <= edges.x_max + 5.0:
                    yl, yr = edges.at(d)
                    yc = a * d * d + b * d + c
                    if not (yl + 0.6 < yc < yr - 0.6):
                        break
                # where the road edges themselves curve away, the road bends: don't guess further
                if edges is not None and getattr(edges, "ok", False) and \
                        abs(edges.right[0]) + abs(edges.left[0]) > 0.004:
                    break
            if d - last_d >= 0.5:
                ys.append((y, d))
                last_d = d
        if len(ys) < 5:
            return []
        # learn the road colour just past our own car on screen: the nearest rows (from 2 m) where
        # the car doesn't hide our lane. With a high or far chase camera the car covers the first
        # ~10 m, so a fixed 2-9 m window would see only our own paintwork (and never learn).
        pick = []
        for y, d in ys:
            if d < 2.0 or d > 25.0:
                continue
            cy = a * d * d + b * d + c
            u0 = int(rows.row_c0[y] + (cy - hw * 0.7) * rows.row_ppm[y])
            u1 = int(rows.row_c0[y] + (cy + hw * 0.7) * rows.row_ppm[y])
            u0, u1 = max(0, u0), min(W, u1)
            if u1 - u0 <= 4:
                continue
            if ego_mask is not None and float((ego_mask[y, u0:u1] == 0).mean()) < 0.6:
                continue
            pick.append((y, u0, u1))
            if len(pick) >= 6 or d - (ys[0][1] if not pick else 0) > 40:
                break
        if pick:
            labs, shs = [], []
            for y, u0, u1 in pick:
                seg = frame[y:y + 1, u0:u1]
                pm = paint_wide[y, u0:u1] == 0
                if ego_mask is not None:
                    pm &= ego_mask[y, u0:u1] == 0
                hsv = cv2.cvtColor(seg, cv2.COLOR_BGR2HSV)[0]
                lab = cv2.cvtColor(seg, cv2.COLOR_BGR2LAB)[0]
                shadow = (hsv[:, 0] >= 95) & (hsv[:, 0] <= 130) & (hsv[:, 2] < 125)
                labs.append(lab[pm])
                shs.append(shadow[pm])
            L = np.concatenate(labs) if labs else np.zeros((0, 3))
            S = np.concatenate(shs) if shs else np.zeros(0, bool)
            if len(L) >= 20:
                self.road.update(L, S)
        if self.road.sun is None and self.road.shade is None:
            return []
        # walk up the lane. All sampled rows are colour-converted in ONE call (they're stacked
        # into a small image) - per-row conversions were the slow part.
        blocked = []
        yrows = np.array([y for y, _ in ys], np.int32)
        strip = np.ascontiguousarray(frame[yrows])                       # (N, W, 3)
        lab_all = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB)
        grey_all = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.int16)
        hsv_all = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        road_all = self.road.is_road(lab_all, hsv_all)
        white_all = (hsv_all[:, :, 2] > 185) & (hsv_all[:, :, 1] < 55)
        ok_all = paint_wide[yrows] == 0
        vis_all = ego_mask[yrows] == 0 if ego_mask is not None else None
        if vis_all is not None:
            ok_all &= vis_all
        for n, (y, d) in enumerate(ys):
            cy = a * d * d + b * d + c
            u0 = int(rows.row_c0[y] + (cy - hw) * rows.row_ppm[y])
            u1 = int(rows.row_c0[y] + (cy + hw) * rows.row_ppm[y])
            u0, u1 = max(0, u0), min(W, u1)
            if u1 - u0 < 4:
                blocked.append((d, 0.0, 0.0, u0, u1, y, 0.0))
                continue
            road = road_all[n, u0:u1]
            ok = ok_all[n, u0:u1].copy()
            wh = white_all[n, u0:u1]
            if wh.any():
                runs = int(np.count_nonzero(np.diff(wh.astype(np.int8)) == 1)) + int(wh[0])
                if runs >= 3:
                    ok &= ~wh                    # crosswalk / hatching stripes: paint, not a car
            n_ok = int(ok.sum())
            frac = float((~road & ok).sum()) / n_ok if n_ok > 3 else 0.0
            g = grey_all[n, u0:u1]
            edges = float(np.abs(g[1:] - g[:-1]).mean())
            # beside our lane (past the lane lines): is it road there? A new road surface (junction,
            # bridge, concrete -> asphalt) changes BOTH sides too; a car only fills our lane.
            ppm, c0 = rows.row_ppm[y], rows.row_c0[y]
            sides = []
            for lo, hi in ((cy - hw - 2.6, cy - hw - 0.7), (cy + hw + 0.7, cy + hw + 2.6)):
                v0, v1 = int(max(0, c0 + lo * ppm)), int(min(W, c0 + hi * ppm))
                if v1 - v0 >= 3:
                    so = ok_all[n, v0:v1]
                    sides.append(float((~road_all[n, v0:v1] & so).sum()) / max(1, int(so.sum())))
                else:
                    sides.append(0.0)
            blocked.append((d, frac, edges, u0, u1, y, min(sides)))
            visible = float(vis_all[n, u0:u1].mean()) > 0.5 if vis_all is not None else True
            if not visible and not self._seen_any:
                self.hidden_to = d               # still under our own car on screen
            else:
                self._seen_any = True
                if self._chain and visible and frac < 0.3:
                    self.clear_to = d
                else:
                    self._chain = False
        self.debug_rows = blocked
        # first block of >= 3 consecutive rows mostly not-road. Some row in it must have edges
        # (bodywork, tyres, lights) unless the block is tall - a big plain car body close up has
        # few edges per row but fills many rows.
        found = None
        first_vis = None
        for i, row in enumerate(blocked):
            if row[0] > self.hidden_to:
                first_vis = i
                break
        run, run_edges, start = 0, 0.0, 0
        for i, (d, frac, edges, u0, u1, y, side) in enumerate(blocked):
            if frac > 0.45:
                if run == 0:
                    start, run_edges, run_sides = i, 0.0, []
                run += 1
                run_edges = max(run_edges, edges)
                run_sides.append(side)
                surface = run >= 3 and min(run_sides) > 0.6 and \
                    float(np.median([r[2] for r in blocked[start:i + 1]])) < 4.0
                if surface:
                    continue                     # the whole road changes colour here - not a car
                if run >= 3 and (run_edges > 2.0 or (run >= 6 and max(run_sides) < 0.3)):
                    # the body floats above the road; walk back toward us over rows that are
                    # partly blocked (tyres, bumper, the shadow under the car) to find where it
                    # actually touches the road
                    k = start
                    while k - 1 >= 0 and blocked[k - 1][1] > 0.1 and blocked[start][0] - blocked[k - 1][0] < 2.5:
                        k -= 1
                    found = blocked[k]
                    break
            else:
                run = 0
        self.hist.append(found[0] if found else None)
        recent = [h for h in self.hist if h is not None]
        if found is None or len(recent) < self.persist:
            return []
        if max(recent[-self.persist:]) - min(recent[-self.persist:]) > 6:
            return []                        # jumping around: noise
        d, frac, edges, u0, u1, y, _side = found
        # the block starts right where our own car stops hiding the road: its bottom (where it
        # touches the road) is hidden, so it's actually CLOSER - somewhere inside the hidden zone
        occluded = first_vis is not None and blocked.index(found) <= first_vis + 1 and self.hidden_to > 1.0
        if occluded:
            d = min(d, self.hidden_to)
        cy = a * d * d + b * d + c
        # screen box: from the found row up to ~1.6 m high
        top = int(max(rows.geo.horizon_y, y - 1.6 * rows.row_ppm[y]))
        box = (int(u0), top, int(u1), int(y))
        return [Obj("obstacle", box, float(min(0.9, 0.4 + frac * 0.5)), x=float(d), y=float(cy),
                    extra={"half_width": hw, "source": "free-space", "occluded": occluded})]
