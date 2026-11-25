"""Lane lines, stop lines and crosswalks from a bird's-eye view of the road.

Why this is better than v3:
* Works in a top-down (bird's-eye) view, so lane lines are parallel and have a
  known width in metres.
* A "top-hat" filter keeps only thin bright paint (10-45 cm wide). Sidewalks,
  kerbs and the sky are wide, so they vanish. Street-light poles never reach the
  ground plane, so they can't be mistaken for lane lines.
* Only paint touching dark asphalt counts.
* Crosswalk ladders and stop bars are found separately and removed before lane
  fitting, then reported as stop targets.
* Every line is fitted as a curve in metres and smoothed over time, with a
  prior from the last frame, so one noisy frame can't jerk the steering.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from ..world import LaneLine, Lanes
from .geometry import Geometry


@dataclass
class LaneDebug:
    bev: np.ndarray | None = None
    mask: np.ndarray | None = None
    candidates: list | None = None


def _fit(xs: np.ndarray, ys: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Weighted quadratic y = a x^2 + b x + c. Falls back to linear for short spans."""
    span = xs.max() - xs.min()
    deg = 2 if span > 18 else 1
    if w is None:
        w = np.ones_like(xs)
    coef = np.polyfit(xs, ys, deg, w=w)
    if deg == 1:
        coef = np.array([0.0, coef[0], coef[1]])
    res = ys - np.polyval(coef, xs)
    return coef, float(np.sqrt(np.average(res ** 2, weights=w)))


class LaneFinder:
    def __init__(self, geo: Geometry):
        self.geo = geo
        self.prev_left: LaneLine | None = None
        self.prev_right: LaneLine | None = None
        self.prev_t = 0.0
        self.last_seen_left = 0.0
        self.last_seen_right = 0.0
        self.debug = LaneDebug()
        mpx = geo.m_per_px_x
        k = max(5, int(round(0.55 / mpx)) | 1)          # wider than paint, narrower than sidewalks
        self.k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        self.k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
        self.side_px = max(3, int(0.45 / mpx))               # look 45 cm left/right for road
        kb = int(2.0 / mpx) | 1                                # 2 m: wider than any paint block
        self.k_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (kb, kb))
        self.k_road = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(1.2 / mpx) | 1, int(1.2 / mpx) | 1))
        self.ego_mask_screen: np.ndarray | None = None

    # ------------------------------------------------------------------
    def set_ego_mask(self, frac_box: list[float]) -> None:
        g = self.geo
        m = np.full((g.H, g.W), 255, np.uint8)
        x1, y1, x2, y2 = frac_box
        m[int(y1 * g.H):int(y2 * g.H), int(x1 * g.W):int(x2 * g.W)] = 0
        self.ego_mask_screen = g.warp(m, cv2.INTER_NEAREST)

    def masks(self, bev: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (paint_lane, yellow, paint_horizontal, road, any_white_paint) uint8 masks in BEV."""
        hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        L, S = hls[:, :, 1], hls[:, :, 2]
        Hh, Sv, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Road surface: grey (low saturation). ER:LC city asphalt is dark, highway
        # concrete is light grey, and Roblox shadows tint the road dark BLUE.
        shadow = (Hh >= 95) & (Hh <= 130) & (V < 120)
        asphalt = ((((Sv < 50) & (V > 18) & (V < 205)) | shadow)).astype(np.uint8) * 255
        asphalt = cv2.morphologyEx(asphalt, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # Brightness relative to the road around it, so tunnels/shadows still work.
        th_h = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, self.k_h)   # thin along x (lane lines)
        th_v = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, self.k_v)   # thin along y (stop bars)
        road_px = L[asphalt > 0]
        road_lum = float(np.median(road_px)) if road_px.size > 500 else 70.0
        t = max(16.0, min(45.0, (255 - road_lum) * 0.22))

        # white paint: brighter than its surroundings (top-hat) and not grass/yellow.
        # In shadow, white paint turns light blue, so blue-ish bright pixels count too.
        grass = (Hh >= 35) & (Hh <= 95) & (Sv > 60)
        # Brighter than the road right around it (works in sun AND in dark shadow,
        # where paint is only ~L=110 on ~L=35 road).
        bg = cv2.morphologyEx(L, cv2.MORPH_OPEN, self.k_bg)
        local = cv2.subtract(L, bg)
        white = (((S < 90) | ((Hh >= 95) & (Hh <= 130))) & ~grass & (local > 28))
        yellow = (Hh >= 14) & (Hh <= 34) & (Sv > 70) & (V > 90)
        paint_lane = (((th_h > t) & white) | (yellow & (cv2.morphologyEx(V, cv2.MORPH_TOPHAT, self.k_h) > t * 0.6)))
        paint_hz = (th_v > t) & white
        road = cv2.dilate(asphalt, self.k_road)
        # A lane line has road on BOTH sides; a curb / sidewalk edge only on one.
        sh = self.side_px
        left_ok = np.zeros_like(asphalt)
        right_ok = np.zeros_like(asphalt)
        left_ok[:, sh:] = asphalt[:, :-sh]
        right_ok[:, :-sh] = asphalt[:, sh:]
        both = cv2.dilate(left_ok & right_ok, np.ones((5, 3), np.uint8))
        pl = (paint_lane.astype(np.uint8) * 255) & road & both
        ph = (paint_hz.astype(np.uint8) * 255) & road
        yl = (yellow.astype(np.uint8) * 255) & pl
        wa = (white.astype(np.uint8) * 255) & road     # wide paint too (crosswalk blocks)
        if self.ego_mask_screen is not None:
            for m in (pl, ph, yl, asphalt, wa):
                m &= self.ego_mask_screen
        return pl, yl, ph, asphalt, wa

    # ------------------------------------------------------------------
    def find_crosswalk_rows(self, paint: np.ndarray, corridor_px: tuple[int, int]) -> np.ndarray:
        """Boolean per BEV row: True where a zebra/ladder crosswalk crosses our corridor."""
        u0, u1 = corridor_px
        band = paint[:, max(0, u0):min(paint.shape[1], u1)] > 0
        if band.size == 0:
            return np.zeros(paint.shape[0], bool)
        trans = np.abs(np.diff(band.astype(np.int8), axis=1)).sum(axis=1)
        fill = band.mean(axis=1)
        rows = (trans >= 6) & (fill > 0.2) & (fill < 0.8)
        # need ~1.5 m of consecutive rows to call it a crosswalk
        n = max(3, int(1.5 / self.geo.m_per_px_y))
        k = np.convolve(rows.astype(np.float32), np.ones(n) / n, mode="same")
        return k > 0.6

    def find_stop_line(self, paint_hz: np.ndarray, corridor_px: tuple[int, int]) -> float | None:
        """Distance (m) to a white stop bar spanning our lane, if any."""
        u0, u1 = corridor_px
        u0, u1 = max(0, u0), min(paint_hz.shape[1], u1)
        if u1 - u0 < 10:
            return None
        band = paint_hz[:, u0:u1] > 0
        cover = band.mean(axis=1)
        rows = np.where(cover > 0.6)[0]
        if rows.size == 0:
            return None
        # group consecutive rows; a stop bar is 0.2 - 0.9 m deep
        groups = np.split(rows, np.where(np.diff(rows) > 2)[0] + 1)
        best = None
        for g in groups:
            depth = (g[-1] - g[0] + 1) * self.geo.m_per_px_y
            if 0.15 <= depth <= 0.9:
                v = float(g[-1])                     # near edge of the bar
                x = self.geo.bev_to_vehicle(np.array([[self.geo.bev_w / 2, v]]))[0, 0]
                if 2.0 < x < 55 and (best is None or x < best):
                    best = float(x)
        return best

    # ------------------------------------------------------------------
    def _trace(self, pts_x: np.ndarray, pts_y: np.ndarray, yel: np.ndarray, start_y: float,
               prior: LaneLine | None, x_start: float = 4.0) -> LaneLine | None:
        """Follow one line forward from lateral position start_y using sliding windows."""
        win = 0.9
        step = 3.0
        xs_all, ys_all, yel_all = [], [], []
        y_c = start_y
        hits = 0
        n_win = 0
        x = x_start
        prev_shift = 0.0
        while x < 58:
            n_win += 1
            if prior is not None and x > 8:
                y_pred = prior.y_at(x) + (start_y - prior.y_at(6.0))
                y_c = 0.5 * y_c + 0.5 * y_pred
            sel = (pts_x >= x) & (pts_x < x + step) & (np.abs(pts_y - y_c) < win)
            if np.count_nonzero(sel) >= 4:
                hits += 1
                ym = float(np.median(pts_y[sel]))
                prev_shift = 0.6 * prev_shift + 0.4 * (ym - y_c)
                y_c = ym
                xs_all.append(pts_x[sel])
                ys_all.append(pts_y[sel])
                yel_all.append(yel[sel])
            else:
                y_c += prev_shift            # keep going through dashes along the curve
            x += step
        if hits < 3:
            return None
        xs = np.concatenate(xs_all)
        ys = np.concatenate(ys_all)
        yl = np.concatenate(yel_all)
        if xs.max() - xs.min() < 6:
            return None
        w = 1.0 / (1.0 + xs / 25.0)          # trust near paint more
        coef, rms = _fit(xs, ys, w)
        if rms > 0.45 or abs(coef[0]) > 0.02:
            return None
        # Dashed or solid: walk along the fitted curve and see how much of it has paint.
        xq = np.arange(xs.min(), xs.max(), 0.5)
        if xq.size >= 6:
            order = np.argsort(xs)
            xs_s = xs[order]
            ys_s = ys[order]
            yq = np.polyval(coef, xq)
            lo = np.searchsorted(xs_s, xq - 0.3)
            hi = np.searchsorted(xs_s, xq + 0.3)
            occ = np.array([np.any(np.abs(ys_s[a:b] - y) < 0.25) if b > a else False
                            for a, b, y in zip(lo, hi, yq)])
            dashed = bool(0.15 < occ.mean() < 0.75)
            if dashed:
                # lane dashes are metres long; crosswalk-ladder ends repeat every ~0.5-1 m
                runs, cur = [], 0
                for o in occ:
                    if o:
                        cur += 1
                    elif cur:
                        runs.append(cur)
                        cur = 0
                if cur:
                    runs.append(cur)
                if runs and np.median(runs) * 0.5 < 1.2 and len(runs) >= 4:
                    return None
        else:
            dashed = False
        color = "yellow" if yl.mean() > 0.35 else "white"
        conf = float(min(1.0, (hits / 6.0)) * max(0.0, 1.0 - rms / 0.45))
        return LaneLine((float(coef[0]), float(coef[1]), float(coef[2])), float(xs.max()), color, dashed, conf)

    def _blend(self, new: LaneLine | None, old: LaneLine | None, alpha: float) -> LaneLine | None:
        if new is None:
            return old
        if old is None:
            return new
        # Reject jumps (e.g. grabbed the next line over)
        if abs(new.y_at(6) - old.y_at(6)) > 1.4:
            if new.conf < old.conf + 0.25:
                return old
            return new
        c = tuple(alpha * o + (1 - alpha) * n for o, n in zip(old.coeffs, new.coeffs))
        return LaneLine(c, new.x_max, new.color, new.dashed, max(new.conf, 0.85 * old.conf))

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, smoothing: float = 0.35, lane_width: float = 3.7,
                t: float | None = None) -> tuple[Lanes, float | None, float | None]:
        """Returns (lanes, stop_line_distance, crosswalk_distance)."""
        t = time.perf_counter() if t is None else t
        g = self.geo
        bev = g.warp(frame)
        pl, yl, ph, asphalt, wa = self.masks(bev)

        # Corridor around our predicted lane, for stop line / crosswalk checks
        cy = 0.0
        if self.prev_left and self.prev_right:
            cy = (self.prev_left.y_at(10) + self.prev_right.y_at(10)) / 2
        cu = int(g.bev_w / 2 + cy / g.m_per_px_x)
        half = int(lane_width * 0.5 / g.m_per_px_x)
        corridor = (cu - half, cu + half)

        cw_rows = self.find_crosswalk_rows(wa, (cu - 3 * half, cu + 3 * half))
        crosswalk = None
        if cw_rows.any():
            vs = np.where(cw_rows)[0]
            x_near = g.bev_to_vehicle(np.array([[g.bev_w / 2, float(vs.max())]]))[0, 0]
            crosswalk = float(x_near) if 2 < x_near < 55 else None
            pl = pl.copy()
            pl[cw_rows] = 0                   # crosswalk paint is not a lane line
        stop_line = self.find_stop_line(ph, corridor)

        # Paint pixels -> vehicle metres (subsample for speed)
        vv, uu = np.nonzero(pl[::2, :])
        vv = vv * 2
        if vv.size > 6000:
            idx = np.random.default_rng(0).choice(vv.size, 6000, replace=False)
            vv, uu = vv[idx], uu[idx]
        pts = g.bev_to_vehicle(np.stack([uu, vv], 1).astype(np.float32)) if vv.size else np.zeros((0, 2))
        is_yel = (yl[vv, uu] > 0) if vv.size else np.zeros(0, bool)
        px, py = (pts[:, 0], pts[:, 1]) if len(pts) else (np.zeros(0), np.zeros(0))

        # Start positions: histogram of lateral offsets in the near field
        cands: list[LaneLine] = []
        # Start lines from peaks in the near band AND the far band: near paint is often
        # hidden under the car, and inside a junction the only paint is across it.
        for lo, hi, far in ((4, 30, False), (22, 58, True)):
            band = (px > lo) & (px < hi)
            if np.count_nonzero(band) < 20:
                continue
            bins = np.arange(-10, 10.01, 0.25)
            hist, edges = np.histogram(py[band], bins=bins)
            hist = np.convolve(hist, [1, 2, 3, 2, 1], mode="same")
            peaks = []
            for i in np.argsort(hist)[::-1]:
                if hist[i] < 8:
                    break
                yv = (edges[i] + edges[i + 1]) / 2
                if all(abs(yv - q) > 1.6 for q in peaks):
                    peaks.append(yv)
                if len(peaks) >= 6:
                    break
            for p in peaks:
                x0 = 4.0
                if far:
                    if any(abs(c.y_at(40) - p) < 0.8 for c in cands):
                        continue
                    x0 = 22.0
                prior = None
                if self.prev_left and abs(self.prev_left.y_at(6) - p) < 1.0:
                    prior = self.prev_left
                elif self.prev_right and abs(self.prev_right.y_at(6) - p) < 1.0:
                    prior = self.prev_right
                if far:
                    prior = None
                    start = p
                    # walk from the far band; fit is then extended back to the car
                    line = self._trace(px, py, is_yel, start, None, x_start=x0)
                else:
                    line = self._trace(px, py, is_yel, p, prior)
                if line is not None:
                    cands.append(line)

        left, right, left2, right2 = self._pick(cands, lane_width)
        alpha = smoothing
        sl = self._blend(left, self.prev_left if t - self.last_seen_left < 0.6 else None, alpha)
        sr = self._blend(right, self.prev_right if t - self.last_seen_right < 0.6 else None, alpha)
        if left is not None:
            self.last_seen_left = t
        elif sl is not None:
            sl = LaneLine(sl.coeffs, sl.x_max, sl.color, sl.dashed, sl.conf * 0.8)
        if right is not None:
            self.last_seen_right = t
        elif sr is not None:
            sr = LaneLine(sr.coeffs, sr.x_max, sr.color, sr.dashed, sr.conf * 0.8)
        if sl is not None and sl.conf < 0.05:
            sl = None
        if sr is not None and sr.conf < 0.05:
            sr = None
        self.prev_left, self.prev_right, self.prev_t = sl, sr, t

        lanes = self._assemble(sl, sr, left2, right2, lane_width)
        self.debug = LaneDebug(bev, pl, cands)
        return lanes, stop_line, crosswalk

    @staticmethod
    def _pick(cands: list[LaneLine], lane_width: float):
        lefts = sorted([c for c in cands if -lane_width * 1.25 < c.y_at(6) < -0.4], key=lambda c: -c.y_at(6))
        rights = sorted([c for c in cands if 0.4 < c.y_at(6) < lane_width * 1.25], key=lambda c: c.y_at(6))
        left = lefts[0] if lefts else None
        right = rights[0] if rights else None
        if left and right:
            w = right.y_at(6) - left.y_at(6)
            if not (2.4 < w < 6.0):          # implausible pair: keep the more confident one
                if left.conf >= right.conf:
                    right = None
                else:
                    left = None
        left2 = next((c for c in cands if left and -lane_width * 1.6 < c.y_at(6) - left.y_at(6) < -2.4), None)
        right2 = next((c for c in cands if right and 2.4 < c.y_at(6) - right.y_at(6) < lane_width * 1.6), None)
        return left, right, left2, right2

    @staticmethod
    def _assemble(left: LaneLine | None, right: LaneLine | None, left2, right2, lane_width: float) -> Lanes:
        lanes = Lanes(left=left, right=right, left2=left2, right2=right2)
        if left and right:
            ca = [(a + b) / 2 for a, b in zip(left.coeffs, right.coeffs)]
            lanes.width_m = float(np.clip(right.y_at(8) - left.y_at(8), 2.5, 6.0))
            lanes.conf = float(min(left.conf, right.conf) * 0.5 + 0.5 * (left.conf + right.conf) / 2)
            lanes.x_max = min(left.x_max, right.x_max)
        elif left or right:
            ln = left or right
            sgn = 1 if left else -1
            ca = [ln.coeffs[0], ln.coeffs[1], ln.coeffs[2] + sgn * lane_width / 2]
            lanes.width_m = lane_width
            lanes.conf = ln.conf * 0.6
            lanes.x_max = ln.x_max
        else:
            return lanes
        a, b, c = ca
        lanes.center = (float(a), float(b), float(c))
        lanes.offset = float(c)
        lanes.heading = float(math.atan(b))
        lanes.curvature = float(2 * a / (1 + (2 * a * 10 + b) ** 2) ** 1.5)
        return lanes
