"""Lane lines found directly in the game picture, then converted to metres.

How it works (all plain image processing):

1. Rows are sampled from the bottom of the screen up to the horizon, spaced so
   each step covers about the same distance on the road.
2. In each row band, runs of paint pixels of a plausible line width become
   "points" (centre x, row, yellow or white).
3. Points are linked band to band into chains. A chain may skip a few bands
   (gaps between dashes) and must keep heading toward the vanishing point the
   way real road lines do, so fences, building edges and poles drop out.
4. Each chain is converted to road coordinates (metres ahead / to the right of
   your car) and fitted with a smooth curve; it gets a colour, dashed/solid
   and a confidence.
5. The lines nearest your car on the left and right are your lane. Results are
   smoothed over time so one bad frame can't jerk the steering.

Also finds stop bars and crosswalks across your lane, in the same picture.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..perception.geometry import Geometry
from ..world import LaneLine, Lanes
from .paint import PaintFinder, PaintMasks


@dataclass
class Chain:
    pts: list = field(default_factory=list)       # [(x, y, yellow_fraction, band_index)]
    last_band: int = 0
    misses: int = 0

    def predict(self, y: float, vx: float, vy: float) -> float:
        if len(self.pts) >= 2:
            (x1, y1, *_), (x2, y2, *_) = self.pts[-2], self.pts[-1]
            if abs(y2 - y1) > 0.5:
                return x2 + (x2 - x1) * (y - y2) / (y2 - y1)
        x0, y0 = self.pts[-1][0], self.pts[-1][1]
        if abs(y0 - vy) < 1:
            return x0
        return vx + (x0 - vx) * (y - vy) / (y0 - vy)


@dataclass
class ScreenLaneResult:
    lanes: Lanes
    stop_line: float | None
    crosswalk: float | None
    chains_screen: list                           # [(points Nx2 int, is_ego_line, is_yellow)]
    masks: PaintMasks | None
    ego_pair_screen: tuple | None                 # ((aL,bL),(aR,bR)) lines x=a*y+b for auto-calibration
    timings: dict


class ScreenLaneFinder:
    def __init__(self, geo: Geometry, hud: list | None = None):
        self.set_geometry(geo, hud)
        self.prev_left: LaneLine | None = None
        self.prev_right: LaneLine | None = None
        self.seen_left = 0.0
        self.seen_right = 0.0
        self.vp_x = geo.W / 2.0
        self.last: ScreenLaneResult | None = None

    # ------------------------------------------------------------------ setup
    def set_geometry(self, geo: Geometry, hud: list | None = None) -> None:
        self.geo = geo
        self.W, self.H = geo.W, geo.H
        self.paint = PaintFinder(geo.W, geo.H, geo.horizon_y, hud)
        yh = geo.horizon_y
        # rows: equal steps in distance -> geometric spacing toward the horizon
        rows = []
        y = self.H - 2.0
        while y - yh > 4:
            rows.append(int(y))
            y = yh + (y - yh) * 0.9
        self.rows = np.array(rows[:48])
        # lateral pixels per metre at each row (perspective)
        ppm = []
        for r in self.rows:
            a = geo.screen_to_vehicle(np.array([[geo.W / 2 - 20, r], [geo.W / 2 + 20, r]], np.float32))
            dy = abs(a[1, 1] - a[0, 1])
            ppm.append(40.0 / dy if dy > 1e-6 else 1.0)
        self.ppm = np.array(ppm)
        self.vp_x = geo.W / 2.0
        # Per-row lookup: a screen row is a line of constant distance on a flat road, and along it
        # the lateral position is linear in x. dist[y] and (x - c0[y]) / ppm_row[y] convert any pixel.
        ys = np.arange(self.H, dtype=np.float32)
        valid = ys > yh + 1.5
        a = geo.screen_to_vehicle(np.stack([np.full(self.H, geo.W / 2 - 50.0, np.float32), ys], 1))
        b = geo.screen_to_vehicle(np.stack([np.full(self.H, geo.W / 2 + 50.0, np.float32), ys], 1))
        dy = np.where(np.abs(b[:, 1] - a[:, 1]) > 1e-6, b[:, 1] - a[:, 1], 1e-6)
        self.row_ppm = np.where(valid, 100.0 / dy, 0.0)
        self.row_c0 = np.where(valid, geo.W / 2 - 50.0 - a[:, 1] * self.row_ppm, geo.W / 2)
        self.row_dist = np.where(valid, a[:, 0], -1e3)
        self.row_valid = valid & (self.row_dist > -40) & (self.row_dist < 150)

    # ------------------------------------------------------------------ points
    def _band_points(self, pm: PaintMasks) -> list[list[tuple]]:
        """Paint runs per row band -> [(x, y, yellow_frac)] per band."""
        out = []
        paint, yel = pm.paint, pm.yellow
        for i, r in enumerate(self.rows):
            half = max(1, int(round(0.18 * self.ppm[i])))          # ~0.35 m tall band
            a, b = max(0, r - half), min(self.H, r + half + 1)
            col = paint[a:b].max(axis=0) > 0
            if not col.any():
                out.append([])
                continue
            ycol = yel[a:b].max(axis=0) > 0
            d = np.diff(np.concatenate([[0], col.astype(np.int8), [0]]))
            starts = np.where(d == 1)[0]
            ends = np.where(d == -1)[0]
            wmax = max(3.0, 0.75 * self.ppm[i])                       # lines are < ~0.6 m wide
            pts = []
            for s, e in zip(starts, ends):
                w = e - s
                if w > wmax:
                    continue
                pts.append(((s + e - 1) / 2.0, float(r), float(ycol[s:e].mean())))
            out.append(pts)
        return out

    def _link(self, bands: list[list[tuple]]) -> list[Chain]:
        vy = self.geo.horizon_y
        vx = self.vp_x
        active: list[Chain] = []
        done: list[Chain] = []
        for bi, pts in enumerate(bands):
            r = self.rows[bi]
            tol = max(3.0, 0.55 * self.ppm[bi])
            used = set()
            # match existing chains, closest first
            pairs = []
            for ci, ch in enumerate(active):
                xp = ch.predict(r, vx, vy)
                for pi, p in enumerate(pts):
                    dx = abs(p[0] - xp)
                    if dx < tol * (1 + 0.5 * ch.misses):
                        pairs.append((dx, ci, pi))
            pairs.sort()
            got_c = set()
            for dx, ci, pi in pairs:
                if ci in got_c or pi in used:
                    continue
                ch = active[ci]
                p = pts[pi]
                ch.pts.append((p[0], p[1], p[2], bi))
                ch.last_band = bi
                ch.misses = 0
                got_c.add(ci)
                used.add(pi)
            still = []
            for ci, ch in enumerate(active):
                if ci not in got_c:
                    ch.misses += 1
                (still if ch.misses <= 4 else done).append(ch)
            active = still
            for pi, p in enumerate(pts):
                if pi not in used:
                    active.append(Chain([(p[0], p[1], p[2], bi)], bi, 0))
        return done + active

    # ------------------------------------------------------------------ fast pixel -> metres
    def to_vehicle(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        yi = np.clip(ys.astype(np.int32), 0, self.H - 1)
        d = self.row_dist[yi]
        lat = (xs - self.row_c0[yi]) / np.where(self.row_ppm[yi] > 0, self.row_ppm[yi], 1.0)
        return d, lat

    # ------------------------------------------------------------------ blobs -> lines
    def _blob_lines(self, pm: PaintMasks) -> list[tuple[LaneLine, np.ndarray]]:
        """Each dash / line piece is a blob. Blobs that line up on the road become one line.

        All blobs are measured at once with array maths (per-blob sums via bincount), so
        hundreds of paint specks cost about a millisecond instead of tens.
        """
        m = cv2.morphologyEx(pm.paint, cv2.MORPH_OPEN, np.ones((2, 1), np.uint8))   # drop 1-row specks
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            return []
        py, px = np.nonzero(lab)
        L = lab[py, px]
        ok = self.row_valid[py]
        py, px, L = py[ok], px[ok], L[ok]
        if L.size == 0:
            return []
        d, lat = self.to_vehicle(px.astype(np.float32), py.astype(np.float32))
        inroad = (np.abs(lat) < 15) & (d > -25) & (d < 90)
        py, px, L, d, lat = py[inroad], px[inroad], L[inroad], d[inroad], lat[inroad]
        if L.size == 0:
            return []
        yl = (pm.yellow[py, px] > 0).astype(np.float64)
        cnt = np.bincount(L, minlength=n).astype(np.float64)
        sd = np.bincount(L, d, n)
        sl = np.bincount(L, lat, n)
        sdd = np.bincount(L, d * d, n)
        sdl = np.bincount(L, d * lat, n)
        sll = np.bincount(L, lat * lat, n)
        sy = np.bincount(L, yl, n)
        # per-blob min/max distance: sort once by label, then reduceat (ufunc.at is slow)
        order = np.argsort(L, kind="stable")
        Ls = L[order]
        ds = d[order]
        first = np.flatnonzero(np.r_[True, Ls[1:] != Ls[:-1]])
        dmin = np.full(n, np.inf)
        dmax = np.full(n, -np.inf)
        dmin[Ls[first]] = np.minimum.reduceat(ds, first)
        dmax[Ls[first]] = np.maximum.reduceat(ds, first)
        with np.errstate(invalid="ignore", divide="ignore"):
            md, ml = sd / cnt, sl / cnt
            vdd = sdd / cnt - md * md
            vdl = sdl / cnt - md * ml
            vll = sll / cnt - ml * ml
            slope = np.where(vdd > 1e-4, vdl / vdd, 0.0)
            resid = np.sqrt(np.maximum(0.0, vll - slope * vdl))       # rms distance from the blob's line
        h = stats[:, cv2.CC_STAT_HEIGHT]
        w = stats[:, cv2.CC_STAT_WIDTH]
        good = (cnt >= 6) & (h >= 3) & ~((w > 6 * h) & (w > 20)) & ((dmax - dmin) >= 0.6) \
            & (np.abs(slope) <= 0.9) & (resid <= 0.35)
        good[0] = False
        ids = np.nonzero(good)[0]
        if ids.size == 0:
            return []
        starts = np.searchsorted(Ls, ids)
        ends = np.searchsorted(Ls, ids, side="right")
        segs = []                    # (x_mid, lat_mid, heading, length_m, d, lat, yellow, screen_pts)
        for k, i in enumerate(ids):
            sel = order[starts[k]:ends[k]]
            span = float(dmax[i] - dmin[i])
            hb = float(slope[i]) if span > 1.5 else 0.0
            segs.append([float(md[i]), float(ml[i]), hb, span, d[sel], lat[sel], float(sy[i] / cnt[i]),
                         np.stack([px[sel], py[sel]], 1)])

        # union blobs that continue each other along the road
        parent = list(range(len(segs)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        order = sorted(range(len(segs)), key=lambda i: segs[i][0])
        for ii, i in enumerate(order):
            xi, li, hi = segs[i][0], segs[i][1], segs[i][2]
            for j in order[ii + 1:]:
                xj, lj, hj = segs[j][0], segs[j][1], segs[j][2]
                gap = xj - xi
                if gap > 30:
                    break
                h = (hi + hj) / 2 if abs(hi - hj) < 0.25 else hi
                if abs(hi - hj) > 0.3:
                    continue
                if abs((li + h * gap) - lj) < 0.45 + 0.015 * gap and (segs[i][6] > 0.4) == (segs[j][6] > 0.4):
                    parent[find(j)] = find(i)
        groups: dict[int, list[int]] = {}
        for i in range(len(segs)):
            groups.setdefault(find(i), []).append(i)
        out = []
        for idxs in groups.values():
            d = np.concatenate([segs[i][4] for i in idxs])
            lat = np.concatenate([segs[i][5] for i in idxs])
            span = d.max() - d.min()
            total = sum(segs[i][3] for i in idxs)
            if span < 3.0 or total < 1.5:
                continue
            w = 1.0 / (1.0 + np.abs(d) / 20.0)
            deg = 2 if span > 18 else 1
            coef = np.polyfit(d, lat, deg, w=w)
            if deg == 1:
                coef = np.array([0.0, coef[0], coef[1]])
            rms = float(np.sqrt(np.average((lat - np.polyval(coef, d)) ** 2, weights=w)))
            if rms > 0.45 or abs(coef[0]) > 0.02:
                continue
            fill = min(1.0, total / max(span, 1e-3))
            yfrac = float(np.average([segs[i][6] for i in idxs], weights=[segs[i][3] for i in idxs]))
            conf = float(min(1.0, total / 8.0) * max(0.0, 1.0 - rms / 0.45) * min(1.0, span / 12.0))
            line = LaneLine((float(coef[0]), float(coef[1]), float(coef[2])), float(d.max()),
                            "yellow" if yfrac > 0.4 else "white", bool(fill < 0.75 and len(idxs) >= 2), conf)
            pts = np.concatenate([segs[i][7] for i in idxs])
            pts = pts[np.argsort(pts[:, 1])]
            out.append((line, pts.astype(np.int32)))
        return out

    # ------------------------------------------------------------------ fitting
    def _chain_to_line(self, ch: Chain) -> tuple[LaneLine, np.ndarray] | None:
        if len(ch.pts) < 3:
            return None
        P = np.array([(p[0], p[1]) for p in ch.pts], np.float32)
        # must head toward the vanishing point: screen slope consistent with perspective
        vy = self.geo.horizon_y
        a, b = np.polyfit(P[:, 1], P[:, 0], 1)                        # x = a*y + b
        x_at_h = a * vy + b
        if abs(x_at_h - self.vp_x) > self.W * 0.45:
            return None
        V = self.geo.screen_to_vehicle(P)
        keep = (V[:, 0] > -25) & (V[:, 0] < 90)
        V = V[keep]
        if len(V) < 3:
            return None
        xs, ys = V[:, 0], V[:, 1]
        span = xs.max() - xs.min()
        if span < 4:
            return None
        w = 1.0 / (1.0 + np.abs(xs) / 20.0)
        deg = 2 if span > 15 else 1
        coef = np.polyfit(xs, ys, deg, w=w)
        if deg == 1:
            coef = np.array([0.0, coef[0], coef[1]])
        res = ys - np.polyval(coef, xs)
        rms = float(np.sqrt(np.average(res ** 2, weights=w)))
        if rms > 0.5 or abs(coef[0]) > 0.02 or abs(coef[1]) > 1.2:
            return None
        bands_spanned = ch.pts[-1][3] - ch.pts[0][3] + 1
        occ = len(ch.pts) / max(1, bands_spanned)
        yel = float(np.mean([p[2] for p in ch.pts]))
        conf = float(min(1.0, len(ch.pts) / 8.0) * max(0.0, 1.0 - rms / 0.5) * min(1.0, span / 15.0))
        line = LaneLine((float(coef[0]), float(coef[1]), float(coef[2])), float(xs.max()),
                        "yellow" if yel > 0.4 else "white", bool(occ < 0.8), conf)
        return line, P.astype(np.int32)

    # ------------------------------------------------------------------ stop bars / crosswalks
    def _cross_paint(self, pm: PaintMasks, lanes: Lanes) -> tuple[float | None, float | None]:
        """Rows where paint runs ACROSS our lane: stop bar (one long bar) or crosswalk (many blocks)."""
        geo = self.geo
        c = lanes.center[2] if lanes.conf > 0 else 0.0
        hw = max(1.6, (lanes.width_m or 3.7) / 2)
        stop = cross = None
        wide = pm.wide
        for r in range(self.H - 1, int(geo.horizon_y) + 2, -2):
            if not self.row_valid[r]:
                continue
            x_m = float(self.row_dist[r])
            if x_m < 1.0:
                continue
            if x_m > 55:
                break
            a, b, cc = lanes.center if lanes.conf > 0 else (0.0, 0.0, c)
            cy = a * x_m * x_m + b * x_m + cc
            ppm, c0 = self.row_ppm[r], self.row_c0[r]
            u0, u1 = int(max(0, c0 + (cy - hw) * ppm)), int(min(self.W, c0 + (cy + hw) * ppm))
            w0, w1 = int(max(0, c0 + (cy - 3 * hw) * ppm)), int(min(self.W, c0 + (cy + 3 * hw) * ppm))
            if u1 - u0 < 6:
                continue
            row = wide[max(0, r - 1):r + 2].max(axis=0) > 0
            lane = row[u0:u1]
            cover = lane.mean()
            if stop is None and cover > 0.7:
                stop = float(x_m)
            if cross is None and w1 - w0 > 10:
                band = row[w0:w1]
                flips = int(np.abs(np.diff(band.astype(np.int8))).sum())
                if flips >= 8 and 0.25 < band.mean() < 0.8:
                    cross = float(x_m)
            if stop is not None and cross is not None:
                break
        if stop is not None and cross is not None and abs(stop - cross) < 1.5:
            stop = None            # the "bar" was the crosswalk itself
        return stop, cross

    # ------------------------------------------------------------------ main
    def process(self, frame: np.ndarray, ego_mask: np.ndarray | None, smoothing: float = 0.35,
                lane_width: float = 3.7, t: float | None = None) -> ScreenLaneResult:
        from ..perception.lanes import LaneFinder
        t = time.perf_counter() if t is None else t
        tm = {}
        t0 = time.perf_counter()
        pm = self.paint.find(frame, ego_mask)
        tm["paint"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        cands = self._blob_lines(pm)
        tm["chains"] = time.perf_counter() - t0
        lines = [c[0] for c in cands]
        left, right, left2, right2 = LaneFinder._pick(lines, lane_width)

        sl = self._blend(left, self.prev_left if t - self.seen_left < 0.6 else None, smoothing)
        sr = self._blend(right, self.prev_right if t - self.seen_right < 0.6 else None, smoothing)
        if left is not None:
            self.seen_left = t
        elif sl is not None:
            sl = LaneLine(sl.coeffs, sl.x_max, sl.color, sl.dashed, sl.conf * 0.85)
        if right is not None:
            self.seen_right = t
        elif sr is not None:
            sr = LaneLine(sr.coeffs, sr.x_max, sr.color, sr.dashed, sr.conf * 0.85)
        sl = sl if (sl is not None and sl.conf > 0.05) else None
        sr = sr if (sr is not None and sr.conf > 0.05) else None
        self.prev_left, self.prev_right = sl, sr
        lanes = LaneFinder._assemble(sl, sr, left2, right2, lane_width)

        # vanishing point follows the ego lane lines (helps linking on curves)
        ego_pair = None
        idx = {id(c[0]): c for c in cands}
        if left is not None and right is not None:
            Lp, Rp = idx[id(left)][1], idx[id(right)][1]
            if len(Lp) >= 4 and len(Rp) >= 4:
                la, lb = np.polyfit(Lp[:, 1], Lp[:, 0], 1)
                ra, rb = np.polyfit(Rp[:, 1], Rp[:, 0], 1)
                if abs(la - ra) > 1e-3:
                    vy = (rb - lb) / (la - ra)
                    vx = la * vy + lb
                    if 0 < vx < self.W:
                        self.vp_x = 0.9 * self.vp_x + 0.1 * vx
                    lres = np.abs(Lp[:, 0] - (la * Lp[:, 1] + lb)).mean()
                    rres = np.abs(Rp[:, 0] - (ra * Rp[:, 1] + rb)).mean()
                    if lres < 2.0 and rres < 2.0 and abs(lanes.curvature) < 0.004:
                        ego_pair = ((float(la), float(lb)), (float(ra), float(rb)))

        stop, cross = self._cross_paint(pm, lanes)
        ego_ids = {id(x) for x in (left, right) if x is not None}
        chains_screen = [(c[1], id(c[0]) in ego_ids, c[0].color == "yellow") for c in cands]
        tm["total"] = tm["paint"] + tm["chains"]
        res = ScreenLaneResult(lanes, stop, cross, chains_screen, pm, ego_pair, tm)
        self.last = res
        return res

    @staticmethod
    def _blend(new: LaneLine | None, old: LaneLine | None, alpha: float) -> LaneLine | None:
        if new is None:
            return old
        if old is None:
            return new
        if abs(new.y_at(8) - old.y_at(8)) > 1.4:
            return new if new.conf > old.conf + 0.25 else old
        c = tuple(alpha * o + (1 - alpha) * n for o, n in zip(old.coeffs, new.coeffs))
        return LaneLine(c, new.x_max, new.color, new.dashed, max(new.conf, 0.85 * old.conf))


def lane_heading_deg(lanes: Lanes) -> float:
    return math.degrees(lanes.heading)


def draw_chains(img: np.ndarray, res: ScreenLaneResult, scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
    """Debug drawing of the linked chains (ego lane lines thicker)."""
    for pts, ego, yel in res.chains_screen:
        p = np.stack([pts[:, 0] * scale_x, pts[:, 1] * scale_y], 1).astype(np.int32)
        col = (0, 220, 255) if yel else (255, 255, 255)
        cv2.polylines(img, [p], False, col if ego else (120, 120, 120), 3 if ego else 1, cv2.LINE_AA)
    return img
