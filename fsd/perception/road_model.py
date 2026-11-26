"""Where is the ROAD, and which part of it is ours (right-hand traffic)?

Lane lines aren't always there: country roads with worn paint, parking lots, the
middle of a junction, dirt tracks, a road with only a yellow centre line. A human
still sees the road's edges and keeps to the right. This module does the same, with
no AI:

1. Road colour. The patch just in front of our bumper is road. Its colour (Lab) is
   learned all the time, sunlit and shaded separately (Roblox shadows are blue).
2. Road edges. For ~25 rows from 3 m to ~60 m ahead we walk left and right from where
   our car is heading until the pixels stop looking like road (paint and small gaps
   are allowed). That gives the left and right edge of the road at each distance, in
   metres, which are fitted with smooth curves.
3. Our side. ER:LC drives on the RIGHT. Facts we use, strongest first:
   * a YELLOW line marks the centre of a two-way road: it must be on our LEFT.
     If our lane's right-hand line is yellow, we're on the wrong side.
   * if there's no centre line, the road is two-way if it's wide enough for two lanes,
     so we keep to the right half; a narrow road is one lane, we keep to its middle.
   * other cars' direction: a car ahead coming toward us is on the other side.

The planner gets: road edges, how wide the road is, whether we're on the wrong side,
and a suggested lateral position when there are no usable lane lines.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class RoadEdges:
    ok: bool = False
    left: tuple[float, float, float] = (0.0, 0.0, -3.5)     # y = a x^2 + b x + c (vehicle m, + right)
    right: tuple[float, float, float] = (0.0, 0.0, 3.5)
    x_max: float = 0.0
    width_m: float = 0.0
    conf: float = 0.0
    samples: list = field(default_factory=list)              # (x, y_left, y_right) for the dashboard

    def at(self, x: float) -> tuple[float, float]:
        l = self.left[0] * x * x + self.left[1] * x + self.left[2]
        r = self.right[0] * x * x + self.right[1] * x + self.right[2]
        return l, r


@dataclass
class SideAdvice:
    wrong_side: bool = False
    reason: str = ""
    target_offset: float | None = None   # wanted lateral position of our centre (m, + right) at ~10 m
    two_way: bool = True
    lanes_our_way: int = 1


class RoadColour:
    def __init__(self):
        self.sun = None
        self.shade = None
        self.n = 0

    def learn(self, lab: np.ndarray, hsv: np.ndarray) -> None:
        """Robust colour statistics (median / MAD) of the road just ahead. Paint that slipped
        through the paint mask (anti-aliased edges of a yellow or white line) is dropped first,
        and the spread is capped - one stripe of yellow in the sample must not make the whole
        world look like road."""
        lab = lab.reshape(-1, 3)
        hsv = hsv.reshape(-1, 3)
        paintlike = ((hsv[:, 1] > 70) & (hsv[:, 0] >= 15) & (hsv[:, 0] <= 40)) | \
                    ((hsv[:, 2] > 185) & (hsv[:, 1] < 50))
        lab = lab[~paintlike]
        hsv = hsv[~paintlike]
        shadow = (hsv[:, 0] >= 95) & (hsv[:, 0] <= 130) & (hsv[:, 2] < 130)
        for name, sel in (("sun", ~shadow), ("shade", shadow)):
            px = lab[sel].astype(np.float64)
            if len(px) < 30:
                continue
            m = np.median(px, axis=0)
            mad = np.median(np.abs(px - m), axis=0) * 1.4826
            sd = np.clip(mad, [2.0, 1.0, 1.0], [12.0, 5.0, 5.0]) + np.array([4.0, 2.0, 2.0])
            cur = getattr(self, name)
            setattr(self, name, (m, sd) if cur is None else (0.9 * cur[0] + 0.1 * m, 0.9 * cur[1] + 0.1 * sd))
        self.n += 1

    def mask(self, lab: np.ndarray, hsv: np.ndarray, k: float = 3.0) -> np.ndarray:
        # (unlike the obstacle finder, shadows are NOT assumed to be road here: tree shadows on
        # grass are just as blue, and they'd move the road edge onto the verge)
        out = np.zeros(lab.shape[:2], bool)
        for cur in (self.sun, self.shade):
            if cur is None:
                continue
            m, s = cur
            lo = np.clip(np.floor(m - k * s), 0, 255).astype(np.uint8)
            hi = np.clip(np.ceil(m + k * s), 0, 255).astype(np.uint8)
            out |= cv2.inRange(lab, lo, hi) > 0
        return out


class RoadModel:
    def __init__(self, n_rows: int = 26, max_gap_m: float = 0.5):
        self.colour = RoadColour()
        self.n_rows = n_rows
        self.max_gap_m = max_gap_m
        self.edges = RoadEdges()
        self._smooth_w: float | None = None

    def _rows(self, rows) -> list[tuple[int, float]]:
        """(image row, distance) pairs from ~3 m to ~60 m, evenly in distance."""
        valid = np.nonzero(rows.row_valid)[0]
        if not len(valid):
            return []
        dist = rows.row_dist[valid]
        out = []
        for d in np.geomspace(3.0, 60.0, self.n_rows):
            i = int(np.argmin(np.abs(dist - d)))
            if abs(dist[i] - d) < max(0.8, 0.15 * d):
                out.append((int(valid[i]), float(dist[i])))
        # unique rows, nearest first
        seen, uniq = set(), []
        for y, d in sorted(out, key=lambda r: r[1]):
            if y not in seen:
                seen.add(y)
                uniq.append((y, d))
        return uniq

    def update(self, frame: np.ndarray, rows, paint: np.ndarray | None, ego_mask: np.ndarray | None,
               heading_y: float = 0.0) -> RoadEdges:
        """rows: the lane finder (row tables). heading_y: lateral position (m) to start the
        search from (where our lane centre is, or 0 = straight ahead of the car)."""
        H, W = frame.shape[:2]
        rws = self._rows(rows)
        if len(rws) < 6:
            self.edges = RoadEdges()
            return self.edges
        ys = np.array([r[0] for r in rws])
        strip = np.ascontiguousarray(frame[ys])
        lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        okpix = np.ones(strip.shape[:2], bool)
        if ego_mask is not None:
            okpix &= ego_mask[ys] == 0
        # learn the road colour from just in front of us: the nearest rows (+-1.1 m around the
        # start column) that our own car doesn't hide - in a chase view the car covers the first
        # metres, so this is usually ~8-15 m ahead
        sel_lab, sel_hsv = [], []
        used = 0
        for k, (y, d) in enumerate(rws):
            if d > 25.0 or used >= 4:
                break
            c = rows.row_c0[y] + heading_y * rows.row_ppm[y]
            half = 1.1 * rows.row_ppm[y]
            a, b = int(max(0, c - half)), int(min(W, c + half))
            if b - a < 3:
                continue
            if okpix[k, a:b].mean() < 0.6:
                continue
            m = okpix[k, a:b].copy()
            if paint is not None:
                m &= paint[y, a:b] == 0
            sel_lab.append(lab[k, a:b][m])
            sel_hsv.append(hsv[k, a:b][m])
            used += 1
        if sel_lab:
            L = np.concatenate(sel_lab)
            Hs = np.concatenate(sel_hsv)
            if len(L) >= 30:
                self.colour.learn(L.reshape(-1, 1, 3), Hs.reshape(-1, 1, 3))
        if self.colour.n < 3:
            self.edges = RoadEdges()
            return self.edges
        road = self.colour.mask(lab, hsv, k=2.5)
        if paint is not None:
            road |= paint[ys] > 0                          # paint is part of the road
        road |= ~okpix                                     # hidden by our car: assume road
        samples = []
        for k, (y, d) in enumerate(rws):
            ppm = rows.row_ppm[y]
            c = int(round(rows.row_c0[y] + heading_y * ppm))
            if not (0 <= c < W) or not road[k, c]:
                continue
            gap = max(2, int(self.max_gap_m * ppm))
            row = road[k]
            left = self._walk(row, c, -1, gap)
            right = self._walk(row, c, +1, gap)
            if left <= 1 or right >= W - 2:
                # ran off the picture: we don't know where the edge is there
                yl = None if left <= 1 else (left - rows.row_c0[y]) / ppm
                yr = None if right >= W - 2 else (right - rows.row_c0[y]) / ppm
            else:
                yl = (left - rows.row_c0[y]) / ppm
                yr = (right - rows.row_c0[y]) / ppm
            samples.append((d, yl, yr))
        self.edges = self._fit(samples)
        return self.edges

    @staticmethod
    def _walk(row: np.ndarray, c: int, step: int, gap: int) -> int:
        """From column c walk in `step` direction while road (gaps up to `gap` px allowed)."""
        W = len(row)
        x = c
        miss = 0
        last_road = c
        while 0 <= x + step < W:
            x += step
            if row[x]:
                last_road = x
                miss = 0
            else:
                miss += 1
                if miss > gap:
                    break
        return last_road

    def _fit(self, samples) -> RoadEdges:
        e = RoadEdges(samples=samples)
        L = [(d, yl) for d, yl, _ in samples if yl is not None]
        R = [(d, yr) for d, _, yr in samples if yr is not None]
        if len(L) < 4 or len(R) < 4:
            return e
        def fit(pts):
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            deg = 2 if xs.max() - xs.min() > 20 else 1
            for _ in range(2):                       # one round of outlier removal
                c = np.polyfit(xs, ys, deg)
                r = np.abs(np.polyval(c, xs) - ys)
                keep = r < max(0.6, 2.5 * np.median(r))
                if keep.sum() >= 4:
                    xs, ys = xs[keep], ys[keep]
            c = np.polyfit(xs, ys, deg)
            if deg == 1:
                c = np.r_[0.0, c]
            return tuple(float(v) for v in c), float(xs.max()), float(np.median(np.abs(np.polyval(c, xs) - ys)))
        lc, lmax, lres = fit(L)
        rc, rmax, rres = fit(R)
        width = (rc[2] - lc[2]) if lc and rc else 0.0
        if not (2.5 < width < 30.0):
            return e
        self._smooth_w = width if self._smooth_w is None else 0.85 * self._smooth_w + 0.15 * width
        conf = float(np.clip(min(len(L), len(R)) / 12.0, 0, 1) * np.clip(1.0 - (lres + rres) / 2.0, 0, 1))
        return RoadEdges(True, lc, rc, min(lmax, rmax), self._smooth_w, conf, samples)


def side_advice(lanes, edges: RoadEdges, objects=(), lane_width: float = 3.7) -> SideAdvice:
    """Are we on the correct (right-hand) side, and where should we be?"""
    adv = SideAdvice()
    L, R = lanes.left, lanes.right
    # 1) yellow line on our RIGHT = we're in an oncoming lane
    if lanes.conf > 0.35 and R is not None and getattr(R, "color", "white") == "yellow" and \
            (L is None or getattr(L, "color", "white") != "yellow"):
        adv.wrong_side = True
        adv.reason = "yellow centre line is on our right - wrong side of the road"
        adv.target_offset = R.y_at(10.0) + lane_width / 2
        return adv
    # 2) cars coming at us in our own lane path (fast closing, ahead, roughly in front)
    for o in objects:
        if getattr(o, "kind", "") == "car" and 5 < o.x < 60 and abs(o.y) < 1.4 and o.vx < -25:
            adv.wrong_side = True
            adv.reason = "oncoming car in our lane"
            adv.target_offset = 3.0
            return adv
    # 3) no usable lane lines: use the road edges
    if lanes.conf < 0.35 and edges.ok and edges.conf > 0.3:
        yl, yr = edges.at(10.0)
        width = yr - yl
        if width >= 2.0 * lane_width * 0.85:          # wide enough for two-way: right half
            adv.two_way = True
            adv.target_offset = yr - min(lane_width / 2 + 0.3, width / 4)
            adv.reason = "no lane lines - keeping to the right half of the road"
        else:                                          # single track: the middle
            adv.two_way = False
            adv.target_offset = (yl + yr) / 2
            adv.reason = "narrow road - keeping to the middle"
    # 4) several lanes our way (white dashed lines on both sides): count them
    n = 1
    if L is not None and getattr(L, "color", "white") == "white" and getattr(L, "dashed", False):
        n += 1
    if R is not None and getattr(R, "color", "white") == "white" and getattr(R, "dashed", False):
        n += 1
    adv.lanes_our_way = n
    return adv
