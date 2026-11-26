"""Roads for the simulator.

A Road is a centre line built from straight and curved pieces, sampled every
0.5 m. Positions along it are (s, d): s = metres along the road, d = metres to
the RIGHT of the reference line. The reference line is the road's left edge
(where ER:LC paints the solid yellow line on a divided highway, or the
centre double-yellow on a two-way street).

Cross-section, left to right (d grows to the right):

    ... median / opposite side | yellow edge | lane 0 | dashed | lane 1 | ... | white edge | shoulder | grass

Everything is in metres; world coordinates are X east, Y north, Z up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Piece:
    kind: str            # "straight" | "arc"
    length: float        # m
    radius: float = 0.0  # m, + = turns right, - = turns left (arc only)


@dataclass
class Marking:
    d: float                     # lateral position of the marking centre (m from reference line)
    width: float = 0.15
    color: str = "white"         # white | yellow
    dash: tuple | None = None    # (on_m, off_m) or None = solid
    double: bool = False         # two parallel lines 0.3 m apart (centre of a two-way street)


@dataclass
class Strip:
    d0: float
    d1: float
    surface: str                 # concrete | asphalt | grass | dirt | sidewalk | shoulder


@dataclass
class Road:
    pieces: list[Piece]
    lanes: int = 2
    lane_width: float = 4.5
    start: tuple = (0.0, 0.0)            # world position of s=0 on the reference line
    heading: float = 0.0                 # rad, 0 = +Y (north), + = clockwise (to the right)
    surface: str = "concrete"            # highway concrete (light grey) or town asphalt
    divided: bool = True                 # yellow left edge + median (highway) vs. two-way street
    median_width: float = 14.0
    shoulder: float = 1.2
    sidewalk: bool = False
    step: float = 0.5
    xs: np.ndarray = field(init=False)
    ys: np.ndarray = field(init=False)
    hs: np.ndarray = field(init=False)   # heading at each sample
    ss: np.ndarray = field(init=False)
    markings: list[Marking] = field(init=False)
    strips: list[Strip] = field(init=False)

    def __post_init__(self):
        self._sample()
        self._cross_section()

    # ------------------------------------------------------------------ geometry
    def _sample(self) -> None:
        x, y = self.start
        h = self.heading
        xs, ys, hs = [x], [y], [h]
        for p in self.pieces:
            n = max(1, int(round(p.length / self.step)))
            ds = p.length / n
            for _ in range(n):
                if p.kind == "arc" and p.radius != 0:
                    h_mid = h + ds / (2 * p.radius)
                    x += ds * math.sin(h_mid)
                    y += ds * math.cos(h_mid)
                    h += ds / p.radius
                else:
                    x += ds * math.sin(h)
                    y += ds * math.cos(h)
                xs.append(x)
                ys.append(y)
                hs.append(h)
        self.xs, self.ys, self.hs = np.array(xs), np.array(ys), np.array(hs)
        self.ss = np.arange(len(xs)) * self.step

    @property
    def length(self) -> float:
        return float(self.ss[-1])

    def _cross_section(self) -> None:
        W, N = self.lane_width, self.lanes
        self.markings = []
        self.strips = []
        road_w = W * N
        if self.divided:
            self.markings.append(Marking(0.25, 0.18, "yellow"))
            for i in range(1, N):
                self.markings.append(Marking(i * W, 0.15, "white", (3.0, 9.0)))
            self.markings.append(Marking(road_w - 0.25, 0.18, "white"))
            # our carriageway, shoulders, median, the opposite carriageway, roadside grass
            self.strips += [
                Strip(-self.shoulder - self.median_width - road_w - self.shoulder, -self.shoulder - self.median_width,
                      self.surface),
                Strip(-self.shoulder - self.median_width, -self.shoulder, "grass"),
                Strip(-self.shoulder, 0.0, self.surface),
                Strip(0.0, road_w, self.surface),
                Strip(road_w, road_w + self.shoulder, self.surface),
                Strip(road_w + self.shoulder, road_w + self.shoulder + 60.0, "grass"),
                Strip(-self.shoulder - self.median_width - road_w - self.shoulder - 60.0,
                      -self.shoulder - self.median_width - road_w - self.shoulder, "grass"),
            ]
            # the opposite carriageway's own markings (seen across the median)
            off = -self.shoulder - self.median_width - self.shoulder
            self.markings.append(Marking(off - 0.25, 0.18, "yellow"))
            for i in range(1, N):
                self.markings.append(Marking(off - i * W, 0.15, "white", (3.0, 9.0)))
            self.markings.append(Marking(off - road_w + 0.25, 0.18, "white"))
        else:
            # two-way street: our lanes to the right of a double yellow, theirs to the left
            self.markings.append(Marking(0.0, 0.12, "yellow", None, double=True))
            for i in range(1, N):
                self.markings.append(Marking(i * W, 0.15, "white", (3.0, 9.0)))
                self.markings.append(Marking(-i * W, 0.15, "white", (3.0, 9.0)))
            self.markings.append(Marking(road_w - 0.3, 0.15, "white"))
            self.markings.append(Marking(-road_w + 0.3, 0.15, "white"))
            side = "sidewalk" if self.sidewalk else "grass"
            self.strips += [
                Strip(-road_w, road_w, self.surface),
                Strip(road_w, road_w + 3.0, side),
                Strip(-road_w - 3.0, -road_w, side),
                Strip(road_w + 3.0, road_w + 60.0, "grass"),
                Strip(-road_w - 60.0, -road_w - 3.0, "grass"),
            ]

    def lane_center(self, lane: int) -> float:
        return (lane + 0.5) * self.lane_width

    # ------------------------------------------------------------------ queries
    def index_at(self, s: float) -> int:
        return int(np.clip(round(s / self.step), 0, len(self.ss) - 1))

    def point(self, s: float, d: float = 0.0) -> tuple[float, float, float]:
        """World (x, y) and heading at road position (s, d)."""
        i = self.index_at(s)
        h = self.hs[i]
        x = self.xs[i] + d * math.cos(h)
        y = self.ys[i] - d * math.sin(h)
        return float(x), float(y), float(h)

    def points(self, s: np.ndarray, d: float | np.ndarray) -> np.ndarray:
        idx = np.clip(np.round(s / self.step).astype(int), 0, len(self.ss) - 1)
        h = self.hs[idx]
        d = np.broadcast_to(d, idx.shape)
        return np.stack([self.xs[idx] + d * np.cos(h), self.ys[idx] - d * np.sin(h)], 1)

    def project(self, x: float, y: float, s_hint: float | None = None, window: float = 40.0) -> tuple[float, float]:
        """World point -> (s, d) on this road (nearest sample; local search around s_hint)."""
        if s_hint is None:
            lo, hi = 0, len(self.ss)
        else:
            lo = max(0, int((s_hint - window) / self.step))
            hi = min(len(self.ss), int((s_hint + window) / self.step) + 1)
        dx = self.xs[lo:hi] - x
        dy = self.ys[lo:hi] - y
        i = lo + int(np.argmin(dx * dx + dy * dy))
        h = self.hs[i]
        # signed lateral offset: + = to the right of the heading
        ex, ey = x - self.xs[i], y - self.ys[i]
        d = ex * math.cos(h) - ey * math.sin(h)
        along = ex * math.sin(h) + ey * math.cos(h)
        return float(self.ss[i] + along), float(d)

    def curvature_at(self, s: float) -> float:
        i = self.index_at(s)
        j = min(len(self.hs) - 1, i + 4)
        if j == i:
            return 0.0
        return float((self.hs[j] - self.hs[i]) / ((j - i) * self.step))


# ---------------------------------------------------------------------- builders
def straight(length: float) -> Piece:
    return Piece("straight", length)


def arc(length: float, radius: float) -> Piece:
    """radius > 0 turns right, < 0 turns left."""
    return Piece("arc", length, radius)


def highway(pieces: list[Piece], lanes: int = 2, **kw) -> Road:
    return Road(pieces, lanes=lanes, lane_width=kw.pop("lane_width", 4.6), surface="concrete", divided=True, **kw)


def town_street(pieces: list[Piece], lanes: int = 1, **kw) -> Road:
    return Road(pieces, lanes=lanes, lane_width=kw.pop("lane_width", 4.2), surface="asphalt", divided=False,
                sidewalk=True, **kw)
