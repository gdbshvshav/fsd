"""The line the car should drive, in car coordinates (x forward, y right, metres).

Normally: the middle of our lane (from the filtered lane). On top of that:

* lateral offset changes (lane change, keep right for an emergency vehicle, moving
  back to our side) are never steps: they're spread over a distance that grows with
  speed using a quintic curve (zero sideways speed AND zero sideways acceleration at
  both ends) - the same shape a smooth human lane change has;
* no lane lines: the road edges give the line (right half of a two-way road);
* nothing at all: keep the current heading for a short while (coast), the planner
  slows down meanwhile and asks for a takeover if it lasts.
"""
from __future__ import annotations

import math

import numpy as np


def quintic(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (10 - 15 * u + 6 * u * u)


class OffsetShaper:
    """Turns a wanted lateral offset into a smooth profile along the road."""

    def __init__(self):
        self.now = 0.0            # offset at the car (m)
        self.target = 0.0
        self.start = 0.0
        self.length = 30.0
        self.done_m = 0.0

    def reset(self) -> None:
        self.__init__()

    def set_target(self, target: float, v: float, min_len: float = 18.0, sec: float = 3.0) -> None:
        if abs(target - self.target) < 0.05:
            return
        self.start = self.now
        self.target = target
        self.length = max(min_len, v * sec * min(1.6, 0.4 + abs(target - self.start) / 3.0))
        self.done_m = 0.0

    def advance(self, dist: float) -> None:
        self.done_m = min(self.length, self.done_m + max(0.0, dist))
        self.now = self.start + (self.target - self.start) * float(quintic(np.array(self.done_m / self.length)))

    def snap_to(self, value: float) -> None:
        """Make `value` the settled offset (no transition in progress)."""
        self.now = self.target = self.start = float(value)
        self.done_m = self.length

    def restart_from(self, value: float) -> None:
        """Continue the transition from `value` (where the car really is now) to the same target
        over the distance that was left."""
        left = max(8.0, self.length - self.done_m)
        self.start = self.now = float(value)
        self.length = left
        self.done_m = 0.0

    def shift(self, delta: float) -> None:
        """The reference itself jumped (e.g. lane re-assigned): move everything with it."""
        self.now += delta
        self.target += delta
        self.start += delta

    def profile(self, xs: np.ndarray) -> np.ndarray:
        u = (self.done_m + xs) / max(self.length, 1e-3)
        return self.start + (self.target - self.start) * quintic(u)


def lane_path(center: tuple[float, float, float], reach: float, offsets: np.ndarray | None = None,
              n: int = 28) -> list[tuple[float, float]]:
    a, b, c = center
    xs = np.linspace(0.0, max(15.0, min(70.0, reach)), n)
    ys = a * xs * xs + b * xs + c
    if offsets is not None:
        ys = ys + offsets(xs) if callable(offsets) else ys + offsets
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def edges_path(edges, target_offset_now: float, reach: float = 40.0, n: int = 20) -> list[tuple[float, float]]:
    """A path parallel to the road's right edge (or mid-road), from RoadEdges."""
    xs = np.linspace(0.0, max(12.0, min(50.0, edges.x_max or reach)), n)
    la, lb, lc = edges.left
    ra, rb, rc = edges.right
    yl = la * xs * xs + lb * xs + lc
    yr = ra * xs * xs + rb * xs + rc
    yl10, yr10 = edges.at(10.0)
    # keep the same relation to the edges as target_offset has at 10 m
    frac = (target_offset_now - yl10) / max(0.5, yr10 - yl10)
    ys = yl + frac * (yr - yl)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def heading_hold_path(yaw_rate: float, v: float, length: float = 25.0, n: int = 14) -> list[tuple[float, float]]:
    """Keep doing what we're doing (constant curvature), for coasting through a gap."""
    k = yaw_rate / max(v, 3.0)
    k = float(np.clip(k, -0.05, 0.05))
    out = []
    for s in np.linspace(0.0, length, n):
        if abs(k) < 1e-5:
            out.append((float(s), 0.0))
        else:
            out.append((float(math.sin(k * s) / k), float((1 - math.cos(k * s)) / k)))
    return out
