"""Turning radius: how tight can THIS car turn, at THIS speed - learned while driving.

ER:LC cars steer very differently: a Falcon Stallion, a Bullhorn pickup, a box truck, a
bus and a motorbike have different wheelbases and wheel locks, and Roblox cars also
reduce the wheel angle as speed rises. Planning a 9 m right turn in a bus that
can't do less than 14 m puts it on the curb. Taking a ramp at 90 km/h in a car whose
lock at that speed only manages a 120 m radius puts it in the barrier.

What this does:

1. LEARN. Whenever the car holds a steady steering value (|s| > 0.2 for 0.6 s) we know
   how much the model says it should turn. The real turn rate is measured
   independently, from the road moving under us (visual odometry) and from the lane
   lines (v x road curvature - rate of change of lane heading). Their ratio, per speed
   band, is the car's real steering strength. At full lock it's the real minimum
   turning radius. Values are kept between sessions per vehicle.
2. ANSWER. min_radius(v): the tightest radius possible at speed v (with 10% margin).
   max_curvature(v): its inverse, used to clamp the steering.
3. PLAN TURNS. turn_plan(radius_wanted, v_now): the radius we can actually do, the
   speed we must be down to for it (both lock and sideways grip), and how far before
   the corner to start turning in, so a wide car starts its turn early instead of
   hitting the inside curb.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .vehicle_model import curvature_from_steer

BANDS = (0.0, 3.0, 6.0, 10.0, 15.0, 22.0, 30.0, 40.0, 200.0)       # m/s


@dataclass
class TurnPlan:
    radius: float              # radius to drive (m)
    speed: float               # m/s we must be at (or below) when the turn starts
    entry_offset: float        # metres before the usual turn-in point to start turning
    limited_by: str            # "junction", "lock", "grip"


class TurningModel:
    def __init__(self):
        self.ratio = [1.0] * (len(BANDS) - 1)       # real curvature / model curvature, per band
        self.n = [0] * (len(BANDS) - 1)
        self.hist: deque = deque(maxlen=90)         # (t, v, s, yaw_measured or None)

    def reset_history(self) -> None:
        self.hist.clear()

    @staticmethod
    def band(v: float) -> int:
        for i in range(len(BANDS) - 1):
            if BANDS[i] <= v < BANDS[i + 1]:
                return i
        return len(BANDS) - 2

    def strength(self, v: float) -> float:
        """Learned ratio at speed v (interpolated between bands that have data)."""
        i = self.band(v)
        if self.n[i] >= 3:
            return self.ratio[i]
        # nearest band with data, else 1
        best, bd = 1.0, 99
        for j, n in enumerate(self.n):
            if n >= 3 and abs(j - i) < bd:
                best, bd = self.ratio[j], abs(j - i)
        return best

    # ------------------------------------------------------------------ learning
    def observe(self, t: float, v: float, s: float, yaw_meas: float | None, p) -> None:
        self.hist.append((t, v, s, yaw_meas))
        if len(self.hist) < 36:
            return
        h = [x for x in self.hist if t - x[0] <= 0.6]
        if len(h) < 20 or h[-1][0] - h[0][0] < 0.45:
            return
        ss = np.array([x[2] for x in h])
        vs = np.array([x[1] for x in h])
        ys = [x[3] for x in h if x[3] is not None]
        if len(ys) < 0.6 * len(h):
            return
        if abs(ss.mean()) < 0.2 or ss.std() > 0.06 or vs.mean() < 1.5 or vs.std() > 0.15 * vs.mean() + 0.3:
            return
        y = float(np.median(ys))
        v_m = float(vs.mean())
        k_model = curvature_from_steer(float(ss.mean()), v_m, p)
        k_real = y / v_m
        if abs(k_model) < 1e-3 or np.sign(k_real) != np.sign(k_model):
            return
        ratio = float(np.clip(k_real / k_model, 0.3, 3.0))
        i = self.band(v_m)
        self.n[i] += 1
        k = 1.0 / min(self.n[i] + 1, 20)
        self.ratio[i] = (1 - k) * self.ratio[i] + k * ratio
        self.hist.clear()                    # one sample per steady stretch

    # ------------------------------------------------------------------ answers
    def max_curvature(self, v: float, p) -> float:
        return abs(curvature_from_steer(1.0, v, p)) * self.strength(v)

    def min_radius(self, v: float, p, margin: float = 1.1) -> float:
        k = self.max_curvature(v, p)
        return math.inf if k < 1e-6 else margin / k

    def turn_plan(self, radius_wanted: float, v_now: float, p) -> TurnPlan:
        """Radius and entry speed for a junction turn."""
        a_lat = p.get("turn_lat_accel", 3.0)
        # slowest speed we'd ever turn at (walking pace) and the planned one
        v_turn = p.get("turn_speed_kph", 18.0) / 3.6
        # grip: v <= sqrt(a_lat * R)
        v_grip = math.sqrt(a_lat * radius_wanted)
        v = min(v_turn, v_grip)
        limited = "grip" if v_grip < v_turn else "junction"
        r_min = self.min_radius(v, p)
        radius = radius_wanted
        if r_min > radius_wanted:
            # can't make it at this speed: slow down (lock grows at low speed) ...
            for v_try in np.arange(v, 1.4, -0.4):
                if self.min_radius(float(v_try), p) <= radius_wanted:
                    v = float(v_try)
                    limited = "lock"
                    break
            else:
                # ... and if even crawling isn't enough, take the turn wider
                v = 1.5
                radius = self.min_radius(v, p)
                limited = "lock"
        # a wider-than-planned radius must start earlier so we end up in the target road:
        # the arc's start moves back by (radius - wanted)
        entry = max(0.0, radius - radius_wanted)
        return TurnPlan(float(radius), float(v), float(entry), limited)

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"ratio": self.ratio, "n": self.n}

    def from_dict(self, d: dict) -> None:
        try:
            r = [float(x) for x in d.get("ratio", [])]
            n = [int(x) for x in d.get("n", [])]
            if len(r) == len(self.ratio) and len(n) == len(self.n):
                self.ratio = [float(np.clip(x, 0.3, 3.0)) for x in r]
                self.n = [min(x, 10) for x in n]
        except (TypeError, ValueError):
            pass


class CamYaw:
    """Car turn rate from the camera's turn rate (visual odometry), undoing the chase camera's
    lag: yaw_car ~= yaw_cam + tau * d(yaw_cam)/dt, smoothed."""

    def __init__(self):
        self.prev = None
        self.prev_t = None
        self.out = None
        self.vals: deque = deque(maxlen=6)

    def update(self, t: float, yaw_cam: float | None, conf: float, tau: float) -> float | None:
        if yaw_cam is None or conf < 0.2 or abs(yaw_cam) > 2.5:
            return None
        self.vals.append(yaw_cam)
        y = float(np.median(self.vals))
        if self.prev is not None and self.prev_t is not None and 0.005 < t - self.prev_t < 0.2:
            dy = (y - self.prev) / (t - self.prev_t)
            est = y + tau * float(np.clip(dy, -3.0, 3.0))
        else:
            est = y
        self.prev, self.prev_t = y, t
        self.out = est if self.out is None else 0.6 * self.out + 0.4 * est
        return self.out
