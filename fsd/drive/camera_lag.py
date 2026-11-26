"""The chase camera swings AFTER the car - undo that before trusting the lanes.

In ER:LC (and the simulator) the camera follows the car's heading with a lag. While
the car turns right, the camera still points a little to the left of where the car
points, so everything ahead appears rotated to the RIGHT: the lane looks like it bends
more than it does and the car looks like it's pointing away from it. Steering on that
picture cuts every corner (that's what the tight town bend in the simulator showed:
1.5 m of error 10 m ahead).

The camera's heading follows the car's like a first-order lag with time constant tau
(param cam_yaw_lag_s), so the angle between them obeys

    d(delta)/dt = yaw_rate - delta / tau

We integrate that with our (filtered) yaw rate and rotate every lane measurement back
by -delta about the point the camera orbits (the car's rear axle).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np


class CameraLag:
    """tau is LEARNED while driving (different cars / camera settings / frame delay lag
    differently): in a steady bend, a camera rotation we didn't undo makes the car sit
    off-centre toward the inside of the bend (it follows a lane that looks rotated). The
    sideways offset of the lane beside the car, divided by the lever arm, tells how much
    rotation is missing; divided by the turn rate, how much lag."""

    def __init__(self, tau0: float | None = None):
        self.delta = 0.0
        self.t = None
        self.tau_learned: float | None = tau0
        self.samples = 0
        self._yaw_hist: list[float] = []

    def reset(self) -> None:
        tau = self.tau_learned
        n = self.samples
        self.__init__(tau)
        self.samples = n

    def tau(self, p) -> float:
        base = p.get("cam_yaw_lag_s", 0.25)
        if p.get("cam_lag_learn", 1.0) > 0.5 and self.tau_learned is not None:
            return self.tau_learned
        return base

    def learn(self, p, c_offset: float, yaw: float, v: float, conf: float, lever: float) -> None:
        """c_offset: lane centre beside the car (m, + right), from the filtered lanes."""
        if p.get("cam_lag_learn", 1.0) <= 0.5:
            return
        self._yaw_hist.append(yaw)
        self._yaw_hist = self._yaw_hist[-60:]
        if conf < 0.6 or v < 5.0 or abs(yaw) < 0.08 or len(self._yaw_hist) < 60:
            return
        ys = self._yaw_hist
        if min(ys) * max(ys) <= 0 or (max(ys) - min(ys)) > 0.5 * abs(sum(ys) / len(ys)):
            return                                   # not a steady bend
        if abs(c_offset) > 1.5:
            return
        tau = self.tau(p)
        d_err = -c_offset / max(3.0, lever)
        tau_needed = (self.delta + d_err) / yaw
        tau_needed = float(np.clip(tau_needed, 0.0, 0.9))
        self.tau_learned = float(np.clip(tau + 0.004 * (tau_needed - tau), 0.05, 0.8))
        self.samples += 1

    def update(self, t: float, yaw: float, tau: float) -> float:
        if self.t is None or tau <= 1e-3:
            self.t = t
            self.delta = 0.0 if tau <= 1e-3 else self.delta
            return self.delta
        dt = float(np.clip(t - self.t, 0.0, 0.2))
        self.t = t
        # exact solution of the first-order lag for this step
        k = float(np.exp(-dt / tau))
        self.delta = self.delta * k + yaw * tau * (1.0 - k)
        self.delta = float(np.clip(self.delta, -0.5, 0.5))
        return self.delta

    @staticmethod
    def _rot(coeffs, delta: float, pivot_back: float):
        a, b, c = coeffs
        # y_true(x) = y_seen(x) - delta * (x + pivot_back)
        return (a, b - delta, c - delta * pivot_back)

    def correct(self, lanes, pivot_back: float = 3.7):
        """Lane measurement in the car's true frame."""
        d = self.delta
        if lanes is None or abs(d) < 1e-4 or lanes.conf <= 0:
            return lanes

        def mv(line):
            if line is None:
                return None
            return replace(line, coeffs=self._rot(line.coeffs, d, pivot_back))
        out = replace(lanes, left=mv(lanes.left), right=mv(lanes.right), left2=mv(lanes.left2),
                      right2=mv(lanes.right2), center=self._rot(lanes.center, d, pivot_back))
        a, b, c = out.center
        out.offset = c
        out.heading = float(np.arctan(b))
        out.curvature = float(2 * a / (1 + (2 * a * 10 + b) ** 2) ** 1.5)
        return out
