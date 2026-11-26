"""How Roblox keyboard steering really works, and how to drive it precisely.

In Roblox car chassis (A-Chassis style, like ER:LC's cars) the A/D keys don't
set a steering angle. They move a hidden steering value s in -1..1:

    holding D   s rises toward +1 at `rise` per second
    holding A   s falls toward -1 at `rise` per second
    no key      s springs back toward 0 at `ret` per second (usually faster)

and the wheels turn by s x (full-lock angle at the current speed).

So "press D 10% of the time" does NOT mean 10% steering: with a fast spring
back it means almost nothing - which is why the old controller ran wide on
highway bends. Instead, SteerObserver keeps its own estimate of s (it knows
exactly which keys it pressed), and every frame picks D, A or nothing so the
estimate lands as close as possible to the steering value the controller wants.
That gives smooth, accurate steering from on/off keys.

`rise` and `ret` differ between vehicles; they are parameters
(steer_rise_per_s, steer_return_per_s) and can be measured automatically
(see learning/steer_id.py).
"""
from __future__ import annotations

import math


class SteerObserver:
    def __init__(self, rise: float = 3.9, ret: float = 6.0):
        self.rise = rise
        self.ret = ret
        self.s = 0.0                # estimated steering value, -1..1

    def reset(self) -> None:
        self.s = 0.0

    def predict(self, key: int, dt: float) -> float:
        """Steering value after holding `key` (-1 = A, 0 = none, +1 = D) for dt."""
        s = self.s
        if key > 0:
            s = min(1.0, s + self.rise * dt) if s >= 0 else min(1.0, s + max(self.rise, self.ret) * dt)
        elif key < 0:
            s = max(-1.0, s - self.rise * dt) if s <= 0 else max(-1.0, s - max(self.rise, self.ret) * dt)
        else:
            if s > 0:
                s = max(0.0, s - self.ret * dt)
            elif s < 0:
                s = min(0.0, s + self.ret * dt)
        return s

    def choose(self, target: float, dt: float) -> int:
        """Key for this frame that brings the steering value closest to target."""
        target = max(-1.0, min(1.0, target))
        best_k, best_e = 0, abs(self.predict(0, dt) - target)
        for k in (1, -1):
            e = abs(self.predict(k, dt) - target)
            if e < best_e - 1e-9:
                best_k, best_e = k, e
        return best_k

    def apply(self, key: int, dt: float) -> None:
        self.s = self.predict(key, dt)


def max_steer_rad(speed_ms: float, lock_deg: float, falloff_ms: float) -> float:
    """Full-lock wheel angle at a speed (Roblox chassis reduce lock as speed rises)."""
    return math.radians(lock_deg) / (1.0 + abs(speed_ms) / max(1.0, falloff_ms))


def steer_value_for_curvature(kappa: float, speed_ms: float, wheelbase: float, lock_deg: float,
                              falloff_ms: float) -> float:
    """Steering value (-1..1) that makes the car follow curvature kappa (1/m, + = right)."""
    delta = math.atan(wheelbase * kappa)
    lim = max_steer_rad(speed_ms, lock_deg, falloff_ms)
    return max(-1.0, min(1.0, delta / lim)) if lim > 1e-6 else 0.0
