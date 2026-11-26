"""A model of the Roblox car, used to predict what it will do with our keys.

Steering: A/D move a hidden steering value s (-1..1) at a fixed rate; it springs back
when released (see control/steer_model.py). The wheels turn s x full lock, and full
lock shrinks with speed. With the kinematic bicycle model that gives a path curvature

    kappa = tan(s * lock(v)) / wheelbase          (1/m, + = right)

Pedals: W gives an acceleration that falls with speed (engine + drag), S brakes hard,
nothing coasts. ER:LC cars differ a lot (a Bullhorn pursuit car vs a bus), so the
pedal map is LEARNED online: for each speed band we keep a running estimate of the
acceleration we get at full W, at coast, and at full S. The longitudinal controller
inverts that map to pick the pedal duty for the acceleration it wants.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

KPH = 3.6


def lock_rad(v: float, lock_deg: float, falloff_ms: float) -> float:
    return math.radians(lock_deg) / (1.0 + abs(v) / max(1.0, falloff_ms))


def curvature_from_steer(s: float, v: float, p) -> float:
    lock = lock_rad(v, p["steer_lock_deg"], p["steer_falloff_ms"])
    return math.tan(max(-1.0, min(1.0, s)) * lock) / max(1.0, p["wheelbase_m"])


def steer_for_curvature(kappa: float, v: float, p) -> float:
    lock = lock_rad(v, p["steer_lock_deg"], p["steer_falloff_ms"])
    if lock < 1e-6:
        return 0.0
    return max(-1.0, min(1.0, math.atan(p["wheelbase_m"] * kappa) / lock))


def max_curvature(v: float, p) -> float:
    """The tightest curve the car can do at this speed (full lock)."""
    return abs(curvature_from_steer(1.0, v, p))


def steer_time_constant(p) -> float:
    """Rough first-order time constant of the key-driven steering (s).

    The steering value moves at `rise` per second, so going from 0 to a typical
    correction (~0.25) takes 0.25/rise s; we use that as the lag the controllers
    must anticipate. Never below one frame."""
    return float(np.clip(0.25 / max(0.5, p["steer_rise_per_s"]), 1 / 60, 0.3))


@dataclass
class PedalBand:
    v_lo: float
    v_hi: float
    full: float           # m/s^2 at full W
    coast: float          # m/s^2 with no key (negative: drag + rolling)
    brake: float          # m/s^2 at full S (negative)
    n_full: int = 0
    n_coast: int = 0
    n_brake: int = 0


class PedalMap:
    """Learned acceleration for full W / coast / full S in speed bands (m/s)."""

    EDGES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 36.0, 44.0, 60.0, 200.0)

    def __init__(self, accel: float = 5.0, top_ms: float = 52.0, brake: float = 9.0):
        self.bands: list[PedalBand] = []
        for lo, hi in zip(self.EDGES[:-1], self.EDGES[1:]):
            mid = min((lo + hi) / 2, top_ms)
            full = max(0.3, accel * (1.0 - (mid / top_ms) ** 2))
            coast = -(0.25 + 0.0009 * mid * mid)
            self.bands.append(PedalBand(lo, hi, full, coast, -brake))
        self._hist: deque = deque(maxlen=90)       # (t, v, pedal)

    def band(self, v: float) -> PedalBand:
        for b in self.bands:
            if b.v_lo <= v < b.v_hi:
                return b
        return self.bands[-1]

    def accel_for(self, pedal: float, v: float) -> float:
        """Expected acceleration (m/s^2) for a pedal duty in -1..1 (linear in duty)."""
        b = self.band(v)
        if pedal >= 0:
            return b.coast + pedal * (b.full - b.coast)
        return b.coast + (-pedal) * (b.brake - b.coast)

    def pedal_for(self, accel: float, v: float) -> float:
        """Pedal duty that gives `accel` at speed v."""
        b = self.band(v)
        if accel >= b.coast:
            span = max(0.2, b.full - b.coast)
            return float(np.clip((accel - b.coast) / span, 0.0, 1.0))
        span = max(0.5, b.coast - b.brake)
        return -float(np.clip((b.coast - accel) / span, 0.0, 1.0))

    def observe(self, t: float, v: float, pedal: float, valid: bool = True) -> None:
        """Feed (time, speed m/s, pedal duty actually sent). Learns from stretches where the
        pedal stayed at one extreme (full W, coast, full S) for >= 0.5 s."""
        if not valid:
            self._hist.clear()
            return
        self._hist.append((t, v, pedal))
        if len(self._hist) < 12:
            return
        t0, v0, _ = self._hist[0]
        if t - t0 < 0.5:
            return
        ps = [h[2] for h in self._hist]
        kind = None
        if min(ps) > 0.92:
            kind = "full"
        elif max(abs(x) for x in ps) < 0.05:
            kind = "coast"
        elif max(ps) < -0.92:
            kind = "brake"
        if kind is None:
            # keep only the last 0.5 s
            while self._hist and t - self._hist[0][0] > 0.5:
                self._hist.popleft()
            return
        ts = np.array([h[0] for h in self._hist])
        vs = np.array([h[1] for h in self._hist])
        if vs.max() - vs.min() > 40:                 # speedometer glitch
            self._hist.clear()
            return
        a = float(np.polyfit(ts - ts[0], vs, 1)[0])
        b = self.band(float(vs.mean()))
        if kind == "full" and v > 0.5:
            b.n_full += 1
            k = 1.0 / min(b.n_full + 1, 12)
            b.full = (1 - k) * b.full + k * float(np.clip(a, 0.1, 15.0))
        elif kind == "coast" and v > 1.0:
            b.n_coast += 1
            k = 1.0 / min(b.n_coast + 1, 12)
            b.coast = (1 - k) * b.coast + k * float(np.clip(a, -4.0, 0.3))
        elif kind == "brake" and v > 2.0:
            b.n_brake += 1
            k = 1.0 / min(b.n_brake + 1, 12)
            b.brake = (1 - k) * b.brake + k * float(np.clip(a, -25.0, -1.0))
        self._hist.clear()

    def summary(self) -> str:
        parts = []
        for b in self.bands:
            if b.n_full or b.n_coast or b.n_brake:
                parts.append(f"{b.v_lo * KPH:.0f}-{b.v_hi * KPH:.0f}: W {b.full:+.1f} coast {b.coast:+.1f} "
                             f"S {b.brake:+.1f}")
        return "; ".join(parts) or "default"


class SteerSim:
    """Simulates the steering value from the keys we choose (same as the actuator does) so the
    controllers can look a few frames ahead."""

    def __init__(self, rise: float, ret: float):
        self.rise = rise
        self.ret = ret

    def step(self, s: float, key: int, dt: float) -> float:
        if key > 0:
            return min(1.0, s + (self.rise if s >= 0 else max(self.rise, self.ret)) * dt)
        if key < 0:
            return max(-1.0, s - (self.rise if s <= 0 else max(self.rise, self.ret)) * dt)
        if s > 0:
            return max(0.0, s - self.ret * dt)
        return min(0.0, s + self.ret * dt)

    def time_to(self, s: float, target: float) -> float:
        """Seconds to move the steering value from s to target."""
        d = target - s
        if abs(d) < 1e-6:
            return 0.0
        towards_zero = (s > 0 and d < 0) or (s < 0 and d > 0)
        rate = max(self.rise, self.ret) if towards_zero else self.rise
        return abs(d) / max(rate, 1e-3)
