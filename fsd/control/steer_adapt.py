"""Learns how strongly THIS car really steers, while driving (no AI).

Every ER:LC vehicle steers differently: a sports car turns sharply, a bus or a
fire truck barely turns and its steering builds up slowly. The steering model
(control/steer_model.py) assumes one typical car. If the real car turns less
than expected, FSD under-steers and runs wide on every bend; if it turns more,
it weaves.

So we compare, all the time:

    expected curvature = what our steering value should give (chassis model)
    measured curvature = yaw rate (from how fast the picture slides sideways) / speed

and keep a running least-squares ratio  gain = measured / expected  (with
forgetting, so it follows a vehicle change). The lateral controller divides its
wanted steering by this gain. The camera's view lags the car a little, so the
expected value is taken from `delay_s` ago before comparing.

Keyboard steering comes in short pulses, so both sides are AVERAGED over a
window first (comparing a pulse with a smooth measurement would always make the
car look weaker than it is).
"""
from __future__ import annotations

import math
from collections import deque


class SteerAdapter:
    def __init__(self, delay_s: float = 0.3, forget: float = 0.995, lo: float = 0.4, hi: float = 2.5,
                 window_s: float = 0.6):
        self.delay_s = delay_s
        self.window_s = window_s
        self.meas: deque[tuple[float, float]] = deque(maxlen=240)
        self.forget = forget
        self.lo, self.hi = lo, hi
        self.hist: deque[tuple[float, float]] = deque(maxlen=240)
        self.num = 0.0
        self.den = 0.0
        self.samples = 0
        self.gain = 1.0

    def reset(self) -> None:
        self.__init__(self.delay_s, self.forget, self.lo, self.hi, self.window_s)

    def expected_curvature(self, s_value: float, speed_ms: float, p) -> float:
        lock = math.radians(p["steer_lock_deg"]) / (1.0 + abs(speed_ms) / max(1.0, p["steer_falloff_ms"]))
        return math.tan(s_value * lock) / max(1.0, p["wheelbase_m"])

    @staticmethod
    def _mean(buf, t0: float, t1: float) -> float | None:
        vals = [k for (tt, k) in buf if t0 <= tt <= t1]
        return sum(vals) / len(vals) if len(vals) >= 3 else None

    def _delayed(self, t: float) -> float | None:
        target = t - self.delay_s
        if not self.hist or self.hist[0][0] > target:
            return None
        prev = self.hist[0]
        for item in self.hist:
            if item[0] >= target:
                t0, k0 = prev
                t1, k1 = item
                if t1 - t0 < 1e-6:
                    return k1
                a = (target - t0) / (t1 - t0)
                return k0 + a * (k1 - k0)
            prev = item
        return self.hist[-1][1]

    def update(self, t: float, speed_ms: float, s_value: float, yaw_rate: float, p,
               trustworthy: bool = True) -> float:
        """Feed one frame. s_value: the estimated Roblox steering value now; yaw_rate: rad/s
        measured from the picture (+ = right). Returns the current gain."""
        self.hist.append((t, self.expected_curvature(s_value, speed_ms, p)))
        if speed_ms > 1.0:
            self.meas.append((t, yaw_rate / speed_ms))
        if not trustworthy or speed_ms < 4.0:
            return self.gain
        if self.hist[0][0] > t - self.delay_s - self.window_s:
            return self.gain                         # not enough history yet
        k_exp = self._mean(self.hist, t - self.delay_s - self.window_s, t - self.delay_s)
        k_meas = self._mean(self.meas, t - self.window_s, t)
        if k_exp is None or k_meas is None or abs(k_exp) < 0.0015:
            return self.gain
        if abs(k_meas) > 0.3:
            return self.gain                         # glitch (camera jump, collision)
        self.num = self.forget * self.num + k_meas * k_exp
        self.den = self.forget * self.den + k_exp * k_exp
        self.samples += 1
        if self.samples >= 20 and self.den > 1e-6:
            g = self.num / self.den
            self.gain = max(self.lo, min(self.hi, g))
        return self.gain
