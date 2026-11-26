"""Pedals: hit the wanted ACCELERATION, using what we've learned about this car.

The speed planner says "I want +1.2 m/s^2 now" (or -2.5). The learned pedal map
(vehicle_model.PedalMap) says which W/S duty gives that at this speed, and a small
feedback loop on speed fixes what the map gets wrong (hills, a heavy truck).

Speed itself comes from SpeedEstimator: the speedometer is read from the screen, so a
reading can be late, missing, or misread (a "1" read as "7"). Between readings the
estimator integrates the acceleration we expect from the pedal; when a reading arrives
it pulls the estimate toward it, and a reading that jumps impossibly far is ignored
unless it repeats. So the controller always has a smooth, current speed - this is the
fix for "it still thinks it's going 25 when it's at 200".
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .vehicle_model import PedalMap

KPH = 3.6


class SpeedEstimator:
    """Speedometer + pedal model + VISUAL speed (road moving under the car, every frame).

    The visual speed is what fixes the lag: it changes the same frame the car speeds up.
    It is used once it's steady (several frames agreeing), scaled by a factor learned
    against fresh speedometer readings, and it wins over a speedometer reading that has
    stopped changing while the road clearly says otherwise."""

    def __init__(self):
        self.v = 0.0
        self.ok = False
        self.t = None
        self.last_meas = None
        self.last_meas_t = -1e9
        self.pending = None
        self.pending_n = 0
        self.var = 25.0
        self.vis: deque = deque(maxlen=40)      # (t, v_visual_raw, conf)
        self.vis_scale = 1.0
        self.vis_scale_n = 0
        self.vis_used = False
        self.last_change_t = -1e9

    def reset(self) -> None:
        self.__init__()

    def update_visual(self, t: float, v_raw: float | None, conf: float, p=None) -> float:
        """Feed the visual odometry result (raw m/s). Returns the estimate."""
        self.vis_used = False
        if v_raw is None or conf < 0.15:
            return self.v
        self.vis.append((t, v_raw, conf))
        recent = [(tt, v, c) for tt, v, c in self.vis if t - tt <= 0.35]
        if len(recent) < 5:
            return self.v
        vs = np.array([v for _, v, _ in recent])
        med = float(np.median(vs))
        mad = float(np.median(np.abs(vs - med)))
        if mad > 1.2 + 0.05 * abs(med):
            return self.v                          # not steady enough this moment
        z = med * self.vis_scale
        if not self.ok:
            return self.v
        stale = t - self.last_change_t > 0.5
        r = (0.9 if self.vis_scale_n >= 15 else 2.5) ** 2
        if stale and abs(z - self.v) > 3.0:
            r *= 0.3                               # the speedometer is stuck: believe the road
        innov = z - self.v
        if not stale and abs(innov) > 5.0 * np.sqrt(self.var + r) + 4.0:
            return self.v
        k = self.var / (self.var + r)
        self.v = max(0.0, self.v + k * innov)
        self.var *= (1 - k)
        self.vis_used = True
        return self.v

    def vis_used_recent(self, t: float) -> bool:
        return bool(self.vis) and t - self.vis[-1][0] < 0.2 and self.vis_scale_n >= 10

    def learn_visual_scale(self, t: float, v_speedo: float, accel: float) -> None:
        """A fresh speedometer reading while not accelerating hard: compare with the road."""
        if t - self.last_change_t > 0.15 or abs(accel) > 1.5 or v_speedo < 5.0:
            return
        recent = [v for tt, v, c in self.vis if t - tt <= 0.25 and c > 0.25]
        if len(recent) < 4:
            return
        med = float(np.median(recent))
        if med < 3.0:
            return
        ratio = v_speedo / med
        if not 0.5 < ratio < 2.0:
            return
        self.vis_scale_n += 1
        k = 1.0 / min(self.vis_scale_n + 1, 50)
        self.vis_scale = float(np.clip((1 - k) * self.vis_scale + k * ratio, 0.5, 2.0))

    def predict(self, t: float, a_expected: float) -> None:
        if self.t is None:
            self.t = t
            return
        dt = float(np.clip(t - self.t, 0.0, 0.25))
        self.t = t
        if self.ok:
            self.v = max(0.0, self.v + a_expected * dt)
            self.var += (0.8 + 0.3 * abs(a_expected)) ** 2 * dt

    def update(self, t: float, v_meas: float | None, valid: bool) -> float:
        if not valid or v_meas is None:
            if t - self.last_meas_t > 1.5:
                self.var = max(self.var, 9.0)
            return self.v
        if v_meas == self.last_meas:
            # same reading again: the screen hasn't changed - or OCR is slow / stuck. The longer
            # it stays the same, the less it's worth (a real car's speed is never exactly constant
            # for long while the pedals are working)
            stale = t - self.last_change_t
            r = 4.0 + (0.0 if stale < 0.4 else 60.0 * (stale - 0.4))
            if self.vis_used_recent(t):
                r *= 4.0
        else:
            r = 0.6
        if not self.ok:
            self.v, self.ok, self.var = v_meas, True, r
        else:
            jump = abs(v_meas - self.v)
            # 4 sigma + what a car can do in the time since the last reading
            allowed = 4.0 * np.sqrt(self.var + r) + 14.0 * max(0.0, t - self.last_meas_t)
            if jump > max(6.0, allowed):
                # impossible jump: accept only if it repeats (consistent new value)
                if self.pending is not None and abs(self.pending - v_meas) < 3.0:
                    self.pending_n += 1
                else:
                    self.pending, self.pending_n = v_meas, 1
                if self.pending_n < 3:
                    return self.v
                self.v, self.var = v_meas, r
                self.pending, self.pending_n = None, 0
            else:
                k = self.var / (self.var + r)
                self.v += k * (v_meas - self.v)
                self.var *= (1 - k)
                self.pending, self.pending_n = None, 0
        if v_meas != self.last_meas:
            self.last_meas_t = t
            self.last_change_t = t
        self.last_meas = v_meas
        self.v = max(0.0, self.v)
        return self.v


class LongitudinalController:
    def __init__(self):
        self.map = PedalMap()
        self.speed = SpeedEstimator()
        self.i = 0.0
        self.pedal = 0.0
        self.debug: dict = {}

    def reset(self) -> None:
        self.i = 0.0
        self.pedal = 0.0

    def step(self, v: float, v_valid: bool, v_target: float, a_target: float, holding: bool, p, dt: float,
             t: float, blind_stop: bool = False) -> float:
        """v: estimated speed (m/s). Returns pedal duty -1 (full S) .. +1 (full W)."""
        if not v_valid:
            # blind: creep or brake only
            cmd = -0.6 if (blind_stop or v_target < 1.0) else 0.3
            return self._out(cmd, dt)
        if holding and v < 0.25:
            self.i = 0.0
            return self._out(0.0, dt)                 # never press S at a standstill (it would reverse)
        err = v_target - v
        self.i = float(np.clip(self.i + err * dt, -6.0, 6.0))
        if abs(err) > 6.0:
            self.i *= 0.98
        a_cmd = a_target + p.get("speed_kp", 0.7) * err + p.get("speed_ki", 0.08) * self.i
        a_cmd = float(np.clip(a_cmd, -p["max_decel"] if a_target > -p["max_decel"] else -15.0, 6.0))
        cmd = self.map.pedal_for(a_cmd, v)
        # coast band: tiny accelerations near target -> no key at all (calmer, like a person)
        if abs(err) < p.get("coast_band_kph", 2.0) / KPH and abs(a_cmd - self.map.accel_for(0.0, v)) < 0.25:
            cmd = 0.0
        if cmd < 0 and v < 0.25:                      # ~1 km/h: let it roll to a stop, never reverse
            cmd = 0.0
        if v_target < 0.05 and 0.25 <= v < 2.0:
            cmd = min(cmd, -0.35)                     # last metres before a stop: firm, no creeping
        self.debug = {"a_cmd": a_cmd, "err": err, "i": self.i}
        return self._out(cmd, dt)

    def _out(self, cmd: float, dt: float) -> float:
        k = min(1.0, dt * 15.0)
        self.pedal += (float(np.clip(cmd, -1, 1)) - self.pedal) * k
        return self.pedal
