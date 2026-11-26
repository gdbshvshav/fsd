"""Steering and speed controllers.

Steering = pure pursuit (aim at a point ahead on the path) + feed-forward from
the road's curve - damping on how fast we're already turning. The damping term is
what stops v3's "overshoot the turn, then swerve" problem. The lookahead grows
with speed so the car is calm on the highway and tight in town.

Speed = PI controller with feed-forward, separate gentle brake. If the
speedometer can't be read, it falls back to cautious fixed pedal values.
"""
from __future__ import annotations

import numpy as np

from ..world import Mode, Plan, World

KPH = 3.6


class LateralController:
    """Pure pursuit + curve feed-forward + yaw damping, in curvature (1/m).

    The wanted curvature is turned into a Roblox steering value (-1..1) with the
    chassis model (wheelbase, full lock that shrinks with speed). The actuator's
    SteerObserver then picks the A/D presses that produce exactly that value.
    """

    def __init__(self):
        from .steer_adapt import SteerAdapter
        self.out = 0.0
        self.adapt = SteerAdapter()

    def reset(self) -> None:
        self.out = 0.0

    def observe(self, t: float, speed_kph: float, s_value: float, yaw_rate: float, p, lanes_ok: bool) -> None:
        """Feed the car's measured response so steering strength adapts to this vehicle."""
        if p.get("steer_adapt", 1.0) > 0.5:
            self.adapt.update(t, speed_kph / KPH, s_value, yaw_rate, p, trustworthy=lanes_ok)

    def step(self, w: World, plan: Plan, p, dt: float) -> float:
        from .steer_model import steer_value_for_curvature
        path = plan.path
        if not path or len(path) < 2:
            return self._limit(0.0, p, dt)
        v = max(0.0, w.speed_kph / KPH)
        ld = float(np.clip(p["lookahead_base_m"] + p["lookahead_gain_s"] * v, 3.0, p["lookahead_max_m"]))
        xs = np.array([q[0] for q in path])
        ys = np.array([q[1] for q in path])
        d = np.hypot(xs, ys)
        if d.max() < ld:
            ld = float(d.max())
        i = int(np.argmax(d >= ld)) if np.any(d >= ld) else len(path) - 1
        tx, ty = xs[i], ys[i]
        L2 = tx * tx + ty * ty
        kappa = 2.0 * ty / max(L2, 1.0)                     # pure-pursuit curvature (+ = right)

        # feed-forward: the road's own curve; feedback: pure pursuit's CORRECTION to it.
        # (Adding the two outright double-counts the bend and cuts every corner.)
        turning = plan.mode == Mode.TURNING
        road_k = 0.0 if turning else w.lanes.curvature
        if turning or w.lanes.conf <= 0:
            k_cmd = p["steer_gain"] * kappa
        else:
            k_cmd = p["steer_ff_gain"] * road_k + p["steer_gain"] * (kappa - road_k)
        # damping: if we're already turning faster than wanted, ease off (stops overshoot / swerve)
        if v > 2.0:
            k_cmd -= p["steer_damping"] * (w.yaw_rate / v - k_cmd)
        k_cmd /= (1.0 + p["steer_speed_scale"] * w.speed_kph)
        if p.get("steer_adapt", 1.0) > 0.5:
            k_cmd /= self.adapt.gain                     # this car turns more/less than the model
        u = steer_value_for_curvature(k_cmd, v, p["wheelbase_m"], p["steer_lock_deg"], p["steer_falloff_ms"])
        return self._limit(float(np.clip(u, -1, 1)), p, dt)

    def _limit(self, cmd: float, p, dt: float) -> float:
        step = p["steer_rate_limit"] * max(dt, 1e-3)
        self.out += float(np.clip(cmd - self.out, -step, step))
        return self.out


class SpeedController:
    def __init__(self):
        self.i = 0.0
        self.out = 0.0

    def reset(self) -> None:
        self.i = 0.0
        self.out = 0.0

    def step(self, w: World, plan: Plan, p, dt: float) -> float:
        target = plan.target_speed_kph
        if not w.speed_valid:
            # Blind: creep or brake only.
            if target < 3 or plan.mode in (Mode.STOPPING, Mode.STOPPED, Mode.STOP_WAIT, Mode.FAULT):
                return self._smooth(-0.6, dt)
            return self._smooth(0.3, dt)
        v = w.speed_kph
        err = target - v
        if plan.stop_at is not None and plan.stop_at < 900 and v > 1:
            # how hard must we brake to stop in time?
            vv = v / KPH
            need = vv * vv / (2 * max(plan.stop_at, 0.3))
            if need > p["comfort_decel"] * 0.6:
                self.i = 0.0
                brake = np.clip(need / p["max_decel"] * 1.3, 0.15, 1.0)
                return self._smooth(-float(brake), dt)
        if target < 0.5 and v < 1.5:
            self.i = 0.0
            return self._smooth(0.0, dt)                     # holding: never press S at a stop
        band = p["coast_band_kph"]
        if err > -band:
            self.i = float(np.clip(self.i + err * dt, -40, 120))
            thr = p["throttle_ff"] * target + p["throttle_kp"] * err + p["throttle_ki"] * self.i
            return self._smooth(float(np.clip(thr, 0.0, 1.0)), dt)
        self.i *= 0.9
        brake = p["brake_kp"] * (-err - band)
        return self._smooth(-float(np.clip(brake, 0.0, 1.0)), dt)

    def _smooth(self, cmd: float, dt: float) -> float:
        k = min(1.0, dt * 12.0)
        self.out += (cmd - self.out) * k
        return self.out
