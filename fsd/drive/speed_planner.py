"""How fast to go - decided by looking AHEAD, and changed smoothly.

The old speed logic reacted to the bend we were already in and jumped its target
the moment something changed, so the car braked late in curves and lurched when a
car pulled out. A good driver instead reads the road ahead:

1. Speed limit along the road ahead (metres 0..~200):
      * the planner's target (limit, following, caution...)
      * every bend: v <= sqrt(sideways_accel / curvature) at that point
      * a stop point: v = 0 there (red light, stop sign, car stopped ahead)
      * what we can see: we must be able to stop within the visible road
2. Backward pass: from each limit, how fast may we be going EARLIER so that we can
   slow down to it at a comfortable deceleration?  v(s) <= sqrt(v(s+ds)^2 + 2 a ds)
3. The speed we want now is the start of that profile; the acceleration we want is
   what gets us there over the next second.
4. Jerk limit: the wanted acceleration itself changes gradually (no lurching), but
   braking for something close is always allowed to ramp up fast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

KPH = 3.6


@dataclass
class SpeedPlan:
    v_target: float = 0.0          # m/s wanted now
    a_target: float = 0.0          # m/s^2 wanted now (after jerk limit)
    limiting: str = ""             # what limits the speed most
    profile_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    profile_v: np.ndarray = field(default_factory=lambda: np.zeros(0))


def lane_curvature_along(center: tuple[float, float, float], xs: np.ndarray) -> np.ndarray:
    a, b, _ = center
    return 2 * a / (1 + (2 * a * xs + b) ** 2) ** 1.5


def path_curvature_profile(path, s_grid: np.ndarray) -> np.ndarray:
    """|curvature| of the planned path at arc lengths s_grid (0 beyond its end)."""
    if not path or len(path) < 3:
        return np.zeros_like(s_grid)
    P = np.asarray(path, np.float64)
    seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 2.0:
        return np.zeros_like(s_grid)
    ss = np.arange(0.0, s[-1], 1.0)
    xs = np.interp(ss, s, P[:, 0])
    ys = np.interp(ss, s, P[:, 1])
    ang = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
    if len(ang) >= 7:
        ang = np.convolve(np.pad(ang, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    k = np.abs(np.gradient(ang, ss)) if len(ss) >= 3 else np.zeros_like(ss)
    out = np.interp(s_grid, ss, k, right=0.0)
    return out


class SpeedPlanner:
    def __init__(self, horizon_m: float = 220.0, ds: float = 2.0):
        self.s = np.arange(0.0, horizon_m + ds, ds)
        self.ds = ds
        self.a = 0.0
        self.plan = SpeedPlan()

    def reset(self) -> None:
        self.a = 0.0

    def step(self, v: float, target_kph: float, stop_at: float | None, path, lane_center, lane_reach: float,
             lane_conf: float, p, dt: float, sight_m: float | None = None, turning: bool = False) -> SpeedPlan:
        s = self.s
        vmax = np.full_like(s, max(0.0, target_kph) / KPH)
        why = np.zeros(len(s), np.int8)        # 0 target, 1 curve, 2 stop, 3 sight
        a_lat = p["curve_lat_accel"]
        # bends: from the planned path (covers turns), and from the lane further out
        k = path_curvature_profile(path, s)
        if lane_center is not None and lane_conf > 0.3 and lane_reach > 10:
            reach = min(lane_reach, 120.0)
            xs = np.minimum(s, reach)
            kl = np.abs(lane_curvature_along(lane_center, xs))
            kl[s > reach] = kl[s <= reach][-1] if np.any(s <= reach) else 0.0
            # trust the far lane shape less (pixel noise): shrink beyond 40 m
            kl *= np.clip(1.0 - (s - 40.0) / 160.0, 0.4, 1.0)
            k = np.maximum(k, kl)
        k = np.maximum(k, 1e-5)
        vc = np.sqrt(a_lat / k)
        vc = np.maximum(vc, (p.get("turn_speed_kph", 18.0) / KPH) if turning else 4.0)
        m = vc < vmax
        vmax = np.where(m, vc, vmax)
        why[m] = 1
        # stop point
        if stop_at is not None and stop_at < s[-1]:
            i = int(np.searchsorted(s, max(0.0, stop_at)))
            vmax[i:] = 0.0
            why[i:] = 2
        # sight distance: be able to stop within what we can see (+ a little, it's a game)
        if sight_m is not None and sight_m < s[-1]:
            i = int(np.searchsorted(s, max(4.0, sight_m)))
            cap = math.sqrt(2 * p["comfort_decel"] * max(4.0, sight_m)) + 4.0
            sub = vmax[:i] > cap
            # only the far part needs to be 0; simpler: everything up to there capped
            vmax[:i] = np.minimum(vmax[:i], cap)
            why[:i][sub] = 3
        # backward pass (comfortable deceleration)
        dec = p["comfort_decel"]
        vb = vmax.copy()
        for i in range(len(s) - 2, -1, -1):
            lim = math.sqrt(vb[i + 1] ** 2 + 2 * dec * self.ds)
            if lim < vb[i]:
                vb[i] = lim
                why[i] = why[i + 1]
        # what we want now: aim at the profile ~0.8 s ahead
        look = max(2.0, v * 0.8)
        v_look = float(np.interp(look, s, vb))
        v_now = float(vb[0])
        # acceleration to reach v_look at distance `look` (kinematics), plus a pull toward v_now
        a_kin = (v_look * v_look - v * v) / (2 * look)
        a_want = a_kin + 0.6 * (min(v_now, v_look) - v)
        a_max = p.get("comfort_accel", 3.0)
        # emergency: a stop point we can't make comfortably -> allow up to max_decel
        a_min = -dec * 1.15
        if stop_at is not None and v > 0.5:
            need = v * v / (2 * max(0.3, stop_at))
            if need > dec * 0.95:
                # beyond max_decel = an emergency: everything the brakes have
                a_min = -min(p["max_decel"], need * 1.15) if need <= p["max_decel"] else -15.0
                a_want = min(a_want, -need * 1.05)
        a_want = float(np.clip(a_want, a_min, a_max))
        # jerk limit (braking may build fast, letting go / accelerating builds slowly)
        j_up = p.get("jerk_limit", 4.0)
        j_brake = j_up * 3.0
        da = a_want - self.a
        if a_want < -0.9 * p["max_decel"]:
            da = a_want - self.a              # emergency: brake NOW, comfort doesn't matter
        elif da < 0:
            da = max(da, -j_brake * dt)
        else:
            da = min(da, j_up * dt)
        self.a += da
        if target_kph < 0.5 and v < 0.3 and (stop_at is None or stop_at < 1.0):
            self.a = min(self.a, 0.0)
        names = ("target", "curve", "stop", "sight")
        self.plan = SpeedPlan(v_now, self.a, names[int(why[0])] if len(why) else "", s, vb)
        return self.plan
