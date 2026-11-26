"""Steering: follow the planned line smoothly, at any speed.

How a person drives a keyboard car well: look a bit ahead, turn in early by about the
right amount for the bend (feed-forward), then make small corrections - and never
correct so hard that the car swings across the lane and back.

The controller works in CURVATURE (1/m), then converts to the Roblox steering value
with the vehicle model:

1. Path geometry (any shape, also 90 deg turns): resample the planned path by arc
   length, find where we are on it, and measure
      e   = how far we are beside it (m, + = we're right of it)
      psi = how much we point away from it (rad, + = we point right of it)
      k_p = its curvature a little ahead (where the car will be once the steering
            has moved - the keys need ~0.1-0.25 s to turn the wheel).
2. Feedback: a Linear-Quadratic Regulator on the car's sideways motion
      e' = v psi,   psi' = v (k - k_p),   k' = (u - k) / tau
   (tau = steering lag). The gains are recomputed for each speed band, so the car
   is firm in town and calm at 200 km/h - the LQR finds the best trade-off between
   "get back to the line" and "don't yank the wheel" automatically.
3. Limits: never ask for more curvature than the tyres / sideways comfort allow at
   this speed, and never more than full lock.
4. Weave guard: if the car starts to swing from side to side (lateral error crossing
   zero every 1-4 s with growing size) the corrections are softened until it settles.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .vehicle_model import curvature_from_steer, max_curvature, steer_for_curvature, steer_time_constant


# ----------------------------------------------------------------------------- path geometry
@dataclass
class PathGeom:
    ok: bool = False
    e: float = 0.0            # we're this far right of the path (m)
    psi: float = 0.0          # we point this much right of the path (rad)
    k_near: float = 0.0       # path curvature here
    k_prev: float = 0.0       # path curvature at the preview point
    length: float = 0.0
    s0: float = 0.0           # arc length of the closest point
    e_look: float = 0.0       # how far right of the path we'd be at `look` m if we drove straight
    look: float = 0.0


def resample(path, step: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path or len(path) < 2:
        return None
    P = np.asarray(path, np.float64)
    seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    keep = np.r_[True, seg > 1e-4]
    P = P[keep]
    if len(P) < 2:
        return None
    seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 1.0:
        return None
    ss = np.arange(0.0, s[-1] + 1e-9, step)
    xs = np.interp(ss, s, P[:, 0])
    ys = np.interp(ss, s, P[:, 1])
    return ss, xs, ys


def path_geometry(path, v: float, preview_s: float, look: float = 0.0) -> PathGeom:
    r = resample(path)
    if r is None:
        return PathGeom()
    ss, xs, ys = r
    n = len(ss)
    # heading of the path at each point (angle from +x toward +y = right)
    hx = np.gradient(xs)
    hy = np.gradient(ys)
    ang = np.unwrap(np.arctan2(hy, hx))
    # smooth the angle a little before differentiating (pixel noise in far points)
    if n >= 7:
        kern = np.ones(5) / 5.0
        angs = np.convolve(np.pad(ang, 2, mode="edge"), kern, mode="valid")
    else:
        angs = ang
    kap = np.gradient(angs, ss) if n >= 3 else np.zeros(n)
    # closest point to the car (origin), but only among points not far behind
    d2 = xs * xs + ys * ys
    i = int(np.argmin(d2))
    # lateral error: signed distance of origin from the path at i (+ = origin right of path)
    tx, ty = math.cos(ang[i]), math.sin(ang[i])
    ox, oy = -xs[i], -ys[i]                     # vector path point -> car
    e = tx * oy - ty * ox                       # cross(t, o): + when car is to the right
    psi = -float(ang[i])                        # car heading 0 vs path heading
    psi = (psi + math.pi) % (2 * math.pi) - math.pi
    s0 = float(ss[i])
    # preview window
    a = s0 + max(0.0, v * preview_s)
    b = a + max(2.0, v * 0.35)
    sel = (ss >= a) & (ss <= b)
    k_prev = float(np.mean(kap[sel])) if np.any(sel) else float(kap[-1])
    near = (ss >= s0) & (ss <= s0 + max(2.0, v * 0.2))
    k_near = float(np.mean(kap[near])) if np.any(near) else float(kap[i])
    e_look = float(e)
    if look > 0:
        sl = min(float(ss[-1]), s0 + look)
        xl = float(np.interp(sl, ss, xs))
        yl = float(np.interp(sl, ss, ys))
        # lateral offset of that path point from our straight-ahead line (car frame: y right)
        e_look = -yl if xl > 0.5 else float(e)
    return PathGeom(True, float(e), psi, k_near, k_prev, float(ss[-1]), s0, e_look, look)


# ----------------------------------------------------------------------------- LQR
def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, iters: int = 400) -> np.ndarray:
    """Discrete LQR gain by iterating the Riccati equation (no scipy needed)."""
    P = Q.copy()
    for _ in range(iters):
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        Pn = A.T @ P @ A - A.T @ P @ B @ K + Q
        if np.max(np.abs(Pn - P)) < 1e-9 * (1.0 + np.max(np.abs(P))):
            P = Pn
            break
        P = Pn
    BtP = B.T @ P
    return np.linalg.solve(R + BtP @ B, BtP @ A)


class GainTable:
    """LQR gains per speed band (computed once per band, cached)."""

    def __init__(self):
        self.cache: dict = {}

    def gains(self, v: float, tau: float, dt: float, softness: float, e_max: float, psi_max: float,
              look: float = 0.0) -> np.ndarray:
        """State: [e_look, psi, k]. e_look = sideways error `look` metres ahead (e + look * psi):
        steering on a point a little ahead - where the lines are actually SEEN - is much less
        sensitive to errors in the lane's estimated curvature than the error beside the car,
        which has to be extrapolated under our own bonnet."""
        vb = round(max(1.0, v) / 1.5) * 1.5
        lb = round(look * 2) / 2
        key = (vb, round(tau, 3), round(dt, 4), round(softness, 2), round(e_max, 2), round(psi_max, 3), lb)
        K = self.cache.get(key)
        if K is not None:
            return K
        A = np.array([[1.0, vb * dt, lb * vb * dt],
                      [0.0, 1.0, vb * dt],
                      [0.0, 0.0, 1.0 - dt / tau]])
        B = np.array([[0.0], [0.0], [dt / tau]])
        # how much error is "a lot" vs how much curvature is "a lot" at this speed:
        # sideways accel v^2 k of ~2.5 m/s^2 is a firm correction
        k_max = 2.5 / max(vb * vb, 4.0)
        Q = np.diag([1.0 / e_max ** 2, 1.0 / psi_max ** 2, 0.0])
        R = np.array([[softness / k_max ** 2]])
        K = dlqr(A, B, Q, R)
        if len(self.cache) > 400:
            self.cache.clear()
        self.cache[key] = K
        return K


# ----------------------------------------------------------------------------- weave guard
class WeaveGuard:
    """Notices side-to-side swinging and softens the steering until it stops."""

    def __init__(self):
        self.hist: deque = deque(maxlen=400)        # (t, e)
        self.soft = 1.0
        self.weaving = False
        self.last_cross: deque = deque(maxlen=8)
        self._sign = 0

    def update(self, t: float, e: float, active: bool) -> float:
        if not active:
            self._sign = 0
            self.soft = max(1.0, self.soft * 0.995)
            return self.soft
        self.hist.append((t, e))
        s = 1 if e > 0.15 else (-1 if e < -0.15 else 0)
        if s != 0 and self._sign != 0 and s != self._sign:
            self.last_cross.append(t)
        if s != 0:
            self._sign = s
        # weaving = 3+ crossings within 6 s, with swings bigger than 0.25 m
        recent = [c for c in self.last_cross if t - c < 6.0]
        amp = max((abs(x[1]) for x in self.hist if t - x[0] < 6.0), default=0.0)
        self.weaving = len(recent) >= 3 and amp > 0.4
        if self.weaving:
            self.soft = min(6.0, self.soft * 1.02)
        else:
            self.soft = max(1.0, self.soft * 0.997)
        return self.soft


# ----------------------------------------------------------------------------- controller
class LateralLQR:
    def __init__(self):
        self.table = GainTable()
        self.guard = WeaveGuard()
        self.k_cmd = 0.0
        self.s_cmd = 0.0
        self.geom = PathGeom()
        self.debug: dict = {}
        self.ie = 0.0                # integral of lateral error (removes steady offsets in long bends)

    def reset(self) -> None:
        self.k_cmd = 0.0
        self.s_cmd = 0.0
        self.ie = 0.0
        self.guard = WeaveGuard()

    def step(self, path, v: float, steer_value: float, strength: float, p, dt: float, t: float,
             turning: bool = False, conf: float = 1.0, kmax: float | None = None) -> float:
        """Returns the wanted Roblox steering value (-1..1)."""
        v = max(0.0, v)
        tau = steer_time_constant(p)
        latency = p.get("control_latency_s", 0.05)
        look = float(np.clip(p.get("steer_look_s", 0.35) * v, 3.0, 12.0)) if not turning else 2.0
        g = path_geometry(path, v, tau + latency, look)
        self.geom = g
        if not g.ok:
            self.s_cmd *= 0.9
            return self.s_cmd
        # current curvature from the steering value we're holding
        k_now = curvature_from_steer(steer_value, v, p) * strength
        # firmness: town / turns tighter, unsure lanes softer
        soft = p.get("steer_softness", 1.0) * self.guard.update(t, g.e_look, not turning)
        if conf < 0.5:
            soft *= 1.0 + (0.5 - conf) * 4.0
        e_max = 0.35 if not turning else 0.25
        psi_max = 0.06 if v > 15 else 0.12
        K = self.table.gains(v, tau, max(1 / 120, min(dt, 0.05)), soft, e_max, psi_max, look)
        z = np.array([g.e_look, g.psi, k_now - g.k_near])
        u_fb = float(-(K @ z)[0])
        k_ff = g.k_prev * p.get("steer_ff_gain", 1.0)
        # integral action: a model that's a bit off (this car steers weaker, the camera leans)
        # would otherwise leave us a constant 20-40 cm off-centre in every long bend
        if not turning and conf > 0.5 and v > 5.0 and abs(g.e) < 0.8:
            self.ie = float(np.clip(self.ie + g.e_look * dt, -4.0, 4.0))
        else:
            self.ie *= 0.98
        k_i = -p.get("steer_ki", 0.35) * self.ie / max(v * v, 25.0)
        k = k_ff + u_fb + k_i
        # limits: full lock, and sideways acceleration (not in tight low-speed turns)
        kmax = kmax if kmax is not None else max_curvature(v, p) * strength
        if v > 6.0:
            kmax = min(kmax, p.get("max_lat_accel", 7.0) / (v * v))
        k = float(np.clip(k, -kmax, kmax))
        self.k_cmd = k
        s = steer_for_curvature(k / max(strength, 0.2), v, p)
        self.s_cmd = s
        self.debug = {"e": g.e, "psi": g.psi, "k_ff": k_ff, "k_fb": u_fb, "k": k, "s": s,
                      "soft": soft, "weave": self.guard.weaving, "k_i": k_i}
        return s
