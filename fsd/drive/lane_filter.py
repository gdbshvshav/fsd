"""A smooth, gap-free estimate of our lane, 60 times a second.

Lane detection finishes 20-40 times a second, its answers jitter by a few
centimetres, and now and then it misses a frame or picks the wrong line. Steering
straight from it makes the wheel twitch, and a single bad answer can yank the car.

This is a Kalman filter on the lane centre line in car coordinates:

    y(x) = c + b x + a x^2          (metres, + = right of the car)

    c = where the lane centre is beside us       b = its direction vs the car
    a = half its curvature                        w = lane width

Between detections it PREDICTS: we know how far we drove (speed) and how much we
turned (yaw), so the line moves in the picture exactly like the old one would.
When a detection arrives it CORRECTS, trusting it more when its confidence is high.

Extra rules:
* A detection whose centre is ~one lane width away from the prediction means we
  crossed a line (lane change / merge): the filter re-bases onto the new lane at
  once instead of fighting it.
* A detection that disagrees wildly is ignored (gate); if several in a row
  disagree, the filter accepts the new picture (it was the filter that was wrong).
* Without detections the estimate coasts for up to ~1.5 s with growing doubt,
  so a missed frame or a crest never makes the car swerve.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..world import LaneLine, Lanes


def _shift(c: float, b: float, a: float, d: float, th: float) -> tuple[float, float, float]:
    return a * d * d + b * d + c, 2 * a * d + b - th, a


class LaneFilter:
    def __init__(self, width0: float = 3.7):
        self.x = np.zeros(4)                       # c, b, a, yaw bias (rad/s)
        self.P = np.diag([4.0, 0.05, 1e-4, 0.01])
        self.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])
        self.changing = False                      # a lane change is in progress (set by the driver)
        self.width = width0
        self.width_var = 1.0
        self.ok = False
        self.t = None
        self.last_meas_t = -1e9
        self.meas_key = None
        self.rejects = 0
        self.switched = 0                          # +1 / -1 on the frame we moved one lane right / left
        self.last_meas: Lanes | None = None
        self.innov = 0.0
        self.conf = 0.0
        self.max_age = 1.6                         # how long to coast without detections
        self.junction = False                      # crossing a junction: coast longer, straight on
        # process noise per second (how fast the true road shape can change under us)
        self.q = np.array([0.04, 0.004, 2e-6])

    def reset(self) -> None:
        self.__init__(self.width)

    # ------------------------------------------------------------------
    def predict(self, t: float, v: float, yaw: float) -> None:
        if self.t is None:
            self.t = t
            return
        dt = float(np.clip(t - self.t, 0.0, 0.25))
        self.t = t
        if not self.ok or dt <= 0:
            return
        if self.junction and t - self.last_meas_t > 0.3:
            # across a junction: the road ahead continues the way it pointed; don't let the
            # (unseen) curvature bend our line
            self.x[2] *= math.exp(-dt / 0.8)
        d = v * dt
        th = yaw * dt
        c, b, a, beta = self.x
        # our yaw estimate may be off by a steady amount (vehicle steers differently than the
        # model, camera leans): that bias is part of the state and learned from the lanes
        c2, b2, a2 = _shift(c, b, a, d, th - beta * dt)
        self.x = np.array([c2, b2, a2, beta])
        F = np.array([[1.0, d, d * d, 0.0],
                      [0.0, 1.0, 2 * d, dt],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        yaw_var = (0.05 * abs(yaw) + 0.01) ** 2 * dt * dt
        Q = np.diag([*(self.q * dt * (1.0 + 0.04 * v)), 0.002 * dt]) + np.diag([0.0, yaw_var, 0.0, 0.0])
        self.P = F @ self.P @ F.T + Q
        age = t - self.last_meas_t
        self.conf = float(self.conf * math.exp(-dt / 1.2)) if age > 0.15 else self.conf

    # ------------------------------------------------------------------
    def update(self, lanes: Lanes, t: float, key=None) -> None:
        """Feed a detection (already moved forward to time t). `key` identifies the
        detection so the same one is never applied twice."""
        self.switched = 0
        if key is not None and key == self.meas_key:
            return
        self.meas_key = key
        if lanes is None or lanes.conf <= 0.05:
            return
        a, b, c = lanes.center
        z = np.array([c, b, a])
        conf = float(np.clip(lanes.conf, 0.05, 1.0))
        # measurement noise: worse when unsure, and worse for the far shape terms
        reach = max(8.0, min(60.0, lanes.x_max or 30.0))
        R = np.diag([(0.12 / conf) ** 2, (0.03 / conf) ** 2 * (30.0 / reach),
                     (0.0012 / conf) ** 2 * (30.0 / reach) ** 2])
        width = lanes.width_m or self.width
        if not self.ok:
            self.x = np.r_[z, self.x[3] if len(self.x) == 4 else 0.0]
            self.P = np.diag([*(np.diag(R) * 4.0), max(self.P[3, 3], 0.01)])
            self.width = width
            self.ok = True
            self.last_meas_t = t
            self.last_meas = lanes
            self.conf = conf
            return
        # crossed a line? (the detector now reports the neighbouring lane)
        dc = z[0] - self.x[0]
        for sgn in (+1, -1):
            if abs(dc - sgn * self.width) < 0.45 * self.width and abs(z[1] - self.x[1]) < 0.12:
                # new lane: take its measured position outright (our dead-reckoned estimate of the
                # old lane may have drifted while we straddled the line)
                self.x[0] = z[0]
                self.switched = sgn
                self.P[0, 0] = R[0, 0]
                self.rejects = 0
                break
        H = self.H
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            Si = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        m2 = float(y @ Si @ y)
        self.innov = math.sqrt(max(0.0, m2))
        # while changing lanes the detector often reports a lane "in between" (we straddle the
        # line): a centre that is neither our lane nor a neighbour is ignored, not believed
        straddle = self.changing and min(abs(y[0]), abs(abs(y[0]) - self.width)) > 0.6
        # two lines much further apart than a lane = the detector paired the wrong lines
        # (e.g. the yellow edge and the far white edge while we straddle the middle line)
        if lanes.left is not None and lanes.right is not None and width > 1.35 * self.width:
            straddle = True
        if m2 > 30.0 or straddle:         # way off: probably a wrong detection
            self.rejects += 1
            if self.rejects < 6 or self.changing:
                return
            # several in a row: the picture is right, we were wrong
            self.x = np.r_[z, self.x[3]]
            self.P = np.diag([*(np.diag(R) * 2.0), self.P[3, 3]])
            self.rejects = 0
            self.last_meas_t = t
            self.last_meas = lanes
            self.conf = conf * 0.7
            return
        self.rejects = 0
        K = self.P @ H.T @ Si
        self.x = self.x + K @ y
        self.x[3] = float(np.clip(self.x[3], -0.4, 0.4))
        self.P = (np.eye(4) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        # width: simple scalar filter, only from confident two-line detections
        if lanes.left is not None and lanes.right is not None and conf > 0.4 and 2.4 < width < 6.5:
            kw = self.width_var / (self.width_var + 0.05 / conf)
            self.width += kw * (width - self.width)
            self.width_var = (1 - kw) * self.width_var + 1e-4
        self.last_meas_t = t
        self.last_meas = lanes
        self.conf = 0.7 * self.conf + 0.3 * conf

    # ------------------------------------------------------------------
    def lanes(self, t: float) -> Lanes:
        """The filtered lane, in the same form the rest of FSD uses."""
        if not self.ok:
            return Lanes()
        age = t - self.last_meas_t
        if age > self.max_age:
            self.ok = False
            return Lanes()
        c, b, a = (float(v) for v in self.x[:3])
        m = self.last_meas
        half = self.width / 2

        def line(src: LaneLine | None, off: float) -> LaneLine:
            col = src.color if src is not None else "white"
            dashed = src.dashed if src is not None else False
            lc = src.conf if src is not None else 0.3
            xm = max(0.0, (src.x_max if src is not None else (m.x_max if m else 30.0)) - 0.0)
            return LaneLine((a, b, c + off), xm, col, dashed, lc)

        left = line(m.left if m else None, -half) if (m is None or m.left is not None or m.right is not None) else None
        right = line(m.right if m else None, +half)
        out = Lanes(left=left, right=right,
                    left2=m.left2 if m else None, right2=m.right2 if m else None,
                    center=(a, b, c), width_m=self.width,
                    conf=self._conf_at(age),
                    curvature=float(2 * a / (1 + (2 * a * 10 + b) ** 2) ** 1.5),
                    heading=float(math.atan(b)), offset=c,
                    x_max=float(max(0.0, (m.x_max if m else 30.0))))
        return out

    def _conf_at(self, age: float) -> float:
        if age < 0.3:
            return float(self.conf)
        fade = max(0.0, 1.0 - (age - 0.3) / max(0.3, self.max_age - 0.3))
        c = float(self.conf * fade)
        if self.junction:
            # inside a junction the lines are SUPPOSED to vanish: keep driving the lane's line
            # across at normal confidence instead of slowing to a crawl
            c = max(c, 0.45 if age < self.max_age - 0.3 else c)
        return c

    @property
    def yaw_bias(self) -> float:
        """How much our yaw estimate reads too high (rad/s), learned from the lanes."""
        return float(self.x[3]) if self.ok else 0.0

    def sigma_offset(self) -> float:
        return float(math.sqrt(max(0.0, self.P[0, 0])))


def filtered(raw: Lanes, f: LaneFilter, t: float, v: float, yaw: float, key) -> Lanes:
    """One call per control frame: predict to t, correct with `raw` if it's new."""
    f.predict(t, v, yaw)
    f.update(raw, t, key)
    out = f.lanes(t)
    if out.conf <= 0 and raw is not None and raw.conf > 0:
        return raw
    # keep the detector's extra lines (adjacent lanes) aligned with the filtered centre
    if raw is not None and raw.conf > 0 and out.conf > 0:
        dc = out.offset - raw.offset
        def mv(l):
            if l is None:
                return None
            a, b, c = l.coeffs
            return replace(l, coeffs=(a, b, c + dc))
        out.left2 = mv(raw.left2)
        out.right2 = mv(raw.right2)
    return out
