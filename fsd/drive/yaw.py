"""How fast is the car really turning?

The old code measured yaw from how the picture slides sideways (phase correlation of
a band near the horizon). In ER:LC that is unreliable:

* the chase camera swings AFTER the car, so the picture first rotates the wrong way
  and then catches up;
* repeating scenery (lamp posts, fences, barriers) gives false matches - a spike of
  60 deg/s on a dead-straight highway;
* a damping term that trusts such a spike steers the car off the road (that's what
  the simulator showed: one spike, then the wrong-signed reading kept the car turning).

So we FUSE three sources, each with what it's good at:

1. the chassis model: yaw = v * kappa(steering value). We know the steering value
   exactly (we pressed the keys), so this is smooth and has no lag. Its weakness:
   if the real car steers stronger/weaker than the model, it's off by a factor.
   That factor is learned (StrengthLearner) from the lanes, not from the picture.
2. the lanes: on a road whose curvature we see, the change of the lane's heading
   relative to the car is (road turning) - (car turning). Slow but drift-free.
3. the picture: used only when it agrees in sign and roughly in size with 1,
   just to nudge the estimate (e.g. sliding on ice-like surfaces, bumps).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .vehicle_model import curvature_from_steer


class StrengthLearner:
    """Learns 'this car turns g times what the model says' from lane geometry.

    While driving a steady bend with good lanes and the car holding the lane
    (lane heading not changing), the car's curvature equals the road's curvature.
    Comparing that with the model curvature for the steering value we're holding
    gives the ratio. Least squares with forgetting; clamped to 0.4..2.5.
    """

    def __init__(self, forget: float = 0.997):
        self.num = 0.0
        self.den = 0.0
        self.forget = forget
        self.gain = 1.0
        self.samples = 0
        self._hist: deque = deque(maxlen=60)

    def update(self, t: float, v: float, model_k: float, road_k: float, lane_heading: float,
               lanes_ok: bool) -> None:
        self._hist.append((t, model_k, road_k, lane_heading, lanes_ok))
        if v < 8.0 or len(self._hist) < 30:
            return
        h = list(self._hist)[-30:]
        if not all(x[4] for x in h) or h[-1][0] - h[0][0] < 0.35:
            return
        ts = np.array([x[0] for x in h])
        mk = np.array([x[1] for x in h])
        rk = np.array([x[2] for x in h])
        hd = np.array([x[3] for x in h])
        # car curvature = road curvature + d(lane heading)/ds   (lane heading turning toward +
        # means the road turns right more than we do)
        dhd = float(np.polyfit(ts - ts[0], hd, 1)[0])            # rad/s
        car_k = float(rk.mean()) - dhd / v
        m = float(mk.mean())
        if abs(m) < 2e-3 or abs(car_k) < 1.5e-3 or np.sign(m) != np.sign(car_k):
            return
        if mk.std() > 0.5 * abs(m) + 1e-3:          # steering not steady enough
            return
        self.num = self.forget * self.num + m * car_k
        self.den = self.forget * self.den + m * m
        self.samples += 1
        if self.samples >= 8 and self.den > 0:
            self.gain = float(np.clip(self.num / self.den, 0.4, 2.5))


class YawFusion:
    def __init__(self):
        self.yaw = 0.0
        self.model = 0.0
        self.image_ok = 0.0          # 0..1: how often the picture agreed lately
        self.strength = StrengthLearner()
        self._last_t = None
        self._prev_heading = None
        self._prev_t = None
        self.lane_yaw = None

    def reset(self) -> None:
        self.__init__()

    def update(self, t: float, v: float, steer_value: float, p, image_yaw: float | None,
               lanes=None) -> float:
        """v in m/s; steer_value = the actuator's estimate of the Roblox steering value."""
        gain = self.strength.gain if p.get("steer_strength_learn", 0.0) > 0.5 else 1.0
        k_model = curvature_from_steer(steer_value, v, p) * gain
        self.model = v * k_model
        est = self.model
        # lanes: yaw = v * road_k - d(heading)/dt
        self.lane_yaw = None
        if lanes is not None and lanes.conf > 0.5 and v > 3.0:
            if self._prev_heading is not None and self._prev_t is not None and 0.005 < t - self._prev_t < 0.2:
                dhd = (lanes.heading - self._prev_heading) / (t - self._prev_t)
                ly = v * lanes.curvature - dhd
                if abs(ly) < 1.5:
                    self.lane_yaw = ly
            self.strength.update(t, v, k_model / max(gain, 1e-3), lanes.curvature,
                                 lanes.heading, True)
            self._prev_heading, self._prev_t = lanes.heading, t
        else:
            self._prev_heading = None
            if lanes is not None:
                self.strength.update(t, v, 0.0, 0.0, 0.0, False)
        if self.lane_yaw is not None:
            est = 0.85 * est + 0.15 * self.lane_yaw          # lane derivative is noisy: small weight
        if image_yaw is not None and v > 2.0:
            tol = 0.08 + 0.5 * abs(self.model)
            agree = abs(image_yaw - self.model) < tol and (abs(self.model) < 0.03 or
                                                           math.copysign(1, image_yaw) == math.copysign(1, self.model))
            self.image_ok = 0.97 * self.image_ok + 0.03 * (1.0 if agree else 0.0)
            if agree:
                est = 0.8 * est + 0.2 * image_yaw
        # light smoothing (keeps the 60 Hz output from stepping with every key pulse)
        dt = 0.016 if self._last_t is None else float(np.clip(t - self._last_t, 0.0, 0.2))
        self._last_t = t
        k = min(1.0, dt / 0.06)
        self.yaw += (est - self.yaw) * k
        return self.yaw
