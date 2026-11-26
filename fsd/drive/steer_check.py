"""Keep our idea of the Roblox steering value honest.

We never SEE the steering value: we only know which keys we pressed, and we assume
how fast A/D move it (rise) and how fast it springs back (return). For the default
car that's right; for a truck or a bus it isn't. Worse, when holding a small steering
value we alternate the key on and off (dithering). If the real car springs back
faster than we assume, the real value quietly sinks to zero while we believe we're
holding 0.2. The simulator's box truck showed it: we "held" 0.20, the truck had 0.04,
and it drifted off the road on a gentle bend.

The fix closes the loop. Whenever we can measure the real turn rate independently,
from the lane lines (road curvature x speed minus how fast the lanes rotate past us),
or from the road under the car once that has proven reliable, we work out what
steering value would cause that turn rate and nudge our estimate toward it. The
key logic then simply presses more (or less) to get what it wants.

It also learns the car's real return/rise behaviour: if the corrections keep pointing
the same way while dithering, the assumed spring-back rate is adjusted (kept per
vehicle with the other learned values).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .vehicle_model import steer_for_curvature


class SteerCheck:
    def __init__(self):
        self.meas_lp: float | None = None
        self.model_lp: float | None = None
        self.corr_hist: deque = deque(maxlen=240)
        self.ret_scale = 1.0          # learned: real spring-back / assumed spring-back
        self.n = 0
        self.last_corr = 0.0

    def update(self, observer, v: float, yaw_meas: float | None, gain: float, p, dt: float) -> float:
        """observer: control.steer_model.SteerObserver (its .s is corrected in place)."""
        self.last_corr = 0.0
        if observer is None or yaw_meas is None or v < 4.0:
            self.meas_lp = None
            return 0.0
        k_meas = yaw_meas / v
        s_meas = steer_for_curvature(k_meas / max(gain, 0.2), v, p)
        # both through the same smoothing so their lag matches (the lane derivative is ~0.15 s late)
        a = min(1.0, dt / 0.15)
        self.meas_lp = s_meas if self.meas_lp is None else self.meas_lp + a * (s_meas - self.meas_lp)
        self.model_lp = observer.s if self.model_lp is None else self.model_lp + a * (observer.s - self.model_lp)
        err = self.meas_lp - self.model_lp
        if abs(err) < 0.02:
            return 0.0
        corr = float(np.clip(err, -0.3, 0.3)) * min(1.0, dt * 3.0)
        observer.s = float(np.clip(observer.s + corr, -1.0, 1.0))
        self.model_lp += corr
        self.last_corr = corr
        # learning the spring-back: we were dithering (small |s|) and keep having to pull our
        # estimate toward zero -> the car returns faster than assumed (and vice versa)
        if 0.05 < abs(observer.s) < 0.5:
            toward_zero = corr * np.sign(observer.s) < 0
            self.corr_hist.append(1.0 if toward_zero else -1.0)
            if len(self.corr_hist) >= 120:
                bias = float(np.mean(self.corr_hist))
                if abs(bias) > 0.4:
                    self.ret_scale = float(np.clip(self.ret_scale * (1.0 + 0.05 * bias), 0.5, 2.0))
                    observer.ret = p["steer_return_per_s"] * self.ret_scale
                    self.n += 1
                    self.corr_hist.clear()
        return corr
