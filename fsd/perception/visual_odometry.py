"""Speed and turn rate from the road itself moving under the car, every frame.

The speedometer is read from the screen, so it is slow: OCR takes a few hundred ms,
and the fast digit reader only works once it has learned every digit. Accelerating
from 25 to 200 km/h shows new digits it hasn't seen yet, so the speed FSD used could
lag by seconds. A person doesn't need the speedometer to know they're speeding up. They
see the road rushing past.

So this module does that:
1. Bird's-eye strips of the road just past our own car (the part the car doesn't hide):
   a NEAR strip (~x0..x0+9 m) and a FAR strip (~x0+9..x0+18 m), lane-wide, at 6 cm/px,
   sampled straight from the frame with a pre-computed remap (~0.5 ms).
2. Each new strip is phase-correlated with the one from the previous frame:
   - the forward shift (metres) / time between frames = SPEED
   - the sideways shift differs between near and far strip when we rotate:
     (shift_far - shift_near) / (distance between them) = how much we TURNED.
   Lane dashes, asphalt texture, cracks and shadows are all features. Roblox has no
   motion blur, so this works at any speed.
3. Ambiguity: dashed lines repeat every ~12 m. One frame at 200 km/h is ~0.9 m (60 fps)
   or ~1.9 m (30 fps), well under half a period, and the result is checked against the
   previous speed (a car can't gain 30 km/h in one frame).
4. Calibration error scales the result (a camera assumed 10% too high reads 10% fast),
   so a scale factor is learned whenever a fresh speedometer reading and a steady
   visual speed are both available.

Output: speed (m/s), turn rate of the CAMERA (rad/s), and a confidence per frame.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VOResult:
    ok: bool = False
    v: float = 0.0            # m/s, scaled by the learned factor
    v_raw: float = 0.0        # m/s straight from the picture
    yaw: float = 0.0          # rad/s (camera)
    conf: float = 0.0
    shift_m: float = 0.0


class VisualOdometry:
    def __init__(self, res_m: float = 0.06, strip_len_m: float = 9.0, half_w_m: float = 2.4):
        self.res = res_m
        self.strip_len = strip_len_m
        self.half_w = half_w_m
        self._maps = None
        self._key = None
        self._prev: tuple | None = None
        self._prev_t: float | None = None
        self.scale = 1.0
        self.scale_n = 0
        self.hist: deque = deque(maxlen=7)
        self.last = VOResult()
        self.x0 = 10.0
        self._win = None
        self._bg = [None, None]

    def reset(self) -> None:
        self._prev = None
        self._prev_t = None
        self.hist.clear()

    # ------------------------------------------------------------------ geometry
    def _build(self, geo, x0: float, c: float, shape) -> None:
        H, W = shape[:2]
        nx = int(round(self.strip_len / self.res))
        ny = int(round(2 * self.half_w / self.res))
        maps = []
        for start in (x0, x0 + self.strip_len):
            # row 0 = far end, last row = near end (like the picture)
            xs = start + self.strip_len - (np.arange(nx) + 0.5) * self.res
            ys = c - self.half_w + (np.arange(ny) + 0.5) * self.res
            gx, gy = np.meshgrid(xs, ys, indexing="ij")
            pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
            scr = geo.vehicle_to_screen(pts).astype(np.float32)
            mx = scr[:, 0].reshape(nx, ny)
            my = scr[:, 1].reshape(nx, ny)
            valid = (mx >= 0) & (mx < W - 1) & (my > geo.horizon_y + 2) & (my < H - 1)
            maps.append((mx, my, float(valid.mean())))
        self._maps = maps
        self._win = cv2.createHanningWindow((ny, nx), cv2.CV_32F)
        self._bg = [None, None]

    def _strip(self, gray: np.ndarray, k: int) -> np.ndarray:
        mx, my, _ = self._maps[k]
        s = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        s = s.astype(np.float32)
        # high-pass: remove lighting gradients / shadows' soft edges, keep texture and paint
        s = s - cv2.GaussianBlur(s, (0, 0), 6)
        # solid lane lines and the road edge look the same however far we drive: they'd pull the
        # match to "no movement". Remove whatever is constant ALONG the road (per column mean),
        # so only dash ends, cracks, texture and shadows' edges - things that move - remain.
        s = s - s.mean(axis=0, keepdims=True)
        return s

    # ------------------------------------------------------------------ per frame
    def update(self, frame: np.ndarray, geo, t: float, hidden_m: float, lane_c: float = 0.0,
               ego_mask: np.ndarray | None = None, v_prior: float | None = None) -> VOResult:
        if geo is None:
            return VOResult()
        x0 = float(np.clip(hidden_m + 1.5, 4.0, 22.0))
        c = float(np.clip(lane_c, -2.0, 2.0))
        # rebuild maps when the geometry or the window changes noticeably
        key = (id(geo), round(x0), round(c * 2) / 2, frame.shape[:2])
        if key != self._key:
            self._build(geo, round(x0), round(c * 2) / 2, frame.shape)
            self._key = key
            self._prev = None
        if self._maps[0][2] < 0.8 or self._maps[1][2] < 0.5:
            return VOResult()
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        near = self._strip(gray, 0)
        far = self._strip(gray, 1)
        prev, prev_t = self._prev, self._prev_t
        self._prev, self._prev_t = (near, far), t
        if prev is None or prev_t is None:
            return VOResult()
        dt = t - prev_t
        if not (0.004 < dt < 0.25):
            return VOResult()
        # things that move WITH us (our car's shadow or bumper edge, a HUD element) give a
        # strong peak at zero shift; when we know we're moving, that peak is skipped
        prior = v_prior if v_prior is not None else (self.last.v if self.last.ok else None)
        skip0 = prior is not None and abs(prior) > 2.0
        (dyn, dxn), rn = phase_shift(prev[0], near, self._win, skip0)
        (dyf, dxf), rf = phase_shift(prev[1], far, self._win, skip0)
        # phaseCorrelate returns (x shift, y shift) = (sideways px, along px).
        # Rows run far->near, so the road moving TOWARD us (driving forward) is +rows.
        fwd_n = dxn * self.res
        fwd_f = dxf * self.res
        conf = float(min(rn, rf))
        res = VOResult(conf=conf)
        if max(rn, rf) < 0.06:
            self.last = res
            return res
        # prefer the stronger strip, check they agree
        if rn >= 0.06 and rf >= 0.06 and abs(fwd_n - fwd_f) > 0.25 + 0.15 * abs(fwd_n):
            conf *= 0.3
        fwd = fwd_n if rn >= rf else fwd_f
        v_raw = fwd / dt
        # plausibility against the prior (no car gains 30 km/h in a frame)
        if v_prior is not None and abs(v_raw - v_prior) > 8.0 + 0.4 * abs(v_prior):
            conf *= 0.2
        if v_raw < -2.0 or v_raw > 110.0:
            self.last = VOResult(conf=0.0)
            return self.last
        # rotation: sideways shift difference between the strips' centres
        lever = self.strip_len
        side_n = dyn * self.res
        side_f = dyf * self.res
        yaw = 0.0
        if rn >= 0.06 and rf >= 0.06:
            yaw = -(side_f - side_n) / lever / dt
        self.hist.append(v_raw)
        v_med = float(np.median(self.hist)) if len(self.hist) >= 3 else v_raw
        res = VOResult(True, v_med * self.scale, v_raw, float(yaw), float(np.clip(conf, 0, 1)), fwd)
        self.last = res
        return res

    def learn_scale(self, v_speedo: float, fresh: bool, steady: bool) -> None:
        """Speedometer fresh and the car not accelerating hard: compare."""
        r = self.last
        if not (fresh and steady and r.ok and r.conf > 0.15 and v_speedo > 5.0 and r.v_raw > 3.0):
            return
        ratio = v_speedo / r.v_raw
        if not 0.5 < ratio < 2.0:
            return
        self.scale_n += 1
        k = 1.0 / min(self.scale_n + 1, 60)
        self.scale = float(np.clip((1 - k) * self.scale + k * ratio, 0.5, 2.0))

    @property
    def calibrated(self) -> bool:
        return self.scale_n >= 20


def phase_shift(a: np.ndarray, b: np.ndarray, win: np.ndarray, skip_zero: bool = False,
                zero_px: float = 1.6) -> tuple[tuple[float, float], float]:
    """Phase correlation like cv2.phaseCorrelate, returns ((dx, dy), response) of b vs a, with
    the option of ignoring the zero-shift peak. Sub-pixel by a parabola through the peak."""
    A = np.fft.rfft2(a * win)
    B = np.fft.rfft2(b * win)
    R = B * np.conj(A)
    R /= np.abs(R) + 1e-9
    r = np.fft.irfft2(R, s=a.shape)
    r = np.fft.fftshift(r)
    h, w = r.shape
    cy, cx = h // 2, w // 2
    if skip_zero:
        z = int(np.ceil(zero_px))
        r2 = r.copy()
        r2[cy - z:cy + z + 1, cx - z:cx + z + 1] = -1.0
    else:
        r2 = r
    iy, ix = np.unravel_index(int(np.argmax(r2)), r2.shape)
    peak = float(r[iy, ix])

    def sub(v_m, v0, v_p):
        d = v_m - 2 * v0 + v_p
        return 0.0 if abs(d) < 1e-9 else 0.5 * (v_m - v_p) / d
    oy = sub(r[(iy - 1) % h, ix], r[iy, ix], r[(iy + 1) % h, ix])
    ox = sub(r[iy, (ix - 1) % w], r[iy, ix], r[iy, (ix + 1) % w])
    # response comparable with cv2's: the peak's share of a small neighbourhood energy
    nb = r[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3]
    resp = float(nb.sum()) if nb.size else peak
    return ((ix - cx + ox), (iy - cy + oy)), float(np.clip(resp, -1, 1))


def turn_radius(v: float, yaw: float) -> float:
    return math.inf if abs(yaw) < 1e-4 else abs(v / yaw)


def hidden_from_box(geo, box, W: int) -> float:
    """Distance (m) of the road hidden behind our own car: ground point at the top of its box."""
    if geo is None or box is None or box[3] <= box[1]:
        return 8.0
    y1 = float(box[1])
    if y1 <= geo.horizon_y + 3:
        return 22.0
    try:
        v = geo.screen_to_vehicle(np.array([[W / 2.0, y1]], np.float32))
        return float(np.clip(v[0][0], 0.0, 22.0))
    except Exception:  # noqa: BLE001
        return 8.0
