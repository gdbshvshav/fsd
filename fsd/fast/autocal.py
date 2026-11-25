"""Keeps the camera calibration right while you drive - nothing to click.

Every frame where both of your lane's lines are found on a straight bit of road,
their on-screen lines (and where they meet, the vanishing point) are stored.
Once enough agree, the median is turned into a calibration with the same maths
as F4 (see perception/calibrate.py). If it differs noticeably from the one in
use - you zoomed, changed field of view, swapped to a truck - it's applied and
saved, so lanes, distances and stop lines stay accurate.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from ..perception.calibrate import from_lines
from ..settings import Calibration


class AutoCalibrator:
    def __init__(self, cal: Calibration, need: int = 45, cooldown_s: float = 8.0, lane_width_m: float = 3.7):
        self.cal = cal
        self.need = need
        self.cooldown = cooldown_s
        self.lane_width_m = lane_width_m
        self.samples: deque = deque(maxlen=150)
        self.last_apply = 0.0
        self.applied = 0
        self.message = ""

    def add(self, pair, W: int, H: int, car_rear_row: float | None, t: float | None = None) -> dict | None:
        """Feed one frame's ego-lane line pair. Returns new calibration fields when it's time to change."""
        t = time.perf_counter() if t is None else t
        if pair is None:
            return None
        (la, lb), (ra, rb) = pair
        if abs(la - ra) < 1e-3:
            return None
        vy = (rb - lb) / (la - ra)
        vx = la * vy + lb
        if not (0.15 * H < vy < 0.7 * H and 0.3 * W < vx < 0.7 * W):
            return None
        self.samples.append((la, lb, ra, rb, vx, vy, car_rear_row if car_rear_row else -1.0))
        if len(self.samples) < self.need or t - self.last_apply < self.cooldown:
            return None
        arr = np.array(self.samples)
        # keep the samples whose vanishing point agrees with the majority (drop curves / lane changes)
        med_vx, med_vy = np.median(arr[:, 4]), np.median(arr[:, 5])
        ok = (np.abs(arr[:, 4] - med_vx) < 0.03 * W) & (np.abs(arr[:, 5] - med_vy) < 0.03 * H)
        if ok.sum() < self.need * 0.6:
            return None
        good = arr[ok]
        L = np.array([np.median(good[:, 0]), np.median(good[:, 1])])
        R = np.array([np.median(good[:, 2]), np.median(good[:, 3])])
        rears = good[:, 6][good[:, 6] > 0]
        rear = float(np.median(rears)) if rears.size else None
        fields, msg = from_lines(L, R, W, H, self.cal, self.lane_width_m, rear)
        if fields is None:
            self.message = msg
            return None
        # only change when it's noticeably different from what's in use
        old = np.array(self.cal.src_quad)
        new = np.array(fields["src_quad"])
        if np.abs(old - new).max() < 0.012 and abs(self.cal.cam_to_bumper_m - fields["cam_to_bumper_m"]) < 1.5:
            self.last_apply = t
            self.samples.clear()
            return None
        self.last_apply = t
        self.samples.clear()
        self.applied += 1
        self.message = "Re-calibrated automatically: " + msg
        return fields
