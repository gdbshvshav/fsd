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


class Bootstrap:
    """Calibrate from scratch when the current calibration is so wrong that no lanes are found
    (another zoom, camera angle or window shape - the defaults were measured on one screen).

    Works on the screen only: finds the painted lines, where they meet, and the two either side
    of our car (calibrate.lane_pair_from_mask), a few times in a row, then turns the agreeing
    result into a calibration with the same maths as F4. The normal AutoCalibrator fine-tunes
    it afterwards."""

    def __init__(self, cal: Calibration, lane_width_m: float = 3.7):
        self.cal = cal
        self.lane_width_m = lane_width_m
        self.low_since: float | None = None
        self.samples: deque = deque(maxlen=12)
        self.last_try = 0.0
        self.last_apply = -1e9
        self.applied = 0
        self.message = ""

    def add(self, paint, car_box, horizon_y: float, W: int, H: int, lanes_conf: float, rear_frac,
            t: float | None = None) -> dict | None:
        from ..perception.calibrate import lane_pair_from_mask
        t = time.perf_counter() if t is None else t
        if lanes_conf > 0.25:
            self.low_since = None
            self.samples.clear()
            return None
        if self.low_since is None:
            self.low_since = t
        # lanes missing for a moment, and not right after a change (give it time to settle)
        if t - self.low_since < 0.8 or t - self.last_apply < 6.0 or t - self.last_try < 0.12:
            return None
        self.last_try = t
        pair = lane_pair_from_mask(paint, car_box, horizon_y)
        if pair is None:
            return None
        (la, lb), (ra, rb) = pair
        vy = (rb - lb) / (la - ra)
        vx = la * vy + lb
        self.samples.append((t, la, lb, ra, rb, vx, vy))
        recent = np.array([s for s in self.samples if t - s[0] < 3.0])
        if len(recent) < 5:
            return None
        mvx, mvy = np.median(recent[:, 5]), np.median(recent[:, 6])
        ok = (np.abs(recent[:, 5] - mvx) < 0.03 * W) & (np.abs(recent[:, 6] - mvy) < 0.03 * H)
        if ok.sum() < 4:
            return None
        g = recent[ok]
        L = np.array([np.median(g[:, 1]), np.median(g[:, 2])])
        R = np.array([np.median(g[:, 3]), np.median(g[:, 4])])
        rear = rear_frac if rear_frac else car_box[3] / H
        fields, msg = from_lines(L, R, W, H, self.cal, self.lane_width_m, rear)
        self.samples.clear()
        if fields is None:
            self.message = msg
            return None
        self.last_apply = t
        self.low_since = None
        if np.abs(np.array(self.cal.src_quad) - np.array(fields["src_quad"])).max() < 0.012 \
                and abs(self.cal.cam_to_bumper_m - fields["cam_to_bumper_m"]) < 1.5:
            return None                           # same as what we have: lanes are just faint here
        self.applied += 1
        self.message = "Lanes weren't matching this camera - re-measured it from the road: " + msg
        return fields
