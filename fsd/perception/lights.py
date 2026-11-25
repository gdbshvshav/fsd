"""Traffic lights: which lamp is lit, which light is ours, and outages.

Fixes v3's "any green/red/yellow blob is a traffic light":
* A light must be a dark housing with ONE small, bright, saturated lamp in it.
  Green trees, red brick, yellow paint and car tail lights fail that test.
* Lamp position inside the housing must match its colour (red top, green bottom).
* Yellow signal BACKS (big flat yellow boxes facing away) are recognised and ignored.
* Only lights above our lane and facing us count; the nearest wins and nearby
  lights at the same junction vote.
* A red that blinks on/off with no green/yellow over ~2 s = OUTAGE -> all-way stop.
"""
from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from ..world import Light, Obj, SignalInfo
from .geometry import Geometry

HOUSING_H_M = 1.3   # ER:LC 3-lamp head height incl. frame (m), for distance estimates


def lamp_masks(hsv: np.ndarray) -> dict[str, np.ndarray]:
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    lit = (V > 170) & (S > 80)
    return {
        "red": lit & ((H <= 9) | (H >= 168)),
        "yellow": lit & (H >= 12) & (H <= 34),
        "green": lit & (H >= 40) & (H <= 100),
    }


def classify_light(crop: np.ndarray) -> tuple[str, float]:
    """Return ('red'|'yellow'|'green'|'off'|'back', confidence) for a light crop.

    Measured on real ER:LC screenshots: a LIT lamp is clipped to full brightness
    (V ~255, saturated), while the yellow housing tops out around V 215-230. So
    "lit" = the very brightest saturated pixels, and their hue gives the colour.
    Lamp position (red top, green bottom) is used as a tie-breaker.
    """
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 4:
        return "off", 0.0
    h, w = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(np.int16), hsv[:, :, 1], hsv[:, :, 2]
    vmax = int(V.max())
    yellow_body = ((H >= 15) & (H <= 35) & (S > 90) & (V > 110)).mean()
    if vmax < 238:
        # nothing lit. A face-on head shows dark lamps; a back/side is flat yellow.
        dark = (V < 90).mean()
        if yellow_body > 0.5 and dark < 0.1:
            return "back", float(min(1.0, yellow_body))
        return "off", 0.5 if dark > 0.15 else 0.3
    lit = (V >= max(238, vmax - 10)) & (S > 90)
    n = int(lit.sum())
    if n < 2:
        return "off", 0.3
    hh = H[lit]
    counts = {
        "red": int(((hh <= 12) | (hh >= 166)).sum()),
        "yellow": int(((hh >= 13) & (hh <= 36)).sum()),
        "green": int(((hh >= 40) & (hh <= 100)).sum()),
    }
    best = max(counts, key=counts.get)
    if counts[best] < 2:
        return "off", 0.3
    conf = 0.55 + 0.35 * counts[best] / max(1, n)
    if h >= w * 1.3:                                   # vertical head: check the lamp sits right
        ys = np.nonzero(lit)[0]
        cy = float(ys.mean()) / h
        expect = {"red": (0.0, 0.5), "yellow": (0.2, 0.8), "green": (0.5, 1.0)}[best]
        conf += 0.1 if expect[0] <= cy <= expect[1] else -0.2
    if best == "yellow" and yellow_body > 0.6 and n > 0.5 * h * w:
        return "back", 0.5                             # whole box glowing yellow = sunlit back
    return best, float(np.clip(conf, 0.0, 1.0))


def _classify_by_hue(hsv: np.ndarray) -> tuple[str, float]:
    m = lamp_masks(hsv)
    counts = {k: int(v.sum()) for k, v in m.items()}
    best = max(counts, key=counts.get)
    area = hsv.shape[0] * hsv.shape[1]
    if counts[best] < max(3, 0.015 * area):
        return "off", 0.3
    second = sorted(counts.values())[-2]
    return best, 0.6 if second < 0.5 * counts[best] else 0.35


def find_light_candidates(frame: np.ndarray, horizon_y: float) -> list[Obj]:
    """Colour fallback when no YOLO: dark housings with one lit lamp, above the road."""
    H, W = frame.shape[:2]
    top = 0
    bottom = int(min(H * 0.75, horizon_y + H * 0.25))
    roi = frame[top:bottom]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lit = np.zeros(roi.shape[:2], np.uint8)
    for mk in lamp_masks(hsv).values():
        lit |= mk.astype(np.uint8)
    lit = cv2.morphologyEx(lit * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(lit)
    dark = hsv[:, :, 2] < 70
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 6 or a > 2500 or w > 3 * h or h > 3 * w:
            continue
        r = max(w, h)
        # housing: dark area around the lamp, ~3 lamps tall
        x1, x2 = max(0, x - r), min(roi.shape[1], x + w + r)
        y1, y2 = max(0, y - 3 * r), min(roi.shape[0], y + h + 3 * r)
        ring = dark[y1:y2, x1:x2]
        if ring.size == 0 or ring.mean() < 0.35:
            continue
        out.append(Obj("light", (int(x1), int(y1 + top), int(x2), int(y2 + top)), 0.4))
    return out


class SignalTracker:
    """Chooses the signal that controls our lane and detects outages."""

    def __init__(self, geo: Geometry, cam_to_bumper: float):
        self.geo = geo
        self.cam_to_bumper = cam_to_bumper
        self.hist: deque[tuple[float, str]] = deque(maxlen=240)
        self.last_state = Light.NONE
        self.last_dist = 999.0
        self.last_seen = 0.0
        self.last_info = SignalInfo()

    def light_position(self, box) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        h = max(1, y2 - y1)
        d_cam = self.geo.distance_from_height(h, HOUSING_H_M)
        lateral = ((x1 + x2) / 2 - self.geo.W / 2) / self.geo.focal_px * d_cam
        return d_cam - self.cam_to_bumper, lateral

    def update(self, frame: np.ndarray, lights: list[Obj], lane_center_y: float,
               flash_window: float = 2.0, t: float | None = None) -> SignalInfo:
        t = time.perf_counter() if t is None else t
        relevant: list[tuple[float, str, float, Obj]] = []
        for ob in lights:
            x1, y1, x2, y2 = ob.box
            state = ob.extra.get("state")
            conf = ob.score
            if state is None or state == "off":
                pad = 2
                crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                state, c2 = classify_light(crop)
                conf = 0.5 * conf + 0.5 * c2
            if state == "back":
                continue
            dist, lat = self.light_position(ob.box)
            ob.x, ob.y = dist, lat
            ob.extra["state"] = state
            if dist < 2 or dist > 120:
                continue
            # must be roughly over our road: within ~7 m sideways of our lane centre
            if abs(lat - lane_center_y) > 7.0 + dist * 0.05:
                continue
            relevant.append((dist, state, conf, ob))

        info = SignalInfo()
        if relevant:
            relevant.sort(key=lambda r: r[0])
            d0 = relevant[0][0]
            group = [r for r in relevant if r[0] < d0 + 8]       # same junction
            votes: dict[str, float] = {}
            for _d, s, c, o in group:
                votes[s] = votes.get(s, 0.0) + c * (1.0 / (1.0 + abs(o.y - lane_center_y)))
            state = max(votes, key=votes.get)
            total = sum(votes.values()) or 1.0
            info.state = {"red": Light.RED, "yellow": Light.YELLOW, "green": Light.GREEN}.get(state, Light.UNKNOWN)
            info.conf = float(votes[state] / total)
            info.distance = float(d0)
            info.box = group[0][3].box
            self.hist.append((t, state))
            self.last_seen = t
            self.last_dist = d0
        elif t - self.last_seen < 0.8 and self.last_state in (Light.RED, Light.FLASHING_RED):
            # Lamp went dark for a moment: could be the off-phase of an outage
            self.hist.append((t, "off"))
            info.state = self.last_state
            info.distance = self.last_dist
            info.conf = 0.4

        # outage detection
        while self.hist and t - self.hist[0][0] > flash_window:
            self.hist.popleft()
        states = [s for _, s in self.hist]
        if states and "green" not in states and "yellow" not in states and "red" in states:
            flips = sum(1 for a, b in zip(states, states[1:]) if (a == "red") != (b == "red"))
            if flips >= 2:
                info.state = Light.FLASHING_RED
                info.distance = self.last_dist
                info.conf = max(info.conf, 0.6)
        self.last_state = info.state
        self.last_info = info
        return info
