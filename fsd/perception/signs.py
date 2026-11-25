"""Stop signs and speed-limit signs."""
from __future__ import annotations

import cv2
import numpy as np

from ..world import Obj
from .geometry import Geometry

STOP_SIGN_H_M = 0.75
SPEED_SIGN_H_M = 0.9
VALID_LIMITS = {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80}


def find_stop_signs(frame: np.ndarray, horizon_y: float) -> list[Obj]:
    """Red octagon with white inside. Colour fallback for when YOLO isn't loaded."""
    H, W = frame.shape[:2]
    y0 = int(max(0, horizon_y - H * 0.2))
    y1 = int(min(H, horizon_y + H * 0.35))
    roi = frame[y0:y1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] <= 8) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 70))
    red = cv2.morphologyEx(red.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 120:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (0.75 < w / max(h, 1) < 1.3):
            continue
        hull = cv2.convexHull(c)
        solidity = a / max(cv2.contourArea(hull), 1)
        approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
        if solidity < 0.8 or not (6 <= len(approx) <= 10):
            continue
        inner = hsv[y + h // 3:y + 2 * h // 3, x + w // 6:x + 5 * w // 6]
        white = ((inner[:, :, 1] < 60) & (inner[:, :, 2] > 170)).mean() if inner.size else 0
        if white < 0.05:
            continue
        out.append(Obj("stop_sign", (x, y + y0, x + w, y + h + y0), 0.5 + min(0.4, white)))
    return out


def find_speed_sign_crops(frame: np.ndarray, horizon_y: float) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """White upright rectangles with dark text on the roadside: candidates for OCR."""
    H, W = frame.shape[:2]
    y0 = int(max(0, horizon_y - H * 0.15))
    y1 = int(min(H, horizon_y + H * 0.3))
    roi = frame[y0:y1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 185)).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 10 or h < 14 or w > W * 0.08:
            continue
        ar = h / w
        if not (1.05 < ar < 1.8):
            continue
        fill = cv2.contourArea(c) / (w * h)
        if fill < 0.6:
            continue
        inner = roi[y:y + h, x:x + w]
        dark = (cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) < 90).mean()
        if not (0.04 < dark < 0.5):
            continue
        box = (x, y + y0, x + w, y + h + y0)
        out.append((box, frame[box[1]:box[3], box[0]:box[2]].copy()))
    return out[:4]


def parse_limit(texts: list[str]) -> int | None:
    import re
    for t in texts:
        for m in re.findall(r"\d{2}", t.replace(" ", "")):
            v = int(m)
            if v in VALID_LIMITS:
                return v
    return None


def sign_position(geo: Geometry, box, real_h: float, cam_to_bumper: float) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    d = geo.distance_from_height(max(1, y2 - y1), real_h)
    lat = ((x1 + x2) / 2 - geo.W / 2) / geo.focal_px * d
    return d - cam_to_bumper, lat
