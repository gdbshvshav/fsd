"""Stop signs and speed-limit signs."""
from __future__ import annotations

import cv2
import numpy as np

from ..world import Obj
from .geometry import Geometry

STOP_SIGN_H_M = 0.75
SPEED_SIGN_H_M = 0.9
VALID_LIMITS = {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80}


def _red_mask(hsv: np.ndarray) -> np.ndarray:
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = ((hue <= 8) | (hue >= 170)) & (sat > 120) & (val > 70)
    return cv2.morphologyEx(red.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _has_pole(img: np.ndarray, cx: float, y_bottom: int, h: int) -> bool:
    """Is there a thin post straight below (a column that differs from its surroundings)?"""
    H, W = img.shape[:2]
    y0 = int(y_bottom + 1)
    y1 = int(min(H, y_bottom + max(4, 3 * h)))
    c = int(round(cx))
    off = max(2, h)
    if y1 - y0 < 3 or c - off < 0 or c + off >= W:
        return True                                      # can't check at the edge: don't veto
    g = cv2.cvtColor(img[y0:y1, c - off:c + off + 1], cv2.COLOR_BGR2GRAY).astype(np.int16)
    centre = g[:, off - 1:off + 2].mean(axis=1)
    sides = (g[:, 0] + g[:, -1]) / 2.0
    return float(np.median(np.abs(centre - sides))) > 10.0


def _judge(hsv: np.ndarray, blob: np.ndarray, w: int, h: int, a: int, W: int) -> float | None:
    """Score for one red blob (its own hsv crop + mask), None if it isn't a stop sign."""
    if not (0.7 < w / max(h, 1) < 1.4) or w > W * 0.12:
        return None
    if a / float(w * h) < 0.55:
        return None
    val = hsv[:, :, 2]
    bright = val[blob]
    if bright.size and np.percentile(bright, 50) > 238:
        return None                                       # a lit lamp, not paint
    if w < 14:
        return 0.42 + 0.02 * w
    c = cv2.findContours(blob.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not c:
        return None
    c = max(c, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    solidity = cv2.contourArea(c) / max(cv2.contourArea(hull), 1)
    approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
    if solidity < 0.75 or not (6 <= len(approx) <= 10):
        return None
    inner = hsv[h // 3:2 * h // 3, w // 6:5 * w // 6]
    white = ((inner[:, :, 1] < 60) & (inner[:, :, 2] > 170)).mean() if inner.size else 0
    if white < 0.05:
        return None
    return 0.5 + min(0.4, float(white))


def find_stop_signs(frame: np.ndarray, horizon_y: float, scale_to: float = 1.0,
                    fine: np.ndarray | None = None) -> list[Obj]:
    """Red octagons with white inside (no AI). Seen 40+ m out: tiny ones (a few px) only need
    the right colour, size and squareness; bigger ones must look like an octagon with white
    lettering. Lit lamps (brake lights, red signal lamps) are brighter than sign paint: skipped.

    The search runs on `frame` (small, fast); if `fine` (a sharper copy of the same picture)
    is given, every candidate is re-measured there. Boxes come back in `frame` pixels x scale_to.
    """
    H, W = frame.shape[:2]
    y0 = int(max(0, horizon_y - H * 0.2))
    y1 = int(min(H, horizon_y + H * 0.35))
    roi = frame[y0:y1]
    if roi.size == 0:
        return []
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = _red_mask(hsv)
    if cv2.countNonZero(red) == 0:
        return []
    kf = fine.shape[1] / W if fine is not None else 1.0
    min_px = 2 if fine is not None else max(3, int(round(4 * W / 960)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if w < min_px or h < min_px or w > W * 0.12:
            continue
        if fine is not None:
            # re-measure in the sharp picture
            m = 2
            fx0, fy0 = int((x - m) * kf), int((y + y0 - m) * kf)
            fx1, fy1 = int((x + w + m) * kf), int((y + y0 + h + m) * kf)
            fx0, fy0 = max(0, fx0), max(0, fy0)
            crop = fine[fy0:fy1, fx0:fx1]
            if crop.size == 0:
                continue
            fh = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            fr = _red_mask(fh)
            fn, flab, fst, _ = cv2.connectedComponentsWithStats(fr, connectivity=8)
            if fn <= 1:
                continue
            j = 1 + int(np.argmax(fst[1:, cv2.CC_STAT_AREA]))
            bx, by, bw, bh, ba = (int(v) for v in fst[j])
            if bw < 4 or bh < 4:
                continue
            sub = fh[by:by + bh, bx:bx + bw]
            score = _judge(sub, flab[by:by + bh, bx:bx + bw] == j, bw, bh, ba, fine.shape[1])
            gx0, gy0 = (fx0 + bx) / kf, (fy0 + by) / kf
            gx1, gy1 = (fx0 + bx + bw) / kf, (fy0 + by + bh) / kf
        else:
            if w < min_px:
                continue
            sub = hsv[y:y + h, x:x + w]
            score = _judge(sub, lab[y:y + h, x:x + w] == i, w, h, a, W)
            gx0, gy0, gx1, gy1 = x, y + y0, x + w, y + h + y0
        if score is None:
            continue
        if w < 14 * (1 / kf if fine is not None else 1):
            if gy1 < horizon_y - H * 0.12:
                continue                                 # tiny and floating high in the sky
            # a sign stands alone on a pole; red autumn leaves come in clusters of blobs
            pad = 2 * max(w, h)
            ax0, ay0 = max(0, x - pad), max(0, y - pad)
            around = red[ay0:y + h + pad, ax0:x + w + pad]
            if cv2.countNonZero(around) > 2.2 * a:
                continue
            if not _has_pole(roi, x + w / 2, y + h, h):
                continue                                 # nothing holding it up: leaves, a roof, a car
        k = scale_to
        out.append(Obj("stop_sign", (int(gx0 * k), int(gy0 * k), int(gx1 * k), int(gy1 * k)),
                       float(min(0.9, score))))
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
