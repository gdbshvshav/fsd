"""Traffic lights without AI: find the lit lamps themselves.

Measured on real ER:LC screenshots: a lit lamp is clipped to full brightness
(V >= ~245) and strongly coloured, while the yellow signal housings stay around
V 215-230 and the sky is a soft blue. So:

1. In the part of the picture above the road, take pixels that are both
   blazing bright and saturated, coloured red, yellow or green.
2. Group them into small round blobs (a lamp is a few pixels to ~20 px).
3. A lamp must sit in a housing: dark (visor/body) or signal-yellow pixels
   around it. Street lights (white), sky and car tail lights fail this.
4. The state is the lamp's colour; vertical heads also get a position check.
5. Each lamp becomes an Obj("light") with a box the size of a signal head, so
   the existing SignalTracker picks the light over your lane, votes between
   heads at the same junction and spots flashing-red outages.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..world import Obj

LAMP_D_M = 0.3          # lamp diameter (m), for distance


def find_lamps(frame: np.ndarray, horizon_y: float, scale_to: float = 1.0,
               ignore: list[tuple[int, int, int, int]] | None = None) -> list[Obj]:
    """frame: picture to search (any size). Boxes are returned multiplied by scale_to."""
    H, W = frame.shape[:2]
    y1 = int(min(H, max(10, horizon_y + 0.06 * H)))
    y0 = int(H * 0.075)                       # below the Roblox / ER:LC top bar icons
    roi = frame[y0:y1]
    if roi.size == 0:
        return []
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    lit = (V >= 243) & (S >= 100)
    red = lit & ((Hh <= 12) | (Hh >= 166))
    yel = lit & (Hh >= 13) & (Hh <= 36)
    grn = lit & (Hh >= 40) & (Hh <= 100)
    any_lit = (red | yel | grn).astype(np.uint8)
    if not any_lit.any():
        return []
    any_lit = cv2.dilate(any_lit, np.ones((2, 2), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(any_lit)
    dark = V < 95
    housing_yel = (Hh >= 15) & (Hh <= 35) & (S > 90) & (V >= 110) & (V < 240)
    out: list[Obj] = []
    max_d = max(6, int(0.03 * W))
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 2 or w > max_d or h > max_d or w > 2.5 * h or h > 2.5 * w:
            continue
        sl = (slice(y, y + h), slice(x, x + w))
        cr, cy_, cg = int(red[sl].sum()), int(yel[sl].sum()), int(grn[sl].sum())
        state = max((("red", cr), ("yellow", cy_), ("green", cg)), key=lambda kv: kv[1])[0]
        r = max(w, h)
        # housing check: ring around the lamp
        ax0, ax1 = max(0, x - 2 * r), min(roi.shape[1], x + w + 2 * r)
        ay0, ay1 = max(0, y - 3 * r), min(roi.shape[0], y + h + 3 * r)
        ring = np.ones((ay1 - ay0, ax1 - ax0), bool)
        ring[y - ay0:y - ay0 + h, x - ax0:x - ax0 + w] = False
        if ring.sum() == 0:
            continue
        hd = float(dark[ay0:ay1, ax0:ax1][ring].mean())
        hy = float(housing_yel[ay0:ay1, ax0:ax1][ring].mean())
        if hd + hy < 0.28:
            continue                       # floating bright dot (sky glint, street light, sign)
        # signal head box: ~3 lamps tall, centred so the lit lamp sits where its colour belongs
        bx0, bx1 = x - r * 0.6, x + w + r * 0.6
        if state == "red":
            by0, by1 = y - r * 0.5, y + h + r * 2.6
        elif state == "yellow":
            by0, by1 = y - r * 1.5, y + h + r * 1.5
        else:
            by0, by1 = y - r * 2.6, y + h + r * 0.5
        box = (int(bx0 * scale_to), int((by0 + y0) * scale_to), int(bx1 * scale_to), int((by1 + y0) * scale_to))
        if ignore and any(_inside(box, ig) for ig in ignore):
            continue
        conf = float(min(0.95, 0.45 + 0.5 * (hd + hy)))
        out.append(Obj("light", box, conf, extra={"state": state, "lamp_px": float(r * scale_to),
                                                   "source": "lamp"}))
    return _merge_heads(out)


def _inside(a, b) -> bool:
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def _merge_heads(objs: list[Obj]) -> list[Obj]:
    """Two lit pixels groups in one head (e.g. red + glare) -> keep the strongest."""
    objs = sorted(objs, key=lambda o: -o.score)
    kept: list[Obj] = []
    for o in objs:
        if any(_overlap(o.box, k.box) > 0.3 for k in kept):
            continue
        kept.append(o)
    return kept


def _overlap(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ar = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / ar
