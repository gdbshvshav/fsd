"""Shows on the camera view which part of the road is OURS and which is oncoming.

* our side of the road (from the centre line to the right road edge): faint blue
* the oncoming side (left of a yellow centre line): faint red
* road edges found without lane lines: thin white curbs

All drawn into the greyed camera picture with OpenCV, so it costs ~1 ms.
"""
from __future__ import annotations

import cv2
import numpy as np


def _poly(coeffs, xs):
    a, b, c = coeffs
    return a * xs * xs + b * xs + c


def draw_road_sides(out: np.ndarray, w, to_view, reach: float = 45.0) -> None:
    """out: BGR picture (view size). to_view: vehicle (x, y) points -> view pixels (N, 2) int32,
    dropping points above the horizon."""
    if w is None:
        return
    ln = w.lanes
    e = getattr(w, "road_edges", None)
    xs = np.linspace(2.0, reach, 24)
    edges_ok = e is not None and getattr(e, "ok", False)
    right_edge = _poly(e.right, xs) if edges_ok else None
    left_edge = _poly(e.left, xs) if edges_ok else None
    centre = None
    if ln.conf > 0.3 and ln.left is not None and ln.left.color == "yellow":
        centre = _poly(ln.left.coeffs, xs)
    elif ln.conf > 0.3 and ln.left2 is not None and ln.left2.color == "yellow":
        centre = _poly(ln.left2.coeffs, xs)
    layer = out.copy()
    drew = False
    # our side: centre line (or our lane's left line) to the right edge (or our lane's right line)
    if ln.conf > 0.3:
        lft = centre if centre is not None else _poly((ln.center[0], ln.center[1], ln.center[2] - ln.width_m / 2), xs)
        rgt = right_edge if right_edge is not None else _poly((ln.center[0], ln.center[1],
                                                                ln.center[2] + ln.width_m / 2), xs)
        if np.all(rgt > lft + 1.0):
            L = to_view(np.stack([xs, lft], 1))
            R = to_view(np.stack([xs, rgt], 1))
            if len(L) > 1 and len(R) > 1:
                cv2.fillPoly(layer, [np.vstack([L, R[::-1]])], (255, 140, 40))
                drew = True
        # oncoming side: left road edge to the yellow line
        if centre is not None and left_edge is not None and np.all(centre > left_edge + 1.0):
            L = to_view(np.stack([xs, left_edge], 1))
            R = to_view(np.stack([xs, centre], 1))
            if len(L) > 1 and len(R) > 1:
                cv2.fillPoly(layer, [np.vstack([L, R[::-1]])], (60, 60, 230))
                drew = True
    elif edges_ok:
        # no lines: show the half we keep to
        mid = (left_edge + right_edge) / 2
        width = float(np.median(right_edge - left_edge))
        lft = mid if width > 6.0 else left_edge
        L = to_view(np.stack([xs, lft], 1))
        R = to_view(np.stack([xs, right_edge], 1))
        if len(L) > 1 and len(R) > 1:
            cv2.fillPoly(layer, [np.vstack([L, R[::-1]])], (255, 140, 40))
            drew = True
    if drew:
        cv2.addWeighted(layer, 0.16, out, 0.84, 0, out)
    if edges_ok:
        for ys in (left_edge, right_edge):
            pts = to_view(np.stack([xs, ys], 1))
            if len(pts) > 1:
                cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (235, 235, 235), 1, cv2.LINE_AA)
