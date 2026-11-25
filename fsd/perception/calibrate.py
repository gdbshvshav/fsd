"""Auto-calibration: work out the bird's-eye trapezoid from a straight lane.

Drive in the middle of a straight lane, press F4 (or run `python -m fsd.tools.calibrate`).
It finds your two lane lines, where they meet (the vanishing point / horizon),
and uses Roblox's camera field of view plus the standard lane width to work out
distances. No clicking needed; a manual click mode exists as a backup.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..settings import Calibration


def _paint_mask(frame: np.ndarray) -> np.ndarray:
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    L, S = hls[:, :, 1], hls[:, :, 2]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k)
    white = (th > 30) & (S < 90)
    yellow = (hsv[:, :, 0] >= 14) & (hsv[:, :, 0] <= 34) & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 90)
    return ((white | yellow).astype(np.uint8)) * 255


def find_lane_pair(frame: np.ndarray, ego_mask: list[float]) -> tuple[np.ndarray, np.ndarray] | None:
    """Return two lines (x = a*y + b) for the ego lane's left and right markings."""
    H, W = frame.shape[:2]
    m = _paint_mask(frame)
    m[: int(H * 0.45)] = 0
    x1, y1, x2, y2 = ego_mask
    m[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)] = 0
    lines = cv2.HoughLinesP(m, 1, np.pi / 180, threshold=60, minLineLength=int(H * 0.06), maxLineGap=int(H * 0.05))
    if lines is None:
        return None
    yb = H * 0.93
    segs = []
    for xa, ya, xb, yb_ in lines.reshape(-1, 4).astype(np.float64):
        if abs(yb_ - ya) < 8:
            continue
        a = (xb - xa) / (yb_ - ya)          # dx/dy
        b = xa - a * ya
        if abs(a) < 0.15 or abs(a) > 6:      # near-vertical on screen (poles) or flat (stop bars)
            continue
        segs.append((a, b, math.hypot(xb - xa, yb_ - ya)))
    if len(segs) < 2:
        return None
    # Robust vanishing point: the point most lane-paint lines pass through.
    A = np.array([[1.0, -a] for a, _, _ in segs])
    bv = np.array([b for _, b, _ in segs])
    wv = np.array([ln for _, _, ln in segs])
    keep = np.ones(len(segs), bool)
    vp = None
    for _ in range(4):
        if keep.sum() < 2:
            return None
        sw = np.sqrt(wv[keep])
        sol, *_ = np.linalg.lstsq(A[keep] * sw[:, None], bv[keep] * sw, rcond=None)
        vp = sol                              # (x0, y0)
        res = np.abs(A @ vp - bv)
        keep = res < max(W * 0.02, np.percentile(res, 60))
    x0, y0 = vp
    left, right = [], []
    for (a, b, ln), k in zip(segs, keep):
        if not k:
            continue
        # snap the line through the VP, keep its bottom position
        xbot = a * yb + b
        a2 = (xbot - x0) / (yb - y0) if abs(yb - y0) > 1 else a
        b2 = xbot - a2 * yb
        if a2 < 0 and xbot < x0:
            left.append((x0 - xbot, a2, b2, ln))
        elif a2 > 0 and xbot > x0:
            right.append((xbot - x0, a2, b2, ln))
    if not left or not right:
        return None

    def pick(c):
        c.sort(key=lambda r: r[0])
        d0 = c[0][0]
        grp = [r for r in c if r[0] < d0 + W * 0.03]
        wsum = sum(r[3] for r in grp)
        return np.array([sum(r[1] * r[3] for r in grp) / wsum, sum(r[2] * r[3] for r in grp) / wsum])
    return pick(left), pick(right)


def auto_calibrate(frames: list[np.ndarray], cal: Calibration, lane_width_m: float = 3.7) -> tuple[dict | None, str]:
    """Returns (new calibration fields, message)."""
    pairs = []
    H = W = 0
    for f in frames:
        H, W = f.shape[:2]
        r = find_lane_pair(f, cal.ego_mask)
        if r is not None:
            pairs.append(r)
    if len(pairs) < max(1, len(frames) // 3):
        return None, "Couldn't see both lane lines. Drive in the middle of a straight, marked lane and try again."
    L = np.median([p[0] for p in pairs], axis=0)
    R = np.median([p[1] for p in pairs], axis=0)
    fields, msg = from_lines(L, R, W, H, cal, lane_width_m)
    if fields:
        msg = f"Calibrated from {len(pairs)} frames: " + msg
    return fields, msg


def from_lines(L: np.ndarray, R: np.ndarray, W: int, H: int, cal: Calibration,
               lane_width_m: float = 3.7, car_rear_row: float | None = None) -> tuple[dict | None, str]:
    """Calibration from the ego lane's left/right lines on screen (x = a*y + b, pixels).

    Camera model: a flat road seen by Roblox's camera. The lane width at one row
    gives the distance to that row; every other row's distance follows from the
    horizon. The bottom of your car (car_rear_row, fraction of screen height)
    tells us where the car is, so the bird's-eye view starts at your FRONT bumper.
    ER:LC's chase camera sits high and far back, so the bottom of the screen is
    road BEHIND the car.
    """
    if abs(L[0] - R[0]) < 1e-6:
        return None, "Lane lines look parallel on screen; is the camera top-down?"
    y_h = (R[1] - L[1]) / (L[0] - R[0])
    x_h = L[0] * y_h + L[1]
    if not (H * 0.15 < y_h < H * 0.7) or not (W * 0.3 < x_h < W * 0.7):
        return None, (f"Odd vanishing point ({x_h / W:.0%}, {y_h / H:.0%}). Use a straight road with the camera "
                      "straight behind the car, then try again.")
    yr = H * 0.93                                   # reference row for the lane width
    wr = (R[0] * yr + R[1]) - (L[0] * yr + L[1])
    if not (W * 0.05 < wr < W * 1.2):
        return None, "The lane's width on screen looks wrong; make sure both of YOUR lane's lines are visible."
    focal = (H / 2) / math.tan(math.radians(cal.vfov_deg) / 2)
    K = focal * lane_width_m / wr * (yr - y_h)      # distance(y) = K / (y - y_h)

    def dist(y: float) -> float:
        return K / max(1e-3, y - y_h)

    def row(d: float) -> float:
        return y_h + K / d

    rear = car_rear_row if car_rear_row is not None else cal.ego_mask[3]
    if rear and H * 0.3 < rear * H < H * 0.98 and rear * H > y_h + 5:
        cam_to_bumper = dist(rear * H) + 4.6        # rear wheels + one car length
    else:
        cam_to_bumper = cal.cam_to_bumper_m
    near = cam_to_bumper + 1.0
    far = cam_to_bumper + 40.0
    yb, yt = row(near), row(far)
    if not (y_h + 3 < yt < yb <= H):
        return None, "Camera is too low or too flat to see the road ahead; zoom out a little."
    xl_b, xr_b = L[0] * yb + L[1], R[0] * yb + R[1]
    xl_t, xr_t = L[0] * yt + L[1], R[0] * yt + R[1]
    fields = {
        "width": int(W), "height": int(H),
        "src_quad": [[float(xl_b / W), float(yb / H)], [float(xr_b / W), float(yb / H)],
                     [float(xr_t / W), float(yt / H)], [float(xl_t / W), float(yt / H)]],
        "quad_width_m": float(lane_width_m),
        "quad_depth_m": round(float(far - near), 2),
        "bumper_to_quad_m": 1.0,
        "cam_to_bumper_m": round(float(cam_to_bumper), 2),
    }
    return fields, (f"horizon at {y_h / H:.0%} of screen, camera {cam_to_bumper:.0f} m behind your front bumper, "
                    f"road view {near - cam_to_bumper:.0f}-{far - cam_to_bumper:.0f} m ahead.")


def apply(cal: Calibration, fields: dict) -> None:
    for k, v in fields.items():
        setattr(cal, k, v)
