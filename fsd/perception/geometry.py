"""Ground-plane geometry: screen pixels <-> bird's-eye view <-> metres.

Vehicle frame used everywhere in the planner:
    x = metres forward from the front bumper, y = metres to the RIGHT.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..settings import Calibration


@dataclass
class Geometry:
    W: int                  # frame width the geometry is built for
    H: int
    bev_w: int              # bird's-eye image size in px
    bev_h: int
    m_per_px_x: float
    m_per_px_y: float
    M: np.ndarray           # screen -> BEV
    Minv: np.ndarray        # BEV -> screen
    bumper_to_bev_bottom: float
    focal_px: float
    horizon_y: float        # screen y of the horizon (vanishing line)

    @classmethod
    def build(cls, cal: Calibration, W: int, H: int, bev_w: int = 400, lateral_m: float = 20.0,
              depth_m: float = 60.0) -> "Geometry":
        """BEV covers `lateral_m` across (centred) and `depth_m` ahead of the trapezoid bottom."""
        q = np.array(cal.src_quad, dtype=np.float32) * np.array([W, H], dtype=np.float32)
        mpx = lateral_m / bev_w
        bev_h = int(round(depth_m / mpx))
        cx = bev_w / 2
        half = cal.quad_width_m / 2 / mpx
        depth = cal.quad_depth_m / mpx
        dst = np.array([[cx - half, bev_h], [cx + half, bev_h],
                        [cx + half, bev_h - depth], [cx - half, bev_h - depth]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(q, dst)
        Minv = cv2.getPerspectiveTransform(dst, q)
        # Horizon: where the two trapezoid sides meet.
        (x1, y1), (x2, y2) = q[0], q[3]
        (x3, y3), (x4, y4) = q[1], q[2]
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) > 1e-6:
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
        else:
            py = H * 0.4
        focal = (H / 2) / math.tan(math.radians(cal.vfov_deg) / 2)
        return cls(W, H, bev_w, bev_h, mpx, mpx, M, Minv, cal.bumper_to_quad_m, focal, float(py))

    # ---- conversions ----
    def warp(self, img: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
        if img.shape[1] != self.W or img.shape[0] != self.H:
            img = cv2.resize(img, (self.W, self.H))
        return cv2.warpPerspective(img, self.M, (self.bev_w, self.bev_h), flags=interp)

    def screen_to_bev(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.M).reshape(-1, 2)

    def bev_to_screen(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.Minv).reshape(-1, 2)

    def bev_to_vehicle(self, pts: np.ndarray) -> np.ndarray:
        """BEV px (u, v) -> vehicle metres (x fwd, y right)."""
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        x = (self.bev_h - p[:, 1]) * self.m_per_px_y + self.bumper_to_bev_bottom
        y = (p[:, 0] - self.bev_w / 2) * self.m_per_px_x
        return np.stack([x, y], axis=1)

    def vehicle_to_bev(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        v = self.bev_h - (p[:, 0] - self.bumper_to_bev_bottom) / self.m_per_px_y
        u = p[:, 1] / self.m_per_px_x + self.bev_w / 2
        return np.stack([u, v], axis=1)

    def screen_to_vehicle(self, pts: np.ndarray) -> np.ndarray:
        return self.bev_to_vehicle(self.screen_to_bev(pts))

    def vehicle_to_screen(self, pts: np.ndarray) -> np.ndarray:
        return self.bev_to_screen(self.vehicle_to_bev(pts))

    def ground_point(self, sx: float, sy: float) -> tuple[float, float] | None:
        """Vehicle-frame position of a screen point that touches the ground (e.g. a car's bottom)."""
        if sy <= self.horizon_y + 2:
            return None
        x, y = self.screen_to_vehicle(np.array([[sx, sy]]))[0]
        if not (0 < x < 400):
            return None
        return float(x), float(y)

    def distance_from_height(self, px_height: float, real_height_m: float) -> float:
        """Pinhole distance from an object's on-screen height (used for signs/lights)."""
        if px_height <= 1:
            return 999.0
        return real_height_m * self.focal_px / px_height
