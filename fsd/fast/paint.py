"""Road paint straight from the game picture (no bird's-eye warp, no AI).

Tuned on real ER:LC screenshots:
* city asphalt is dark grey, highway concrete light grey, and Roblox shadows
  turn both a dark BLUE (H~111, S~190, V~50) - paint in shadow becomes light blue
* white paint is clearly brighter than the road right next to it; we measure
  that locally (top-hat), so sun, shade and tunnels all work
* yellow paint: hue 14-34, saturated
* a lane line has road on BOTH sides; a kerb or sidewalk edge only on one
* paint far away is a few pixels wide, near the bottom it's 20+: the filter
  size grows with distance, row band by row band
* the HUD (speedometer, minimap, top bar) and your own car are blanked out
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# HUD areas of ER:LC (fractions x1, y1, x2, y2) that never contain road paint
DEFAULT_HUD = [
    [0.42, 0.79, 0.58, 1.00],   # speedometer + controls
    [0.00, 0.80, 0.10, 1.00],   # minimap
    [0.00, 0.00, 1.00, 0.06],   # top bar (Roblox menu, money)
    [0.96, 0.44, 1.00, 0.56],   # passengers list
]


@dataclass
class PaintMasks:
    paint: np.ndarray        # thin paint that could be a lane line (0/255)
    yellow: np.ndarray       # yellow part of `paint`
    wide: np.ndarray         # all bright paint incl. wide blocks (crosswalks, stop bars, arrows)
    road: np.ndarray         # road surface (0/255)
    y0: int                  # first row analysed (just below the horizon)


class PaintFinder:
    """Finds road paint in a frame of a fixed size. Rebuild when the size changes."""

    def __init__(self, width: int, height: int, horizon_y: float, hud: list | None = None):
        self.W, self.H = width, height
        self.horizon_y = float(horizon_y)
        self.y0 = int(min(height - 10, max(0, horizon_y + 3)))
        self.hud = hud if hud is not None else DEFAULT_HUD
        self._static = np.full((height, width), 255, np.uint8)
        for x1, y1, x2, y2 in self.hud:
            self._static[int(y1 * height):int(y2 * height), int(x1 * width):int(x2 * width)] = 0
        # row bands (far / middle / near) with a paint-filter width for each
        span = height - self.y0
        cuts = [self.y0, self.y0 + int(span * 0.12), self.y0 + int(span * 0.35), height]
        base = max(3, width // 160)
        self.bands = []
        for i, k in enumerate((base, base * 2 + 1, base * 4 + 1)):
            y_a, y_b = cuts[i], cuts[i + 1]
            if y_b - y_a < 2:
                continue
            k = k | 1
            self.bands.append((y_a, y_b, k,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)),
                               cv2.getStructuringElement(cv2.MORPH_RECT, (k * 3 | 1, 1)),
                               cv2.getStructuringElement(cv2.MORPH_RECT, (k * 6 | 1, max(3, k * 2) | 1)),
                               max(1, int(k * 0.9))))

    def find(self, frame: np.ndarray, ego_mask: np.ndarray | None = None) -> PaintMasks:
        H, W = frame.shape[:2]
        y0 = self.y0
        roi = frame[y0:]
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        L, Sl = hls[:, :, 1], hls[:, :, 2]
        Hh, Sv, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        shadow = (Hh >= 95) & (Hh <= 130) & (V < 125)
        road = (((Sv < 52) & (V > 16) & (V < 210)) | shadow).astype(np.uint8) * 255
        grass = (Hh >= 33) & (Hh <= 95) & (Sv > 55)
        yellow_c = (Hh >= 14) & (Hh <= 34) & (Sv > 70) & (V > 90)
        whiteish = ((Sl < 95) | ((Hh >= 95) & (Hh <= 130))) & ~grass

        paint = np.zeros_like(L)
        yel = np.zeros_like(L)
        wide = np.zeros_like(L)
        for y_a, y_b, k, k_line, k_bg, k_wide, side in self.bands:
            a, b = y_a - y0, y_b - y0
            Lb = L[a:b]
            # thin bright stripes: brighter than the road right next to them
            th = cv2.morphologyEx(Lb, cv2.MORPH_TOPHAT, k_line)
            local = cv2.subtract(Lb, cv2.morphologyEx(Lb, cv2.MORPH_OPEN, k_bg))
            thr = 22 if k > 6 else 18
            thin_white = (th > thr) & (local > thr) & whiteish[a:b]
            Vb = V[a:b]
            thv = cv2.morphologyEx(Vb, cv2.MORPH_TOPHAT, k_line)
            thin_yel = yellow_c[a:b] & (thv > thr * 0.6)
            # road on both sides (kills kerbs / sidewalk edges)
            rb = road[a:b]
            left = np.zeros_like(rb)
            right = np.zeros_like(rb)
            s = side + max(1, k // 2)
            left[:, s:] = rb[:, :-s]
            right[:, :-s] = rb[:, s:]
            both = left & right
            band_paint = ((thin_white | thin_yel).astype(np.uint8) * 255) & both
            paint[a:b] = band_paint
            yel[a:b] = (thin_yel.astype(np.uint8) * 255) & both
            # wide paint (crosswalk blocks, stop bars, arrows): bright vs a big neighbourhood
            loc_w = cv2.subtract(Lb, cv2.morphologyEx(Lb, cv2.MORPH_OPEN, k_wide))
            # ...and not road-coloured itself: plain road next to a dark car or shadow is
            # "brighter than its surroundings" too, but it's still road, not paint
            not_road = (rb == 0)
            wide[a:b] = (((loc_w > 26) & whiteish[a:b] & not_road).astype(np.uint8) * 255) & \
                cv2.dilate(rb, np.ones((3, 3), np.uint8))

        stat = self._static[y0:] if self._static.shape == (H, W) else None
        for m in (paint, yel, wide):
            if stat is not None:
                m &= stat
            if ego_mask is not None:
                m &= ~ego_mask[y0:]
        full = lambda m: np.vstack([np.zeros((y0, W), np.uint8), m])  # noqa: E731
        return PaintMasks(full(paint), full(yel), full(wide), full(road), y0)
