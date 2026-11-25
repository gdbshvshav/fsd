"""Exact outline of YOUR car on screen - no AI.

Two classic image techniques, combined:

1. Stillness. In the chase camera your car stays in the same place on screen
   while the road, grass and everything else slide past. Each update compares
   the area around the car with the previous one; pixels that keep NOT changing
   while the scene IS moving are your car. After about a second of driving this
   gives the car's true outline, black body included.
2. GrabCut (graph-cut colour segmentation) turns those seeds into a clean,
   sharp-edged outline and keeps working while you're stopped.

Recomputed ~6x a second, so it follows the car when the camera swings or you
switch vehicles.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

# Part of the game picture that always contains your car (fractions of width/height).
WORK = (0.36, 0.38, 0.64, 0.78)


class EgoShape:
    def __init__(self, period: float = 0.15, work_w: int = 200):
        self.period = period
        self.work_w = work_w
        self.mask: np.ndarray | None = None              # 0/255 in `region` pixels
        self.region: tuple[int, int, int, int] | None = None
        self.bbox: tuple[int, int, int, int] | None = None
        self._last = 0.0
        self._prev: np.ndarray | None = None             # previous grey work area
        self._still: np.ndarray | None = None            # 0..1: how reliably each pixel stays still
        self._moving_updates = 0
        self._prev_fg: np.ndarray | None = None
        self.method = ""

    def update(self, frame: np.ndarray, ego_box: tuple[int, int, int, int], t: float | None = None) -> None:
        t = time.perf_counter() if t is None else t
        if t - self._last < self.period:
            return
        self._last = t
        H, W = frame.shape[:2]
        rx1, ry1, rx2, ry2 = int(WORK[0] * W), int(WORK[1] * H), int(WORK[2] * W), int(WORK[3] * H)
        x1, y1, x2, y2 = ego_box
        rx1, ry1 = max(0, min(rx1, x1 - 10)), max(0, min(ry1, y1 - 10))
        rx2, ry2 = min(W, max(rx2, x2 + 10)), min(H, max(ry2, y2 + 10))
        region = (rx1, ry1, rx2, ry2)
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return
        scale = self.work_w / crop.shape[1]
        small = cv2.resize(crop, (self.work_w, max(8, int(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        grey = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.int16)

        # ---- 1. stillness while the scene moves ----
        if self.region != region or self._prev is None or self._prev.shape != grey.shape:
            self._still = np.full((sh, sw), 0.5, np.float32)
            self._moving_updates = 0
        else:
            diff = np.abs(grey - self._prev)
            border = np.concatenate([diff[:4].ravel(), diff[-4:].ravel(), diff[:, :4].ravel(), diff[:, -4:].ravel()])
            if np.percentile(border, 75) > 5:               # the world is moving past us
                d8 = cv2.dilate(np.clip(diff, 0, 255).astype(np.uint8), np.ones((3, 3), np.uint8))
                self._still = 0.8 * self._still + 0.2 * (d8 < 6).astype(np.float32)
                self._moving_updates += 1
        self._prev = grey
        self.region = region

        # ---- 2. GrabCut seeds ----
        gc = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
        e = max(2, int(0.04 * min(sh, sw)))
        gc[:e, :] = cv2.GC_BGD
        gc[-e:, :] = cv2.GC_BGD
        gc[:, :e] = cv2.GC_BGD
        gc[:, -e:] = cv2.GC_BGD
        bx1, by1 = int((x1 - rx1) * scale), int((y1 - ry1) * scale)
        bx2, by2 = int((x2 - rx1) * scale), int((y2 - ry1) * scale)
        inner = np.zeros((sh, sw), bool)
        inner[e:-e, e:-e] = True
        if self._moving_updates >= 6:
            # Anything that moved is definitely NOT the car (this teaches GrabCut the road's
            # colours, including flat road that only looks still). Still pixels are "probably
            # car"; a column through the middle of the car is certain.
            self.method = "stillness + grabcut"
            gc[inner & (self._still >= 0.6)] = cv2.GC_PR_FGD
            gc[inner & (self._still < 0.35)] = cv2.GC_BGD
            w3 = max(1, (bx2 - bx1) // 8)
            mx = (bx1 + bx2) // 2
            col = np.zeros((sh, sw), bool)
            col[max(0, by1 + (by2 - by1) // 6): by2 - (by2 - by1) // 6, max(0, mx - w3): mx + w3] = True
            gc[col & (self._still >= 0.6)] = cv2.GC_FGD
        else:
            self.method = "grabcut"
            w3 = max(1, (bx2 - bx1) // 6)
            mx = (bx1 + bx2) // 2
            gc[max(e, by1):min(sh - e, by2), max(e, bx1):min(sw - e, bx2)] = cv2.GC_PR_FGD
            gc[by1 + (by2 - by1) // 8: by2 - (by2 - by1) // 8, mx - w3: mx + w3] = cv2.GC_FGD
            if self._prev_fg is not None and self._prev_fg.shape == gc.shape:
                core = cv2.erode(self._prev_fg, np.ones((5, 5), np.uint8)) > 0
                gc[core & inner] = cv2.GC_FGD
        if not np.any((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)) or not np.any(gc == cv2.GC_BGD):
            return
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(small, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return
        fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        # never leak far outside the car's box (flat road can look like part of the car);
        # the box itself follows the outline, so a bigger car still gets room to grow
        lim = np.zeros_like(fg)
        mxp, myp = int(0.2 * (bx2 - bx1)), int(0.2 * (by2 - by1))
        lim[max(0, by1 - myp):by2 + myp, max(0, bx1 - mxp):bx2 + mxp] = 255
        fg &= lim
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, cents = cv2.connectedComponentsWithStats(fg)
        if n <= 1:
            return
        # the car: the biggest blob near the middle of its box
        mx, my = (bx1 + bx2) / 2, (by1 + by2) / 2
        best, best_score = None, -1e18
        for i in range(1, n):
            score = stats[i, cv2.CC_STAT_AREA] - 40 * np.hypot(cents[i][0] - mx, cents[i][1] - my)
            if score > best_score:
                best, best_score = i, score
        fg = np.where(lab == best, 255, 0).astype(np.uint8)
        if fg.sum() / 255 < 0.1 * max(1, (bx2 - bx1) * (by2 - by1)):
            return                                     # implausible: keep the previous outline
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        fg = np.zeros_like(fg)
        cv2.drawContours(fg, cnts, -1, 255, -1)        # fill windows / holes
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        self._prev_fg = fg
        full = cv2.resize(fg, (rx2 - rx1, ry2 - ry1), interpolation=cv2.INTER_LINEAR)
        full = cv2.GaussianBlur(full, (5, 5), 0)
        self.mask = np.where(full > 127, 255, 0).astype(np.uint8)
        ys, xs = np.nonzero(self.mask)
        if xs.size:
            self.bbox = (rx1 + int(xs.min()), ry1 + int(ys.min()), rx1 + int(xs.max()), ry1 + int(ys.max()))

    def paste(self, target_mask: np.ndarray, sx: float, sy: float) -> None:
        """Draw the outline into a mask of another size (scale sx, sy)."""
        if self.mask is None or self.region is None:
            return
        x1, y1, x2, y2 = self.region
        a, b = int(x1 * sx), int(y1 * sy)
        w, h = max(1, int((x2 - x1) * sx)), max(1, int((y2 - y1) * sy))
        m = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_LINEAR)
        H, W = target_mask.shape[:2]
        a2, b2 = min(W, a + w), min(H, b + h)
        if a2 <= max(0, a) or b2 <= max(0, b):
            return
        sub = m[max(0, -b):max(0, -b) + (b2 - max(0, b)), max(0, -a):max(0, -a) + (a2 - max(0, a))]
        target_mask[max(0, b):b2, max(0, a):a2] |= np.where(sub > 127, 255, 0).astype(np.uint8)
