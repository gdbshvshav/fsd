"""Bird's-eye mini map (bottom-right of the camera view), Tesla style.

Shows, from above with a slight tilt, what the driving core believes:
our lane (filtered), the lines' colours (yellow centre = the other side of the road),
the road edges when there are no lines, the planned path, the stop point, other
road users, and our car. Everything is eased so the picture glides.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from .anim import Spring

ROAD = (40, 43, 52)
EDGE = (90, 94, 104)
WHITE_LINE = (210, 214, 222)
YELLOW_LINE = (250, 200, 40)
PATH = (56, 132, 255)
CAR = (150, 154, 164)
LEAD = (90, 150, 255)
EGO = (230, 232, 238)


class MiniMap:
    def __init__(self):
        self.coef = [Spring(0.0, 3.0) for _ in range(3)]        # lane centre a, b, c
        self.width = Spring(3.7, 2.0)
        self.edge_l = Spring(-6.0, 2.0)
        self.edge_r = Spring(6.0, 2.0)
        self.alpha = Spring(0.0, 2.0)
        self.cars: dict[int, tuple[Spring, Spring]] = {}
        self.path = None

    def _proj(self, rect: pygame.Rect, x: float, y: float) -> tuple[float, float]:
        """Vehicle metres -> panel pixels. Slight perspective: far things narrower."""
        span = 70.0
        u = (x + 8.0) / span                       # 0 at 8 m behind, 1 at 62 m ahead
        scale = 1.0 - 0.45 * max(0.0, min(1.2, u))
        px = rect.centerx + y * rect.w / 22.0 * scale
        py = rect.bottom - 6 - u * (rect.h - 12)
        return px, py

    def draw(self, screen: pygame.Surface, rect: pygame.Rect, w, plan, dt: float) -> None:
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, (12, 13, 17, 200), panel.get_rect(), border_radius=14)
        pygame.draw.rect(panel, (255, 255, 255, 22), panel.get_rect(), 1, border_radius=14)
        R = panel.get_rect()
        have = w is not None and w.lanes.conf > 0.1
        a = self.alpha.step(1.0 if have else 0.35, dt)
        if have:
            for s, v in zip(self.coef, w.lanes.center):
                s.step(v, dt)
            self.width.step(w.lanes.width_m or 3.7, dt)
        else:
            for s, v in zip(self.coef, (0.0, 0.0, 0.0)):
                s.step(v, dt)
        A, B, C = (s.x for s in self.coef)
        half = self.width.x / 2
        xs = np.linspace(-8.0, 62.0, 30)

        def line(off):
            return [self._proj(R, x, A * x * x + B * x + C + off) for x in xs]

        # road edges (if known) else two lanes each side
        e = w.road_edges if w is not None else None
        if e is not None and getattr(e, "ok", False):
            yl, yr = e.at(10.0)
            self.edge_l.step(yl - C, dt)
            self.edge_r.step(yr - C, dt)
        else:
            self.edge_l.step(-half - 3.7, dt)
            self.edge_r.step(half + 3.7, dt)
        left_edge = line(self.edge_l.x)
        right_edge = line(self.edge_r.x)
        pygame.draw.polygon(panel, (*ROAD, int(255 * a)), left_edge + right_edge[::-1])
        pygame.draw.lines(panel, (*EDGE, int(200 * a)), False, left_edge, 2)
        pygame.draw.lines(panel, (*EDGE, int(200 * a)), False, right_edge, 2)
        # lane lines
        for side, off in (("left", -half), ("right", half)):
            ln = getattr(w.lanes, side) if w is not None else None
            col = YELLOW_LINE if (ln is not None and ln.color == "yellow") else WHITE_LINE
            pts = line(off)
            if ln is not None and ln.dashed:
                for i in range(0, len(pts) - 1, 2):
                    pygame.draw.line(panel, (*col, int(230 * a)), pts[i], pts[i + 1], 2)
            else:
                pygame.draw.lines(panel, (*col, int(230 * a)), False, pts, 2)
        # planned path ribbon
        if plan is not None and plan.path and plan.mode.value != "off":
            P = np.asarray(plan.path, np.float64)
            if self.path is None or self.path.shape != P.shape:
                self.path = P.copy()
            k = 1 - math.exp(-8 * dt)
            self.path += (P - self.path) * k
            L = [self._proj(R, x, y - 0.8) for x, y in self.path if -2 < x < 62]
            Rr = [self._proj(R, x, y + 0.8) for x, y in self.path if -2 < x < 62]
            if len(L) > 1:
                pygame.draw.polygon(panel, (*PATH, 120), L + Rr[::-1])
        # stop point
        if plan is not None and plan.stop_at is not None and plan.stop_at < 60:
            d = plan.stop_at
            yc = A * d * d + B * d + C
            pygame.draw.line(panel, (235, 64, 64), self._proj(R, d, yc - half), self._proj(R, d, yc + half), 4)
        # other road users
        seen = set()
        if w is not None:
            for o in w.objects:
                if o.kind not in ("car", "emergency", "person", "cone", "obstacle") or not (-8 < o.x < 62):
                    continue
                tid = o.track_id if o.track_id >= 0 else id(o)
                seen.add(tid)
                sx, sy = self.cars.setdefault(tid, (Spring(o.x, 4.0), Spring(o.y, 4.0)))
                x, y = sx.step(o.x, dt), sy.step(o.y, dt)
                lead = plan is not None and plan.lead is not None and plan.lead.track_id == o.track_id
                if o.kind in ("car", "emergency", "obstacle"):
                    col = LEAD if lead else ((235, 64, 64) if o.kind == "emergency" else CAR)
                    self._box(panel, R, x, y, 4.5, 1.9, col)
                elif o.kind == "person":
                    pygame.draw.circle(panel, (240, 200, 80), self._proj(R, x, y), 4)
                else:
                    px, py = self._proj(R, x, y)
                    pygame.draw.polygon(panel, (255, 140, 0), [(px, py - 6), (px - 4, py + 2), (px + 4, py + 2)])
        for tid in [t for t in self.cars if t not in seen]:
            del self.cars[tid]
        # our car
        self._box(panel, R, -4.6, 0.0, 4.6, 1.9, EGO)
        screen.blit(panel, rect.topleft)

    def _box(self, panel, R, x, y, length, width, col) -> None:
        pts = [self._proj(R, x, y - width / 2), self._proj(R, x + length, y - width / 2),
               self._proj(R, x + length, y + width / 2), self._proj(R, x, y + width / 2)]
        pygame.draw.polygon(panel, col, pts, 0)
        pygame.draw.polygon(panel, (20, 20, 24), pts, 1)
