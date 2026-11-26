"""Animated heads-up display drawn over the camera view (and the 3D view).

    +--------------------------------------------------------------+
    | (speed ring)  [ mode banner, slides + fades ]  <  >  (light)  |
    |  87 KPH        why: following car 32 m           blinkers     |
    +--------------------------------------------------------------+
                                     toasts slide in on the right ->
    steering wheel + pedal bars (bottom left)     mini-map (bottom right)

Every value goes through a spring or a tween (ui/anim.py), so the speed needle
sweeps, the banner colour fades, blinkers breathe, and messages slide in and out.
"""
from __future__ import annotations

import math
import time

import pygame

from ..world import Light, Mode
from .anim import ColorTween, Counter, Pulse, Spring, Toasts, Tween, ease_out_back, lerp_color

WHITE = (236, 238, 242)
DIM = (140, 144, 154)
BLUE = (56, 132, 255)
RED = (235, 64, 64)
YELLOW = (250, 200, 40)
GREEN = (60, 210, 110)
GLASS = (12, 13, 17)

MODE_COLOR = {
    Mode.OFF: (60, 64, 74), Mode.CRUISE: BLUE, Mode.FOLLOW: (70, 150, 255), Mode.STOPPING: (255, 150, 40),
    Mode.STOPPED: (220, 90, 60), Mode.STOP_WAIT: (220, 90, 60), Mode.TURNING: (120, 110, 255),
    Mode.LANE_CHANGE: (120, 110, 255), Mode.YIELD: (255, 120, 60), Mode.CAUTION: (230, 170, 40),
    Mode.FAULT: RED,
}

LIGHT_COLOR = {Light.RED: RED, Light.YELLOW: YELLOW, Light.GREEN: GREEN, Light.FLASHING_RED: RED}


def glass(surface: pygame.Surface, rect: pygame.Rect, alpha: int = 190, radius: int = 14,
          border: tuple | None = (255, 255, 255, 22)) -> None:
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(s, (*GLASS, alpha), s.get_rect(), border_radius=radius)
    if border:
        pygame.draw.rect(s, border, s.get_rect(), 1, border_radius=radius)
    surface.blit(s, rect.topleft)


def arc(surface: pygame.Surface, color, center, radius: float, a0: float, a1: float, width: int,
        steps: int | None = None) -> None:
    """Thick anti-aliased-ish arc from angle a0 to a1 (radians, 0 = right, clockwise on screen)."""
    if abs(a1 - a0) < 1e-3:
        return
    steps = steps or max(6, int(abs(a1 - a0) * radius / 4))
    cx, cy = center
    pts_o, pts_i = [], []
    for k in range(steps + 1):
        a = a0 + (a1 - a0) * k / steps
        pts_o.append((cx + math.cos(a) * radius, cy + math.sin(a) * radius))
        pts_i.append((cx + math.cos(a) * (radius - width), cy + math.sin(a) * (radius - width)))
    poly = pts_o + pts_i[::-1]
    if len(color) == 4:
        tmp = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(tmp, color, poly)
        surface.blit(tmp, (0, 0))
    else:
        pygame.draw.polygon(surface, color, poly)


class Hud:
    def __init__(self, dash):
        self.d = dash
        self.speed = Counter(0.0, 2.2)
        self.target = Spring(0.0, 1.5)
        self.steer = Spring(0.0, 3.0, 0.8)
        self.throttle = Spring(0.0, 4.0)
        self.brake = Spring(0.0, 4.0)
        self.mode_col = ColorTween(MODE_COLOR[Mode.OFF], 0.4)
        self.mode_text = ""
        self.old_text = ""
        self.swap = Tween(1.0, 0.35, "cubic")
        self.left = Pulse(1.6)
        self.right = Pulse(1.6)
        self.light_col = ColorTween((40, 42, 50), 0.3)
        self.light_show = Tween(0.0, 0.3, "cubic")
        self.warn = Pulse(2.2)
        self.toasts = Toasts(3)
        self.engage_ring = Tween(1.0, 0.9, "expo")
        self._engaged = False
        self._last_alert = ""
        self._last_log = ""
        self.fps_spring = Spring(0.0, 1.0)

    # ------------------------------------------------------------------
    def draw(self, screen: pygame.Surface, snap, w, plan, dt: float) -> None:
        d = self.d
        W, H = d.W, d.H
        engaged = snap.engaged
        mode = plan.mode if plan else Mode.OFF
        if engaged and not self._engaged:
            self.engage_ring.start, self.engage_ring.end = 0.0, 0.0
            self.engage_ring.to(1.0)
            self.toasts.push("FSD engaged", BLUE, 2.5)
        if not engaged and self._engaged:
            self.toasts.push("FSD off - you're driving", (160, 164, 174), 2.5)
        self._engaged = engaged
        self._messages(snap, plan)
        bar_h = max(56, int(H * 0.14))
        glass(screen, pygame.Rect(8, 8, W - 16, bar_h), 185, 16)
        self._speed(screen, w, plan, engaged, 8 + bar_h // 2 + 6, 8 + bar_h // 2, bar_h // 2 - 4, dt)
        x = 8 + bar_h + 22
        x = self._limit_sign(screen, w, x, 8 + bar_h // 2, bar_h)
        x = self._banner(screen, plan, mode, engaged, x, 8 + bar_h // 2, W, dt)
        x = self._blinkers(screen, plan, x, 8 + bar_h // 2)
        self._light(screen, w, x, 8 + bar_h // 2, bar_h, W)
        self._wrong_side(screen, plan, W, H)
        self._engage_flash(screen, W, H)
        self._toasts(screen, W, bar_h + 18)
        self._controls(screen, snap, plan, W, H, dt)

    # ------------------------------------------------------------------
    def _messages(self, snap, plan) -> None:
        alert = plan.alert if plan is not None else ""
        if alert and alert != self._last_alert:
            col = RED if ("TAKE OVER" in alert or "Wrong side" in alert) else (120, 110, 255) if (
                "turn" in alert.lower() or "lane" in alert.lower()) else BLUE
            self.toasts.push(alert, col, 4.0)
        self._last_alert = alert
        log = list(snap.status.get("log", []))
        if log:
            last = log[-1]
            if last != self._last_log:
                txt = last[9:] if len(last) > 9 and last[2] == ":" else last
                if any(k in txt for k in ("Blinker", "error", "Disengaged", "calibrat", "learned", "Fix")):
                    self.toasts.push(txt, (150, 154, 164), 3.5)
            self._last_log = last

    def _speed(self, screen, w, plan, engaged, cx, cy, r, dt) -> None:
        d = self.d
        unit = w.speed_unit if w else "kph"
        valid = bool(w and w.speed_valid)
        k = 1 / 1.609344 if unit == "mph" else 1.0
        v = self.speed.step((w.speed_kph if w else 0.0) * k, dt)
        tgt = self.target.step((plan.target_speed_kph if (plan and engaged) else 0.0) * k, dt)
        vmax = 220.0 * k
        a0 = math.radians(135)
        span = math.radians(270)
        arc(screen, (40, 43, 52), (cx, cy), r, a0, a0 + span, 5)
        frac = max(0.0, min(1.0, v / vmax))
        col = BLUE if engaged else (170, 174, 184)
        if valid:
            arc(screen, col, (cx, cy), r, a0, a0 + span * frac, 5)
        if engaged and tgt > 1:
            ta = a0 + span * max(0.0, min(1.0, tgt / vmax))
            p1 = (cx + math.cos(ta) * (r + 3), cy + math.sin(ta) * (r + 3))
            p2 = (cx + math.cos(ta) * (r - 9), cy + math.sin(ta) * (r - 9))
            pygame.draw.line(screen, WHITE, p1, p2, 3)
        txt = f"{v:.0f}" if valid else "--"
        f = d.f_big if r < 50 else d.f_huge
        d._text(f, txt, WHITE if valid else DIM, (cx, cy - 2), "center")
        d._text(d.f_small, unit.upper(), DIM, (cx, cy + r * 0.55), "center")

    def _limit_sign(self, screen, w, x, cy, bar_h) -> int:
        d = self.d
        if w is None or not w.speed_limit_kph:
            return x
        unit = w.speed_unit
        lim = w.speed_limit_kph / 1.609344 if unit == "mph" else w.speed_limit_kph
        sign = pygame.Rect(0, 0, int(bar_h * 0.56), int(bar_h * 0.74))
        sign.midleft = (x, cy)
        pygame.draw.rect(screen, WHITE, sign, border_radius=5)
        pygame.draw.rect(screen, (25, 25, 25), sign.inflate(-5, -5), 2, border_radius=4)
        d._text(d.f_small, f"{lim:.0f}", (20, 20, 20), sign.center, "center")
        return sign.right + 14

    def _banner(self, screen, plan, mode, engaged, x, cy, W, dt) -> int:
        d = self.d
        from .dashboard import MODE_TEXT
        label = MODE_TEXT.get(mode, mode.value)
        sub = plan.reason if (plan is not None and engaged and plan.reason) else ""
        text = label if not sub else f"{label} · {sub}"
        text = text[:48]
        if text != self.mode_text:
            self.old_text = self.mode_text
            self.mode_text = text
            self.swap.start, self.swap.end = 0.0, 0.0
            self.swap.to(1.0)
        self.mode_col.to(MODE_COLOR.get(mode, BLUE) if engaged or mode == Mode.FAULT else MODE_COLOR[Mode.OFF])
        col = self.mode_col.value
        if mode == Mode.FAULT:
            col = lerp_color((110, 25, 25), RED, self.warn.value() if self.warn.on else 1.0)
            self.warn.set(True)
        w = max(180, int(W * 0.36))
        pill = pygame.Rect(0, 0, w, 34)
        pill.midleft = (x, cy)
        pygame.draw.rect(screen, col, pill, border_radius=17)
        # text swap: old slides up and fades, new rises from below with a little overshoot
        u = self.swap.value
        clip = screen.get_clip()
        screen.set_clip(pill)
        if self.old_text and u < 1:
            s = d.f_small.render(self.old_text, True, WHITE)
            s.set_alpha(int(255 * (1 - u)))
            screen.blit(s, s.get_rect(center=(pill.centerx, pill.centery - 14 * u)))
        s = d.f_small.render(self.mode_text, True, WHITE)
        s.set_alpha(int(255 * min(1.0, u * 1.5)))
        off = 14 * (1 - ease_out_back(u))
        screen.blit(s, s.get_rect(center=(pill.centerx, pill.centery + off)))
        screen.set_clip(clip)
        return pill.right + 16

    def _blinkers(self, screen, plan, x, cy) -> int:
        on_l = plan is not None and plan.turn_pending == "left"
        on_r = plan is not None and plan.turn_pending == "right"
        self.left.set(on_l)
        self.right.set(on_r)
        s = 10
        for k, (pl, sgn) in enumerate(((self.left, -1), (self.right, 1))):
            v = pl.value()
            col = lerp_color((55, 58, 68), GREEN, v)
            bx = x + k * (2 * s + 10) + (0 if sgn < 0 else s)
            tip = (bx - s, cy) if sgn < 0 else (bx + s, cy)
            base = bx if sgn < 0 else bx
            pygame.draw.polygon(screen, col, [tip, (base, cy - s), (base, cy + s)])
            if v > 0.05:
                glow = pygame.Surface((4 * s, 4 * s), pygame.SRCALPHA)
                pygame.draw.circle(glow, (*GREEN, int(70 * v)), (2 * s, 2 * s), 2 * s)
                screen.blit(glow, (tip[0] - 2 * s, cy - 2 * s))
        return x + 2 * (2 * s + 10) + 10

    def _light(self, screen, w, x, cy, bar_h, W) -> None:
        d = self.d
        st = w.signal.state if w is not None else Light.NONE
        show = st not in (Light.NONE,) and w is not None and w.signal.distance < 200
        self.light_show.to(1.0 if show else 0.0)
        a = self.light_show.value
        if a < 0.02 or x > W - 60:
            return
        if show:
            self.light_col.to(LIGHT_COLOR.get(st, DIM))
        col = self.light_col.value
        if st == Light.FLASHING_RED:
            col = lerp_color((60, 20, 20), RED, 0.5 + 0.5 * math.cos(time.perf_counter() * 2 * math.pi * 1.2))
        r = int(bar_h * 0.22 * (0.6 + 0.4 * a))
        cx = x + r + 4
        glow = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*col, int(80 * a)), (2 * r, 2 * r), 2 * r)
        screen.blit(glow, (cx - 2 * r, cy - 2 * r))
        pygame.draw.circle(screen, col, (cx, cy), r)
        if show:
            label = "all-way stop" if st == Light.FLASHING_RED else f"{w.signal.distance:.0f} m"
            d._text(d.f_small, label, WHITE, (cx + r + 8, cy), "midleft")

    def _wrong_side(self, screen, plan, W, H) -> None:
        bad = plan is not None and plan.alert and ("Wrong side" in plan.alert or "TAKE OVER" in plan.alert)
        if not bad:
            if plan is None or plan.mode != Mode.FAULT:
                self.warn.set(False)
        else:
            self.warn.set(True)
        v = self.warn.value()
        if v < 0.03:
            return
        key = (W, H)
        if getattr(self, "_edge_key", None) != key:
            edge = pygame.Surface((W, H), pygame.SRCALPHA)
            for i in range(14):
                a = int(110 * (1 - i / 14))
                pygame.draw.rect(edge, (235, 50, 50, a), (i, i, W - 2 * i, H - 2 * i), 1, border_radius=18)
            self._edge, self._edge_key = edge, key
        self._edge.set_alpha(int(255 * v))
        screen.blit(self._edge, (0, 0))

    def _engage_flash(self, screen, W, H) -> None:
        u = self.engage_ring.value
        if u >= 1.0:
            return
        s = pygame.Surface((W, H), pygame.SRCALPHA)
        a = int(140 * (1 - u))
        inset = int(20 * (1 - u))
        pygame.draw.rect(s, (*BLUE, a), (inset, inset, W - 2 * inset, H - 2 * inset), 4, border_radius=20)
        screen.blit(s, (0, 0))

    def _toasts(self, screen, W, y0) -> None:
        d = self.d
        y = y0
        now = time.perf_counter()
        for t in self.toasts.alive():
            a = t.alpha(now)
            txt = d.f_small.render(t.text[:70], True, WHITE)
            w = txt.get_width() + 34
            h = txt.get_height() + 14
            slide = t.slide.value
            x = W - 12 - w * min(1.0, slide) + (1 - min(1.0, slide)) * 30
            box = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(box, (*GLASS, int(210 * a)), box.get_rect(), border_radius=h // 2)
            pygame.draw.circle(box, (*t.color, int(255 * a)), (h // 2, h // 2), 5)
            txt.set_alpha(int(255 * a))
            box.blit(txt, (24, 7))
            screen.blit(box, (int(x), y))
            y += h + 6

    def _controls(self, screen, snap, plan, W, H, dt) -> None:
        """Steering wheel + throttle / brake bars, bottom left (only while engaged)."""
        d = self.d
        steer = plan.steer if plan is not None else 0.0
        pedal = plan.pedal if plan is not None else 0.0
        sv = self.steer.step(steer if snap.engaged else 0.0, dt)
        th = self.throttle.step(max(0.0, pedal) if snap.engaged else 0.0, dt)
        br = self.brake.step(max(0.0, -pedal) if snap.engaged else 0.0, dt)
        r = max(18, int(H * 0.045))
        foot = d.line_h * 2 + 12
        cx, cy = 20 + r, H - foot - r - 10
        glass(screen, pygame.Rect(cx - r - 10, cy - r - 10, 2 * r + 20 + 40, 2 * r + 20), 150, 12)
        ang = sv * math.radians(90)
        pygame.draw.circle(screen, (200, 204, 214), (cx, cy), r, 3)
        for base in (math.radians(180), 0.0, math.radians(90)):
            a = base + ang
            p = (cx + math.cos(a) * r, cy + math.sin(a) * r)
            pygame.draw.line(screen, (200, 204, 214), (cx, cy), p, 3)
        pygame.draw.circle(screen, (200, 204, 214), (cx, cy), max(3, r // 4))
        # bars
        bx = cx + r + 12
        bh = 2 * r
        for i, (val, col) in enumerate(((th, GREEN), (br, RED))):
            rect = pygame.Rect(bx + i * 12, cy - r, 8, bh)
            pygame.draw.rect(screen, (45, 48, 58), rect, border_radius=4)
            fill = int(bh * max(0.0, min(1.0, val)))
            if fill > 0:
                pygame.draw.rect(screen, col, (rect.x, rect.bottom - fill, 8, fill), border_radius=4)
