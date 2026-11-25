"""Tesla-style FSD display (pygame).

Rules from the spec, so it never flickers:
* one window, one render loop, fixed 60 fps
* the road is drawn from a smoothed model every frame, never rebuilt from noisy data
* one turn-preview path per Q/E request (kept until done/cancelled)
* everything shown is eased toward the latest perception; if perception drops,
  the last good state is held briefly, then faded out
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from ..world import Light, Mode  # noqa: E402

BG = (14, 15, 18)
ROAD = (34, 36, 42)
LINE = (120, 124, 134)
WHITE = (236, 238, 242)
DIM = (130, 134, 144)
BLUE = (56, 132, 255)
BLUE_SOFT = (56, 132, 255, 90)
RED = (235, 64, 64)
YELLOW = (250, 200, 40)
GREEN = (60, 210, 110)
GREY_CAR = (150, 154, 164)
LEAD_CAR = (90, 150, 255)

MODE_TEXT = {
    Mode.OFF: "FSD available - press F1", Mode.CRUISE: "Full Self-Driving", Mode.FOLLOW: "Following",
    Mode.STOPPING: "Stopping", Mode.STOPPED: "Stopped", Mode.STOP_WAIT: "Stop - checking cross traffic",
    Mode.TURNING: "Turning", Mode.LANE_CHANGE: "Changing lanes", Mode.YIELD: "Yielding to emergency vehicle",
    Mode.CAUTION: "Caution - slowing", Mode.FAULT: "TAKE OVER",
}


class Ease:
    """Critically damped smoothing for display values."""

    def __init__(self, v=0.0, rate=8.0):
        self.v = np.array(v, dtype=np.float64)
        self.rate = rate

    def to(self, target, dt):
        k = 1 - math.exp(-self.rate * dt)
        self.v = self.v + (np.array(target, dtype=np.float64) - self.v) * k
        return self.v


class Dashboard:
    W, H = 1280, 720

    def __init__(self, app, monitor: int = 1, overlay: bool = False):
        self.app = app
        st = app.settings
        pygame.init()
        pygame.display.set_caption("ER:LC FSD")
        try:
            sizes = pygame.display.get_desktop_sizes()
        except Exception:  # noqa: BLE001
            sizes = [(1920, 1080)]
        # Two screens: fill the second one. One screen: small floating window
        # in the top-right corner that stays above Roblox (drag it to move).
        self.compact = overlay or not (0 < monitor < len(sizes))
        if self.compact:
            self.W, self.H = int(st.ui_size[0]), int(st.ui_size[1])
            x, y = st.ui_pos or (sizes[0][0] - self.W - 16, 64)
        else:
            self.W, self.H = sizes[monitor]
            x, y = sum(s[0] for s in sizes[:monitor]), 0
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"
        self.win_pos = [int(x), int(y)]
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.NOFRAME)
        self.hwnd = None
        self._drag: tuple[int, int] | None = None
        self._visible = True
        self._last_top = 0.0
        self._win_setup(st.ui_on_top, st.ui_opacity if self.compact else 1.0)
        self.clock = pygame.time.Clock()
        name = "segoeui" if os.name == "nt" else None
        # readable minimum sizes for the small floating window
        self.f_huge = pygame.font.SysFont(name, max(56, int(self.H * 0.15)), bold=True)
        self.f_big = pygame.font.SysFont(name, max(20, int(self.H * 0.05)), bold=True)
        self.f_med = pygame.font.SysFont(name, max(16, int(self.H * 0.032)))
        self.f_small = pygame.font.SysFont(name, max(13, int(self.H * 0.024)))
        self.line_h = self.f_small.get_linesize()
        self.road = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        # smoothed road model
        self.e_left = Ease([0, 0, -1.85], 6)
        self.e_right = Ease([0, 0, 1.85], 6)
        self.e_center = Ease([0, 0, 0], 6)
        self.e_path = Ease(np.zeros((24, 2)), 9)
        self.e_speed = Ease(0, 6)
        self.e_alpha = Ease(1.0, 3)
        self.cars: dict[int, Ease] = {}
        self.last_good = 0.0
        self.prompt_text = ""
        self.typing = False
        self.preview_cache: tuple[str | None, list] = (None, [])
        self.blink = 0.0
        self.show_camera = not self.compact
        self._resize_at = 0.0

    # ---------------- window management (Windows) ----------------
    def _user32(self):
        if os.name != "nt":
            return None
        import ctypes
        return ctypes.windll.user32

    def _win_setup(self, on_top: bool, opacity: float) -> None:
        u = self._user32()
        if u is None:
            return
        try:
            self.hwnd = pygame.display.get_wm_info()["window"]
            self.on_top = on_top
            if opacity < 0.999:
                GWL_EXSTYLE, WS_EX_LAYERED, LWA_ALPHA = -20, 0x00080000, 0x2
                ex = u.GetWindowLongW(self.hwnd, GWL_EXSTYLE)
                u.SetWindowLongW(self.hwnd, GWL_EXSTYLE, ex | WS_EX_LAYERED)
                u.SetLayeredWindowAttributes(self.hwnd, 0, int(255 * max(0.3, min(1.0, opacity))), LWA_ALPHA)
            self._keep_on_top(force=True)
        except Exception:  # noqa: BLE001
            self.hwnd = None

    def _keep_on_top(self, force: bool = False) -> None:
        """Re-assert 'topmost' every couple of seconds; never steals focus from Roblox."""
        u = self._user32()
        if u is None or self.hwnd is None or not getattr(self, "on_top", False):
            return
        now = time.time()
        if not force and now - self._last_top < 2.0:
            return
        self._last_top = now
        HWND_TOPMOST, NOSIZE, NOMOVE, NOACTIVATE = -1, 0x0001, 0x0002, 0x0010
        u.SetWindowPos(self.hwnd, HWND_TOPMOST, 0, 0, 0, 0, NOSIZE | NOMOVE | NOACTIVATE)

    def _move_window(self, x: int, y: int) -> None:
        u = self._user32()
        if u is None or self.hwnd is None:
            return
        HWND_TOPMOST, NOSIZE, NOACTIVATE = -1, 0x0001, 0x0010
        u.SetWindowPos(self.hwnd, HWND_TOPMOST, int(x), int(y), 0, 0, NOSIZE | NOACTIVATE)
        self.win_pos = [int(x), int(y)]

    def _cursor(self) -> tuple[int, int]:
        import ctypes
        from ctypes import wintypes
        pt = wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def _set_visible(self, show: bool) -> None:
        u = self._user32()
        if u is None or self.hwnd is None or show == self._visible:
            return
        self._visible = show
        u.ShowWindow(self.hwnd, 4 if show else 0)       # SW_SHOWNOACTIVATE / SW_HIDE
        if show:
            self._keep_on_top(force=True)

    # ---------------- projection: vehicle metres -> dashboard pixels ----------------
    CAM_BACK, CAM_H = 13.0, 5.5

    def proj(self, x: float, y: float, z: float = 0.0) -> tuple[float, float]:
        """Chase view from 13 m behind and 5.5 m above the car. z = height above the road."""
        W, H = self.road_w, self.H
        f = H * 1.05
        d = max(0.5, x + self.CAM_BACK)
        horizon = H * 0.30
        return W / 2 + y * f / d, horizon + (self.CAM_H - z) * f / d

    # ---------------- main loop ----------------
    def run(self) -> None:
        dt = 1 / 60
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    running = self._key(ev) and running
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.compact and self.hwnd:
                    cx, cy = self._cursor()
                    self._drag = (cx - self.win_pos[0], cy - self.win_pos[1])
                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self._drag:
                    self._drag = None
                    self.app.settings.ui_pos = list(self.win_pos)     # remember where you put it
                    self.app.settings.save()
            if self._drag:
                cx, cy = self._cursor()
                self._move_window(cx - self._drag[0], cy - self._drag[1])
            self._set_visible(self.app.ui_visible)
            self._keep_on_top()
            try:
                self.draw(dt)
            except Exception:  # noqa: BLE001 - a drawing glitch must never close the app
                from ..app import log_error
                log_error("dashboard draw")
                self.screen.fill(BG)
            pygame.display.flip()
            dt = self.clock.tick(60) / 1000.0
        self.app.shutdown()
        pygame.quit()

    def _key(self, ev) -> bool:
        if self.app.prompt_open:
            if ev.key == pygame.K_RETURN:
                self.app.submit_note(self.prompt_text)
                self.prompt_text = ""
            elif ev.key == pygame.K_ESCAPE:
                self.app.submit_note("")
                self.prompt_text = ""
            elif ev.key == pygame.K_BACKSPACE:
                self.prompt_text = self.prompt_text[:-1]
            elif ev.unicode and ev.unicode.isprintable() and len(self.prompt_text) < 240:
                self.prompt_text += ev.unicode
            return True
        if ev.key == pygame.K_c:
            self.show_camera = not self.show_camera
        if ev.key == pygame.K_ESCAPE and (ev.mod & pygame.KMOD_SHIFT):
            return False
        return True

    # ---------------- drawing ----------------
    def draw(self, dt: float) -> None:
        snap = self.app.snapshot()
        w, plan = snap.world, snap.plan
        self.road_w = int(self.W * (0.55 if self.compact else 0.64))
        self.screen.fill(BG)
        self.road.fill((0, 0, 0, 0))
        now = time.time()
        self.blink = (self.blink + dt) % 1.0

        if w is not None and w.lanes.conf > 0.05:
            self.last_good = now
            ln = w.lanes
            if ln.left:
                self.e_left.to(ln.left.coeffs, dt)
            else:
                self.e_left.to([ln.center[0], ln.center[1], ln.center[2] - ln.width_m / 2], dt)
            if ln.right:
                self.e_right.to(ln.right.coeffs, dt)
            else:
                self.e_right.to([ln.center[0], ln.center[1], ln.center[2] + ln.width_m / 2], dt)
            self.e_center.to(ln.center, dt)
        # hold the last good road 1 s, then fade to a neutral straight road
        held = now - self.last_good
        self.e_alpha.to(1.0 if held < 1.0 else 0.35, dt)
        if held > 1.5:
            self.e_left.to([0, 0, -1.85], dt)
            self.e_right.to([0, 0, 1.85], dt)
            self.e_center.to([0, 0, 0], dt)

        self._draw_road(dt, w, plan)
        self.screen.blit(self.road, (0, 0))
        self._draw_panel(snap, w, plan)
        if self.show_camera and snap.debug_img is not None:
            self._draw_camera(snap.debug_img)
        if self.app.prompt_open:
            self._draw_prompt()

    def _poly_pts(self, coeffs, x0=-8.0, x1=80.0, n=44, dy=0.0):
        a, b, c = coeffs
        xs = np.linspace(x0, x1, n)
        return [self.proj(x, a * x * x + b * x + c + dy) for x in xs]

    def _draw_road(self, dt, w, plan) -> None:
        alpha = float(self.e_alpha.v)
        L, R = self.e_left.v, self.e_right.v
        # road surface (two lanes either side)
        left_far = self._poly_pts(L, dy=-3.7)
        right_far = self._poly_pts(R, dy=3.7)
        pygame.draw.polygon(self.road, (*ROAD, int(255 * alpha)), left_far + right_far[::-1])
        # ego lane tint
        pygame.draw.polygon(self.road, (44, 48, 58, int(255 * alpha)), self._poly_pts(L) + self._poly_pts(R)[::-1])
        for coeffs, line in ((L, w.lanes.left if w else None), (R, w.lanes.right if w else None)):
            col = YELLOW if (line and line.color == "yellow") else LINE
            pts = self._poly_pts(coeffs)
            if line and line.dashed:
                for i in range(0, len(pts) - 1, 2):
                    pygame.draw.line(self.road, (*col, int(230 * alpha)), pts[i], pts[i + 1], 4)
            else:
                pygame.draw.lines(self.road, (*col, int(230 * alpha)), False, pts, 4)

        # stop line
        if w is not None:
            d = min(w.stop_line_dist, plan.stop_at if plan and plan.stop_at is not None else 999)
            if d < 80:
                c = float(np.polyval(self.e_center.v, d))
                p1, p2 = self.proj(d, c - 1.9), self.proj(d, c + 1.9)
                pygame.draw.line(self.road, (*RED, 230), p1, p2, 6)

        # planned path ribbon
        if plan is not None and plan.path:
            path = np.array(plan.path[:24], dtype=np.float64)
            if len(path) < 24:
                path = np.vstack([path, np.repeat(path[-1:], 24 - len(path), 0)])
            sm = self.e_path.to(path, dt)
            col = BLUE if plan.mode not in (Mode.OFF,) else (90, 94, 104)
            left = [self.proj(x, y - 0.9) for x, y in sm if x >= 0]
            right = [self.proj(x, y + 0.9) for x, y in sm if x >= 0]
            if len(left) > 1:
                pygame.draw.polygon(self.road, (*col, 110), left + right[::-1])

        # turn preview: computed once per request, kept until done/cancelled
        if plan is not None:
            if plan.turn_pending != self.preview_cache[0] or (plan.turn_preview and not self.preview_cache[1]):
                self.preview_cache = (plan.turn_pending, list(plan.turn_preview))
            if plan.turn_pending is None:
                self.preview_cache = (None, [])
            pts = self.preview_cache[1]
            if pts:
                on = self.blink < 0.6
                scr = [self.proj(x, y) for x, y in pts]
                for i in range(0, len(scr) - 1, 2):
                    pygame.draw.line(self.road, (*BLUE, 255 if on else 140), scr[i], scr[i + 1], 6)

        # other road users
        if w is not None:
            seen = set()
            for o in sorted(w.objects, key=lambda o: -o.x):
                if o.kind not in ("car", "person", "cone", "emergency") or not (0 < o.x < 80):
                    continue
                tid = o.track_id
                seen.add(tid)
                e = self.cars.setdefault(tid, Ease([o.x, o.y], 10))
                x, y = e.to([o.x, o.y], dt)
                lead = plan is not None and plan.lead is not None and plan.lead.track_id == tid
                if o.kind in ("car", "emergency"):
                    self._car(x, y, LEAD_CAR if lead else (RED if o.kind == "emergency" else GREY_CAR))
                elif o.kind == "person":
                    cx, cy = self.proj(x, y)
                    pygame.draw.circle(self.road, (240, 200, 80), (int(cx), int(cy) - 10), max(4, int(160 / (x + 9))))
                else:
                    cx, cy = self.proj(x, y)
                    r = max(3, int(120 / (x + 9)))
                    pygame.draw.polygon(self.road, (255, 140, 0), [(cx, cy - 3 * r), (cx - r, cy), (cx + r, cy)])
            for tid in [t for t in self.cars if t not in seen]:
                del self.cars[tid]

        # traffic light marker over the road
        if w is not None and w.signal.state not in (Light.NONE,) and w.signal.distance < 120:
            d = w.signal.distance
            c = float(np.polyval(self.e_center.v, d))
            px, py = self.proj(d, c)
            col = {Light.RED: RED, Light.YELLOW: YELLOW, Light.GREEN: GREEN}.get(w.signal.state, RED)
            if w.signal.state == Light.FLASHING_RED and self.blink > 0.5:
                col = (90, 30, 30)
            r = max(6, int(300 / (d + 9)))
            pygame.draw.circle(self.road, col, (int(px), int(py - 5 * r)), r)

        # soft fade into the distance
        top = int(self.H * 0.30)
        band = int(self.H * 0.10)
        for i in range(band):
            a = int(255 * (1 - i / band))
            pygame.draw.line(self.road, (*BG, a), (0, top + i), (self.road_w, top + i))

        # ego car
        self._car(0.0, 0.0, (225, 228, 234), ego=True)

    def _car(self, x, y, col, ego=False) -> None:
        """Simple shaded 3D box car. x = front bumper distance for others; the ego car sits behind x=0."""
        l, wd, h = 4.6, 1.9, 1.45
        x0, x1 = (-l, 0.0) if ego else (x, x + l)
        y0, y1 = y - wd / 2, y + wd / 2

        def shade(c, k):
            return tuple(int(min(255, max(0, v * k))) for v in c[:3])
        P = self.proj
        # body sides (only the side facing the camera is visible)
        if y1 < 0:
            pygame.draw.polygon(self.road, shade(col, 0.6), [P(x0, y1), P(x1, y1), P(x1, y1, h * 0.6), P(x0, y1, h * 0.6)])
        if y0 > 0:
            pygame.draw.polygon(self.road, shade(col, 0.6), [P(x0, y0), P(x1, y0), P(x1, y0, h * 0.6), P(x0, y0, h * 0.6)])
        # rear face and lower body top
        pygame.draw.polygon(self.road, shade(col, 0.75), [P(x0, y0), P(x0, y1), P(x0, y1, h * 0.6), P(x0, y0, h * 0.6)])
        pygame.draw.polygon(self.road, shade(col, 1.0), [P(x0, y0, h * 0.6), P(x0, y1, h * 0.6),
                                                          P(x1, y1, h * 0.6), P(x1, y0, h * 0.6)])
        # cabin
        cx0, cx1 = x0 + l * 0.22, x0 + l * 0.72
        cy0, cy1 = y0 + 0.18, y1 - 0.18
        pygame.draw.polygon(self.road, shade(col, 0.45), [P(cx0 - 0.4, cy0, h * 0.6), P(cx0 - 0.4, cy1, h * 0.6),
                                                           P(cx0, cy1, h), P(cx0, cy0, h)])
        pygame.draw.polygon(self.road, shade(col, 0.9), [P(cx0, cy0, h), P(cx0, cy1, h), P(cx1, cy1, h), P(cx1, cy0, h)])
        if ego:
            # tail lights
            for yy in (y0 + 0.25, y1 - 0.25):
                a, b = P(x0, yy - 0.2, h * 0.45), P(x0, yy + 0.2, h * 0.45)
                pygame.draw.line(self.road, (220, 40, 40), a, b, 4)

    def _text(self, font, text, color, pos, anchor="topleft") -> pygame.Rect:
        surf = font.render(text, True, color)
        r = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, r)
        return r

    def _draw_panel(self, snap, w, plan) -> None:
        x0 = self.road_w + int(self.W * 0.02)
        maxc = 34 if self.compact else 70
        y = int(self.H * 0.04)
        engaged = snap.engaged
        # status pill
        pill = pygame.Rect(x0, y, self.W - x0 - int(self.W * 0.02), max(26, int(self.H * 0.07)))
        mode = plan.mode if plan else Mode.OFF
        pcol = BLUE if engaged else (60, 64, 74)
        if mode == Mode.FAULT:
            pcol = RED if self.blink < 0.5 else (120, 30, 30)
        pygame.draw.rect(self.screen, pcol, pill, border_radius=pill.h // 2)
        self._text(self.f_med, MODE_TEXT.get(mode, mode.value), WHITE, pill.center, "center")
        y += pill.h + int(self.H * 0.03)

        # speed
        spd = float(self.e_speed.to(w.speed_kph if w else 0, 1 / 60)) if w else 0.0
        unit = (w.speed_unit if w else "kph")
        shown = spd / 1.609344 if unit == "mph" else spd
        r = self._text(self.f_huge, f"{shown:.0f}", WHITE, (x0, y))
        self._text(self.f_med, unit.upper() if w and w.speed_valid else f"{unit.upper()} (not read)", DIM,
                   (r.right + 10, r.bottom - int(self.H * 0.04)))
        # speed limit sign
        lim = (w.speed_limit_kph if w and w.speed_limit_kph else None)
        if plan is not None and engaged:
            lim = lim or None
        if lim:
            sign = pygame.Rect(x0 + int(self.W * 0.24), y + 6, int(self.H * 0.09), int(self.H * 0.12))
            pygame.draw.rect(self.screen, WHITE, sign, border_radius=6)
            pygame.draw.rect(self.screen, (20, 20, 20), sign.inflate(-6, -6), 2, border_radius=5)
            limv = lim / 1.609344 if unit == "mph" else lim
            self._text(self.f_small, "LIMIT", (20, 20, 20), (sign.centerx, sign.y + 8), "midtop")
            self._text(self.f_big, f"{limv:.0f}", (20, 20, 20), (sign.centerx, sign.bottom - 4), "midbottom")
        y = r.bottom + int(self.H * 0.01)
        if plan is not None and engaged:
            tgt = plan.target_speed_kph / 1.609344 if unit == "mph" else plan.target_speed_kph
            self._text(self.f_small, f"set {tgt:.0f} · {plan.reason}"[:maxc], DIM, (x0, y))
        y += self.line_h + 6

        # turn signals
        if plan is not None:
            on = self.blink < 0.5
            lc = GREEN if (plan.turn_pending == "left" and on) else (50, 54, 62)
            rc = GREEN if (plan.turn_pending == "right" and on) else (50, 54, 62)
            s = int(self.H * 0.035)
            pygame.draw.polygon(self.screen, lc, [(x0, y + s), (x0 + s, y), (x0 + s, y + 2 * s)])
            pygame.draw.polygon(self.screen, rc, [(x0 + 4 * s, y), (x0 + 5 * s, y + s), (x0 + 4 * s, y + 2 * s)])
            if plan.alert:
                self._text(self.f_small, plan.alert[:maxc - 6], WHITE, (x0 + 6 * s, y + s), "midleft")
            y += 2 * s + 8

        # traffic light
        if w is not None and w.signal.state != Light.NONE:
            col = {Light.RED: RED, Light.YELLOW: YELLOW, Light.GREEN: GREEN, Light.FLASHING_RED: RED}.get(
                w.signal.state, DIM)
            pygame.draw.circle(self.screen, col, (x0 + 14, y + 14), 12)
            label = "OUTAGE - all-way stop" if w.signal.state == Light.FLASHING_RED else w.signal.state.value
            self._text(self.f_small, f"{label}  {w.signal.distance:.0f} m", WHITE, (x0 + 36, y + 14), "midleft")
            y += 36
        if w is not None and w.stop_sign_dist < 900:
            pygame.draw.polygon(self.screen, RED, [(x0 + 14 + 12 * math.cos(a), y + 14 + 12 * math.sin(a))
                                                   for a in np.linspace(math.pi / 8, 2 * math.pi + math.pi / 8, 9)])
            self._text(self.f_small, f"stop sign  {w.stop_sign_dist:.0f} m", WHITE, (x0 + 36, y + 14), "midleft")
            y += 36

        # learning log + status
        if not self.compact:
            y = int(self.H * 0.40) + (int(self.W * 0.30 * 9 / 16) + int(self.H * 0.03) if self.show_camera else 0)
        st = snap.status
        if snap.training_on:
            self._text(self.f_small, f"● TRAINING RECORDER ON ({snap.training_count} frames)", RED, (x0, y))
            y += int(self.H * 0.035)
        bottom = self.H - 2 * self.line_h - 8
        for line in list(st.get("log", []))[-6:]:
            if y + self.line_h > bottom:
                break
            self._text(self.f_small, line[9:][:maxc] if self.compact else line[:maxc], DIM, (x0, y))
            y += self.line_h
        foot = (f"{st.get('fps', 0):.0f} fps · {st.get('detector', '')} · params v{st.get('params', 0)} · "
                f"{st.get('ocr', '')}")
        self._text(self.f_small, foot[:(60 if self.compact else 90)], (90, 94, 104),
                   (int(self.W * 0.01), self.H - self.line_h - 2))
        keys = ("F1 FSD · Q/E turn · F9 hide/show this · drag to move" if self.compact else
                "F1 FSD · Q/E turn · F2 train rec · F3 camera · F4 calibrate · F6 undo fix · F7 note · F9 hide · C cam")
        self._text(self.f_small, keys, (90, 94, 104), (int(self.W * 0.01), self.H - 2 * self.line_h - 4))

    def _draw_camera(self, img) -> None:
        h, w = img.shape[:2]
        tw = int(self.W * 0.30)
        th = int(h * tw / w)
        surf = pygame.image.frombuffer(img[:, :, ::-1].tobytes(), (w, h), "RGB")
        surf = pygame.transform.smoothscale(surf, (tw, th))
        x, y = self.W - tw - int(self.W * 0.015), int(self.H * 0.40)
        self.screen.blit(surf, (x, y))
        pygame.draw.rect(self.screen, (60, 64, 74), (x, y, tw, th), 1)

    def _draw_prompt(self) -> None:
        box = pygame.Rect(int(self.W * 0.12), int(self.H * 0.32), int(self.W * 0.76), int(self.H * 0.3))
        shade = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 160))
        self.screen.blit(shade, (0, 0))
        pygame.draw.rect(self.screen, (30, 32, 38), box, border_radius=14)
        self._text(self.f_big, "What went wrong?", WHITE, (box.x + 24, box.y + 18))
        self._text(self.f_small, "Type a quick note, Enter to send, Esc to skip. Gemini reviews the last 30 s.",
                   DIM, (box.x + 24, box.y + 18 + int(self.H * 0.06)))
        field = pygame.Rect(box.x + 24, box.bottom - int(self.H * 0.11), box.w - 48, int(self.H * 0.07))
        pygame.draw.rect(self.screen, (18, 19, 23), field, border_radius=8)
        cursor = "|" if self.blink < 0.5 else ""
        self._text(self.f_med, (self.prompt_text + cursor)[-70:], WHITE, (field.x + 12, field.centery), "midleft")
