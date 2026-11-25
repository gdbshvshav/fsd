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
# SDL minimizes its window when another app (Roblox) takes focus; never do that.
os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"
os.environ.setdefault("SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS", "0")
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


def wintypes_hwnd(v: int):
    import ctypes
    return ctypes.c_void_p(v)


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
            if self.W == 640 and self.H == 360:          # old default -> bigger default for the camera view
                self.W, self.H = 800, 450
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
        self.view = "camera"          # "camera" = your game, greyed, with what FSD sees highlighted; "3d" = map view
        self._rect_t = 0.0
        self._resize_at = 0.0

    # ---------------- window management (Windows) ----------------
    _u32 = None

    def _user32(self):
        """user32 with proper 64-bit argument types (without them, HWND_TOPMOST=-1
        gets truncated on 64-bit Windows and 'always on top' silently does nothing)."""
        if os.name != "nt":
            return None
        if Dashboard._u32 is None:
            import ctypes
            from ctypes import wintypes
            u = ctypes.WinDLL("user32", use_last_error=True)
            u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, wintypes.UINT]
            u.SetWindowPos.restype = wintypes.BOOL
            u.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
            u.GetWindowLongW.restype = ctypes.c_long
            u.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
            u.SetWindowLongW.restype = ctypes.c_long
            u.SetLayeredWindowAttributes.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_ubyte, wintypes.DWORD]
            u.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
            u.IsIconic.argtypes = [wintypes.HWND]
            Dashboard._u32 = u
        return Dashboard._u32

    def _win_setup(self, on_top: bool, opacity: float) -> None:
        u = self._user32()
        if u is None:
            return
        try:
            self.hwnd = pygame.display.get_wm_info()["window"]
            self.on_top = on_top
            # tool window: no taskbar button, can't be minimized by "show desktop"/alt-tab shuffles
            GWL_EXSTYLE, WS_EX_TOOLWINDOW = -20, 0x00000080
            ex = u.GetWindowLongW(self.hwnd, GWL_EXSTYLE)
            u.SetWindowLongW(self.hwnd, GWL_EXSTYLE, ex | WS_EX_TOOLWINDOW)
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
        if not force and now - self._last_top < 0.5:
            return
        self._last_top = now
        if self._visible and u.IsIconic(self.hwnd):
            u.ShowWindow(self.hwnd, 4)                   # got minimized: bring it back without taking focus
        HWND_TOPMOST, NOSIZE, NOMOVE, NOACTIVATE, SHOWWINDOW = wintypes_hwnd(-1), 0x0001, 0x0002, 0x0010, 0x0040
        u.SetWindowPos(self.hwnd, HWND_TOPMOST, 0, 0, 0, 0, NOSIZE | NOMOVE | NOACTIVATE | SHOWWINDOW)

    def _move_window(self, x: int, y: int) -> None:
        u = self._user32()
        if u is None or self.hwnd is None:
            return
        HWND_TOPMOST, NOSIZE, NOACTIVATE = wintypes_hwnd(-1), 0x0001, 0x0010
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
        if ev.key == pygame.K_v:
            self.view = "3d" if self.view == "camera" else "camera"
            self.app.want_debug_img = self.view == "3d"
        if ev.key == pygame.K_ESCAPE and (ev.mod & pygame.KMOD_SHIFT):
            return False
        return True

    # ---------------- drawing ----------------
    def draw(self, dt: float) -> None:
        snap = self.app.snapshot()
        w, plan = snap.world, snap.plan
        self.blink_t = getattr(self, "blink_t", 0.0) + dt
        if time.time() - self._rect_t > 1.0 and self.hwnd:
            self._rect_t = time.time()
            from ..winput import window_rect
            self.app.ui_rect = window_rect(self.hwnd)     # so FSD ignores its own window
        if self.view == "camera":
            self.blink = (self.blink + dt) % 1.0
            self._draw_camera_view(snap, w, plan)
            if self.app.prompt_open:
                self._draw_prompt()
            return
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

    # ================= camera view: your game, greyed, with highlights =================
    def _draw_camera_view(self, snap, w, plan) -> None:
        """Your game in grey; everything FSD detects shown in its REAL colours,
        cut out like a green screen and following it: your car, lane lines,
        stop lines / crosswalks, other cars, people, cones, lights and signs."""
        import cv2
        Wv, Hv = self.W, self.H
        img = self.app.view_frame(Wv, Hv)
        if img is None:
            self.screen.fill(BG)
            self._text(self.f_med, "Waiting for the game screen...", DIM, (Wv // 2, Hv // 2), "center")
            return
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor((g.astype(np.float32) * 0.38 + 14).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        per = self.app.perception
        geo = per.geo
        mask = np.zeros((Hv, Wv), np.uint8)
        if w is not None and geo is not None:
            pw, ph = self.app.proc_size
            sx, sy = Wv / pw, Hv / ph
            S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], np.float64)

            def to_view(pts_vehicle):
                p = geo.vehicle_to_screen(np.asarray(pts_vehicle, np.float32))
                keep = (p[:, 1] > geo.horizon_y + 1) & (p[:, 1] < ph)
                p = p[keep]
                return np.stack([p[:, 0] * sx, p[:, 1] * sy], 1).astype(np.int32)

            # planned path: a faint blue carpet on the grey road
            if plan is not None and plan.path and plan.mode.value != "off":
                xs = np.array([q[0] for q in plan.path])
                ys = np.array([q[1] for q in plan.path])
                k = xs >= 1.0
                if k.sum() > 2:
                    L = to_view(np.stack([xs[k], ys[k] - 0.9], 1))
                    R = to_view(np.stack([xs[k], ys[k] + 0.9], 1))
                    if len(L) > 1 and len(R) > 1:
                        tint = out.copy()
                        cv2.fillPoly(tint, [np.vstack([L, R[::-1]])], (255, 120, 30))
                        cv2.addWeighted(tint, 0.30, out, 0.70, 0, out)
            if plan is not None and plan.turn_preview and self.blink < 0.6:
                pts = to_view(np.array(plan.turn_preview, np.float32))
                if len(pts) > 1:
                    cv2.polylines(out, [pts], False, (255, 150, 50), max(3, Hv // 140), cv2.LINE_AA)

            # lane paint: the real paint pixels, but only along the lines FSD is using
            dbg = per.lanes.debug
            if dbg.paint_all is not None:
                M = S @ geo.Minv
                paint = cv2.warpPerspective(dbg.paint_all, M, (Wv, Hv), flags=cv2.INTER_NEAREST)
                near = np.zeros_like(mask)
                ln = w.lanes
                for side in ("left", "right", "left2", "right2"):
                    line = getattr(ln, side)
                    if line is None:
                        continue
                    xs = np.linspace(1.5, min(58.0, max(8.0, line.x_max)), 40)
                    pts = to_view(np.stack([xs, [line.y_at(x) for x in xs]], 1))
                    if len(pts) > 1:
                        cv2.polylines(near, [pts], False, 255, max(6, Hv // 45))
                # stop bars and crosswalks across our lane
                c = ln.center[2] if ln.conf > 0 else 0.0
                for d in (w.stop_line_dist, w.crosswalk_dist):
                    if d < 60:
                        band = to_view([[d - 1.2, c - 6], [d - 1.2, c + 6], [d + 3.5, c + 6], [d + 3.5, c - 6]])
                        if len(band) == 4:
                            cv2.fillPoly(near, [band], 255)
                mask |= cv2.dilate(paint & near, np.ones((3, 3), np.uint8))

            # your car + everything detected: boxes cut out in real colour
            def cut(box, shrink=0.04):
                x1, y1, x2, y2 = box
                dx, dy = (x2 - x1) * shrink, (y2 - y1) * shrink
                a, b = int((x1 + dx) * sx), int((y1 + dy) * sy)
                c2, d2 = int((x2 - dx) * sx), int((y2 - dy) * sy)
                if c2 > a and d2 > b:
                    cv2.rectangle(mask, (a, b), (c2, d2), 255, -1)
            cut(per.ego_box, 0.0)
            for o in w.objects:
                if o.kind == "light" and o.extra.get("state") not in ("red", "yellow", "green"):
                    continue
                cut(o.box, 0.0 if o.kind == "light" else 0.04)
        out[mask > 0] = img[mask > 0]
        # lead car + stop point: small labels only
        if w is not None and geo is not None and plan is not None:
            pw, ph = self.app.proc_size
            sx, sy = Wv / pw, Hv / ph
            if plan.lead is not None:
                x1, y1, x2, y2 = plan.lead.box
                cv2.putText(out, f"{plan.lead.x:.0f} m", (int(x1 * sx), max(12, int(y1 * sy) - 5)), 0, 0.5,
                            (255, 170, 60), 1, cv2.LINE_AA)
        surf = pygame.image.frombuffer(out[:, :, ::-1].tobytes(), (Wv, Hv), "RGB")
        self.screen.blit(surf, (0, 0))
        self._draw_hud(snap, w, plan)

    def _draw_hud(self, snap, w, plan) -> None:
        """Speed, FSD state, light, turn signal: a slim bar across the top."""
        Wv, Hv = self.W, self.H
        bar_h = max(48, int(Hv * 0.13))
        bar = pygame.Surface((Wv, bar_h), pygame.SRCALPHA)
        bar.fill((10, 11, 14, 200))
        self.screen.blit(bar, (0, 0))
        engaged = snap.engaged
        mode = plan.mode if plan else Mode.OFF
        unit = w.speed_unit if w else "kph"
        spd = float(self.e_speed.to(w.speed_kph if w else 0, 1 / 60)) if w else 0.0
        shown = spd / 1.609344 if unit == "mph" else spd
        x = 12
        r = self._text(self.f_big, f"{shown:.0f}", WHITE if (w and w.speed_valid) else DIM, (x, bar_h // 2), "midleft")
        self._text(self.f_small, unit.upper() if (w and w.speed_valid) else "speed ?", DIM, (r.right + 6, bar_h // 2),
                   "midleft")
        x = r.right + 70
        if w is not None and w.speed_limit_kph:
            lim = w.speed_limit_kph / 1.609344 if unit == "mph" else w.speed_limit_kph
            sign = pygame.Rect(x, (bar_h - int(bar_h * 0.8)) // 2, int(bar_h * 0.62), int(bar_h * 0.8))
            pygame.draw.rect(self.screen, WHITE, sign, border_radius=4)
            self._text(self.f_small, f"{lim:.0f}", (20, 20, 20), sign.center, "center")
            x = sign.right + 12
        pill = pygame.Rect(x, bar_h // 2 - 15, max(150, int(Wv * 0.3)), 30)
        pcol = BLUE if engaged else (60, 64, 74)
        if mode == Mode.FAULT:
            pcol = RED if self.blink < 0.5 else (120, 30, 30)
        pygame.draw.rect(self.screen, pcol, pill, border_radius=15)
        label = MODE_TEXT.get(mode, mode.value)
        if engaged and plan is not None and plan.reason:
            label = f"{label} · {plan.reason}"
        self._text(self.f_small, label[:44], WHITE, pill.center, "center")
        x = pill.right + 14
        if plan is not None:
            on = self.blink < 0.5
            s = 9
            lc = GREEN if (plan.turn_pending == "left" and on) else (60, 64, 74)
            rc = GREEN if (plan.turn_pending == "right" and on) else (60, 64, 74)
            cy = bar_h // 2
            pygame.draw.polygon(self.screen, lc, [(x, cy), (x + s, cy - s), (x + s, cy + s)])
            pygame.draw.polygon(self.screen, rc, [(x + 3 * s + 6, cy), (x + 2 * s + 6, cy - s), (x + 2 * s + 6, cy + s)])
            x += 3 * s + 20
        if w is not None and w.signal.state != Light.NONE and x < Wv - 30:
            col = {Light.RED: RED, Light.YELLOW: YELLOW, Light.GREEN: GREEN, Light.FLASHING_RED: RED}.get(
                w.signal.state, DIM)
            if not (w.signal.state == Light.FLASHING_RED and self.blink > 0.5):
                pygame.draw.circle(self.screen, col, (x + 10, bar_h // 2), 10)
            self._text(self.f_small, f"{w.signal.distance:.0f} m", WHITE, (x + 26, bar_h // 2), "midleft")
        # bottom line: newest message + keys
        st = snap.status
        log = list(st.get("log", []))
        msg = (plan.alert if plan is not None and plan.alert else (log[-1][9:] if log else ""))
        foot = pygame.Surface((Wv, self.line_h * 2 + 8), pygame.SRCALPHA)
        foot.fill((10, 11, 14, 170))
        self.screen.blit(foot, (0, Hv - foot.get_height()))
        self._text(self.f_small, msg[:90], WHITE, (10, Hv - 2 * self.line_h - 4))
        keys = (f"F1 FSD · Q/E turn · F9 hide · V view   brain {st.get('fps', 0):.0f}/s · "
                f"vision {st.get('det_fps', 0):.0f}/s {st.get('det_ms', 0):.0f}ms · {st.get('device', '')} · "
                f"v{st.get('params', 0)}")
        if st.get("warn"):
            keys = st["warn"] + " · " + keys
        self._text(self.f_small, keys[:110], (140, 144, 154), (10, Hv - self.line_h - 2))
        if snap.training_on:
            self._text(self.f_small, f"● REC {snap.training_count}", RED, (Wv - 10, bar_h + 6), "topright")

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
