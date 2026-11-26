"""Draws a simulated scene the way ER:LC looks through its chase camera.

Colours were sampled from real ER:LC screenshots (BGR):
    highway concrete   ~(152,151,163)     town asphalt     ~(82,89,101)
    Roblox shadow      x(0.72,0.42,0.25) per channel - dark and BLUE
    grass              ~(55,105,65)       sky              ~(231,172,92)
    white paint        ~(232,232,236)     yellow paint     ~(40,190,230)
    signal housing     ~(0,190,225) yellow frame, black body, lamps clipped to 255

Method: painter's algorithm. Ground (road strips, grass, junction, shadows,
paint) is drawn as quads along the road, nearest-plane clipped, grouped by
colour so each colour is one fillPoly call. 3D things (cars, signal heads,
poles, signs, trees, your car) are boxes/billboards sorted far to near.
A ER:LC-like HUD (speedometer "NN KM/H", minimap, top bar) is drawn last so the
real HUD masking and speedometer reading get exercised too.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .camera import ChaseCamera
from .road import Road
from .vehicle import Vehicle
from .world import Junction, OtherCar, Prop, Scene, SpeedSign, tail_light_on

SURFACE = {
    "concrete": (152, 151, 163),
    "asphalt": (82, 89, 101),
    "grass": (55, 105, 65),
    "dirt": (65, 75, 95),
    "sidewalk": (170, 172, 178),
    "shoulder": (140, 140, 150),
}
WHITE = (232, 232, 236)
YELLOW = (40, 190, 230)
SHADOW_MUL = np.array([0.72, 0.42, 0.25], np.float32)
HOUSING = (0, 190, 225)
BODY = (25, 25, 28)
LAMP = {"red": (40, 40, 255), "yellow": (0, 205, 255), "green": (90, 255, 60), "off": (45, 45, 45)}
NEAR = 0.6          # near clipping plane (m)
FAR = 220.0         # draw distance (m)


class Renderer:
    def __init__(self, cam: ChaseCamera):
        self.cam = cam
        self._sky = None

    # ------------------------------------------------------------------ helpers
    def _sky_img(self) -> np.ndarray:
        W, H = self.cam.width, self.cam.height
        if self._sky is None or self._sky.shape[:2] != (H, W):
            top = np.array([231, 165, 80], np.float32)
            bottom = np.array([240, 205, 170], np.float32)
            t = np.linspace(0, 1, H, dtype=np.float32)[:, None]
            col = top * (1 - t) + bottom * t
            self._sky = np.repeat(col[:, None, :], W, axis=1).astype(np.uint8)
        return self._sky.copy()

    def _project_polys(self, polys3d: list[np.ndarray]) -> list[np.ndarray]:
        """Clip each 3D polygon to the near plane and project to pixels.

        Items may be single polygons (k,3) or batches of quads (n,4,3): batches are
        projected in one go and only the few quads crossing the near plane are clipped."""
        out = []
        cam = self.cam
        singles = []
        for P in polys3d:
            if P.ndim == 3:
                n = P.shape[0]
                C = cam.to_camera(P.reshape(-1, 3)).reshape(n, 4, 3)
                z = C[:, :, 2]
                front = (z >= NEAR).all(axis=1) & (z.min(axis=1) < FAR)
                part = (z >= NEAR).any(axis=1) & ~front
                if front.any():
                    Cf = C[front].reshape(-1, 3)
                    px = cam.to_screen(Cf).reshape(-1, 4, 2)
                    px = np.clip(px, -4 * cam.width, 5 * cam.width)
                    out.extend(np.round(px).astype(np.int32))
                for k in np.nonzero(part)[0]:
                    singles.append(P[k])
            else:
                singles.append(P)
        for P in singles:
            C = cam.to_camera(P)
            z = C[:, 2]
            if (z < NEAR).all() or (z > FAR).all():
                continue
            if (z < NEAR).any():
                C = _clip_near(C, NEAR)
                if len(C) < 3:
                    continue
            px = cam.to_screen(C)
            if not np.isfinite(px).all():
                continue
            px = np.clip(px, -4 * cam.width, 5 * cam.width)
            out.append(np.round(px).astype(np.int32))
        return out

    def _ground_quads(self, road: Road, s0: float, s1: float, d0: float, d1: float, step: float) -> list[np.ndarray]:
        n = max(1, int(math.ceil((s1 - s0) / step)))
        ss = np.linspace(s0, s1, n + 1)
        L = road.points(ss, d0)
        R = road.points(ss, d1)
        q = np.zeros((n, 4, 3))
        q[:, 0, :2] = L[:-1]
        q[:, 1, :2] = R[:-1]
        q[:, 2, :2] = R[1:]
        q[:, 3, :2] = L[1:]
        return [q]                     # one (n,4,3) batch; _project_polys handles batches

    def _ranges(self, s_cam: float, road: Road) -> list[tuple[float, float, float]]:
        """Road stretches to draw with a step that grows with distance."""
        a = max(0.0, s_cam - 30.0)
        b = min(road.length, s_cam + FAR)
        out = []
        for lo, hi, st in ((a, s_cam + 30, 1.5), (s_cam + 30, s_cam + 80, 3.0), (s_cam + 80, b, 8.0)):
            lo, hi = max(lo, a), min(hi, b)
            if hi > lo:
                out.append((lo, hi, st))
        return out

    # ------------------------------------------------------------------ main
    def render(self, scene: Scene, ego: Vehicle, s_cam: float, t: float, hud_speed_kph: float | None = None,
               show_hud: bool = True) -> np.ndarray:
        cam = self.cam
        W, H = cam.width, cam.height
        img = self._sky_img()
        hy = int(max(0, min(H, cam.horizon_y)))
        img[hy:] = SURFACE["grass"]
        road = scene.road
        ranges = self._ranges(s_cam, road)

        # ---- ground strips, grouped by colour ----
        by_col: dict[tuple, list[np.ndarray]] = {}
        for st in road.strips:
            col = SURFACE.get(st.surface, SURFACE["grass"])
            for lo, hi, step in ranges:
                by_col.setdefault(col, []).extend(self._ground_quads(road, lo, hi, st.d0, st.d1, step))
        # junction cross streets (drawn over our road's strips)
        jq = []
        for j in scene.junctions:
            if s_cam - 20 < j.s < s_cam + FAR:
                jq += self._cross_street(road, j)
        for col, polys in by_col.items():
            cv2.fillPoly(img, self._project_polys(polys), col)
        if jq:
            cv2.fillPoly(img, self._project_polys(jq), SURFACE[road.surface])

        # ---- paint ----
        white, yellow = [], []
        for m in road.markings:
            offs = (-0.15, 0.15) if m.double else (0.0,)
            for o in offs:
                for lo, hi, step in ranges:
                    if m.dash is None:
                        quads = self._ground_quads(road, lo, hi, m.d + o - m.width / 2, m.d + o + m.width / 2, step)
                    else:
                        on, off = m.dash
                        period = on + off
                        quads = []
                        k0 = math.floor(lo / period)
                        k1 = math.ceil(hi / period)
                        for k in range(k0, k1 + 1):
                            a, b = k * period, k * period + on
                            a, b = max(a, lo), min(b, hi)
                            if b > a:
                                quads += self._ground_quads(road, a, b, m.d + o - m.width / 2, m.d + o + m.width / 2,
                                                            max(1.0, b - a))
                    (yellow if m.color == "yellow" else white).extend(quads)
        for j in scene.junctions:
            if s_cam - 20 < j.s < s_cam + FAR:
                white += self._junction_paint(road, j)
        if white:
            cv2.fillPoly(img, self._project_polys(white), WHITE)
        if yellow:
            cv2.fillPoly(img, self._project_polys(yellow), YELLOW)

        # ---- shadows: darken + tint blue, paint included (like Roblox) ----
        sh = []
        for s in scene.shadows:
            if s.s1 > s_cam - 10 and s.s0 < s_cam + FAR:
                sh += self._ground_quads(road, max(s.s0, s_cam - 10), min(s.s1, s_cam + FAR), s.d0, s.d1, 2.0)
        if sh:
            mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask, self._project_polys(sh), 255)
            sel = mask > 0
            img[sel] = (img[sel].astype(np.float32) * SHADOW_MUL).astype(np.uint8)

        # ---- 3D things, far to near ----
        items = []
        for p in scene.props:
            if s_cam - 5 < p.s < s_cam + 160:
                items.append((p.s, "prop", p))
        for sg in scene.signs:
            if s_cam < sg.s < s_cam + 160:
                items.append((sg.s, "sign", sg))
        for j in scene.junctions:
            if s_cam - 5 < j.s < s_cam + FAR:
                items.append((j.s + j.width, "junction", j))
        for c in scene.cars:
            if s_cam - 5 < c.s < s_cam + 180:
                items.append((c.s, "car", c))
        items.sort(key=lambda it: -it[0])
        for _, kind, obj in items:
            if kind == "prop":
                self._prop(img, road, obj)
            elif kind == "sign":
                self._speed_sign(img, road, obj)
            elif kind == "junction":
                self._junction_objects(img, road, obj, t)
            elif kind == "car":
                self._other_car(img, road, obj)
        self._ego(img, ego)
        if show_hud:
            self._hud(img, hud_speed_kph if hud_speed_kph is not None else ego.speed_kph)
        return img

    # ------------------------------------------------------------------ junctions
    def _cross_street(self, road: Road, j: Junction) -> list[np.ndarray]:
        x0, y0, h = road.point(j.s, 0)
        ex, ey = math.cos(h), -math.sin(h)          # across the road (to the right)
        fx, fy = math.sin(h), math.cos(h)           # along the road
        span = 120.0
        quads = []
        for k in range(-6, 6):
            a, b = k * span / 6, (k + 1) * span / 6
            p = [(x0 + a * ex, y0 + a * ey), (x0 + b * ex, y0 + b * ey),
                 (x0 + b * ex + j.width * fx, y0 + b * ey + j.width * fy),
                 (x0 + a * ex + j.width * fx, y0 + a * ey + j.width * fy)]
            quads.append(np.array([[q[0], q[1], 0.0] for q in p]))
        return quads

    def _junction_paint(self, road: Road, j: Junction) -> list[np.ndarray]:
        quads = []
        rw = road.lanes * road.lane_width
        # stop bar across our lanes
        s = j.stop_line_s
        quads += self._ground_quads(road, s - 0.5, s, 0.3, rw - 0.3, 0.5)
        if j.crosswalk:
            for s0 in (j.s - 3.8, j.s + j.width + 0.8):
                d = 0.3
                while d < rw - 0.3:
                    quads += self._ground_quads(road, s0, s0 + 3.0, d, d + 0.6, 3.0)
                    d += 1.2
        return quads

    def _junction_objects(self, img: np.ndarray, road: Road, j: Junction, t: float) -> None:
        rw = road.lanes * road.lane_width
        far_s = j.s + j.width + 1.0
        if j.control == "signal":
            state = j.signal_state(t)
            # pole on the right, mast arm over our lanes, a head over each lane
            self._box(img, road, far_s, rw + 1.5, 0.3, 0.3, 0.0, 7.0, (150, 150, 155))
            self._box(img, road, far_s, rw / 2 - 1.0, rw + 3.0, 0.25, 6.6, 6.85, (140, 140, 145))
            for lane in range(road.lanes):
                d = road.lane_center(lane)
                self._signal_head(img, road, far_s - 0.3, d, 4.9, state)
        elif j.control == "stop":
            self._stop_sign(img, road, j.stop_line_s + 0.5, rw + 1.0)

    def _signal_head(self, img, road: Road, s: float, d: float, z0: float, state: str) -> None:
        cam = self.cam
        x, y, h = road.point(s, d)
        ex, ey = math.cos(h), -math.sin(h)
        w, hh = 0.62, 1.45            # yellow box, black face, 0.3 m lamps (as measured in game)
        corners = []
        for a, b in ((-w / 2, z0 + hh), (w / 2, z0 + hh), (w / 2, z0), (-w / 2, z0)):
            corners.append([x + a * ex, y + a * ey, b])
        C = cam.to_camera(np.array(corners))
        if (C[:, 2] < NEAR).any():
            return
        P = cam.to_screen(C).astype(np.int32)
        cv2.fillPoly(img, [P], HOUSING)
        inner = []
        for a, b in ((-w / 2 + 0.08, z0 + hh - 0.08), (w / 2 - 0.08, z0 + hh - 0.08), (w / 2 - 0.08, z0 + 0.08),
                     (-w / 2 + 0.08, z0 + 0.08)):
            inner.append([x + a * ex, y + a * ey, b])
        Pi = cam.to_screen(cam.to_camera(np.array(inner))).astype(np.int32)
        cv2.fillPoly(img, [Pi], BODY)
        for i, name in enumerate(("red", "yellow", "green")):
            zc = z0 + hh - 0.27 - i * 0.45
            c = cam.to_camera(np.array([[x, y, zc]]))[0]
            px = cam.to_screen(c[None])[0]
            r = max(1, int(round(0.15 * cam.focal / c[2])))
            col = LAMP[name] if state == name else LAMP["off"]
            cv2.circle(img, (int(px[0]), int(px[1])), r, col, -1, cv2.LINE_AA if r > 2 else cv2.LINE_8)

    def _stop_sign(self, img, road: Road, s: float, d: float) -> None:
        self._box(img, road, s, d, 0.08, 0.08, 0.0, 2.2, (150, 150, 155))
        cam = self.cam
        x, y, _ = road.point(s, d)
        c = cam.to_camera(np.array([[x, y, 2.5]]))[0]
        if c[2] < NEAR:
            return
        px = cam.to_screen(c[None])[0]
        r = 0.38 * cam.focal / c[2]
        if r < 1:
            return
        pts = np.array([[px[0] + r * math.cos(a), px[1] + r * math.sin(a)]
                        for a in np.linspace(math.pi / 8, 2 * math.pi + math.pi / 8, 9)[:-1]], np.int32)
        cv2.fillPoly(img, [pts], (40, 40, 200))
        if r > 6:
            fs = r / 22
            cv2.putText(img, "STOP", (int(px[0] - r * 0.75), int(px[1] + r * 0.25)), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (240, 240, 240), max(1, int(fs * 2)), cv2.LINE_AA)

    def _speed_sign(self, img, road: Road, sg: SpeedSign) -> None:
        self._box(img, road, sg.s, sg.d, 0.08, 0.08, 0.0, 2.0, (150, 150, 155))
        cam = self.cam
        x, y, _ = road.point(sg.s, sg.d)
        c = cam.to_camera(np.array([[x, y, 2.4]]))[0]
        if c[2] < NEAR:
            return
        px = cam.to_screen(c[None])[0]
        hh = 0.45 * cam.focal / c[2]
        ww = hh * 0.75
        if hh < 2:
            return
        p1 = (int(px[0] - ww), int(px[1] - hh))
        p2 = (int(px[0] + ww), int(px[1] + hh))
        cv2.rectangle(img, p1, p2, (245, 245, 245), -1)
        cv2.rectangle(img, p1, p2, (20, 20, 20), max(1, int(hh / 12)))
        if hh > 8:
            fs = hh / 28
            cv2.putText(img, str(sg.limit), (int(px[0] - ww * 0.8), int(px[1] + hh * 0.55)), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (15, 15, 15), max(1, int(fs * 2.5)), cv2.LINE_AA)

    # ------------------------------------------------------------------ boxes
    def _box(self, img, road: Road, s: float, d: float, along: float, across: float, z0: float, z1: float,
             col: tuple, heading_offset: float = 0.0) -> None:
        x, y, h = road.point(s, d)
        self._box_world(img, x, y, h + heading_offset, along, across, z0, z1, col)

    def _box_world(self, img, x, y, h, along, across, z0, z1, col, top_col=None, rear_col=None) -> None:
        fx, fy = math.sin(h), math.cos(h)
        ex, ey = math.cos(h), -math.sin(h)
        corners = []
        for a, b in ((-along / 2, -across / 2), (-along / 2, across / 2), (along / 2, across / 2),
                     (along / 2, -across / 2)):
            corners.append((x + a * fx + b * ex, y + a * fy + b * ey))
        bot = [[c[0], c[1], z0] for c in corners]
        top = [[c[0], c[1], z1] for c in corners]
        faces = [
            ([bot[0], bot[1], top[1], top[0]], rear_col or _shade(col, 0.8)),   # rear
            ([bot[2], bot[3], top[3], top[2]], _shade(col, 0.8)),               # front
            ([bot[1], bot[2], top[2], top[1]], _shade(col, 0.65)),              # right side
            ([bot[3], bot[0], top[0], top[3]], _shade(col, 0.65)),              # left side
            ([top[0], top[1], top[2], top[3]], top_col or col),                 # roof
        ]
        cam = self.cam
        drawn = []
        for poly, c in faces:
            P = np.array(poly)
            C = cam.to_camera(P)
            drawn.append((C[:, 2].mean(), P, c))
        drawn.sort(key=lambda t: -t[0])
        for _, P, c in drawn:
            pp = self._project_polys([P])
            if pp:
                cv2.fillPoly(img, pp, c)

    def _wheels(self, img, x: float, y: float, h: float, length: float, width: float) -> None:
        """Four black tyres touching the road (cars sit on them, the body floats 0.3 m up)."""
        fx, fy = math.sin(h), math.cos(h)
        ex, ey = math.cos(h), -math.sin(h)
        for a in (-0.32, 0.32):
            for b in (-0.42, 0.42):
                cx = x + a * length * fx + b * width * ex
                cy = y + a * length * fy + b * width * ey
                self._box_world(img, cx, cy, h, 0.7, 0.3, 0.0, 0.65, (22, 22, 24))

    def _other_car(self, img, road: Road, car: OtherCar) -> None:
        d = road.lane_center(car.lane)
        x, y, h = road.point(car.s + car.length / 2, d)
        self._wheels(img, x, y, h, car.length, car.width)
        self._box_world(img, x, y, h, car.length, car.width, 0.3, car.height * 0.62, car.color)
        self._box_world(img, x - 0.3 * math.sin(h), y - 0.3 * math.cos(h), h, car.length * 0.5, car.width * 0.85,
                        car.height * 0.62, car.height, _shade(car.color, 0.7), top_col=(70, 40, 35))
        # tail lights: bright red, brighter when braking
        cam = self.cam
        rx, ry, _ = road.point(car.s, d)
        ex, ey = math.cos(h), -math.sin(h)
        bright = (30, 30, 255) if tail_light_on(car) else (40, 40, 200)
        for side in (-0.7, 0.7):
            c = cam.to_camera(np.array([[rx + side * ex, ry + side * ey, car.height * 0.5]]))[0]
            if c[2] < NEAR:
                continue
            px = cam.to_screen(c[None])[0]
            r = max(1, int(0.12 * cam.focal / c[2]))
            cv2.circle(img, (int(px[0]), int(px[1])), r, bright, -1)
        # tyres
        for side in (-0.85, 0.85):
            self._box_world(img, rx + side * ex + 0.6 * math.sin(h), ry + side * ey + 0.6 * math.cos(h), h,
                            0.7, 0.25, 0.0, 0.6, (20, 20, 22))

    def _prop(self, img, road: Road, p: Prop) -> None:
        if p.kind == "tree":
            self._box(img, road, p.s, p.d, 0.5 * p.size, 0.5 * p.size, 0.0, 3.0 * p.size, (40, 60, 90))
            cam = self.cam
            x, y, _ = road.point(p.s, p.d)
            c = cam.to_camera(np.array([[x, y, 5.5 * p.size]]))[0]
            if c[2] < NEAR:
                return
            px = cam.to_screen(c[None])[0]
            r = int(2.6 * p.size * cam.focal / c[2])
            if r >= 1:
                cv2.circle(img, (int(px[0]), int(px[1])), r, (50, 115, 60), -1)
                cv2.circle(img, (int(px[0] - r * 0.3), int(px[1] - r * 0.3)), max(1, r // 2), (60, 140, 75), -1)
        elif p.kind == "pole":
            self._box(img, road, p.s, p.d, 0.25, 0.25, 0.0, 9.0, (165, 165, 170))
            self._box(img, road, p.s, p.d - 1.5, 0.2, 3.0, 8.8, 9.0, (165, 165, 170))

    # ------------------------------------------------------------------ your car + HUD
    def _ego(self, img, ego: Vehicle) -> None:
        st, sp = ego.state, ego.spec
        h = st.heading
        back = 0.9
        cx = st.x + (sp.length / 2 - back) * math.sin(h)
        cy = st.y + (sp.length / 2 - back) * math.cos(h)
        body = (25, 22, 20)
        self._box_world(img, cx, cy, h, sp.length, sp.width, 0.25, sp.height * 0.6, body, rear_col=(18, 16, 15))
        self._box_world(img, cx - 0.3 * math.sin(h), cy - 0.3 * math.cos(h), h, sp.length * 0.55, sp.width * 0.82,
                        sp.height * 0.6, sp.height, (60, 30, 25), top_col=(120, 45, 25))
        cam = self.cam
        rx = st.x - back * math.sin(h)
        ry = st.y - back * math.cos(h)
        ex, ey = math.cos(h), -math.sin(h)
        for side in (-0.7, 0.7):
            c = cam.to_camera(np.array([[rx + side * ex, ry + side * ey, sp.height * 0.5]]))[0]
            if c[2] > NEAR:
                px = cam.to_screen(c[None])[0]
                cv2.ellipse(img, (int(px[0]), int(px[1])), (max(2, int(0.25 * cam.focal / c[2])), 2), 0, 0, 360,
                            (40, 40, 220), -1)

    def _hud(self, img, speed_kph: float) -> None:
        H, W = img.shape[:2]
        # speedometer box, bottom centre, like ER:LC
        x1, y1, x2, y2 = int(0.44 * W), int(0.80 * H), int(0.56 * W), int(0.93 * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 58, 55), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (235, 235, 235), 2)
        txt = f"{int(round(max(0.0, speed_kph)))} KM/H"
        fs = (y2 - y1) / 95
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.putText(img, txt, ((x1 + x2 - tw) // 2, y1 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, fs, (245, 245, 245), 2,
                    cv2.LINE_AA)
        cv2.putText(img, "D", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (245, 245, 245), 2, cv2.LINE_AA)
        # minimap and top bar
        cv2.rectangle(img, (int(0.01 * W), int(0.83 * H)), (int(0.09 * W), int(0.98 * H)), (70, 80, 70), -1)
        cv2.rectangle(img, (int(0.005 * W), int(0.01 * H)), (int(0.11 * W), int(0.05 * H)), (30, 30, 30), -1)
        cv2.rectangle(img, (int(0.9 * W), int(0.01 * H)), (int(0.995 * W), int(0.05 * H)), (30, 30, 30), -1)


def _shade(col, k):
    return tuple(int(max(0, min(255, c * k))) for c in col)


def _clip_near(C: np.ndarray, near: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a polygon (camera coords) against z >= near."""
    out = []
    n = len(C)
    for i in range(n):
        a, b = C[i], C[(i + 1) % n]
        ina, inb = a[2] >= near, b[2] >= near
        if ina:
            out.append(a)
        if ina != inb:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.array(out) if out else np.zeros((0, 3))
