"""FSD drives the real Liberty County map.

    python -m fsd.sim.erlc list
    python -m fsd.sim.erlc run highway_east --video out
    python -m fsd.sim.erlc run all

Same loop as fsd.sim (perception -> planner -> controllers -> key modulator ->
keyboard car physics), but the picture is the actual ER:LC map drawn from the
chase camera, and the score comes from the map's own lane paint.

Units: the car and FSD work in metres (x east, y north); the map is in studs
(x east, y up, z south). 1 stud = 0.28 m.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ...control.controllers import LateralController, SpeedController
from ...fast.scaler import FramePair
from ...params import defaults
from ...planning.planner import Planner
from ...settings import Settings
from ..camera import ChaseCamera
from ..render import Renderer as FlatRenderer
from ..runner import Metrics, calibration_for
from ..vehicle import KeyModulator, Vehicle, VehicleSpec
from .glrender import GLRenderer, sky_gradient
from .mapdata import STUD_M, SHAPE_BOX, SHAPE_CYL_X, ErlcMap
from .roads import RoadTruth


@dataclass
class Route:
    name: str
    x: float                      # spawn point (studs)
    z: float
    hx: float                     # rough heading (studs, unit vector)
    hz: float
    lane: int = 0                 # 0 = the lane we spawn in; +1 one lane to the right ...
    duration_s: float = 40.0
    limit_kph: float = 90.0
    start_kph: float = 50.0
    vehicle: VehicleSpec = field(default_factory=VehicleSpec)
    description: str = ""


ROUTES: dict[str, Route] = {}


def _add(r: Route) -> None:
    ROUTES[r.name] = r


# Real places on the map (found from the road paint). Headings follow the lane direction.
_add(Route("highway_east", 2006.0, 2175.0, 1.0, 0.0, duration_s=45, limit_kph=100, start_kph=70,
           description="Southern highway, eastbound, right lane: long straight, then the curve north."))
_add(Route("highway_west", 2006.0, 2128.0, -1.0, 0.0, duration_s=45, limit_kph=100, start_kph=70,
           description="Southern highway, westbound: straight into the long bend toward the interchange."))
_add(Route("highway_curve_ne", 1201.0, -857.0, 0.82, 0.57, duration_s=40, limit_kph=100, start_kph=70,
           description="Highway on the long sweeping bend north-east of the interchange."))
_add(Route("interchange_sw", -505.0, -54.0, -0.70, 0.71, duration_s=35, limit_kph=90, start_kph=60,
           description="Diagonal highway through the big interchange, heading south-west."))
_add(Route("town_south_north", -2180.0, 180.0, 0.0, -1.0, duration_s=35, limit_kph=56, start_kph=35,
           description="Southern town main street, northbound past junctions."))
_add(Route("town_north_west", -2259.0, -2436.0, -1.0, 0.0, duration_s=30, limit_kph=56, start_kph=35,
           description="Northern town grid, westbound."))


# ---------------------------------------------------------------- ego car as parts
def car_instances(cx, cy, cz, hx, hz, spec: VehicleSpec, braking: bool):
    """Boxes for our own car (studs). (cx, cz) = centre of the car, cy = ground."""
    S = 1 / STUD_M
    L, W, H = spec.length * S, spec.width * S, spec.height * S
    fwd = np.array([hx, 0.0, hz])
    right = np.array([-hz, 0.0, hx])
    up = np.array([0.0, 1.0, 0.0])
    R = np.stack([right, up, fwd], 1)          # local x = right, y = up, z = forward
    out = []

    def box(off, size, rgb, shape=SHAPE_BOX, glow=0.0):
        p = np.array([cx, cy, cz]) + R @ np.asarray(off, float)
        out.append((shape, p[None], R[None], np.asarray(size, float)[None], np.array([[*rgb, glow]], np.float32)))

    body = (0.10, 0.12, 0.30)
    box((0, 0.3 * S + H * 0.31, 0), (W, H * 0.62 - 0.3 * S, L), body)
    box((0, H * 0.81, -0.3 * S), (W * 0.85, H * 0.38, L * 0.5), (0.08, 0.09, 0.22))
    box((0, H * 0.81, -0.3 * S), (W * 0.86, H * 0.30, L * 0.48), (0.12, 0.16, 0.22))   # windows band
    tl = (1.0, 0.25, 0.2) if braking else (0.85, 0.08, 0.08)
    for side in (-0.36, 0.36):
        box((side * W, H * 0.48, -L / 2 - 0.02), (W * 0.18, H * 0.08, 0.1), tl, glow=1.0 if braking else 0.0)
    for a in (-0.32, 0.32):
        for b in (-0.42, 0.42):           # tyres: cylinders whose axis is our right vector
            box((b * W, 0.33 * S, a * L), (0.3 * S, 0.66 * S, 0.66 * S), (0.08, 0.08, 0.09), shape=SHAPE_CYL_X)
    return out


class ErlcDrive:
    def __init__(self, route: Route, emap: ErlcMap, truth: RoadTruth, params: dict | None = None,
                 video: str | None = None, dt: float = 1 / 30, physics_hz: int = 60, width: int = 1920,
                 verbose: bool = False, log=print):
        from ...fast.pipeline import FastPerception
        self.route, self.map, self.truth, self.log = route, emap, truth, log
        self.p = dict(defaults())
        if params:
            self.p.update(params)
        self.dt = dt
        self.sub = max(1, int(round(physics_hz * dt)))
        self.verbose = verbose
        height = int(round(width * 512 / 960))
        self.cam = ChaseCamera(width=width, height=height)
        self.gl = GLRenderer(emap, width, height, vfov=self.cam.vfov_deg)
        self.car = Vehicle(route.vehicle)
        # spawn: centre of the chosen lane, heading along the lane lines
        hx, hz = route.hx, route.hz
        x, z = route.x, route.z
        lane = truth.lane_at(x, z, hx, hz)
        if lane is None:
            x, z = truth.beside_yellow(x, z, hx, hz)
            lane = truth.lane_at(x, z, hx, hz)
        if lane is not None:
            lo, ro, _, _ = lane
            shift = (lo + ro) / 2 + route.lane * (ro - lo)
            x, z = x + shift * (-hz), z + shift * hx
        self.car.state.heading = math.atan2(hx, -hz)        # metres frame: forward = (sin h, cos h), y north
        self.car.state.x, self.car.state.y = x * STUD_M, -z * STUD_M
        self.car.state.v = route.start_kph / 3.6
        self.ground = emap.ground_at(x, z)
        self.cam.follow(*self._rear_axle(), self.car.state.heading, 0.9, dt)
        settings = Settings()
        settings.calibration = calibration_for(self.cam, route.vehicle.length, route.vehicle.width,
                                               route.vehicle.height, 3.7)
        self.per = FastPerception(settings, use_detector=False, use_ocr=False, live=False)
        self.planner = Planner()
        self.lat = LateralController()
        self.lon = SpeedController()
        self.keys = KeyModulator(self.p["steer_rise_per_s"], self.p["steer_return_per_s"])
        self.keys.dt = self.dt / self.sub
        self.video = None
        if video:
            Path(video).parent.mkdir(parents=True, exist_ok=True)
            self.video = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), int(round(1 / dt)), (960, 512))
        self.m = Metrics(route.name)
        self.t = 0.0
        self._lane_err: list[float] = []
        self._speeds: list[float] = []
        self._lost = 0.0
        self._edge_now = False
        self.last_world = None
        self.last_plan = None

    # ------------------------------------------------------------ geometry helpers
    def _rear_axle(self) -> tuple[float, float]:
        return self.car.state.x, self.car.state.y

    def _centre_studs(self) -> tuple[float, float, float, float]:
        st, sp = self.car.state, self.car.spec
        c = sp.length / 2 - 0.9
        xm = st.x + c * math.sin(st.heading)
        ym = st.y + c * math.cos(st.heading)
        hx, hz = math.sin(st.heading), -math.cos(st.heading)
        return xm / STUD_M, -ym / STUD_M, hx, hz

    def _camera_studs(self):
        cam = self.cam
        cx, cz = cam.x / STUD_M, -cam.y / STUD_M
        eye = np.array([cx, self.ground + cam.height_m / STUD_M, cz])
        fx, fz = math.sin(cam.yaw), -math.cos(cam.yaw)
        p = math.radians(cam.pitch_deg)
        d = np.array([fx * math.cos(p), -math.sin(p), fz * math.cos(p)])
        return eye, eye + d * 50.0

    # ------------------------------------------------------------ one frame
    def frame(self, speed_kph: float, braking: bool) -> np.ndarray:
        x, z, hx, hz = self._centre_studs()
        eye, tgt = self._camera_studs()
        img = self.gl.render(eye, tgt, car_instances(x, self.ground, z, hx, hz, self.car.spec, braking))
        img = sky_gradient(img, self.cam.horizon_y)
        FlatRenderer._hud(None, img, speed_kph)
        return img

    def step(self) -> np.ndarray:
        x, z, hx, hz = self._centre_studs()
        self.ground = 0.7 * self.ground + 0.3 * self.map.ground_at(x, z, y_hint=self.ground)
        braking = getattr(self, "_last_pedal", 0.0) < -0.1
        img = self.frame(self.car.speed_kph, braking)
        pair = FramePair(img, work_w=960, fine_w=1920)
        w = self.per.process(pair.work, self.p, t=self.t, fine=pair.fine)
        w.speed_kph = self.car.speed_kph
        w.speed_valid = True
        w.speed_limit_kph = self.route.limit_kph
        plan = self.planner.step(w, self.p, True, now=self.t)
        self.lat.observe(self.t, self.car.speed_kph, self.keys.obs.s, w.yaw_rate, self.p, w.lanes.conf > 0.4)
        steer = self.lat.step(w, plan, self.p, self.dt)
        pedal = self.lon.step(w, plan, self.p, self.dt)
        for _ in range(self.sub):
            W_, A_, S_, D_ = self.keys.keys(steer, pedal, self.car.speed_kph)
            self.car.step(self.dt / self.sub, W_, A_, S_, D_)
        self._last_pedal = pedal
        self.cam.follow(*self._rear_axle(), self.car.state.heading, 0.9, self.dt)
        self._score()
        self.last_world, self.last_plan = w, plan
        if self.verbose and int(self.t / self.dt) % 15 == 0:
            self.log(f"t={self.t:5.1f} v={self.car.speed_kph:5.1f} tgt={plan.target_speed_kph:5.1f} "
                     f"{plan.mode.value:9s} steer={steer:+.2f} pedal={pedal:+.2f} conf={w.lanes.conf:.2f} "
                     f"err={self._lane_err[-1] if self._lane_err else float('nan'):+.2f} | {plan.reason}")
        if self.video is not None:
            from ...tools.inspect_image import draw
            out = draw(pair.work, w, self.per)
            cv2.putText(out, f"{self.route.name} t={self.t:4.1f}s {self.car.speed_kph:5.1f} km/h {plan.mode.value}: "
                             f"{plan.reason}", (8, out.shape[0] - 12), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            self.video.write(cv2.resize(out, (960, 512)))
        self.t += self.dt
        return pair.work

    def _score(self) -> None:
        x, z, hx, hz = self._centre_studs()
        self._speeds.append(self.car.speed_kph)
        lane = self.truth.lane_at(x, z, hx, hz)
        if lane is None:
            if self.map.on_road(x, z, self.ground):
                self._lost = 0.0           # junction or a stretch whose paint isn't in the export
                return
            self._lost += self.dt
            if self._lost > 1.5:
                self.m.offroad = True
                self.m.notes.append(f"off the road surface at t={self.t:.1f}s (left the road?)")
            return
        self._lost = 0.0
        lo, ro, ly, ry = lane
        centre = (lo + ro) / 2
        err_m = -centre * STUD_M                         # + = we're right of the lane centre
        self._lane_err.append(abs(err_m))
        half = self.car.spec.width / 2 / STUD_M
        if lo > -half or ro < half:
            self.m.lane_departure_s += self.dt
        touching_yellow = (ly and lo > -half) or (ry and ro < half)
        if touching_yellow and not self._edge_now:
            self.m.edge_violations += 1
            self.m.notes.append(f"wheels on the yellow line at t={self.t:.1f}s")
        self._edge_now = bool(touching_yellow)

    def run(self) -> Metrics:
        t0 = time.perf_counter()
        n = int(self.route.duration_s / self.dt)
        for _ in range(n):
            self.step()
            if self.m.offroad:
                break
        m = self.m
        if self._lane_err:
            m.mean_abs_lane_err = float(np.mean(self._lane_err))
            m.max_abs_lane_err = float(np.max(self._lane_err))
        m.mean_speed_kph = float(np.mean(self._speeds)) if self._speeds else 0.0
        m.fps = n / max(1e-6, time.perf_counter() - t0)
        if self.video is not None:
            self.video.release()
        return m


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m fsd.sim.erlc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("names", nargs="+")
    r.add_argument("--video", default=None)
    r.add_argument("--verbose", action="store_true")
    r.add_argument("--seconds", type=float, default=None)
    s = sub.add_parser("shot", help="save one picture from a route's start")
    s.add_argument("name")
    s.add_argument("--out", default="erlc_shot.png")
    a = ap.parse_args()
    if a.cmd == "list":
        for n, rt in ROUTES.items():
            print(f"{n:18s} {rt.description}")
        return
    emap = ErlcMap.load()
    truth = RoadTruth(emap)
    if a.cmd == "shot":
        d = ErlcDrive(ROUTES[a.name], emap, truth)
        cv2.imwrite(a.out, d.frame(d.car.speed_kph, False))
        print("saved", a.out, "| renderer:", d.gl.renderer_name)
        return
    names = list(ROUTES) if a.names == ["all"] else a.names
    total = 0.0
    for n in names:
        rt = ROUTES[n]
        if a.seconds:
            rt = Route(**{**rt.__dict__, "duration_s": a.seconds})
        vid = str(Path(a.video) / f"{n}.mp4") if a.video else None
        m = ErlcDrive(rt, emap, truth, video=vid, verbose=a.verbose).run()
        sc = m.score()
        total += sc
        print(f"{'PASS' if m.passed() else 'FAIL'} {n:16s} score {sc:7.2f} | lane err {m.mean_abs_lane_err:.2f}/"
              f"{m.max_abs_lane_err:.2f} m | departures {m.lane_departure_s:.1f}s | yellow {m.edge_violations} | "
              f"{m.mean_speed_kph:.0f} km/h | {m.fps:.1f} fps")
        for note in m.notes[:6]:
            print("      -", note)
    print(f"TOTAL {total:.2f}")


if __name__ == "__main__":
    main()
