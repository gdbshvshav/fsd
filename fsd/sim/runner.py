"""Closed-loop simulation: picture -> real FSD -> keys -> car -> next picture.

The same code that drives in Roblox drives here:
    FastPerception (vision)  ->  Planner  ->  Lateral/Speed controllers
    -> sigma-delta key timing (like the real actuator, 60 Hz)  ->  car physics

Only two things are faked: the speedometer reading (we know the true speed)
and the speed limit (the scenario tells us), so these runs test lanes, lights,
signs, obstacles, planning and control.

Scoring (lower is better), all from the car's true position on the road:
    lane error        mean / max distance of the car's centre from its lane centre
    lane departures   seconds with a wheel over a lane line it shouldn't cross
    edge / median     any wheel over the yellow edge or off the road = big penalty
    red light         crossing a stop bar on red (or not fully stopping at a sign)
    stop accuracy     how far from the stop bar it stopped (aim: 0.5 - 3 m before)
    collisions        touching another car
    comfort           steering key flips per second, hard brakes
    progress          too slow for no reason
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..fast.scaler import FramePair
from ..control.controllers import LateralController, SpeedController
from ..params import defaults
from ..planning.planner import Planner
from ..settings import Calibration, Settings
from .camera import ChaseCamera
from .render import Renderer
from .scenarios import SCENARIOS, Scenario, quick_names
from .vehicle import KeyModulator, Vehicle


@dataclass
class Metrics:
    scenario: str = ""
    duration_s: float = 0.0
    distance_m: float = 0.0
    mean_abs_lane_err: float = 0.0
    max_abs_lane_err: float = 0.0
    lane_departure_s: float = 0.0
    edge_violations: int = 0
    offroad: bool = False
    red_light_violations: int = 0
    stop_sign_violations: int = 0
    stops: list = field(default_factory=list)          # metres before the stop bar where we stopped
    collisions: int = 0
    min_gap_m: float = 999.0
    steer_flips_per_s: float = 0.0
    hard_brakes: int = 0
    mean_speed_kph: float = 0.0
    fps: float = 0.0
    lane_conf_mean: float = 0.0
    modes: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def score(self) -> float:
        s = 0.0
        s += 3.0 * self.mean_abs_lane_err + 1.0 * max(0.0, self.max_abs_lane_err - 1.0)
        s += 2.0 * self.lane_departure_s
        s += 25.0 * self.edge_violations + (60.0 if self.offroad else 0.0)
        s += 30.0 * self.red_light_violations + 30.0 * self.stop_sign_violations
        s += 50.0 * self.collisions
        for d in self.stops:
            if d < 0:
                s += 10.0 + 3.0 * abs(d)              # stopped past the bar
            elif d > 4.0:
                s += 1.0 * (d - 4.0)                  # stopped way short
        s += 0.3 * self.steer_flips_per_s
        s += 0.5 * self.hard_brakes
        return round(s, 3)

    def passed(self) -> bool:
        return (self.edge_violations == 0 and not self.offroad and self.red_light_violations == 0
                and self.stop_sign_violations == 0 and self.collisions == 0 and self.max_abs_lane_err < 1.6)


def calibration_for(cam: ChaseCamera, car_len: float, car_w: float, car_h: float, lane_w: float = 3.7) -> Calibration:
    """The exact calibration of the sim camera (what F4 / self-calibration should find)."""
    cal = Calibration()
    W, H = cam.width, cam.height
    cam_to_bumper = cam.back_m + car_len
    near, far = cam_to_bumper + 1.0, cam_to_bumper + 40.0

    def scr(right, fwd, up=0.0):
        save = (cam.x, cam.y, cam.yaw)
        cam.x = cam.y = cam.yaw = 0.0
        c = cam.to_camera(np.array([[right, fwd, up]]))
        p = cam.to_screen(c)[0]
        cam.x, cam.y, cam.yaw = save
        return float(p[0] / W), float(p[1] / H)

    hw = lane_w / 2
    cal.src_quad = [list(scr(-hw, near)), list(scr(hw, near)), list(scr(hw, far)), list(scr(-hw, far))]
    cal.quad_width_m = lane_w
    cal.quad_depth_m = far - near
    cal.bumper_to_quad_m = 1.0
    cal.cam_to_bumper_m = cam_to_bumper
    cal.vfov_deg = cam.vfov_deg
    rear = cam.back_m
    pts = [scr(-car_w / 2, rear, 0), scr(car_w / 2, rear, 0), scr(-car_w / 2, rear, car_h),
           scr(car_w / 2, rear + car_len, car_h)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cal.ego_mask = [min(xs) - 0.01, min(ys) - 0.02, max(xs) + 0.01, max(ys) + 0.01]
    cal.speedo_roi = None
    return cal


class SimRun:
    def __init__(self, sc: Scenario, params: dict | None = None, video: str | None = None, dt: float = 1 / 30,
                 physics_hz: int = 60, verbose: bool = False, fine: bool = True):
        """fine=True renders at 1920x1024 and feeds the pipeline a work + fine pair, exactly like
        the app does with a real capture (so small things - lamps, signs - are as sharp as in game)."""
        from ..fast.pipeline import FastPerception
        self.sc = sc
        self.p = dict(defaults())
        if params:
            self.p.update(params)
        self.dt = dt
        self.sub = max(1, int(round(physics_hz * dt)))
        self.verbose = verbose
        self.scene = sc.build()
        self.road = self.scene.road
        self.car = Vehicle(sc.vehicle)
        x, y, h = self.road.point(sc.start_s, self.road.lane_center(sc.start_lane) + sc.start_offset)
        self.car.state.x, self.car.state.y = x, y
        self.car.state.heading = h + math.radians(sc.start_angle_deg)
        self.car.state.v = sc.start_speed_kph / 3.6
        self.fine = fine
        self.cam = ChaseCamera(width=1920, height=1024) if fine else ChaseCamera()
        self.cam.follow(x, y, self.car.state.heading, 0.9, dt)
        self.renderer = Renderer(self.cam)
        settings = Settings()
        exact = calibration_for(self.cam, sc.vehicle.length, sc.vehicle.width, sc.vehicle.height,
                                self.p["lane_width_m"])
        if sc.exact_calibration:
            settings.calibration = exact
        else:
            settings.calibration.ego_mask = exact.ego_mask
        self.settings = settings
        self.per = FastPerception(settings, use_detector=False, use_ocr=False, live=False)
        self.planner = Planner()
        self.lat = LateralController()
        self.lon = SpeedController()
        from ..drive.driver import Driver
        self.driver = Driver()
        self.planner.turns.model = self.driver.turning
        self.keys = KeyModulator(self.p["steer_rise_per_s"], self.p["steer_return_per_s"])
        self.keys.dt = self.dt / self.sub
        self.video = None
        if video:
            Path(video).parent.mkdir(parents=True, exist_ok=True)
            self.video = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), int(round(1 / dt)),
                                         (960, int(round(self.cam.height * 960 / self.cam.width))))
        self.m = Metrics(sc.name)
        self.t = 0.0
        self.s = sc.start_s
        self.target_lane = sc.start_lane
        self._prev_steer_key = 0
        self._flips = 0
        self._lane_err = []
        self._speeds = []
        self._conf = []
        self._passed_bars: set = set()
        self._stopped_near: dict = {}
        self._was_stopped_for_sign: set = set()

    # ------------------------------------------------------------------ helpers
    def _car_sd(self) -> tuple[float, float]:
        # centre of the car's footprint
        st, sp = self.car.state, self.car.spec
        c = sp.length / 2 - 0.9
        x = st.x + c * math.sin(st.heading)
        y = st.y + c * math.cos(st.heading)
        return self.road.project(x, y, self.s)

    def _front_s(self) -> float:
        fx, fy = self.car.front_bumper()
        return self.road.project(fx, fy, self.s)[0]

    def _check_rules(self, front_s: float, v: float) -> None:
        for k, j in enumerate(self.scene.junctions):
            bar = j.stop_line_s
            gap = bar - front_s
            if v < 0.3 and -1.0 < gap < 15.0:
                self._stopped_near[k] = gap
                if j.control == "stop" or j.signal.mode == "flashing":
                    self._was_stopped_for_sign.add(k)
            if k not in self._passed_bars and front_s > bar:
                self._passed_bars.add(k)
                state = j.signal_state(self.t)
                if j.control == "signal" and j.signal.mode != "flashing" and state == "red":
                    self.m.red_light_violations += 1
                    self.m.notes.append(f"ran a red light at t={self.t:.1f}s")
                if (j.control == "stop" or (j.control == "signal" and j.signal.mode == "flashing")) \
                        and k not in self._was_stopped_for_sign:
                    self.m.stop_sign_violations += 1
                    self.m.notes.append(f"didn't stop at the bar at t={self.t:.1f}s")
                if k in self._stopped_near:
                    self.m.stops.append(round(self._stopped_near[k], 2))

    def _check_space(self, d: float) -> None:
        r = self.road
        st, sp = self.car.state, self.car.spec
        # lateral extent of the car on the road (approx, using heading vs road)
        _, _, h_road = r.point(self.s, 0)
        rel = (st.heading - h_road + math.pi) % (2 * math.pi) - math.pi
        half = sp.width / 2 * abs(math.cos(rel)) + sp.length / 2 * abs(math.sin(rel))
        left, right = d - half, d + half
        lane_c = r.lane_center(self.target_lane)
        err = d - lane_c
        if getattr(self, "_lc", False):
            err = 0.0                       # mid lane change: no "error"
        self._lane_err.append(abs(err))
        lane_l = lane_c - r.lane_width / 2
        lane_r = lane_c + r.lane_width / 2
        if (left < lane_l - 0.05 or right > lane_r + 0.05) and not getattr(self, "_lc", False):
            self.m.lane_departure_s += self.dt
        edge_l = 0.25 + 0.09 if r.divided else -r.lanes * r.lane_width
        edge_r = r.lanes * r.lane_width
        if left < edge_l - 0.1 or right > edge_r + r.shoulder:
            if not getattr(self, "_edge_now", False):
                self.m.edge_violations += 1
                self.m.notes.append(f"wheel over the {'yellow edge / median' if left < edge_l else 'road edge'} "
                                    f"at t={self.t:.1f}s")
            self._edge_now = True
        else:
            self._edge_now = False
        if left < -r.shoulder - 0.5 or right > edge_r + r.shoulder + 1.0:
            self.m.offroad = True

    def _check_cars(self, front_s: float, d: float) -> None:
        r = self.road
        for c in self.scene.cars:
            if abs(r.lane_center(c.lane) - d) < (c.width + self.car.spec.width) / 2:
                gap = c.s - front_s
                if gap > -c.length:
                    self.m.min_gap_m = min(self.m.min_gap_m, gap)
                    if gap < 0.2 and not getattr(c, "_hit", False):
                        c._hit = True
                        self.m.collisions += 1
                        self.m.notes.append(f"hit a car at t={self.t:.1f}s")

    # ------------------------------------------------------------------ loop
    def step(self) -> np.ndarray:
        self.scene.step(self.dt, self.t)
        img = self.renderer.render(self.scene, self.car, self.s, self.t)
        if self.fine:
            pair = FramePair(img, work_w=960, fine_w=1920)
            img = pair.work
            w = self.per.process(img, self.p, t=self.t, fine=pair.fine)
        else:
            w = self.per.process(img, self.p, t=self.t)
        self.last_world = w
        w.speed_kph = self.car.speed_kph
        w.speed_valid = True
        w.speed_limit_kph = self.sc.limit_kph
        new_core = self.p.get("drive_core", 1.0) > 0.5
        if new_core:
            self.driver.sense(w, self.p, self.t, self.keys.obs.s, lanes_key=w.lanes_t)
        self.planner.frame = img
        plan = self.planner.step(w, self.p, True, now=self.t)
        self.last_plan = plan
        if new_core:
            steer, pedal = self.driver.act(w, plan, self.p, self.dt, self.t)
        else:
            self.lat.observe(self.t, self.car.speed_kph, self.keys.obs.s, w.yaw_rate, self.p, w.lanes.conf > 0.4)
            steer = self.lat.step(w, plan, self.p, self.dt)
            pedal = self.lon.step(w, plan, self.p, self.dt)
        for _ in range(self.sub):
            W_, A_, S_, D_ = self.keys.keys(steer, pedal, self.car.speed_kph)
            k = (1 if D_ else 0) - (1 if A_ else 0)
            if k != 0 and self._prev_steer_key != 0 and k != self._prev_steer_key:
                self._flips += 1
            if k != 0:
                self._prev_steer_key = k
            self.car.step(self.dt / self.sub, W_, A_, S_, D_)
            if S_ and self.car.state.v > 5 and pedal < -0.7:
                pass
        if pedal < -0.7:
            self.m.hard_brakes += 1 if getattr(self, "_last_pedal", 0) >= -0.7 else 0
        self._last_pedal = pedal
        self.cam.follow(self.car.state.x, self.car.state.y, self.car.state.heading, 0.9, self.dt)
        s, d = self._car_sd()
        self.s = s
        front_s = self._front_s()
        # a lane change FSD decided on (keep right, merge, wrong side): the new lane is the target
        from ..world import Mode as _Mode
        if plan.mode == _Mode.LANE_CHANGE:
            self._lc = True
        elif getattr(self, "_lc", False):
            self._lc = False
            self.target_lane = int(np.clip(d // self.road.lane_width, 0, self.road.lanes - 1))
            self.m.notes.append(f"changed to lane {self.target_lane} at t={self.t:.1f}s")
        self._check_rules(front_s, self.car.state.v)
        self._check_space(d)
        self._check_cars(front_s, d)
        self._speeds.append(self.car.speed_kph)
        self._conf.append(w.lanes.conf)
        self.m.modes[plan.mode.value] = self.m.modes.get(plan.mode.value, 0) + 1
        if self.verbose and int(self.t / self.dt) % 15 == 0:
            print(f"t={self.t:5.1f} s={s:6.1f} d={d:+5.2f} v={self.car.speed_kph:5.1f} tgt={plan.target_speed_kph:5.1f} "
                  f"{plan.mode.value:9s} steer={steer:+.2f} pedal={pedal:+.2f} conf={w.lanes.conf:.2f} "
                  f"light={w.signal.state.value} | {plan.reason}")
        if self.video is not None:
            self.video.write(self._overlay(img, w, plan, steer, pedal))
        self.t += self.dt
        return img

    def _overlay(self, img, w, plan, steer, pedal) -> np.ndarray:
        from ..tools.inspect_image import draw
        out = draw(img, w, self.per)
        cv2.putText(out, f"{self.sc.name}  t={self.t:4.1f}s  {self.car.speed_kph:5.1f} km/h  {plan.mode.value}: "
                         f"{plan.reason}", (8, out.shape[0] - 30), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"steer {steer:+.2f}  pedal {pedal:+.2f}", (8, out.shape[0] - 10), 0, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def run(self) -> Metrics:
        t0 = time.perf_counter()
        n = int(self.sc.duration_s / self.dt)
        for _ in range(n):
            self.step()
            if self.s > self.road.length - 40:
                self.m.notes.append("reached the end of the road")
                break
            if self.m.offroad:
                self.m.notes.append("left the road - stopping the run")
                break
        wall = time.perf_counter() - t0
        m = self.m
        m.duration_s = round(self.t, 2)
        m.distance_m = round(self.car.state.odometer, 1)
        m.mean_abs_lane_err = round(float(np.mean(self._lane_err)) if self._lane_err else 0.0, 3)
        m.max_abs_lane_err = round(float(np.max(self._lane_err)) if self._lane_err else 0.0, 3)
        m.lane_departure_s = round(m.lane_departure_s, 2)
        m.steer_flips_per_s = round(self._flips / max(1e-3, self.t), 2)
        m.mean_speed_kph = round(float(np.mean(self._speeds)) if self._speeds else 0.0, 1)
        m.fps = round(len(self._speeds) / max(1e-3, wall), 1)
        m.lane_conf_mean = round(float(np.mean(self._conf)) if self._conf else 0.0, 2)
        if self.sc.expect_stop and not m.stops and m.min_gap_m > 50 and not any("stop" in n for n in m.notes):
            if not any(v < 1.0 for v in self._speeds):
                m.notes.append("expected a full stop but never stopped")
                m.stop_sign_violations += 1
        if self.video is not None:
            self.video.release()
        return m


def run_many(names: list[str], params: dict | None = None, video_dir: str | None = None,
             verbose: bool = False, fine: bool = True) -> list[Metrics]:
    out = []
    for n in names:
        sc = SCENARIOS[n]
        vid = str(Path(video_dir) / f"{n}.mp4") if video_dir else None
        m = SimRun(sc, params, vid, verbose=verbose, fine=fine).run()
        out.append(m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m fsd.sim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("names", nargs="+", help="scenario names, 'all' or 'quick'")
    r.add_argument("--video", default=None, help="folder for mp4 videos")
    r.add_argument("--verbose", action="store_true")
    r.add_argument("--coarse", action="store_true", help="render at 960 only (faster, blurrier lamps/signs)")
    r.add_argument("--params", default=None, help="JSON file of parameter overrides")
    r.add_argument("--json", default=None, help="write results here")
    t = sub.add_parser("tune")
    t.add_argument("--rounds", type=int, default=3)
    t.add_argument("--names", nargs="*", default=None)
    a = ap.parse_args()
    if a.cmd == "list":
        for n, sc in SCENARIOS.items():
            print(f"{n:22s} {sc.description}")
        return
    if a.cmd == "tune":
        from .tune import tune
        tune(a.names or quick_names(), rounds=a.rounds)
        return
    names = list(SCENARIOS) if a.names == ["all"] else quick_names() if a.names == ["quick"] else a.names
    params = json.loads(Path(a.params).read_text()) if a.params else None
    results = run_many(names, params, a.video, a.verbose, fine=not a.coarse)
    total = 0.0
    for m in results:
        sc = m.score()
        total += sc
        flag = "PASS" if m.passed() else "FAIL"
        print(f"{flag} {m.scenario:22s} score {sc:7.2f} | lane err {m.mean_abs_lane_err:.2f}/{m.max_abs_lane_err:.2f} m "
              f"| departures {m.lane_departure_s:.1f}s | edge {m.edge_violations} | red {m.red_light_violations} "
              f"| stop {m.stop_sign_violations} {m.stops} | hits {m.collisions} gap {m.min_gap_m:.1f} "
              f"| {m.mean_speed_kph:.0f} km/h | {m.fps:.0f} fps")
        for n in m.notes:
            print("      -", n)
    print(f"TOTAL {total:.2f}  ({sum(m.passed() for m in results)}/{len(results)} passed)")
    if a.json:
        Path(a.json).write_text(json.dumps([asdict(m) | {"score": m.score()} for m in results], indent=1))


if __name__ == "__main__":
    main()
