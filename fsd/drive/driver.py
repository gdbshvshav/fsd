"""The driving core used by the app and the simulator.

    driver.sense(w, p, now, steer_value)   # after perception: fix yaw, speed, lanes
    plan = planner.step(w, p, engaged, now) # decide (unchanged rules: lights, signs...)
    steer, pedal = driver.act(w, plan, p, dt, now)

sense() replaces three noisy measurements in the World with filtered ones:
  * yaw rate   - chassis model + lanes + (only when it agrees) the picture
  * speed      - speedometer reading + what the pedals should be doing in between
  * lanes      - Kalman-filtered, predicted between detections, re-based on lane change
act() plans the speed ahead (bends, stops, jerk limit) and runs the LQR steering and
the acceleration controller.
"""
from __future__ import annotations

from ..world import Mode, Plan, World
from .camera_lag import CameraLag
from .lane_filter import LaneFilter, filtered
from .lateral import LateralLQR
from .longitudinal import LongitudinalController
from .speed_planner import SpeedPlanner
from .supervisor import Supervisor
from .telemetry import Telemetry
from .yaw import YawFusion

KPH = 3.6


class Driver:
    def __init__(self):
        self.yaw = YawFusion()
        self.lanes = LaneFilter()
        self.lat = LateralLQR()
        self.lon = LongitudinalController()
        self.speed = SpeedPlanner()
        self.tel = Telemetry()
        self.sup = Supervisor()
        self.camlag = CameraLag()
        self.loop_fps: float | None = None          # set by the app
        self.raw_speed_kph = 0.0
        self.raw_yaw = 0.0
        self.last_pedal = 0.0
        self.last_steer = 0.0
        self.steer_value = 0.0
        self.lane_switch = 0
        self.junction_until = -1e9
        self.debug: dict = {}

    def reset(self) -> None:
        self.sup.reset()
        self.lat.reset()
        self.lon.reset()
        self.speed.reset()
        self.last_pedal = 0.0
        self.last_steer = 0.0

    # ------------------------------------------------------------------
    def sense(self, w: World, p, now: float, steer_value: float, lanes_key=None) -> World:
        self.steer_value = steer_value
        # speed: predict with the pedal we sent, correct with the speedometer
        sp = self.lon.speed
        a_exp = self.lon.map.accel_for(self.last_pedal, sp.v)
        sp.predict(now, a_exp)
        self.raw_speed_kph = w.speed_kph
        v = sp.update(now, w.speed_kph / KPH if w.speed_valid else None, w.speed_valid)
        if sp.ok and w.speed_valid:
            w.speed_kph = v * KPH
        self.lon.map.observe(now, v, self.last_pedal, valid=w.speed_valid)
        # yaw
        self.raw_yaw = w.yaw_rate
        v = w.speed_kph / KPH
        w.yaw_rate = self.yaw.update(now, v, steer_value, p, self.raw_yaw, self.lanes.lanes(now) if self.lanes.ok else w.lanes)
        # junction crossing: stop bar / crosswalk just ahead -> the lines will vanish for a while
        if min(w.stop_line_dist, w.crosswalk_dist) < 14.0:
            self.junction_until = now + float(min(7.0, 45.0 / max(v, 3.0)))
        in_junction = now < self.junction_until
        self.lanes.junction = in_junction
        self.lanes.max_age = float(min(7.0, 45.0 / max(v, 3.0))) if in_junction else 1.6
        # lanes
        key = lanes_key if lanes_key is not None else (round(w.lanes.offset, 4), round(w.lanes.heading, 5), w.lanes.conf)
        # the chase camera lags the car's turning: rotate what it saw back into the car's frame
        self.camlag.update(now, w.yaw_rate - self.lanes.yaw_bias, self.camlag.tau(p))
        raw = self.camlag.correct(w.lanes)
        w.lanes = filtered(raw, self.lanes, now, v, w.yaw_rate, key)
        # the lane filter learns how far our yaw estimate is off; everyone else gets the corrected one
        w.yaw_rate = w.yaw_rate - self.lanes.yaw_bias
        self.lane_switch = self.lanes.switched
        if w.lanes.conf > 0 and not self.lanes.changing:
            look = float(min(12.0, max(3.0, p.get("steer_look_s", 0.35) * v)))
            self.camlag.learn(p, w.lanes.offset, w.yaw_rate, v, w.lanes.conf, look + 3.7)
        w.lane_switch = self.lane_switch
        self.tel.add_sense(now, raw, w.lanes, self.raw_yaw, w.yaw_rate, self.raw_speed_kph, w.speed_kph)
        return w

    # ------------------------------------------------------------------
    def act(self, w: World, plan: Plan, p, dt: float, now: float) -> tuple[float, float]:
        v = max(0.0, w.speed_kph / KPH)
        turning = plan.mode == Mode.TURNING
        self.lanes.changing = plan.lane_change or plan.mode == Mode.LANE_CHANGE
        if plan.mode == Mode.OFF:
            self.reset()
            return 0.0, 0.0
        # ---- steering ----
        # (the lane filter's yaw-bias state already absorbs a car that steers weaker/stronger than
        # the model; a learned strength factor on top fought it, so it's only used if asked for)
        strength = self.yaw.strength.gain if p.get("steer_strength_learn", 0.0) > 0.5 else 1.0
        s = self.lat.step(plan.path, v, self.steer_value, strength, p, dt, now, turning, w.lanes.conf)
        sup = self.sup.update(now, dt, self.lat.geom.e if self.lat.geom.ok else 0.0, s, turning, w.speed_valid,
                              self.lanes.sigma_offset() if self.lanes.ok else 0.0, self.lanes.rejects,
                              self.lat.guard.weaving, self.loop_fps, w.speed_kph)
        target_kph = plan.target_speed_kph
        if sup.cap_kph < target_kph:
            target_kph = sup.cap_kph
            plan.target_speed_kph = sup.cap_kph
            if sup.reasons:
                plan.reason = f"{plan.reason} · {sup.reasons[0]}" if plan.reason else sup.reasons[0]
        # ---- speed ----
        sight = None          # the planner already slows when the visible road suddenly gets short
        sp = self.speed.step(v, target_kph, plan.stop_at, plan.path, w.lanes.center if w.lanes.conf > 0 else None,
                             w.lanes.x_max, w.lanes.conf, p, dt, sight, turning)
        holding = plan.mode in (Mode.STOPPED, Mode.STOP_WAIT) or (plan.target_speed_kph < 0.5 and v < 0.25)
        pedal = self.lon.step(v, w.speed_valid, sp.v_target, sp.a_target, holding, p, dt, now,
                              blind_stop=plan.mode in (Mode.STOPPING, Mode.STOPPED, Mode.STOP_WAIT, Mode.FAULT))
        self.last_pedal = pedal
        self.last_steer = s
        self.debug = {**self.lat.debug, **self.lon.debug, "v_target": sp.v_target * KPH, "a_target": sp.a_target,
                      "limit": sp.limiting, "yaw": w.yaw_rate, "strength": strength,
                      "lane_sigma": self.lanes.sigma_offset()}
        self.tel.add_act(now, plan, s, pedal, self.debug)
        return s, pedal

    def status(self) -> dict:
        d = self.debug
        return {"drive_e": d.get("e", 0.0), "drive_soft": d.get("soft", 1.0), "drive_weave": d.get("weave", False),
                "drive_limit": d.get("limit", ""), "drive_v_target": d.get("v_target", 0.0),
                "drive_strength": d.get("strength", 1.0), "pedal_map": self.lon.map.summary(),
                "cam_lag": self.camlag.tau_learned}
