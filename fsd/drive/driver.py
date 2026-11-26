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
from .steer_check import SteerCheck
from .telemetry import Telemetry
from .turning import CamYaw, TurningModel
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
        self.turning = TurningModel()          # this car's real steering / turning radius
        self.camyaw = CamYaw()
        self.vo_yaw: float | None = None
        self.vo_trust = 0.0
        self._p = None
        self._last_sense = None
        self.steer_check = SteerCheck()
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
    def sense(self, w: World, p, now: float, steer_value: float, lanes_key=None, observer=None) -> World:
        self._p = p
        dt_s = 1 / 60 if self._last_sense is None else float(min(0.2, max(1e-3, now - self._last_sense)))
        self._last_sense = now
        self.steer_value = steer_value
        # speed: predict with the pedal we sent, correct with the speedometer
        sp = self.lon.speed
        a_exp = self.lon.map.accel_for(self.last_pedal, sp.v)
        sp.predict(now, a_exp)
        self.raw_speed_kph = w.speed_kph
        v = sp.update(now, w.speed_kph / KPH if w.speed_valid else None, w.speed_valid)
        vo = getattr(w, "vo", None)
        if vo is not None and getattr(vo, "ok", False) and p.get("visual_speed", 1.0) > 0.5:
            sp.learn_visual_scale(now, v, a_exp)
            v = sp.update_visual(now, vo.v_raw, vo.conf, p)
        if sp.ok and w.speed_valid:
            w.speed_kph = v * KPH
        elif sp.ok and sp.vis_used and sp.vis_scale_n >= 15:
            # speedometer unreadable right now, but the road says how fast we go
            w.speed_kph = v * KPH
            w.speed_valid = True
        self.lon.map.observe(now, v, self.last_pedal, valid=w.speed_valid)
        # yaw
        self.raw_yaw = w.yaw_rate
        v = w.speed_kph / KPH
        # the road under the car turning (visual odometry) is a much better "picture yaw" than the
        # horizon band; undo the chase camera's lag on it
        vo = getattr(w, "vo", None)
        self.vo_yaw = None
        if vo is not None and getattr(vo, "ok", False):
            self.vo_yaw = self.camyaw.update(now, vo.yaw, vo.conf, self.camlag.tau(p))
        # trust the road-motion yaw only once it has proven itself against the lane lines
        if self.vo_yaw is not None and self.yaw.lane_yaw is not None:
            agree = abs(self.vo_yaw - self.yaw.lane_yaw) < 0.04 + 0.2 * abs(self.yaw.lane_yaw)
            self.vo_trust = 0.98 * self.vo_trust + 0.02 * (1.0 if agree else 0.0)
        use_vo = self.vo_yaw is not None and self.vo_trust > 0.7
        img_yaw = self.vo_yaw if use_vo else self.raw_yaw
        gain = self.turning.strength(v)
        w.yaw_rate = self.yaw.update(now, v, steer_value, p, img_yaw,
                                     self.lanes.lanes(now) if self.lanes.ok else w.lanes, gain=gain)
        # learn this car's real turning (independent measurements only: lanes, then the road)
        meas = self.yaw.lane_yaw if self.yaw.lane_yaw is not None else (self.vo_yaw if use_vo else None)
        if meas is None and self.lanes.ok and self.lanes.conf > 0.35:
            # the lane filter has been correcting our yaw for a while: model yaw minus its learned
            # bias is the car's real turn rate (works with weak lanes, unlike the derivative above)
            meas = w.yaw_rate - self.lanes.yaw_bias
        self.turning.observe(now, v, steer_value, meas, p)
        # keep the steering-value estimate honest with an independent turn-rate measurement
        indep = self.yaw.lane_yaw if self.yaw.lane_yaw is not None else (self.vo_yaw if use_vo else None)
        if p.get("steer_check", 1.0) > 0.5:
            self.steer_check.update(observer, v, indep, gain, p, dt_s)
            if observer is not None:
                self.steer_value = observer.s
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
        strength = self.turning.strength(v)
        if p.get("steer_strength_learn", 0.0) > 0.5:
            strength *= self.yaw.strength.gain
        s = self.lat.step(plan.path, v, self.steer_value, strength, p, dt, now, turning, w.lanes.conf,
                          kmax=self.turning.max_curvature(v, p))
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

    # ------------------------------------------------------------------ what it learned
    def learned(self) -> dict:
        """What's worth keeping between sessions (per vehicle): camera lag, pedal response."""
        return {"cam_lag": self.camlag.tau_learned, "cam_lag_n": self.camlag.samples,
                "pedal": [(b.full, b.coast, b.brake, b.n_full, b.n_coast, b.n_brake) for b in self.lon.map.bands],
                "lane_width": self.lanes.width, "turning": self.turning.to_dict(),
                "steer_return_scale": self.steer_check.ret_scale}

    def restore(self, d: dict) -> None:
        if not d:
            return
        try:
            if d.get("cam_lag") is not None:
                self.camlag.tau_learned = float(d["cam_lag"])
                self.camlag.samples = int(d.get("cam_lag_n", 0))
            for b, row in zip(self.lon.map.bands, d.get("pedal", [])):
                b.full, b.coast, b.brake = float(row[0]), float(row[1]), float(row[2])
                b.n_full, b.n_coast, b.n_brake = (min(int(x), 6) for x in row[3:6])
            if d.get("steer_return_scale"):
                self.steer_check.ret_scale = float(d["steer_return_scale"])
            if d.get("turning"):
                self.turning.from_dict(d["turning"])
            if d.get("lane_width"):
                self.lanes.width = float(d["lane_width"])
        except (TypeError, ValueError, IndexError):
            pass

    def status(self) -> dict:
        d = self.debug
        return {"drive_e": d.get("e", 0.0), "drive_soft": d.get("soft", 1.0), "drive_weave": d.get("weave", False),
                "drive_limit": d.get("limit", ""), "drive_v_target": d.get("v_target", 0.0),
                "drive_strength": d.get("strength", 1.0), "pedal_map": self.lon.map.summary(),
                "cam_lag": self.camlag.tau_learned,
                "min_radius_town": self.turning.min_radius(5.0, self._p) if self._p else None,
                "visual_speed": self.lon.speed.vis_used}
