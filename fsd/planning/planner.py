"""The driving brain: decides mode, path and target speed every frame.

Rules (from the ER:LC reference doc):
* Green: go if clear.  Yellow: stop if we comfortably can, else clear the junction.
* Red: stop at the line.  Flashing red (outage): full stop, wait, go when clear.
* Stop sign: full stop at the stop bar (not at the sign), pause, go when clear.
* Follow the car ahead at a time gap; stop for people and cones in our path.
* Emergency lights: slow, keep right, stop if they're coming at us.
* Unsure what we see: slow down. Lost the road: stop and ask for takeover.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..world import Light, Mode, Obj, Plan, World
from .junction_memory import JunctionMemory
from .autonomy import Autonomy
from .lead_memory import LeadMemory
from .turns import TurnManager
from ..drive.path_planner import OffsetShaper, edges_path, heading_hold_path
from ..drive.passing import PassingPlanner
from ..drive.side_keeper import SideKeeper

KPH = 3.6


class Planner:
    def __init__(self):
        self.turns = TurnManager()
        self.mode = Mode.OFF
        self.stop_wait_until = 0.0
        self.stop_wait_started = 0.0
        self.cleared_sign_until = 0.0     # ignore the stop sign we just obeyed
        self.cleared_flash_until = 0.0
        self.no_lane_since: float | None = None
        self.last_t = time.perf_counter()
        self.prev_offset: float | None = None
        self.lc_corr = 0.0
        self.shaper = OffsetShaper()
        self._lc_was = False
        self._lc_switched = False
        self.last_limit_kph: float | None = None
        self.red_committed = False        # we decided to run a yellow
        self.last_stop_reason = ""
        self.junction = JunctionMemory()
        self.sight_hist: list[float] = []
        self.lead_mem = LeadMemory()
        self.auto = Autonomy()
        self.side = SideKeeper()
        self.passing = PassingPlanner()
        self.side_decision = None
        self.frame = None            # latest work picture (set by the app / sim) for road-end checks

    def reset(self) -> None:
        self.__init__()

    # ------------------------------------------------------------------
    def corridor_objects(self, w: World, path: list[tuple[float, float]], half: float) -> list[Obj]:
        if not path:
            return []
        px = np.array([p[0] for p in path])
        py = np.array([p[1] for p in path])
        out = []
        for o in w.objects:
            if o.kind not in ("car", "person", "cone", "emergency", "obstacle") or not (0 < o.x < 120):
                continue
            yp = float(np.interp(o.x, px, py))
            if abs(o.y - yp) < half + o.extra.get("half_width", 0.9):
                out.append(o)
        return sorted(out, key=lambda o: o.x)

    def adjacent_free(self, w: World, lane_w: float) -> dict[str, bool]:
        free = {"left": True, "right": True}
        for o in w.objects:
            if o.kind not in ("car", "emergency"):
                continue
            rel = o.y - w.lanes.offset
            if -10 < o.x < 30:
                if -1.6 * lane_w < rel < -0.5 * lane_w:
                    free["left"] = False
                if 0.5 * lane_w < rel < 1.6 * lane_w:
                    free["right"] = False
        return free

    def cross_traffic_clear(self, w: World) -> bool:
        for o in w.objects:
            if o.kind in ("car", "emergency") and 0 < o.x < 28 and abs(o.y) < 18:
                if abs(o.y) > 2.0 and (o.vx < -1.0 or o.track_id < 0):
                    return False
                if abs(o.y) <= 2.0 and o.x < 10:
                    return False
            if o.kind == "person" and 0 < o.x < 15 and abs(o.y) < 6:
                return False
        return True

    # ------------------------------------------------------------------
    def build_path(self, w: World, p, extra_shift: float) -> list[tuple[float, float]]:
        ln = w.lanes
        sd = self.side_decision
        edges = w.road_edges
        if ln.conf < p["lane_min_conf"] and sd is not None and sd.no_lines_offset is not None \
                and edges is not None and getattr(edges, "ok", False) and not self.turns.s.lane_change:
            return edges_path(edges, sd.no_lines_offset + (float(extra_shift(np.array([0.0]))[0])
                                                          if callable(extra_shift) else extra_shift))
        if ln.conf <= 0:
            return heading_hold_path(w.yaw_rate, w.speed_kph / KPH)
        a, b, c = ln.center
        # the chase camera swings after the car: while turning it points behind the car's
        # heading, so everything looks rotated toward the inside of the bend. Undo that.
        if p.get("drive_core", 1.0) <= 0.5:          # (the new core corrects the lanes itself)
            b = b - w.yaw_rate * p.get("cam_yaw_lag_s", 0.0)
        xs = np.linspace(0, max(15.0, min(60.0, ln.x_max)), 24)
        off = extra_shift(xs) if callable(extra_shift) else extra_shift
        ys = a * xs * xs + b * xs + c + p["lane_center_bias_m"] + off
        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    def stop_point(self, w: World, p, for_light: bool) -> float:
        """Metres to where the front bumper should stop."""
        cands = []
        jm = self.junction.bar_distance()
        if jm < 900:                       # tracked bar: measured, or dead-reckoned once out of view
            cands.append(jm - p["stop_line_offset_m"])
        elif w.stop_line_dist < 900:
            cands.append(w.stop_line_dist - p["stop_line_offset_m"])
        if w.crosswalk_dist < 900 and (not cands or w.crosswalk_dist < cands[0] + 2.0):
            cands.append(w.crosswalk_dist - p["stop_line_offset_m"] - 0.5)
        if cands:
            return max(0.0, min(cands))
        if for_light and w.signal.distance < 900:
            return max(0.0, w.signal.distance - p["light_to_stop_m"])
        if w.stop_sign_dist < 900:
            return max(0.0, w.stop_sign_dist - p["stop_sign_fallback_m"])
        return 999.0

    # ------------------------------------------------------------------
    def step(self, w: World, p, engaged: bool, now: float | None = None) -> Plan:
        now = time.perf_counter() if now is None else now
        dt = float(np.clip(now - self.last_t, 0.0, 0.2))
        self.last_t = now
        plan = Plan()
        if not engaged:
            self.mode = Mode.OFF
            plan.mode = Mode.OFF
            plan.turn_pending = self.turns.s.pending
            plan.turn_preview = self.turns.s.preview
            plan.alert = self.turns.s.message
            return plan

        v = w.speed_kph / KPH
        lane_w = w.lanes.width_m or p["lane_width_m"]
        jm = self.junction
        jm.update(now, v, w.stop_line_dist, w.crosswalk_dist, w.stop_sign_dist,
                  w.signal.state.value, w.signal.distance)

        # ---------- speed limit ----------
        if w.speed_limit_kph:
            self.last_limit_kph = w.speed_limit_kph
        limit = self.last_limit_kph or p["default_speed_limit_kph"]
        cruise = min(limit + p["speed_over_limit_kph"], p["max_speed_kph"])
        if w.tunnel:
            cruise = min(cruise, max(40.0, limit))

        # ---------- turns / lane changes ----------
        self.auto.step(w, self.turns, p, now, self.frame)
        self.turns.update(w, p, dt, now, self.adjacent_free(w, lane_w))
        ts = self.turns.s
        # right-hand traffic: wrong side / keep right / roads without lines
        sd = self.side.update(w.lanes, w.road_edges, w.objects, p, now, v, ts.lane_change or ts.active)
        self.side_decision = sd
        if sd.move_right and ts.pending is None:
            why = "back to our side" if sd.wrong_side else (sd.reason or "keeping right")
            msg = self.turns.force_lane_change("right", now, self.adjacent_free(w, lane_w).get("right", False), why)
            if msg:
                plan.alert = msg
        ts = self.turns.s

        # lateral offset from the lane centre: lane changes and keep-right moves are shaped
        # along the road (quintic over a speed-dependent distance), and when the lane under us
        # re-assigns (we crossed the line) the whole offset moves with it - no double jump.
        switch = getattr(w, "lane_switch", 0)
        if switch == 0 and self.prev_offset is not None and ts.lane_change:
            jump = w.lanes.offset - self.prev_offset           # old core: detect it from the jump
            if abs(jump - ts.lc_dir * lane_w) < 1.2:
                switch = ts.lc_dir
        self.prev_offset = w.lanes.offset
        if switch:
            self.shaper.shift(-switch * lane_w)
            self._lc_switched = True
            if ts.lane_change and w.lanes.conf > 0.3:
                # glide from where the car actually is in the new lane
                self.shaper.restart_from(float(np.clip(-w.lanes.offset, -lane_w, lane_w)))
        if ts.lane_change and not self._lc_was:
            self._lc_switched = False
            self.shaper.set_target(self.shaper.target + ts.lc_dir * lane_w, v,
                                   sec=p["lane_change_s"])
        self._lc_was = ts.lane_change
        want = 0.0 if not ts.lane_change else self.shaper.target
        if w.emergency_near and not ts.active:
            want += 0.9                                       # keep right for emergency vehicles
        if not ts.lane_change:
            self.shaper.set_target(want, v, min_len=10.0, sec=1.5)
        self.shaper.advance(v * dt)
        if ts.lane_change and self._lc_switched and abs(self.shaper.now - self.shaper.target) < 0.15 \
                and self.shaper.done_m >= self.shaper.length - 0.5:
            self.turns.finish_lane_change()
            ts = self.turns.s
            self.shaper.snap_to(self.shaper.now)
        shift = self.shaper.profile if not ts.active else 0.0
        path = self.turns.path_override(p, w) or self.build_path(w, p, shift)
        plan.path = path
        plan.turn_pending = ts.pending
        plan.turn_preview = ts.preview
        plan.lane_change = ts.lane_change
        plan.alert = ts.message or plan.alert
        if sd.wrong_side:
            plan.alert = "Wrong side of the road - moving right"

        target = cruise
        mode = Mode.CRUISE
        reason = f"limit {limit:.0f}"
        stop_at: float | None = None

        # ---------- curves: slow down so sideways accel stays comfortable ----------
        curv = abs(w.lanes.curvature)
        if ts.active:
            target = min(target, p["turn_speed_kph"], (ts.turn_speed * KPH) if ts.turn_speed else 999.0)
            mode, reason = Mode.TURNING, f"turning {ts.pending}"
        elif ts.pending is not None and ts.junction_d is not None and ts.turn_speed:
            # approaching a turn: be down to the speed this car can turn at by the turn-in point
            v_ok = math.sqrt(ts.turn_speed ** 2 + 2 * p["comfort_decel"] * max(0.0, ts.junction_d)) * KPH
            if v_ok < target:
                target, reason = v_ok, f"slowing for the {ts.pending} turn"
        elif curv > 1e-3:
            v_curve = math.sqrt(p["curve_lat_accel"] / curv) * KPH
            if v_curve < target:
                target, reason = v_curve, "curve"
        # ---------- sight distance: lane lines suddenly end much nearer than usual = a bend or
        #            crest is coming that we can't see round. Slow so we could stop in what we see.
        if w.lanes.conf > 0 and not ts.active:
            self.sight_hist.append(w.lanes.x_max)
            self.sight_hist = self.sight_hist[-90:]
            usual = float(np.median(self.sight_hist)) if len(self.sight_hist) >= 15 else 0.0
            sight = float(np.median(self.sight_hist[-5:]))
            if usual > 20 and sight < 0.6 * usual:
                v_sight = math.sqrt(2 * p["comfort_decel"] * max(4.0, sight - 2.0) + 36.0) * KPH
                if v_sight < target:
                    target, reason = v_sight, f"bend ahead (see {sight:.0f} m)"
        if ts.lane_change:
            mode, reason = Mode.LANE_CHANGE, f"lane change {ts.pending}"

        # ---------- lane confidence ----------
        if w.lanes.conf < p["lane_min_conf"] and not ts.active and not ts.lane_change:
            if self.no_lane_since is None:
                self.no_lane_since = now
            edges_ok = w.road_edges is not None and getattr(w.road_edges, "ok", False) and sd.no_lines_offset is not None
            if edges_ok:
                target = min(target, 40.0)
                mode, reason = Mode.CAUTION, sd.reason or "no lane lines - following the road edges"
            elif w.lanes.conf <= 0.01 and now - self.no_lane_since > 2.0:
                mode, reason = Mode.FAULT, "can't see the road - take over"
                plan.alert = "TAKE OVER: road not visible"
                stop_at = 0.0
            else:
                target = min(target, 25.0)
                mode, reason = Mode.CAUTION, "lane unclear"
        else:
            self.no_lane_since = None

        # ---------- traffic lights ----------
        sig = w.signal.state
        if sig in (Light.RED, Light.YELLOW) and not ts.active:
            d = self.stop_point(w, p, for_light=True)
            if d < 900:
                need = v * v / (2 * max(d, 0.1))
                can_stop = need <= p["max_decel"] * 0.75
                t_line = d / max(v, 0.1)
                if sig == Light.YELLOW and (self.red_committed or not can_stop or t_line < p["yellow_commit_s"]):
                    self.red_committed = True
                    reason = "yellow - clearing junction"
                elif sig == Light.RED and self.red_committed and d < 3:
                    reason = "light turned red behind the line - clearing"
                else:
                    stop_at = d if stop_at is None else min(stop_at, d)
                    reason = f"{sig.value} light"
        else:
            self.red_committed = False if sig in (Light.GREEN, Light.NONE) else self.red_committed

        # ---------- stop signs and outages (full stop, then go when clear) ----------
        needs_full_stop = ((w.stop_sign_dist < 900 or jm.full_stop_needed) and now > self.cleared_sign_until) or \
                          (sig == Light.FLASHING_RED and now > self.cleared_flash_until)
        if needs_full_stop and not ts.active:
            d = self.stop_point(w, p, for_light=(sig == Light.FLASHING_RED))
            if d >= 900:
                d = jm.stop_point(p["stop_line_offset_m"], p["stop_sign_fallback_m"])
            if d < 900:
                if self.mode == Mode.STOP_WAIT or (d < 1.0 and v < 0.6):
                    if self.mode != Mode.STOP_WAIT:
                        self.stop_wait_started = now
                    mode = Mode.STOP_WAIT
                    stop_at = 0.0
                    waited = now - self.stop_wait_started
                    if waited >= p["stop_sign_wait_s"]:
                        if self.cross_traffic_clear(w):
                            self.cleared_sign_until = now + 8.0
                            self.cleared_flash_until = now + 8.0
                            jm.mark_stopped()
                            mode, stop_at, reason = Mode.CRUISE, None, "clear - going"
                        else:
                            reason = "waiting for cross traffic"
                    else:
                        reason = f"full stop {p['stop_sign_wait_s'] - waited:.1f}s"
                else:
                    stop_at = d if stop_at is None else min(stop_at, d)
                    reason = jm.full_stop_reason or ("stop sign" if w.stop_sign_dist < 900 else "outage: all-way stop")
        elif self.mode == Mode.STOP_WAIT and not needs_full_stop:
            pass

        # ---------- obstacles in our path ----------
        lead = None
        for o in self.corridor_objects(w, path, p["corridor_half_width_m"]):
            if o.kind in ("person", "cone"):
                d = o.x - p["pedestrian_buffer_m"] if o.kind == "person" else o.x - 3.0
                stop_at = max(0.0, d) if stop_at is None else min(stop_at, max(0.0, d))
                reason = f"{o.kind} ahead"
                break
            lead = o
            break
        lead = self.lead_mem.update(now, v, lead, w.clear_ahead_m, w.extra_hidden_m)
        if lead is not None:
            v_lead = max(0.0, v + lead.vx)
            gap = lead.x
            desired = p["follow_min_gap_m"] + p["follow_time_gap_s"] * v
            v_follow = v_lead + 0.5 * (gap - desired)
            if gap < p["follow_min_gap_m"] + 1:
                stop_at = 0.0 if stop_at is None else min(stop_at, 0.0)
            elif v_lead < 2.0 and v > v_lead + 1.0:
                # stopped (or nearly) car ahead: a real stop point, so the speed planner brakes as
                # hard as it must - up to everything - to stop behind it
                sp = max(0.0, gap - p["follow_min_gap_m"])
                stop_at = sp if stop_at is None else min(stop_at, sp)
            if v_follow * KPH < target:
                target = max(0.0, v_follow * KPH)
                if mode in (Mode.CRUISE, Mode.CAUTION):
                    mode = Mode.FOLLOW
                reason = f"following car {gap:.0f} m"
            plan.lead = lead

        # ---------- stuck behind a stopped car: go around it (dashed lane line only) ----------
        pd = self.passing.update(w, lead, now, v, p, ts.lane_change, ts.active, reason)
        if pd.go and ts.pending is None:
            msg = self.turns.force_lane_change(pd.go, now, True, "passing a stopped car")
            if msg:
                plan.alert = msg
                self.side.passed_at = now
        elif pd.message:
            plan.alert = pd.message
        if pd.waiting_for and lead is not None and v < 1.0:
            reason = f"{reason} · {pd.waiting_for}"

        # ---------- emergency vehicles ----------
        if w.emergency_near:
            target = min(target, 20.0)
            mode, reason = Mode.YIELD, "emergency vehicle - yielding"
            for o in w.objects:
                if o.kind in ("car", "emergency") and o.x < 25 and o.vx < -2:
                    stop_at = 0.0

        # ---------- stopping profile ----------
        if stop_at is not None and mode != Mode.FAULT:
            v_stop = math.sqrt(2 * p["comfort_decel"] * max(0.0, stop_at)) * KPH
            if v_stop < target:
                target = v_stop
                if mode not in (Mode.STOP_WAIT, Mode.YIELD):
                    mode = Mode.STOPPED if (stop_at < 1.0 and v < 0.7) else Mode.STOPPING
            plan.stop_at = stop_at
        if mode == Mode.FAULT:
            target = 0.0
            plan.stop_at = 0.0

        self.mode = mode
        plan.mode = mode
        plan.target_speed_kph = float(max(0.0, target))
        plan.reason = reason
        return plan
