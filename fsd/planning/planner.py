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
from .turns import TurnManager

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
        self.last_limit_kph: float | None = None
        self.red_committed = False        # we decided to run a yellow
        self.last_stop_reason = ""

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
            if o.kind not in ("car", "person", "cone", "emergency") or not (0 < o.x < 120):
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
        if ln.conf <= 0:
            return [(float(x), 0.0) for x in np.linspace(0, 30, 16)]
        a, b, c = ln.center
        xs = np.linspace(0, max(15.0, min(60.0, ln.x_max)), 24)
        ys = a * xs * xs + b * xs + c + p["lane_center_bias_m"] + extra_shift
        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    def stop_point(self, w: World, p, for_light: bool) -> float:
        """Metres to where the front bumper should stop."""
        cands = []
        if w.stop_line_dist < 900:
            cands.append(w.stop_line_dist - p["stop_line_offset_m"])
        if w.crosswalk_dist < 900:
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

        # ---------- speed limit ----------
        if w.speed_limit_kph:
            self.last_limit_kph = w.speed_limit_kph
        limit = self.last_limit_kph or p["default_speed_limit_kph"]
        cruise = min(limit + p["speed_over_limit_kph"], p["max_speed_kph"])
        if w.tunnel:
            cruise = min(cruise, max(40.0, limit))

        # ---------- turns / lane changes ----------
        self.turns.update(w, p, dt, now, self.adjacent_free(w, lane_w))
        ts = self.turns.s

        # lane-change bookkeeping: when the ego lane re-assigns mid-change, keep the target fixed
        if ts.lane_change and self.prev_offset is not None:
            jump = w.lanes.offset - self.prev_offset
            if abs(jump - ts.lc_dir * lane_w) < 1.2:
                self.lc_corr -= ts.lc_dir * lane_w
        if not ts.lane_change:
            self.lc_corr = 0.0
        self.prev_offset = w.lanes.offset

        shift = self.turns.lateral_shift(lane_w) + self.lc_corr
        if w.emergency_near and not ts.active:
            shift += 0.9                                      # keep right for emergency vehicles
        path = self.turns.path_override(p, w) or self.build_path(w, p, shift)
        plan.path = path
        plan.turn_pending = ts.pending
        plan.turn_preview = ts.preview
        plan.alert = ts.message

        target = cruise
        mode = Mode.CRUISE
        reason = f"limit {limit:.0f}"
        stop_at: float | None = None

        # ---------- curves: slow down so sideways accel stays comfortable ----------
        curv = abs(w.lanes.curvature)
        if ts.active:
            target = min(target, p["turn_speed_kph"])
            mode, reason = Mode.TURNING, f"turning {ts.pending}"
        elif curv > 1e-3:
            v_curve = math.sqrt(p["curve_lat_accel"] / curv) * KPH
            if v_curve < target:
                target, reason = v_curve, "curve"
        if ts.lane_change:
            mode, reason = Mode.LANE_CHANGE, f"lane change {ts.pending}"

        # ---------- lane confidence ----------
        if w.lanes.conf < p["lane_min_conf"] and not ts.active:
            if self.no_lane_since is None:
                self.no_lane_since = now
            if w.lanes.conf <= 0.01 and now - self.no_lane_since > 2.0:
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
        needs_full_stop = (w.stop_sign_dist < 900 and now > self.cleared_sign_until) or \
                          (sig == Light.FLASHING_RED and now > self.cleared_flash_until)
        if needs_full_stop and not ts.active:
            d = self.stop_point(w, p, for_light=(sig == Light.FLASHING_RED))
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
                            mode, stop_at, reason = Mode.CRUISE, None, "clear - going"
                        else:
                            reason = "waiting for cross traffic"
                    else:
                        reason = f"full stop {p['stop_sign_wait_s'] - waited:.1f}s"
                else:
                    stop_at = d if stop_at is None else min(stop_at, d)
                    reason = "stop sign" if w.stop_sign_dist < 900 else "outage: all-way stop"
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
        if lead is not None:
            v_lead = max(0.0, v + lead.vx)
            gap = lead.x
            desired = p["follow_min_gap_m"] + p["follow_time_gap_s"] * v
            v_follow = v_lead + 0.5 * (gap - desired)
            if gap < p["follow_min_gap_m"] + 1:
                stop_at = 0.0 if stop_at is None else min(stop_at, 0.0)
            if v_follow * KPH < target:
                target = max(0.0, v_follow * KPH)
                if mode in (Mode.CRUISE, Mode.CAUTION):
                    mode = Mode.FOLLOW
                reason = f"following car {gap:.0f} m"
            plan.lead = lead

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
