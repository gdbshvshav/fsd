"""Q / E turn requests: preview, lane changes and intersection turns.

* Press Q (left) or E (right): the request is queued and a preview path is shown
  at the next junction where that turn fits.
* Same key again cancels; the other key switches direction.
* Junction ahead (stop line, crosswalk, signal or stop sign) -> we turn there,
  after the light/stop rules say go.
* No junction close but a dashed line on that side -> lane change instead
  (how you move into a turn lane or take a highway exit).
* Solid line and no junction -> "waiting for a turn" and we stay in lane.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

_SEQ = itertools.count(1)

import numpy as np


@dataclass
class TurnState:
    pending: str | None = None        # "left" | "right"
    since: float = 0.0
    junction_d: float | None = None   # metres to the junction entry (stop line etc.)
    active: bool = False              # turning now
    yaw_done: float = 0.0             # rad turned so far
    started: float = 0.0
    lane_change: bool = False
    lc_progress: float = 0.0          # 0..1
    lc_dir: int = 0
    message: str = ""
    preview: list = field(default_factory=list)
    seq: int = field(default_factory=lambda: next(_SEQ))   # identifies this request
    radius: float = 0.0               # radius this car will actually drive for the turn (m)
    turn_speed: float = 0.0           # m/s to be at when the turn starts (lock + grip)
    limited_by: str = ""


def arc_path(d_entry: float, radius: float, direction: str, n: int = 30, straight_after: float = 15.0,
             angle: float = math.pi / 2) -> list[tuple[float, float]]:
    """Straight to d_entry, then a quarter-circle, then a short straight. Vehicle frame (x fwd, y right)."""
    sgn = 1.0 if direction == "right" else -1.0
    pts = [(float(x), 0.0) for x in np.linspace(0, max(0.0, d_entry), 8)]
    for th in np.linspace(0, angle, n)[1:]:
        x = d_entry + radius * math.sin(th)
        y = sgn * radius * (1 - math.cos(th))
        pts.append((float(x), float(y)))
    ex, ey = pts[-1]
    hx, hy = math.cos(angle), sgn * math.sin(angle)
    for s in np.linspace(2, straight_after, 6):
        pts.append((float(ex + hx * s), float(ey + hy * s)))
    return pts


class TurnManager:
    def __init__(self):
        self.s = TurnState()

    # ---- driver input ----
    def press(self, direction: str, now: float) -> str:
        st = self.s
        if st.active:
            return "Already turning"
        if st.pending == direction:
            self.cancel()
            return f"{direction.title()} turn cancelled"
        st.pending = direction
        st.since = now
        st.junction_d = None
        self._cue_d, self._cue_hits, self._cue_first = None, 0, 0.0
        st.message = f"{direction.title()} turn queued"
        return st.message

    def force_lane_change(self, direction: str, now: float, free: bool, why: str = "back to our side") -> str:
        """Move one lane over now (whatever the line type) - used to get back to the right
        side of the road. Waits (returns a message) while the target lane isn't free."""
        st = self.s
        if st.active or st.lane_change:
            return ""
        if not free:
            st.message = f"Waiting for a gap to the {direction}"
            return st.message
        st.pending = direction
        st.since = now
        st.seq = next(_SEQ)
        st.lane_change = True
        st.lc_progress = 0.0
        st.lc_dir = -1 if direction == "left" else 1
        st.message = f"Moving {direction} - {why}"
        return st.message

    def finish_lane_change(self) -> None:
        if self.s.lane_change:
            resume = self._resume
            self.cancel()
            self.s.message = "Lane change complete"
            if resume is not None:
                # we changed lane to get ready for a turn: the turn is still wanted
                self.s.pending, self.s.since, self.s.junction_d = resume
                self.s.message = f"In lane for the {resume[0]} turn"
            self._resume = None

    _resume = None        # (pending, since, junction_d) of a turn waiting for a lane change
    hold_left = False     # set by the planner: don't start a left turn yet (oncoming traffic)

    def lane_change_for_turn(self, direction: str, now: float, free: bool) -> str:
        """Change lane toward the turn while keeping the turn queued."""
        st = self.s
        if st.active or st.lane_change or not free or st.pending is None:
            return ""
        self._resume = (st.pending, st.since, st.junction_d)
        st.seq = next(_SEQ)
        st.lane_change = True
        st.lc_progress = 0.0
        st.lc_dir = -1 if direction == "left" else 1
        st.message = f"Moving {direction} - getting in lane for the turn"
        return st.message

    _cue_d = None         # junction cue tracking (see update)
    _cue_hits = 0
    _cue_first = 0.0

    def cancel(self) -> None:
        self.s = TurnState()
        self._cue_d, self._cue_hits, self._cue_first = None, 0, 0.0

    # ---- per-frame ----
    model = None          # drive.turning.TurningModel (set by the app/simulator): this car's real lock

    def plan_turn(self, p, v: float):
        """(radius, speed m/s, how much earlier to start, what limits it) for the pending turn."""
        st = self.s
        wanted = p["turn_right_radius_m"] if st.pending == "right" else p["turn_left_radius_m"]
        if self.model is None:
            return wanted, p["turn_speed_kph"] / 3.6, 0.0, "junction"
        tp = self.model.turn_plan(wanted, v, p)
        return tp.radius, tp.speed, tp.entry_offset, tp.limited_by

    def update(self, w, p, dt: float, now: float, adjacent_free: dict[str, bool]) -> None:
        st = self.s
        if st.pending is None:
            return
        if now - st.since > p["turn_timeout_s"] and not st.active:
            self.cancel()
            self.s.message = "Turn request expired"
            return
        v = max(0.0, w.speed_kph / 3.6)

        if st.active:
            # integrate yaw: prefer camera motion, fall back to the geometry of the arc
            radius = st.radius or (p["turn_right_radius_m"] if st.pending == "right" else p["turn_left_radius_m"])
            yaw = abs(w.yaw_rate) if abs(w.yaw_rate) > 0.02 else v / radius
            st.yaw_done += yaw * dt
            straight_again = w.lanes.conf > 0.45 and abs(w.lanes.heading) < math.radians(12)
            if (st.yaw_done > math.radians(70) and straight_again) or st.yaw_done > math.radians(105) \
                    or now - st.started > 14:
                done_dir = st.pending
                self.cancel()
                self.s.message = f"{done_dir.title()} turn complete"
            return

        if st.lane_change:
            # the planner finishes the change when the car is actually in the new lane
            # (finish_lane_change); this is only a safety net if that never happens
            st.lc_progress += dt / p["lane_change_s"]
            if st.lc_progress >= 3.0:
                self.cancel()
                self.s.message = "Lane change ended"
            return

        # nearest junction cue
        cues = [d for d in (w.stop_line_dist, w.crosswalk_dist, w.stop_sign_dist) if d < 900]
        if w.signal.state.value not in ("none", "unknown") and w.signal.distance < 900:
            cues.append(max(2.0, w.signal.distance - p["light_to_stop_m"]))
        # A junction must be CONFIRMED: seen in several frames, at distances that agree with us driving
        # toward it. One frame of paint (a false crosswalk, a shadow) used to start a turn "in 0 m"
        # and swerve the car toward a barrier.
        if cues:
            d_now = min(cues)
            pred = None if self._cue_d is None else self._cue_d - v * dt
            if pred is not None and abs(d_now - pred) < 4.0 + 0.3 * v * dt + 0.15 * d_now:
                self._cue_hits += 1
            else:
                self._cue_hits = 1
                self._cue_first = d_now
            self._cue_d = d_now
            if self._cue_hits >= 6 and (self._cue_first > 6.0 or v < 4.0):
                st.junction_d = d_now
        else:
            if self._cue_d is not None:
                self._cue_d -= v * dt
            if st.junction_d is not None:
                st.junction_d -= v * dt                  # dead-reckon once paint is under the car

        if st.junction_d is not None and st.junction_d < p["turn_preview_m"]:
            radius, v_turn, early, why = self.plan_turn(p, v)
            st.radius, st.turn_speed, st.limited_by = radius, v_turn, why
            # a car that can't turn as tight as the junction wants starts its turn earlier
            entry = st.junction_d + p["turn_entry_m"] + (p["lane_width_m"] if st.pending == "left" else 0.0) - early
            st.preview = arc_path(max(0.0, entry), radius, st.pending)
            extra = f" (wide car: {radius:.0f} m radius, {v_turn * 3.6:.0f} km/h)" if why == "lock" else ""
            st.message = f"Turning {st.pending} at the junction in {max(0, st.junction_d):.0f} m{extra}"
            if entry <= 0.5 and st.pending == "left" and self.hold_left:
                st.message = "Waiting for oncoming traffic before turning left"
                return
            # a solid line on the turn side that carries on past the corner = no road to turn into
            side = w.lanes.right if st.pending == "right" else w.lanes.left
            if entry <= 0.5 and side is not None and not side.dashed and side.conf > 0.5 \
                    and side.x_max > max(0.0, st.junction_d) + 12.0 and w.lanes.conf > 0.5:
                st.message = f"No opening to the {st.pending} yet - staying in lane"
                st.preview = []
                return
            if entry <= 0.5:
                st.active = True
                st.started = now
                st.yaw_done = 0.0
                st.message = f"Turning {st.pending}"
            return

        # No junction close: lane change if the line on that side is dashed and clear
        line = w.lanes.left if st.pending == "left" else w.lanes.right
        other = w.lanes.left2 if st.pending == "left" else w.lanes.right2
        if line is not None and line.dashed and other is not None and w.lanes.conf > 0.4:
            if adjacent_free.get(st.pending, False):
                st.seq = next(_SEQ)
                st.lane_change = True
                st.lc_progress = 0.0
                st.lc_dir = -1 if st.pending == "left" else 1
                st.message = f"Changing lanes {st.pending}"
            else:
                st.message = f"Waiting for a gap to the {st.pending}"
            st.preview = []
            return
        st.preview = []
        st.message = f"{st.pending.title()} turn queued - no turn ahead yet, staying in lane"

    def path_override(self, p, w) -> list[tuple[float, float]] | None:
        """While turning: the arc from where we are now."""
        st = self.s
        if not st.active:
            return None
        radius = st.radius or (p["turn_right_radius_m"] if st.pending == "right" else p["turn_left_radius_m"])
        remaining = max(0.05, math.pi / 2 - st.yaw_done)
        return arc_path(0.0, radius, st.pending, angle=remaining)

    def lateral_shift(self, lane_width: float) -> float:
        """Lane change: smooth lateral offset to add to the lane centre (m, + = right)."""
        st = self.s
        if not st.lane_change:
            return 0.0
        k = st.lc_progress
        s = k * k * (3 - 2 * k)                 # smoothstep: gentle start and finish
        return st.lc_dir * lane_width * s
