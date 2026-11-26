"""Going around a stopped or very slow car - like a person would, not waiting forever.

When FSD stops behind something it waits. That's right at a red light or in a queue,
but behind a parked car, a crashed car or a stopped police car (common in ER:LC
roleplay) it would wait for ever. Rules for passing:

1. The thing ahead has been stopped (or crawling below 8 km/h) for a while: 4 s,
   or 8 s if we're near a junction (a queue at a light looks exactly like this).
2. Never at / just before a red light, stop sign or stop line, never while turning.
3. Only to the LEFT, across a DASHED WHITE line (a lane going our way). Crossing a
   yellow centre line into oncoming traffic is not done automatically - FSD asks you
   (message + blinker suggestion) instead.
4. The left lane must be free from 25 m behind to 60 m ahead (no car seen there).
5. Afterwards the normal keep-right rule brings us back.

The decision is returned to the planner, which runs the lane change with the same
smooth shaping as any other (signal, quintic move, re-base when we cross the line).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PassDecision:
    go: str | None = None          # "left" when we should start passing now
    message: str = ""
    waiting_for: str = ""          # why we're not passing (for the dashboard)


class PassingPlanner:
    def __init__(self, wait_s: float = 4.0, wait_junction_s: float = 8.0):
        self.wait_s = wait_s
        self.wait_junction_s = wait_junction_s
        self.stuck_since: float | None = None
        self.last_pass = -1e9
        self.asked = -1e9

    def reset(self) -> None:
        self.__init__(self.wait_s, self.wait_junction_s)

    def update(self, w, lead, now: float, v: float, p, lane_change_active: bool, turning: bool,
               stop_reason: str = "") -> PassDecision:
        d = PassDecision()
        if p.get("auto_pass", 1.0) <= 0.5:
            return d
        lead_slow = lead is not None and lead.x < 30.0 and (v + lead.vx) < 8 / 3.6 and v < 3.0
        if not lead_slow or lane_change_active or turning:
            self.stuck_since = None
            return d
        self.stuck_since = self.stuck_since if self.stuck_since is not None else now
        near_junction = min(w.stop_line_dist, w.crosswalk_dist, w.stop_sign_dist, w.signal.distance) < 45.0
        if w.signal.state.value in ("red", "yellow", "flashing_red") or "light" in stop_reason \
                or "stop" in stop_reason.lower() and "sign" in stop_reason.lower():
            d.waiting_for = "traffic control ahead"
            return d
        wait = self.wait_junction_s if near_junction else self.wait_s
        if now - self.stuck_since < wait:
            d.waiting_for = f"waiting {wait - (now - self.stuck_since):.0f} s before passing"
            return d
        if now - self.last_pass < 15.0:
            return d
        L = w.lanes.left
        if L is None or w.lanes.conf < 0.4:
            d.waiting_for = "can't see the lane lines to pass"
            return d
        if L.color == "yellow" or not L.dashed:
            if now - self.asked > 20.0:
                self.asked = now
                d.message = "Blocked by a stopped car - passing needs the other side of the road: press Q to go around"
            d.waiting_for = "solid / yellow line on the left"
            return d
        lane_w = w.lanes.width_m or p["lane_width_m"]
        for o in w.objects:
            if o.kind not in ("car", "emergency", "obstacle", "person"):
                continue
            rel = o.y - w.lanes.offset
            if -1.6 * lane_w < rel < -0.4 * lane_w and -25.0 < o.x < 60.0:
                d.waiting_for = "left lane not clear"
                return d
        d.go = "left"
        d.message = "Passing the stopped car on the left"
        self.last_pass = now
        self.stuck_since = None
        return d
