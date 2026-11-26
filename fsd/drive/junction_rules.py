"""Junction manners: the right lane for the turn, and giving way when turning left.

Two things a normal ER:LC driver does that FSD didn't:

1. GET IN LANE. With a turn queued (Q/E) on a road with more than one lane our way,
   move to the lane on that side well before the junction (left turn: leftmost lane
   going our way; right turn: rightmost lane). Only over dashed white lines, never into
   the oncoming side, never in the last 30 m (too late: then the turn is made from
   where we are).

2. LEFT TURNS GIVE WAY. Turning left crosses the oncoming lanes. Before turning in we
   look for cars coming toward us: on our left (the other side of the centre line),
   moving toward us faster than we move, and close enough to arrive within ~4 s.
   If there is one, we hold at the turn-in point ("waiting for oncoming traffic") and
   go once the gap is there. A right turn only checks for people crossing (handled by
   the planner's crossing check).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JunctionAdvice:
    change_lane: str | None = None       # "left" / "right": move toward the turn side now
    hold: bool = False                   # don't start the left turn yet
    message: str = ""


class JunctionRules:
    def __init__(self, gap_s: float = 4.0):
        self.gap_s = gap_s
        self.last_change = -1e9
        self.hold_since: float | None = None

    def reset(self) -> None:
        self.__init__(self.gap_s)

    def oncoming_gap(self, w, v: float, lane_w: float) -> float:
        """Seconds until the nearest oncoming car reaches us (inf if none)."""
        best = float("inf")
        for o in w.objects:
            if o.kind not in ("car", "emergency"):
                continue
            rel = o.y - w.lanes.offset
            if not (-3.2 * lane_w < rel < -0.45 * lane_w):     # on the other side, to our left
                continue
            closing = -(o.vx)            # our frame: an oncoming car's x shrinks faster than v
            ground = v + o.vx            # its own speed along our direction (negative = toward us)
            if ground > -1.0 or o.x < -5 or o.x > 120:
                continue
            if closing <= 0.5:
                continue
            best = min(best, max(0.0, o.x) / closing)
        return best

    def update(self, w, ts, v: float, now: float, lane_w: float, p) -> JunctionAdvice:
        adv = JunctionAdvice()
        if ts.pending not in ("left", "right") or ts.lane_change:
            self.hold_since = None
            return adv
        # 1) get in lane for the turn
        if not ts.active and ts.junction_d is not None and 30.0 < ts.junction_d < 160.0 \
                and now - self.last_change > 6.0 and w.lanes.conf > 0.5:
            side = w.lanes.left if ts.pending == "left" else w.lanes.right
            if side is not None and side.dashed and side.color == "white":
                adv.change_lane = ts.pending
                adv.message = f"Moving to the {ts.pending} lane for the turn"
                self.last_change = now
        # 2) left turns give way to oncoming traffic
        if ts.pending == "left" and not ts.active and ts.junction_d is not None and ts.junction_d < 12.0:
            gap = self.oncoming_gap(w, v, lane_w)
            if gap < self.gap_s:
                self.hold_since = self.hold_since if self.hold_since is not None else now
                adv.hold = True
                adv.message = f"Waiting for oncoming traffic ({gap:.1f} s away)"
            else:
                self.hold_since = None
        return adv
