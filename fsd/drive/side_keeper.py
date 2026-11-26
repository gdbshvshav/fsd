"""Right-hand traffic: stay on OUR side of the road.

ER:LC (like the US) drives on the right. Rules this applies, with hysteresis so a
single misread frame never moves the car:

* Yellow centre line on our right = we're in an oncoming lane. Ask for a move to
  the right immediately (signal E), whatever the line type.
* An oncoming car in our own path = same.
* No lane lines but road edges visible: two-way road -> drive in the right half,
  one lane's width in from the right edge; narrow road -> the middle.
* Several lanes our way on a fast road: after passing, move back to the right lane
  when it's free for a while ("keep right except to pass"). Off by default in town.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..perception.road_model import RoadEdges, SideAdvice, side_advice


@dataclass
class SideDecision:
    wrong_side: bool = False
    move_right: bool = False          # ask the turn manager for a move to the right
    no_lines_offset: float | None = None   # where to put the car when there are no lane lines
    reason: str = ""


class SideKeeper:
    def __init__(self):
        self.wrong_since: float | None = None
        self.left_lane_since: float | None = None
        self.right_free_since: float | None = None
        self.last = SideDecision()
        self.advice = SideAdvice()
        self._last_request = -1e9
        self.passed_at = -1e9            # set by the planner when we pulled out to pass

    def reset(self) -> None:
        self.__init__()

    def update(self, lanes, edges: RoadEdges | None, objects, p, now: float, v: float,
               lane_change_active: bool) -> SideDecision:
        d = SideDecision()
        adv = side_advice(lanes, edges or RoadEdges(), objects, lanes.width_m or p["lane_width_m"])
        self.advice = adv
        # wrong side: must persist 0.6 s
        if adv.wrong_side:
            self.wrong_since = self.wrong_since if self.wrong_since is not None else now
            if now - self.wrong_since > 0.6:
                d.wrong_side = True
                d.reason = adv.reason
                if not lane_change_active and now - self._last_request > 4.0:
                    d.move_right = True
                    self._last_request = now
        else:
            self.wrong_since = None
        # no lane lines: drive by the road edges
        if adv.target_offset is not None and lanes.conf < 0.35 and not adv.wrong_side:
            d.no_lines_offset = adv.target_offset
            d.reason = d.reason or adv.reason
        # after going around a stopped car (any speed): back to the right lane once it's behind us
        if 6.0 < now - self.passed_at < 40.0 and lanes.conf > 0.5 and not lane_change_active:
            R = lanes.right
            if R is not None and R.dashed and R.color == "white":
                clear = not any(o.kind in ("car", "emergency", "obstacle") and -12 < o.x < 40
                                and 0.5 * lanes.width_m < o.y - lanes.offset < 1.6 * lanes.width_m for o in objects)
                if clear and now - self._last_request > 5.0:
                    d.move_right = True
                    d.reason = "back to the right lane after passing"
                    self._last_request = now
                    self.passed_at = -1e9
        # keep right except to pass
        if p.get("keep_right", 1.0) > 0.5 and v > 70 / 3.6 and lanes.conf > 0.5 and not lane_change_active:
            R = lanes.right
            in_left_lane = R is not None and R.dashed and R.color == "white" and lanes.right2 is not None
            if in_left_lane:
                self.left_lane_since = self.left_lane_since if self.left_lane_since is not None else now
                right_free = not any(o.kind in ("car", "emergency", "obstacle") and -15 < o.x < 70
                                     and 0.5 * lanes.width_m < o.y - lanes.offset < 1.6 * lanes.width_m
                                     for o in objects)
                if right_free:
                    self.right_free_since = self.right_free_since if self.right_free_since is not None else now
                else:
                    self.right_free_since = None
                if now - self.left_lane_since > 25 and self.right_free_since is not None and \
                        now - self.right_free_since > 6 and now - self._last_request > 20:
                    d.move_right = True
                    d.reason = "keeping right - passing lane not needed"
                    self._last_request = now
            else:
                self.left_lane_since = None
                self.right_free_since = None
        self.last = d
        return d
