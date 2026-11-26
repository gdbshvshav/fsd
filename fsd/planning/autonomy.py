"""Driving on its own for hours: turn signals, merges when a lane ends, and picking
a way when the road ends.

Turn signals (ER:LC: Q = left blinker, E = right blinker, each press toggles it)
-------------------------------------------------------------------------------
`Blinker` keeps track of what the game's blinker is showing. When YOU press Q/E
the game toggles it and we follow along (the key hook tells us). When FSD wants
a blinker (a lane change, a merge, a turn it chose) it taps Q/E itself, and taps
again to switch it off when the manoeuvre is done - unless you switched it on, in
which case it's yours to cancel (ER:LC doesn't auto-cancel).

Lane ends / merges
------------------
Our lane is ending when its two lines converge ahead: the lane is clearly narrower
at 30-40 m than at 8-12 m, and one line is the one moving in. Seen on real ER:LC
roads where 2 lanes become 1 and on on-ramps joining the highway. We then ask for
a lane change to the side that continues (the side whose line is NOT closing in),
with the blinker, and wait for a gap like any lane change.

Road ends
---------
At a T-junction or a dead end the lane lines stop, a junction cue (stop line,
stop sign, crosswalk) is ahead, and past it the corridor we'd drive into is not
road. Then we pick a turn: the side where there's road (seen as road-coloured
on that side past the junction), preferring right, and queue it with the blinker
exactly as if you had pressed Q/E.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Blinker:
    def __init__(self):
        self.state: str | None = None       # what the game shows (our belief)
        self.owner: str = ""                # "driver" | "fsd" | ""
        self.taps: list[str] = []           # keys to tap (consumed by the app / sim)

    def driver_pressed(self, side: str) -> None:
        """You pressed Q/E: the game toggled that blinker."""
        if self.state == side:
            self.state, self.owner = None, ""
        else:
            self.state, self.owner = side, "driver"

    def want(self, side: str | None, by: str = "fsd") -> list[str]:
        """Make the blinker show `side` (None = off). Returns the keys tapped now."""
        taps: list[str] = []
        if side == self.state:
            return taps
        if side is None and self.owner == "driver":
            return taps                      # yours: we don't cancel what you set
        if self.state is not None:
            taps.append("Q" if self.state == "left" else "E")      # toggle the old one off
            self.state, self.owner = None, ""
        if side is not None:
            taps.append("Q" if side == "left" else "E")
            self.state, self.owner = side, by
        self.taps += taps
        return taps

    def pop_taps(self) -> list[str]:
        t, self.taps = self.taps, []
        return t


@dataclass
class LaneEnd:
    ending: bool = False
    side: str = ""            # which line is closing in ("left" / "right")
    merge_to: str = ""        # where to go ("left" / "right")
    near_w: float = 0.0
    far_w: float = 0.0
    frames: int = 0


def lane_width_at(lanes, x: float) -> float | None:
    if lanes.left is None or lanes.right is None:
        return None
    return float(np.polyval(lanes.right.coeffs, x) - np.polyval(lanes.left.coeffs, x))


class LaneEndDetector:
    def __init__(self, need_frames: int = 6):
        self.need = need_frames
        self.st = LaneEnd()

    def update(self, w) -> LaneEnd:
        L = w.lanes
        st = self.st
        if L.conf < 0.4 or L.left is None or L.right is None or L.x_max < 25:
            st.frames = max(0, st.frames - 1)
            st.ending = st.frames >= self.need
            return st
        near_x, far_x = 10.0, min(38.0, L.x_max - 2)
        wn, wf = lane_width_at(L, near_x), lane_width_at(L, far_x)
        if wn is None or wf is None or wn < 2.4:
            return st
        if wf < 0.62 * wn:
            # which line moves in? compare each line's lateral change against the lane centre's
            dl = float(np.polyval(L.left.coeffs, far_x) - np.polyval(L.left.coeffs, near_x))
            dr = float(np.polyval(L.right.coeffs, far_x) - np.polyval(L.right.coeffs, near_x))
            # inward motion of each line (right line moving left, left line moving right)
            closing = "right" if -dr > dl else "left"
            st.side = closing
            st.merge_to = "left" if closing == "right" else "right"
            st.near_w, st.far_w = wn, wf
            st.frames = min(self.need * 3, st.frames + 1)
        else:
            st.frames = max(0, st.frames - 1)
        st.ending = st.frames >= self.need
        return st


def road_beyond(frame: np.ndarray | None, w, geo=None) -> dict[str, float]:
    """Rough 'is there road' past the junction ahead: fraction of road-grey pixels in three
    screen regions just above the stop line (straight on, left, right). Without a picture
    returns neutral values."""
    if frame is None:
        return {"straight": 0.5, "left": 0.5, "right": 0.5}
    H, W = frame.shape[:2]
    hz = int(H * 0.34)
    band = frame[hz + int(0.02 * H):hz + int(0.12 * H)]
    hsv = cv2_hsv(band)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    road = ((s < 40) & (v > 35) & (v < 150)).astype(np.float32)
    thirds = np.array_split(road, 3, axis=1)
    return {"left": float(thirds[0].mean()), "straight": float(thirds[1].mean()),
            "right": float(thirds[2].mean())}


def cv2_hsv(img):
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


class Autonomy:
    """Ties it together; called by the planner every frame."""

    def __init__(self):
        self.blinker = Blinker()
        self.lane_end = LaneEndDetector()
        self.last_choice_t = -1e9
        self.message = ""

    def step(self, w, turns, p, now: float, frame=None) -> None:
        ts = turns.s
        # 1) blinker follows whatever manoeuvre is going on
        if ts.pending is not None and (ts.lane_change or ts.active or ts.junction_d is not None
                                       or ts.message.startswith("Waiting for a gap")):
            self.blinker.want(ts.pending)
        elif ts.pending is None and self.blinker.owner == "fsd":
            self.blinker.want(None)
        if p.get("auto_merge", 1.0) < 0.5 and p.get("auto_route", 1.0) < 0.5:
            return
        if ts.pending is not None:
            return
        # 2) our lane ends -> merge
        le = self.lane_end.update(w)
        if p.get("auto_merge", 1.0) > 0.5 and le.ending:
            turns.press(le.merge_to, now)
            turns.s.message = f"Lane ends - merging {le.merge_to}"
            self.message = turns.s.message
            self.blinker.want(le.merge_to)
            return
        # 3) road ends ahead -> choose a turn
        if p.get("auto_route", 1.0) > 0.5 and now - self.last_choice_t > 20.0:
            cue = min(w.stop_line_dist, w.stop_sign_dist, w.crosswalk_dist)
            if cue < 35 and (w.lanes.x_max < cue + 6 or w.lanes.conf < 0.3):
                rb = road_beyond(frame, w)
                if rb["straight"] < 0.25:
                    side = "right" if rb["right"] >= rb["left"] * 0.8 else "left"
                    turns.press(side, now)
                    turns.s.message = f"Road ends - turning {side}"
                    self.message = turns.s.message
                    self.last_choice_t = now
                    self.blinker.want(side)
