"""Keeps the car ahead in mind when the camera stops seeing it.

With ER:LC's chase camera, a car that's close in front of us disappears BEHIND
OUR OWN CAR on screen (the camera looks over our roof). Detection also drops
out for a frame here and there. If the planner forgot the lead car the moment
that happens it would drive straight into it - which is exactly what the
simulator showed. So:

* every frame the lead (nearest object in our corridor) is recorded with its
  distance and its speed over the ground (our speed + its closing speed);
* when it's not seen, it's predicted forward: its distance changes by
  (its speed - our speed) x dt, and its speed is assumed to fade toward 0
  (a car we can't see might be braking - assume the worse case slowly);
* it's forgotten when it's been unseen for too long AND it wasn't close, or once
  we've seen clear road where it should be for a while.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..world import Obj


@dataclass
class Remembered:
    x: float               # m ahead of our bumper
    y: float               # m sideways
    v_ground: float        # its own speed over the ground (m/s)
    seen_t: float
    kind: str
    half_width: float


class LeadMemory:
    def __init__(self, keep_far_s: float = 0.6, keep_close_s: float = 4.0, close_m: float = 15.0,
                 fade_s: float = 2.0, confirm_n: int = 3):
        self.confirm_n = confirm_n      # frames an object must be tracked before it's remembered
        self.keep_far_s = keep_far_s
        self.keep_close_s = keep_close_s
        self.close_m = close_m
        self.fade_s = fade_s
        self.lead: Remembered | None = None
        self.last_t: float | None = None
        self.clear_frames = 0

    def reset(self) -> None:
        self.lead = None
        self.last_t = None
        self.clear_frames = 0

    def update(self, t: float, speed_ms: float, seen: Obj | None, corridor_visible_m: float = 0.0,
               hidden_m: float = 0.0) -> Obj | None:
        """seen: the lead found this frame (or None). corridor_visible_m: how far ahead the
        road in our lane is positively seen as clear (0 if unknown). Returns the lead to use."""
        dt = 0.0 if self.last_t is None else max(0.0, min(0.3, t - self.last_t))
        self.last_t = t
        if self.lead is not None:
            self._predict(dt, speed_ms)
        if seen is not None and self.lead is None and seen.extra.get("track_n", 99) < self.confirm_n:
            return seen                 # react to it, but don't remember a maybe-phantom yet
        if seen is not None:
            self.clear_frames = 0
            L = self.lead
            same = L is not None and abs(seen.y - L.y) < 2.2
            if same and seen.extra.get("occluded"):
                # its bottom is hidden behind our car: the measured distance is only an upper bound
                L.x = min(seen.x, L.x)
                L.y = seen.y
                L.seen_t = t
                return self._as_obj(L, speed_ms)
            if same and L.x < self.close_m and seen.x > L.x + 1.5:
                # further than where a close car can be: with the chase camera a near car is half
                # hidden by our own and its roof reads as far-away road. Only let the remembered
                # car recede slowly (a car really pulling away is followed, a misread barely moves it).
                L.x += min(seen.x - L.x, 3.0 * dt)
                L.y = seen.y
                L.seen_t = t
                return self._as_obj(L, speed_ms)
            self.lead = Remembered(seen.x, seen.y, max(0.0, speed_ms + seen.vx), t, seen.kind,
                                   seen.extra.get("half_width", 0.9))
            return seen
        if self.lead is None:
            return None
        L = self.lead
        unseen = t - L.seen_t
        keep = self.keep_close_s if L.x < self.close_m else self.keep_far_s
        if L.x < max(hidden_m, 6.0) and speed_ms < 1.0:
            keep = 1e9          # parked behind something we can't see any more: wait to SEE it gone
        # "seen clear where it should be" only counts when that spot is in plain view (well past
        # the part of the road our own car hides) and the clear road reaches well beyond it -
        # a car right in front is half hidden and its roof reads as road
        if corridor_visible_m > L.x + 8.0 and L.x > hidden_m + 4.0:
            self.clear_frames += 1
        else:
            self.clear_frames = 0
        if unseen > keep or L.x < -2.0 or self.clear_frames >= 8:
            self.lead = None
            return None
        return self._as_obj(L, speed_ms)

    def _predict(self, dt: float, speed_ms: float) -> None:
        """The lead keeps going at a speed that fades toward 0 (it may be braking)."""
        L = self.lead
        L.v_ground *= max(0.0, 1.0 - dt / self.fade_s)
        L.x += (L.v_ground - speed_ms) * dt

    @staticmethod
    def _as_obj(L: Remembered, speed_ms: float) -> Obj:
        o = Obj(L.kind, (0, 0, 0, 0), 0.5, x=max(0.0, L.x), y=L.y,
                extra={"half_width": L.half_width, "remembered": True})
        o.vx = L.v_ground - speed_ms
        return o
