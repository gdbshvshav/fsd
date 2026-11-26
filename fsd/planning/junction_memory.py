"""Remembers what's ahead at the next junction, even when the camera loses it.

Why: in ER:LC's chase camera the stop bar slides under our own car a good
15-20 m before we reach it, and stop signs leave the side of the picture as we
pull up. A planner that only looks at the current frame forgets the bar right
when it matters and rolls through. This module keeps:

* the stop bar distance - measured when visible, dead-reckoned from our speed
  when not (odometry), dropped once we're past it;
* whether the next bar needs a FULL stop (a stop sign or a flashing-red/outage
  light was seen for it), kept until we've actually stopped there or passed it;
* the last red/yellow light distance the same way.

Everything is in metres from our front bumper.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Tracked:
    dist: float = 999.0        # metres ahead (999 = nothing)
    seen_t: float = -1e9       # last time it was actually measured
    hits: int = 0              # how many frames in a row it was measured
    alt: float = 999.0         # a competing measurement that doesn't fit the track
    alt_hits: int = 0
    odo_seen: float = 0.0      # odometer when last measured

    def valid(self) -> bool:
        return self.dist < 900


class JunctionMemory:
    def __init__(self, keep_s: float = 6.0, max_jump_m: float = 6.0):
        self.keep_s = keep_s            # how long a lost bar is dead-reckoned
        self.max_jump_m = max_jump_m    # a measurement further than this from the prediction is a new bar
        self.bar = Tracked()
        self.sign = Tracked()
        self.light = Tracked()
        self.full_stop_needed = False   # stop sign / outage at the next bar
        self.full_stop_reason = ""
        self.stopped_here = False       # we already made the full stop at this bar
        self.last_t: float | None = None
        self.odometer = 0.0

    def reset(self) -> None:
        self.__init__(self.keep_s, self.max_jump_m)

    # ------------------------------------------------------------------
    def _advance(self, tr: Tracked, moved: float, t: float, keep: float) -> None:
        """Dead-reckon. A lost track is kept while we haven't driven far since we saw it
        (standing at a red for a minute must not forget the bar), and dropped once passed."""
        if not tr.valid():
            return
        tr.dist -= moved
        driven = self.odometer - tr.odo_seen
        if tr.dist < -3.0 or (t - tr.seen_t > keep and driven > 25.0) or t - tr.seen_t > 120.0:
            tr.dist, tr.hits = 999.0, 0

    def _measure(self, tr: Tracked, d: float, t: float, alpha: float = 0.5) -> None:
        if d >= 900:
            tr.hits = 0
            return
        if tr.valid() and abs(d - tr.dist) < self.max_jump_m:
            tr.dist = alpha * d + (1 - alpha) * tr.dist
            tr.alt_hits = 0
        elif tr.valid() and d > tr.dist + self.max_jump_m and tr.dist > -1.0:
            # further away than the bar we're tracking: that's the NEXT junction (or the far
            # crosswalk). Deal with it after we've passed this one.
            return
        elif tr.valid() and t - tr.seen_t < 0.5:
            # doesn't fit a track we saw just now: switch only if it keeps coming back
            if tr.alt < 900 and abs(d - tr.alt) < self.max_jump_m:
                tr.alt_hits += 1
            else:
                tr.alt_hits = 1
            tr.alt = d
            if tr.alt_hits < 3:
                return
            tr.dist, tr.hits, tr.alt_hits = d, 0, 0
        else:
            tr.dist = d
        tr.seen_t = t
        tr.odo_seen = self.odometer
        tr.hits += 1

    # ------------------------------------------------------------------
    def update(self, t: float, speed_ms: float, stop_line: float, crosswalk: float, stop_sign: float,
               light_state: str, light_dist: float) -> None:
        dt = 0.0 if self.last_t is None else max(0.0, min(0.3, t - self.last_t))
        self.last_t = t
        moved = max(0.0, speed_ms) * dt
        self.odometer += moved

        bar_before = self.bar.dist
        self._advance(self.bar, moved, t, self.keep_s)
        self._advance(self.sign, moved, t, self.keep_s)
        self._advance(self.light, moved, t, 3.0)

        # the bar: the stop line when we see one (a crosswalk alone is too easily confused)
        self._measure(self.bar, stop_line, t)
        self._measure(self.sign, stop_sign, t, alpha=0.4)
        if light_state in ("red", "yellow") and light_dist < 900:
            self._measure(self.light, light_dist, t)

        # full-stop requirement
        if stop_sign < 70 and self.sign.hits >= 2:
            self._require("stop sign")
        if light_state == "flashing_red":
            self._require("outage: all-way stop")

        # passed the bar -> everything about this junction is done
        passed = bar_before < 900 and bar_before - moved < -1.0
        if passed or (self.full_stop_needed and not self.bar.valid() and not self.sign.valid()
                      and light_state != "flashing_red"):
            self.full_stop_needed = False
            self.full_stop_reason = ""
            self.stopped_here = False

    def _require(self, why: str) -> None:
        if not self.stopped_here:
            self.full_stop_needed = True
            self.full_stop_reason = why

    def mark_stopped(self) -> None:
        """Called by the planner once the full stop + wait is done."""
        self.stopped_here = True
        self.full_stop_needed = False

    # ------------------------------------------------------------------
    def bar_distance(self) -> float:
        return self.bar.dist if self.bar.valid() else 999.0

    def stop_point(self, offset_m: float, sign_fallback_m: float) -> float:
        """Where the front bumper should come to rest for the remembered junction."""
        if self.bar.valid():
            return max(0.0, self.bar.dist - offset_m)
        if self.sign.valid():
            return max(0.0, self.sign.dist - sign_fallback_m)
        return 999.0
