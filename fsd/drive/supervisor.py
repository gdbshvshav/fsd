"""Safety supervisor: notices when driving is going wrong BEFORE the car leaves the road.

It watches the driving core's own numbers and, when something looks off, caps the
speed (slower = more time, smaller errors) and tells you why. Checks:

* drifting: the lateral error keeps growing for ~0.7 s while the steering is
  already pushing hard the other way (car not responding as the model expects -
  ice-like surface, damaged car, wrong vehicle settings);
* steering saturated: full lock held for over a second outside a turn;
* lanes unreliable: the lane filter keeps rejecting detections, or its own
  uncertainty is large;
* speed unknown: no speedometer reading for a while -> 50 km/h cap;
* slow PC: the control loop runs below 15 checks/s -> cap that scales with rate;
* weaving: the steering weave guard is active.

Caps ease in and out (never a sudden brake), and each reason is reported once.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class SupervisorState:
    cap_kph: float = 999.0
    reasons: list[str] = field(default_factory=list)
    level: int = 0            # 0 fine, 1 careful, 2 concerned, 3 take over soon


class Supervisor:
    def __init__(self):
        self.hist: deque = deque(maxlen=180)       # (t, e, s_cmd)
        self.cap = 999.0
        self.sat_since: float | None = None
        self.no_speed_since: float | None = None
        self.state = SupervisorState()
        self._said: dict[str, float] = {}
        self.messages: list[str] = []

    def reset(self) -> None:
        self.__init__()

    def _say(self, key: str, text: str, now: float) -> None:
        if now - self._said.get(key, -1e9) > 20.0:
            self._said[key] = now
            self.messages.append(text)

    def update(self, now: float, dt: float, e: float, s_cmd: float, turning: bool, speed_valid: bool,
               lane_sigma: float, lane_rejects: int, weaving: bool, fps: float | None, v_kph: float) -> SupervisorState:
        self.hist.append((now, e, s_cmd))
        caps: list[tuple[float, str, int]] = []
        # drifting against the steering
        recent = [h for h in self.hist if now - h[0] <= 0.7]
        if len(recent) >= 10 and not turning:
            e0, e1 = recent[0][1], recent[-1][1]
            pushing = sum(1 for h in recent if h[2] * e1 < 0 and abs(h[2]) > 0.25) / len(recent)
            if abs(e1) > 0.45 and abs(e1) > abs(e0) + 0.15 and pushing > 0.7:
                caps.append((max(30.0, v_kph * 0.7), "drifting although steering against it", 2))
        # saturated steering outside turns
        if abs(s_cmd) > 0.95 and not turning:
            self.sat_since = self.sat_since if self.sat_since is not None else now
            if now - self.sat_since > 1.0:
                caps.append((35.0, "steering at full lock", 2))
        else:
            self.sat_since = None
        # lanes
        if lane_rejects >= 3 or lane_sigma > 0.6:
            caps.append((max(35.0, v_kph * 0.8), "lane lines unreliable", 1))
        # speedometer
        if not speed_valid:
            self.no_speed_since = self.no_speed_since if self.no_speed_since is not None else now
            if now - self.no_speed_since > 2.0:
                caps.append((50.0, "speed not readable", 1))
        else:
            self.no_speed_since = None
        if fps is not None and 0 < fps < 15:
            caps.append((max(25.0, fps * 4.0), f"only {fps:.0f} checks/s", 1))
        if weaving:
            caps.append((max(40.0, v_kph * 0.85), "swinging in the lane - calming", 1))
        target = min([c[0] for c in caps], default=999.0)
        # ease: caps come in at 15 km/h per second, go away at 8 km/h per second
        if target < self.cap:
            self.cap = max(target, (self.cap if self.cap < 900 else v_kph + 5) - 15.0 * dt)
        else:
            self.cap = min(target, self.cap + 8.0 * dt) if self.cap < 900 else 999.0
            if target > 900 and self.cap > v_kph + 15.0:
                self.cap = 999.0              # no longer limiting: the speed planner takes over smoothly
        reasons = [c[1] for c in caps]
        for c in caps:
            self._say(c[1], f"Careful: {c[1]} - slowing to {c[0]:.0f} km/h", now)
        self.state = SupervisorState(self.cap, reasons, max([c[2] for c in caps], default=0))
        return self.state

    def pop_messages(self) -> list[str]:
        m, self.messages = self.messages, []
        return m
