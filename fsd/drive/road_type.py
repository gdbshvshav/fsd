"""What kind of road are we on, and how fast is normal there when no sign says?

Before this, FSD drove 60 km/h everywhere until it read a speed-limit sign, then kept
that limit until the next sign. After a highway exit or a turn into town, it would
carry 105 km/h into a residential street.

ER:LC's Liberty County roads fall into three kinds, each with a usual limit (signs
are in mph):

  * town   - junctions, crosswalks, stop lines and signals keep showing up;
             one lane each way with a yellow centre line             -> 35 mph (56 km/h)
  * rural  - one lane each way, yellow centre, no junctions for a while -> 45 mph (72 km/h)
  * highway- two or more lanes our way (dashed white between them), wide,
             no junctions for a while                                  -> 65 mph (105 km/h)

Evidence is accumulated with time constants (seconds of driving), so one odd frame
changes nothing. A sign we read always wins, but a sign's limit is dropped when the
road type clearly changes afterwards (we turned onto a different road), or after a
junction turn.
"""
from __future__ import annotations

from dataclasses import dataclass

KPH_PER_MPH = 1.609344

DEFAULTS_MPH = {"town": 35.0, "rural": 45.0, "highway": 65.0}


@dataclass
class RoadTypeState:
    kind: str = "unknown"
    conf: float = 0.0
    limit_kph: float | None = None


class RoadTypeClassifier:
    def __init__(self):
        self.score = {"town": 0.0, "rural": 0.0, "highway": 0.0}
        self.last_junction_t = -1e9
        self.state = RoadTypeState()
        self.kind_since = 0.0

    def reset(self) -> None:
        self.__init__()

    def update(self, w, now: float, dt: float, v: float) -> RoadTypeState:
        if min(w.stop_line_dist, w.crosswalk_dist, w.stop_sign_dist) < 60 or \
                (w.signal.state.value not in ("none", "unknown") and w.signal.distance < 120):
            self.last_junction_t = now
        since_junction = now - self.last_junction_t
        ln = w.lanes
        ev = {"town": 0.0, "rural": 0.0, "highway": 0.0}
        if ln.conf > 0.45:
            lanes_our_way = 1
            for side in (ln.left, ln.right):
                if side is not None and side.color == "white" and side.dashed:
                    lanes_our_way += 1
            yellow_near = any(s is not None and s.color == "yellow" for s in (ln.left, ln.left2))
            if lanes_our_way >= 2 and since_junction > 20:
                ev["highway"] += 1.0
            if lanes_our_way == 1 and yellow_near:
                if since_junction < 25:
                    ev["town"] += 1.0
                else:
                    ev["rural"] += 1.0
        if since_junction < 8:
            ev["town"] += 1.5
        if v > 85 / 3.6 and since_junction > 20:
            ev["highway"] += 0.5
        # leaky integration (~8 s memory)
        k = min(1.0, dt / 8.0)
        for key in self.score:
            self.score[key] += (ev[key] - self.score[key]) * k
        best = max(self.score, key=self.score.get)
        total = sum(self.score.values()) + 1e-6
        conf = self.score[best] / total
        if self.score[best] < 0.25 or conf < 0.5:
            best_kind = "unknown"
        else:
            best_kind = best
        if best_kind != self.state.kind:
            self.kind_since = now
        limit = DEFAULTS_MPH.get(best_kind)
        self.state = RoadTypeState(best_kind, float(conf), limit * KPH_PER_MPH if limit else None)
        return self.state
