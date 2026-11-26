"""Everything in a simulated drive besides the road itself.

* Junction: a cross street over our road, with stop bar, crosswalks and
  optionally ER:LC-style signals (yellow housing, black body, three lamps)
  hanging over our lanes on the far side, or stop signs.
* Signal timing: green -> yellow -> red cycle, fixed state, or flashing red (outage).
* Other cars: drive along a lane with a speed plan (cruise, brake at a time,
  parked); they have bright red tail lights like ER:LC cars.
* Shadows: dark blue patches on the road (Roblox shadow tint), the case that
  broke lane finding in the real game.
* Props: trees and light poles along the road (clutter for the vision).
* Speed-limit signs.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .road import Road


@dataclass
class SignalPlan:
    mode: str = "cycle"            # cycle | fixed | flashing
    fixed: str = "red"             # for mode == fixed
    green_s: float = 10.0
    yellow_s: float = 3.0
    red_s: float = 10.0
    offset_s: float = 0.0          # where in the cycle we start
    flash_period_s: float = 1.0

    def state(self, t: float) -> str:
        if self.mode == "fixed":
            return self.fixed
        if self.mode == "flashing":
            return "red" if (t % self.flash_period_s) < self.flash_period_s / 2 else "off"
        cyc = self.green_s + self.yellow_s + self.red_s
        u = (t + self.offset_s) % cyc
        if u < self.green_s:
            return "green"
        if u < self.green_s + self.yellow_s:
            return "yellow"
        return "red"


@dataclass
class Junction:
    s: float                        # where the cross street's near edge crosses our road
    width: float = 14.0             # cross street width (m)
    control: str = "signal"         # signal | stop | none
    signal: SignalPlan = field(default_factory=SignalPlan)
    crosswalk: bool = True

    @property
    def stop_line_s(self) -> float:
        return self.s - (4.5 if self.crosswalk else 1.5)

    def signal_state(self, t: float) -> str:
        return self.signal.state(t) if self.control == "signal" else "none"


@dataclass
class OtherCar:
    lane: int
    s: float
    speed: float                    # m/s target cruise
    color: tuple = (40, 40, 160)    # BGR body
    brake_at: float | None = None   # sim time at which it brakes to a stop
    brake_decel: float = 6.0
    parked: bool = False
    length: float = 4.7
    width: float = 1.95
    height: float = 1.5
    v: float = field(default=-1.0)

    def step(self, dt: float, t: float) -> None:
        if self.parked:
            self.v = 0.0
            return
        if self.v < 0:
            self.v = self.speed
        if self.brake_at is not None and t >= self.brake_at:
            self.v = max(0.0, self.v - self.brake_decel * dt)
        else:
            self.v += max(-3.0 * dt, min(2.0 * dt, self.speed - self.v))
        self.s += self.v * dt


@dataclass
class Shadow:
    s0: float
    s1: float
    d0: float
    d1: float
    strength: float = 1.0           # 1 = full Roblox shadow


@dataclass
class Prop:
    kind: str                       # tree | pole
    s: float
    d: float
    size: float = 1.0


@dataclass
class SpeedSign:
    s: float
    limit: int                      # number printed on the sign (mph, like ER:LC)
    d: float = 0.0                  # set by the scene to just off the right edge


@dataclass
class Scene:
    road: Road
    junctions: list[Junction] = field(default_factory=list)
    cars: list[OtherCar] = field(default_factory=list)
    shadows: list[Shadow] = field(default_factory=list)
    props: list[Prop] = field(default_factory=list)
    signs: list[SpeedSign] = field(default_factory=list)
    sky: tuple = (235, 190, 120)        # BGR
    time_of_day: str = "day"            # day | dusk (darker)

    def step(self, dt: float, t: float) -> None:
        for c in self.cars:
            c.step(dt, t)

    def add_roadside(self, seed: int = 0, every: float = 25.0) -> None:
        """Trees and light poles along both sides (visual clutter, like ER:LC)."""
        rnd = random.Random(seed)
        r = self.road
        edge = r.lanes * r.lane_width + r.shoulder
        s = 10.0
        while s < r.length:
            self.props.append(Prop("tree", s + rnd.uniform(-5, 5), edge + rnd.uniform(6, 25), rnd.uniform(0.8, 1.4)))
            left = -r.shoulder - (r.median_width / 2 if r.divided else 3.0)
            self.props.append(Prop("tree", s + rnd.uniform(-5, 5), left - rnd.uniform(0, 4) if r.divided
                                   else -edge - rnd.uniform(6, 25), rnd.uniform(0.8, 1.4)))
            if rnd.random() < 0.5:
                self.props.append(Prop("pole", s, edge + 2.0, 1.0))
            s += every * rnd.uniform(0.6, 1.4)
        for sg in self.signs:
            sg.d = edge + 1.5

    def lead_car(self, ego_s: float, ego_lane_d: float) -> OtherCar | None:
        best = None
        for c in self.cars:
            if abs(self.road.lane_center(c.lane) - ego_lane_d) < self.road.lane_width / 2 and c.s > ego_s:
                if best is None or c.s < best.s:
                    best = c
        return best


def tail_light_on(car: OtherCar) -> bool:
    """Brake lights glow brighter when the car is braking."""
    return car.brake_at is not None or car.v < 0.5


def seconds_to_cross(j: Junction, speed: float) -> float:
    return j.width / max(speed, 0.1)


def heading_vector(h: float) -> tuple[float, float]:
    return math.sin(h), math.cos(h)
