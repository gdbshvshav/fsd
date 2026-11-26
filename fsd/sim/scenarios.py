"""Ready-made test drives. Each one targets something that has gone wrong (or
could go wrong) in the real game.

    highway_straight     cruise, hold the lane
    highway_curves       S-bends at speed (overshoot / swerve check)
    highway_yellow_edge  left lane next to the yellow edge on a right-hand bend:
                         the exact spot where FSD drove into the median
    highway_shadows      lanes under heavy Roblox shadow (blue tint)
    recover_offset       starts 1.3 m off-centre and angled: must settle smoothly
    town_red_light       red at a junction: stop at the line, go on green
    town_signal_cycle    arrive while the light cycles green -> yellow -> red
    town_stop_sign       full stop at the bar, then go
    outage_flashing      flashing red (outage) = all-way stop
    lead_car_brakes      car ahead brakes hard on the highway
    parked_car           a car stopped in our lane
    truck_curves         heavy truck physics on bends
    town_tight_bend      a slow, tight town corner
    uncalibrated         default calibration that doesn't match: self-calibration must fix it
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from .road import arc, highway, straight, town_street
from .vehicle import SPECS, VehicleSpec
from .world import Junction, OtherCar, Scene, Shadow, SignalPlan, SpeedSign


@dataclass
class Scenario:
    name: str
    build: Callable[[], Scene]
    start_s: float = 15.0
    start_lane: int = 0
    start_offset: float = 0.0          # m to the right of the lane centre
    start_angle_deg: float = 0.0       # + = pointing right of the road
    start_speed_kph: float = 50.0
    duration_s: float = 30.0
    limit_kph: float = 90.0
    vehicle: VehicleSpec = field(default_factory=VehicleSpec)
    exact_calibration: bool = True     # False: start from the default calibration (tests self-calibration)
    description: str = ""
    expect_stop: bool = False          # the drive must include a full stop (light / sign / car)


def _hw(pieces, shadows=(), cars=(), lanes=2, signs=(), seed=1):
    def build():
        sc = Scene(highway(pieces, lanes=lanes), cars=[OtherCar(**c) for c in cars],
                   shadows=[Shadow(*s) for s in shadows], signs=[SpeedSign(*sg) for sg in signs])
        sc.add_roadside(seed)
        return sc
    return build


def _town(pieces, junctions=(), shadows=(), cars=(), seed=2):
    def build():
        sc = Scene(town_street(pieces, lanes=1), junctions=[j() for j in junctions],
                   cars=[OtherCar(**c) for c in cars], shadows=[Shadow(*s) for s in shadows])
        sc.add_roadside(seed, every=18.0)
        return sc
    return build


SCENARIOS: dict[str, Scenario] = {}


def _add(sc: Scenario) -> None:
    SCENARIOS[sc.name] = sc


_add(Scenario("highway_straight", _hw([straight(1400)]), start_lane=1, start_speed_kph=70, duration_s=25,
              limit_kph=100, description="Cruise and hold the lane on a straight highway."))
_add(Scenario("highway_curves", _hw([straight(120), arc(260, 320), straight(80), arc(260, -300), straight(900)]),
              start_lane=1, start_speed_kph=70, duration_s=32, limit_kph=100,
              description="S-bends at speed: no overshoot, no swerving."))
_add(Scenario("highway_yellow_edge", _hw([straight(60), arc(420, 450), straight(900)], seed=5), start_lane=0,
              start_speed_kph=60, duration_s=30, limit_kph=100,
              description="Left lane by the yellow edge on a right-hand bend (where FSD hit the median)."))
_add(Scenario("highway_shadows", _hw([straight(100), arc(300, -500), straight(900)],
                                     shadows=[(30, 90, -2, 9.5), (130, 170, 4.0, 9.5), (210, 300, -3, 6.0)]),
              start_lane=1, start_speed_kph=65, duration_s=26, limit_kph=100,
              description="Heavy blue Roblox shadows over the lanes."))
_add(Scenario("recover_offset", _hw([straight(900)]), start_lane=1, start_offset=1.3, start_angle_deg=4.0,
              start_speed_kph=55, duration_s=20, limit_kph=90,
              description="Start off-centre and angled; settle smoothly into the lane."))


def _red_then_green():
    return Junction(140.0, control="signal", signal=SignalPlan("cycle", green_s=12, yellow_s=3, red_s=22,
                                                                offset_s=15.0))


_add(Scenario("town_red_light", _town([straight(500)], junctions=[_red_then_green]), start_speed_kph=40,
              duration_s=40, limit_kph=56, expect_stop=True,
              description="Red at the junction: stop at the bar, go on green."))
_add(Scenario("town_signal_cycle", _town([straight(500)], junctions=[
    lambda: Junction(110.0, control="signal", signal=SignalPlan("cycle", green_s=6, yellow_s=3, red_s=12, offset_s=0.0))]),
    start_speed_kph=45, duration_s=36, limit_kph=56,
    description="Light cycles while we approach: stop on yellow if comfortable, else clear."))
_add(Scenario("town_stop_sign", _town([straight(400)], junctions=[lambda: Junction(90.0, control="stop")]),
              start_speed_kph=40, duration_s=30, limit_kph=56, expect_stop=True,
              description="Full stop at the bar (not at the sign), then go."))
_add(Scenario("outage_flashing", _town([straight(400)], junctions=[
    lambda: Junction(90.0, control="signal", signal=SignalPlan("flashing"))]), start_speed_kph=40,
    duration_s=30, limit_kph=56, expect_stop=True, description="Flashing red: all-way stop, then go."))
_add(Scenario("lead_car_brakes", _hw([straight(1500)], cars=[dict(lane=1, s=60.0, speed=22.0, brake_at=9.0,
                                                                   color=(40, 40, 150))]),
              start_lane=1, start_speed_kph=75, duration_s=26, limit_kph=100, expect_stop=True,
              description="Car ahead brakes hard: keep the gap, never hit it."))
_add(Scenario("parked_car", _hw([straight(800)], cars=[dict(lane=1, s=130.0, speed=0.0, parked=True,
                                                             color=(150, 150, 150))]),
              start_lane=1, start_speed_kph=60, duration_s=26, limit_kph=90, expect_stop=True,
              description="A car stopped in our lane: stop behind it."))
_add(Scenario("truck_curves", _hw([straight(100), arc(250, 260), straight(60), arc(250, -260), straight(700)]),
              start_lane=1, start_speed_kph=55, duration_s=32, limit_kph=80, vehicle=SPECS["truck"],
              description="Heavy truck: slow steering response on bends."))
_add(Scenario("town_tight_bend", _town([straight(80), arc(70, 45), straight(300)]), start_speed_kph=35,
              duration_s=26, limit_kph=50, description="Tight town corner at low speed."))
_add(Scenario("uncalibrated", _hw([straight(1200)]), start_lane=1, start_speed_kph=60, duration_s=30,
              limit_kph=90, exact_calibration=False,
              description="Default calibration that doesn't match this camera: self-calibration must fix it."))


def quick_names() -> list[str]:
    """A fast subset for tests."""
    return ["highway_curves", "highway_yellow_edge", "town_red_light", "lead_car_brakes"]


def angle(deg: float) -> float:
    return math.radians(deg)
