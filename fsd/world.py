"""Shared data types: what perception sees and what the planner decides."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Light(str, Enum):
    NONE = "none"
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    FLASHING_RED = "flashing_red"   # outage: treat as all-way stop
    UNKNOWN = "unknown"


@dataclass
class LaneLine:
    coeffs: tuple[float, float, float]   # y_right = a*x^2 + b*x + c  (vehicle metres)
    x_max: float                         # how far ahead it was seen (m)
    color: str = "white"                 # "white" | "yellow"
    dashed: bool = False
    conf: float = 0.0

    def y_at(self, x: float) -> float:
        a, b, c = self.coeffs
        return a * x * x + b * x + c


@dataclass
class Lanes:
    left: LaneLine | None = None
    right: LaneLine | None = None
    left2: LaneLine | None = None        # next line out on the left (adjacent lane)
    right2: LaneLine | None = None
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)   # ego lane centre polynomial
    width_m: float = 3.7
    conf: float = 0.0
    curvature: float = 0.0               # 1/m at ~10 m ahead (+ = curving right)
    heading: float = 0.0                 # rad, lane direction vs car (+ = lane points right)
    offset: float = 0.0                  # m, lane centre's lateral position (+ = centre is to our right)
    x_max: float = 0.0


@dataclass
class Obj:
    kind: str                 # car, person, cone, light, stop_sign, speed_sign, emergency
    box: tuple[int, int, int, int]       # screen px x1,y1,x2,y2
    score: float
    x: float = 999.0          # vehicle-frame metres (forward)
    y: float = 0.0            # (right)
    vx: float = 0.0           # relative velocity m/s along x (negative = closing)
    track_id: int = -1
    extra: dict = field(default_factory=dict)


@dataclass
class SignalInfo:
    state: Light = Light.NONE
    distance: float = 999.0   # m to the light (or stop line if seen)
    conf: float = 0.0
    box: tuple[int, int, int, int] | None = None


@dataclass
class World:
    t: float = 0.0
    frame_id: int = 0
    speed_kph: float = 0.0
    speed_valid: bool = False
    speed_unit: str = "kph"
    lanes: Lanes = field(default_factory=Lanes)
    objects: list[Obj] = field(default_factory=list)
    signal: SignalInfo = field(default_factory=SignalInfo)
    stop_sign_dist: float = 999.0
    stop_line_dist: float = 999.0
    crosswalk_dist: float = 999.0
    speed_limit_kph: float | None = None
    yaw_rate: float = 0.0           # rad/s from image motion (+ = turning right)
    emergency_near: bool = False
    road_conf: float = 0.0          # how sure we are we're on a road
    tunnel: bool = False            # dark scene
    horizon_y: float = 0.0          # measured horizon (screen px), for camera aim
    vanish_x: float | None = None   # lane vanishing point x (screen px)
    clear_ahead_m: float = 0.0      # our lane is SEEN to be clear up to here (0 = unknown / hidden)
    extra_hidden_m: float = 0.0     # the first metres of our lane hidden behind our own car on screen
    lanes_t: float = 0.0            # when the lane detection used here was taken (perception time)
    lane_switch: int = 0            # +1/-1: this frame our lane became the one to the right/left
    road_edges: object = None       # perception.road_model.RoadEdges: where the road surface ends
    timings: dict = field(default_factory=dict)


class Mode(str, Enum):
    OFF = "off"
    CRUISE = "cruise"
    FOLLOW = "follow"
    STOPPING = "stopping"          # braking for light / sign / obstacle
    STOPPED = "stopped"            # holding at a line
    STOP_WAIT = "stop_wait"        # stop sign / flashing red pause & clear check
    TURNING = "turning"
    LANE_CHANGE = "lane_change"
    YIELD = "yield"                # emergency vehicle
    CAUTION = "caution"            # unclear scene: slow
    FAULT = "fault"                # lost perception: stop


@dataclass
class Plan:
    mode: Mode = Mode.OFF
    target_speed_kph: float = 0.0
    path: list[tuple[float, float]] = field(default_factory=list)   # vehicle metres
    steer: float = 0.0
    pedal: float = 0.0
    reason: str = ""
    stop_at: float | None = None     # m ahead we intend to stop
    lead: Obj | None = None
    turn_pending: str | None = None  # "left" | "right"
    turn_preview: list[tuple[float, float]] = field(default_factory=list)
    alert: str = ""
    lane_change: bool = False        # a lane change is in progress (whatever the mode shows)
