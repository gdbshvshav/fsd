"""Keyboard car physics, modelled on how Roblox cars (A-Chassis style, which
ER:LC's vehicles behave like) respond to W/A/S/D:

* Steering is not instant: holding A/D swings the wheels toward full lock at a
  fixed rate; letting go returns them to straight at a (faster) rate. Full
  lock gets smaller at speed.
* W accelerates hard at low speed and less near top speed.
* S brakes hard while moving forward; at a standstill it would reverse
  (the FSD actuator never does that).
* Rolling resistance + air drag slow the car when coasting.
* Movement follows a kinematic bicycle model (wheelbase ~2.9 m).

Different vehicles = different VehicleSpec numbers (sedan, SUV, truck, sports).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class VehicleSpec:
    name: str = "sedan"
    wheelbase: float = 2.9
    length: float = 4.7
    width: float = 1.95
    height: float = 1.5
    max_steer_deg: float = 32.0           # full lock at walking speed
    steer_speed_falloff: float = 28.0     # m/s at which full lock has halved
    steer_rate: float = 3.9               # steering value per second while A/D is held (fraction of lock)
    return_rate: float = 6.0              # steering value per second back to centre when released
    accel: float = 5.2                    # m/s^2 at 0 speed with W held
    top_speed: float = 52.0               # m/s (~187 km/h)
    brake: float = 9.0                    # m/s^2 with S held
    roll: float = 0.25                    # m/s^2 rolling resistance
    drag: float = 0.0009                  # air drag coefficient (a = drag * v^2)
    reverse_speed: float = 6.0


SPECS = {
    "sedan": VehicleSpec(),
    "suv": VehicleSpec("suv", 3.0, 4.9, 2.0, 1.8, 30.0, 26.0, 3.4, 5.5, 4.6, 48.0, 8.5),
    "truck": VehicleSpec("truck", 4.2, 7.5, 2.4, 3.0, 28.0, 20.0, 2.6, 4.5, 3.0, 36.0, 7.0, 0.3, 0.0012),
    "sports": VehicleSpec("sports", 2.6, 4.4, 1.9, 1.25, 33.0, 34.0, 5.0, 7.0, 7.5, 70.0, 11.0, 0.2, 0.0007),
}


@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0       # rad, 0 = +Y (north), + = clockwise
    v: float = 0.0             # m/s, + forward
    steer: float = 0.0         # road-wheel angle (rad), + = right
    steer_value: float = 0.0   # Roblox steering value -1..1 (what A/D actually move)
    yaw_rate: float = 0.0
    odometer: float = 0.0


@dataclass
class Vehicle:
    spec: VehicleSpec = field(default_factory=VehicleSpec)
    state: VehicleState = field(default_factory=VehicleState)

    def max_steer(self) -> float:
        v = abs(self.state.v)
        return math.radians(self.spec.max_steer_deg) / (1.0 + v / self.spec.steer_speed_falloff)

    def step(self, dt: float, w: bool, a: bool, s: bool, d: bool) -> None:
        st, sp = self.state, self.spec
        # ---- steering value ramps while A/D is held, springs back to 0 when released;
        #      wheel angle = value x full lock at this speed (Roblox chassis behaviour)
        sv = st.steer_value
        if d and not a:
            sv = min(1.0, sv + (sp.steer_rate if sv >= 0 else max(sp.steer_rate, sp.return_rate)) * dt)
        elif a and not d:
            sv = max(-1.0, sv - (sp.steer_rate if sv <= 0 else max(sp.steer_rate, sp.return_rate)) * dt)
        else:
            sv = max(0.0, sv - sp.return_rate * dt) if sv > 0 else min(0.0, sv + sp.return_rate * dt)
        st.steer_value = sv
        st.steer = sv * self.max_steer()
        # ---- longitudinal
        acc = 0.0
        if w and not s:
            acc += sp.accel * max(0.0, 1.0 - max(0.0, st.v) / sp.top_speed)
        if s:
            if st.v > 0.3:
                acc -= sp.brake
            else:
                acc -= sp.accel * 0.6                 # reversing
        # resistance always opposes motion
        if abs(st.v) > 0.01:
            res = sp.roll + sp.drag * st.v * st.v
            acc -= math.copysign(res, st.v)
        v_new = st.v + acc * dt
        if st.v > 0 and v_new < 0 and not s:
            v_new = 0.0                                # resistance never reverses the car
        if s and st.v > 0.3 and v_new < 0:
            v_new = 0.0                                # braking stops at zero
        v_new = max(-sp.reverse_speed, min(sp.top_speed, v_new))
        # ---- kinematic bicycle
        v_avg = 0.5 * (st.v + v_new)
        st.yaw_rate = v_avg / sp.wheelbase * math.tan(st.steer)
        st.heading += st.yaw_rate * dt
        st.x += v_avg * math.sin(st.heading) * dt
        st.y += v_avg * math.cos(st.heading) * dt
        st.v = v_new
        st.odometer += abs(v_avg) * dt

    @property
    def speed_kph(self) -> float:
        return self.state.v * 3.6

    def corners(self) -> list[tuple[float, float]]:
        """Footprint corners in world coords (rear-left, rear-right, front-right, front-left).

        The state position is the REAR axle; the footprint extends ~0.9 m behind it."""
        st, sp = self.state, self.spec
        back = 0.9
        fwd = sp.length - back
        hw = sp.width / 2
        sh, ch = math.sin(st.heading), math.cos(st.heading)
        out = []
        for lon, lat in ((-back, -hw), (-back, hw), (fwd, hw), (fwd, -hw)):
            out.append((st.x + lon * sh + lat * ch, st.y + lon * ch - lat * sh))
        return out

    def front_bumper(self) -> tuple[float, float]:
        st, sp = self.state, self.spec
        fwd = sp.length - 0.9
        return st.x + fwd * math.sin(st.heading), st.y + fwd * math.cos(st.heading)


class KeyModulator:
    """The real FSD actuator's sigma-delta key timing, frame by frame (60 Hz).

    Turns steer/pedal commands in -1..1 into held keys, exactly like
    fsd.winput.Actuator.tick does on Windows, so the simulator sees the same
    key pattern the game would.
    """

    def __init__(self, rise: float = 3.9, ret: float = 6.0):
        from ..control.steer_model import SteerObserver
        from ..winput import Actuator
        self._sd = Actuator._sd
        self.obs = SteerObserver(rise, ret)
        self.acc_s = 0.0
        self.acc_p = 0.0
        self.dt = 1 / 60

    allow_reverse = False

    def keys(self, steer: float, pedal: float, speed_kph: float, enabled: bool = True) -> tuple[bool, bool, bool, bool]:
        if not enabled:
            self.acc_s = self.acc_p = 0.0
            self.obs.reset()
            return False, False, False, False
        s = self.obs.choose(steer, self.dt)
        self.obs.apply(s, self.dt)
        if pedal < 0 and speed_kph < 1.5 and not self.allow_reverse:
            pedal = 0.0                       # never reverse at a standstill (unless asked to back up)
        self.acc_p, p = self._sd(self.acc_p, max(-1.0, min(1.0, pedal)))
        return p > 0, s < 0, p < 0, s > 0     # W, A, S, D
