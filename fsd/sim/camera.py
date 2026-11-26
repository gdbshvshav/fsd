"""The chase camera, like ER:LC's default vehicle camera.

It sits high and well behind the car, pitched down so the horizon is about a
third of the way down the screen, and the car sits in the middle of the
picture. Like Roblox's camera it doesn't snap to the car's heading: it swings
after it with a short lag, so on a bend the car turns on screen first and the
view catches up - FSD has to cope with that in the real game too.

Pinhole camera, vertical field of view 70 degrees (Roblox default).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ChaseCamera:
    width: int = 960
    height: int = 512
    vfov_deg: float = 70.0
    height_m: float = 5.0         # above the road
    back_m: float = 12.0          # behind the car's rear
    pitch_deg: float = 12.6       # looking down
    lag_s: float = 0.25           # how slowly it swings after the car's heading
    yaw: float = 0.0
    x: float = 0.0
    y: float = 0.0
    _init: bool = False

    @property
    def focal(self) -> float:
        return (self.height / 2) / math.tan(math.radians(self.vfov_deg) / 2)

    def follow(self, car_x: float, car_y: float, car_heading: float, rear_offset: float, dt: float) -> None:
        """car_x/y: rear axle; rear_offset: metres from rear axle back to the bumper."""
        if not self._init:
            self.yaw = car_heading
            self._init = True
        else:
            k = 1.0 - math.exp(-dt / max(1e-3, self.lag_s))
            dyaw = (car_heading - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += dyaw * k
        back = self.back_m + rear_offset
        self.x = car_x - back * math.sin(self.yaw)
        self.y = car_y - back * math.cos(self.yaw)

    # ------------------------------------------------------------------ projection
    def to_camera(self, P: np.ndarray) -> np.ndarray:
        """World points (N,3) [x east, y north, z up] -> camera coords (N,3) [right, down, forward]."""
        P = np.asarray(P, np.float64).reshape(-1, 3)
        dx = P[:, 0] - self.x
        dy = P[:, 1] - self.y
        dz = P[:, 2] - self.height_m
        sy, cy = math.sin(self.yaw), math.cos(self.yaw)
        fwd = dx * sy + dy * cy
        right = dx * cy - dy * sy
        up = dz
        p = math.radians(self.pitch_deg)
        sp, cp = math.sin(p), math.cos(p)
        # pitch down: rotate (fwd, up) by p
        z = fwd * cp - up * sp
        yv = -(fwd * sp + up * cp)            # image y points down
        return np.stack([right, yv, z], 1)

    def to_screen(self, C: np.ndarray) -> np.ndarray:
        """Camera coords -> pixels (points must have z > 0)."""
        f = self.focal
        return np.stack([self.width / 2 + f * C[:, 0] / C[:, 2], self.height / 2 + f * C[:, 1] / C[:, 2]], 1)

    @property
    def horizon_y(self) -> float:
        return self.height / 2 - self.focal * math.tan(math.radians(self.pitch_deg))
