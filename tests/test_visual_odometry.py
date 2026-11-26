"""Visual odometry on a synthetic textured road seen through the real ER:LC default calibration:
the measured speed and turn rate must match the motion we applied."""
import math

import cv2
import numpy as np

from fsd.perception.geometry import Geometry
from fsd.perception.visual_odometry import VisualOdometry, hidden_from_box
from fsd.settings import Calibration

W, H = 960, 540
RES = 0.05                     # ground texture: 5 cm per pixel


def ground_texture(seed=3, size_m=(400.0, 60.0)):
    rng = np.random.default_rng(seed)
    nx, ny = int(size_m[0] / RES), int(size_m[1] / RES)
    tex = rng.normal(90, 18, (nx, ny)).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (0, 0), 1.2)
    # dashed lane lines every 12 m (3 m paint), like ER:LC
    for lane_y in (-1.85, 1.85):
        j = int((lane_y + size_m[1] / 2) / RES)
        for x0 in np.arange(0, size_m[0], 12.0):
            tex[int(x0 / RES):int((x0 + 3) / RES), j - 2:j + 2] = 230
    return np.clip(tex, 0, 255).astype(np.uint8), size_m


def render(geo, tex, size_m, car_x, car_y, heading):
    """Picture of the ground for a car at (car_x, car_y) heading `heading` (rad, + right)."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.full((H, W), 140, np.uint8)
    below = ys > geo.horizon_y + 1
    pts = np.stack([xs[below], ys[below]], 1)
    veh = geo.screen_to_vehicle(pts)
    ch, sh = math.cos(heading), math.sin(heading)
    wx = car_x + veh[:, 0] * ch - veh[:, 1] * sh
    wy = car_y + veh[:, 0] * sh + veh[:, 1] * ch
    ti = (wx / RES).astype(np.int64)
    tj = ((wy + size_m[1] / 2) / RES).astype(np.int64)
    ok = (ti >= 0) & (ti < tex.shape[0]) & (tj >= 0) & (tj < tex.shape[1])
    vals = np.full(len(ti), 140, np.uint8)
    vals[ok] = tex[ti[ok], tj[ok]]
    out[below] = vals
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def test_speed_from_the_road():
    geo = Geometry.build(Calibration(), W, H)
    tex, size = ground_texture()
    vo = VisualOdometry()
    v = 30.0                                  # 108 km/h
    dt = 1 / 30
    got = []
    for i in range(12):
        img = render(geo, tex, size, 10.0 + v * dt * i, 0.0, 0.0)
        r = vo.update(img, geo, i * dt, hidden_m=8.0, lane_c=0.0, v_prior=v)
        if r.ok:
            got.append(r.v_raw)
    assert len(got) >= 8
    assert abs(float(np.median(got)) - v) < 0.1 * v


def test_turn_rate_from_the_road():
    geo = Geometry.build(Calibration(), W, H)
    tex, size = ground_texture(5)
    vo = VisualOdometry()
    v, yaw = 12.0, 0.3
    dt = 1 / 30
    x = y = h = 0.0
    x = 20.0
    yaws = []
    for i in range(12):
        img = render(geo, tex, size, x, y, h)
        r = vo.update(img, geo, i * dt, hidden_m=8.0, lane_c=0.0, v_prior=v)
        if r.ok and i > 1:
            yaws.append(r.yaw)
        h += yaw * dt
        x += v * math.cos(h) * dt
        y += v * math.sin(h) * dt
    assert len(yaws) >= 6
    assert abs(float(np.median(yaws)) - yaw) < 0.12


def test_standing_still_reads_zero():
    geo = Geometry.build(Calibration(), W, H)
    tex, size = ground_texture(7)
    vo = VisualOdometry()
    img = render(geo, tex, size, 30.0, 0.0, 0.0)
    out = [vo.update(img, geo, i / 30, 8.0, 0.0, v_prior=0.0) for i in range(6)]
    assert all(abs(r.v_raw) < 0.5 for r in out if r.ok)


def test_hidden_from_box():
    geo = Geometry.build(Calibration(), W, H)
    box = (430, int(0.47 * H), 530, int(0.66 * H))
    d = hidden_from_box(geo, box, W)
    assert 2.0 < d < 22.0
