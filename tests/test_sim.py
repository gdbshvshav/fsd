"""Simulator smoke tests: the closed loop runs, keeps the lane, and the picture looks like ER:LC."""
import dataclasses

import cv2
import numpy as np

from fsd.fast.lamps import find_lamps
from fsd.sim.camera import ChaseCamera
from fsd.sim.render import Renderer
from fsd.sim.road import Road, straight
from fsd.sim.runner import SimRun, calibration_for
from fsd.sim.scenarios import SCENARIOS
from fsd.sim.vehicle import SPECS, Vehicle


def test_straight_highway_keeps_lane():
    sc = dataclasses.replace(SCENARIOS["highway_straight"], duration_s=6.0)
    m = SimRun(sc, fine=False).run()
    assert m.collisions == 0 and m.edge_violations == 0
    assert m.max_abs_lane_err < 1.0


def test_vehicle_steering_ramp_and_return():
    car = Vehicle(SPECS["sedan"])
    car.state.v = 15.0
    for _ in range(12):                    # 0.2 s of D
        car.step(1 / 60, True, False, False, True)
    assert 0.5 < car.state.steer_value < 1.0
    for _ in range(30):                    # release
        car.step(1 / 60, True, False, False, False)
    assert abs(car.state.steer_value) < 1e-6


def test_render_has_road_horizon_and_hud():
    cam = ChaseCamera()
    road = Road([straight(300)], lanes=2)
    from fsd.sim.world import Scene
    sc = Scene(road)
    car = Vehicle(SPECS["sedan"])
    x, y, h = road.point(10, road.lane_center(1))
    car.state.x, car.state.y, car.state.heading = x, y, h
    cam.follow(x, y, h, 0.9, 1 / 30)
    img = Renderer(cam).render(sc, car, 10.0, 0.0)
    assert img.shape == (512, 960, 3)
    hz = int(cam.horizon_y)
    sky = img[: hz - 20].reshape(-1, 3).mean(0)
    assert sky[0] > sky[2]                 # blue-ish sky above the horizon (BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    assert (hsv[int(0.85 * 512), int(0.5 * 960), 2]) > 40     # HUD box drawn


def test_exact_calibration_is_sane():
    cam = ChaseCamera()
    cal = calibration_for(cam, 4.6, 1.9, 1.5)
    q = np.array(cal.src_quad)
    assert (q[:, 1][:2] > q[:, 1][2:]).all()        # near edge lower on screen than far edge
    assert 0.3 < cal.ego_mask[1] < 0.6


def test_tail_lights_are_not_a_red_light():
    img = np.full((1024, 1920, 3), (90, 100, 105), np.uint8)
    img[:340] = (230, 180, 100)
    cv2.rectangle(img, (900, 330), (1020, 380), (25, 25, 25), -1)     # rear of a dark car
    cv2.circle(img, (915, 345), 4, (40, 40, 255), -1)                 # two lit tail lights
    cv2.circle(img, (1005, 345), 4, (40, 40, 255), -1)
    assert [o for o in find_lamps(img, 350) if o.extra["state"] == "red"] == []


def test_signal_head_is_a_red_light():
    img = np.full((1024, 1920, 3), (230, 180, 100), np.uint8)
    cv2.rectangle(img, (950, 280), (966, 318), (0, 190, 225), -1)     # yellow housing
    cv2.rectangle(img, (952, 282), (964, 316), (25, 25, 28), -1)      # black face
    cv2.circle(img, (958, 288), 4, (40, 40, 255), -1)                 # lit red lamp
    found = find_lamps(img, 350)
    assert any(o.extra["state"] == "red" for o in found)
