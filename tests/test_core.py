"""Unit tests that run anywhere (no Roblox, no GPU): python -m pytest -q"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from fsd.control.controllers import LateralController, SpeedController
from fsd.params import SPEC, ParamStore, clamp_changes, defaults
from fsd.perception.geometry import Geometry
from fsd.perception.lanes import LaneFinder
from fsd.perception.lights import SignalTracker, classify_light
from fsd.planning.planner import Planner
from fsd.planning.turns import TurnManager, arc_path
from fsd.settings import Calibration
from fsd.winput import Actuator, InputMonitor, VK
from fsd.world import LaneLine, Lanes, Light, Mode, Obj, SignalInfo, World

P = defaults()


def straight_world(speed=50.0, **kw) -> World:
    w = World(speed_kph=speed, speed_valid=True)
    w.lanes = Lanes(left=LaneLine((0, 0, -1.85), 50, conf=0.9), right=LaneLine((0, 0, 1.85), 50, conf=0.9),
                    center=(0, 0, 0), width_m=3.7, conf=0.9, x_max=50)
    for k, v in kw.items():
        setattr(w, k, v)
    return w


# ---------------------------------------------------------------- params
def test_clamp_and_versions(tmp_path):
    assert clamp_changes({"steer_gain": 99, "nope": 1})["steer_gain"] == SPEC["steer_gain"].hi
    store = ParamStore(tmp_path)
    v1 = store.version
    v2 = store.apply({"steer_gain": 1.5}, "test")
    assert v2 == v1 + 1 and store["steer_gain"] == 1.5
    assert store.rollback() == v1 and store["steer_gain"] == SPEC["steer_gain"].default
    store2 = ParamStore(tmp_path)                      # reload from disk
    assert store2.version == v1


# ---------------------------------------------------------------- actuator
def test_sigma_delta_average():
    a = Actuator()
    a.enabled = True
    a.speed_kph = 30
    a.command(0.3, -0.25)
    s, sv = [], []
    for _ in range(600):
        s.append(a.tick())
        sv.append(a.steer_obs.s)
    a.release_all()
    pedal = np.mean([x[1] for x in s])
    # steering: the Roblox steering value (as the observer tracks it) settles on the command
    assert abs(np.mean(sv[120:]) - 0.3) < 0.06 and abs(pedal + 0.25) < 0.02


def test_no_reverse_at_standstill():
    a = Actuator()
    a.enabled = True
    a.speed_kph = 0.0
    a.command(0, -1)
    assert all(a.tick()[1] == 0 for _ in range(60))
    a.release_all()


def test_takeover_and_hotkeys():
    hits = []
    mon = InputMonitor({VK["F1"]: lambda: hits.append("f1")}, lambda vk: hits.append("drive"))
    mon.feed(VK["F1"], True)
    mon.feed(VK["F1"], True)          # key repeat must not toggle twice
    mon.feed(VK["F1"], False)
    mon.feed(VK["W"], True)
    assert hits == ["f1", "drive"]


# ---------------------------------------------------------------- lights
def _signal_crop(lit: str) -> np.ndarray:
    img = np.zeros((90, 36, 3), np.uint8)
    img[:] = (0, 190, 225)                              # yellow frame (BGR), like ER:LC (V~225)
    img[4:86, 6:30] = (20, 20, 20)                      # black body
    pos = {"red": 17, "yellow": 45, "green": 72}
    col = {"red": (40, 40, 255), "yellow": (0, 200, 255), "green": (80, 255, 60)}
    for k, y in pos.items():
        cv2.circle(img, (18, y), 9, col[k] if k == lit else (45, 45, 45), -1)
    return img


@pytest.mark.parametrize("state", ["red", "yellow", "green"])
def test_classify_light_ignores_yellow_frame(state):
    s, conf = classify_light(_signal_crop(state))
    assert s == state and conf > 0.5


def test_light_back_ignored():
    back = np.zeros((90, 36, 3), np.uint8)
    back[:] = (0, 190, 225)
    assert classify_light(back)[0] == "back"


def test_flashing_red_outage():
    geo = Geometry.build(Calibration(), 1920, 1080)
    tr = SignalTracker(geo, 7.0)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    t = 0.0
    info = None
    for i in range(40):
        lit = "red" if (i // 5) % 2 == 0 else "off"
        crop = _signal_crop(lit if lit != "off" else "none")
        frame[200:290, 950:986] = crop
        ob = Obj("light", (950, 200, 986, 290), 0.9)
        info = tr.update(frame, [ob], 0.0, 2.0, t)
        t += 0.1
    assert info.state == Light.FLASHING_RED


# ---------------------------------------------------------------- lanes
def test_lane_finder_synthetic_road():
    cal = Calibration(ego_mask=[0, 0, 0, 0])
    geo = Geometry.build(cal, 1920, 1080)
    bev = np.full((geo.bev_h, geo.bev_w, 3), (70, 70, 70), np.uint8)   # grey asphalt
    for y_m, dashed in ((-1.85, False), (1.85, True)):
        u = int(geo.bev_w / 2 + y_m / geo.m_per_px_x)
        for v in range(0, geo.bev_h, 1):
            if dashed and (v // 60) % 2:
                continue
            cv2.line(bev, (u - 2, v), (u + 2, v), (235, 235, 235), 1)
    frame = cv2.warpPerspective(bev, geo.Minv, (1920, 1080))
    frame[frame.sum(axis=2) == 0] = (60, 140, 60)                     # grass outside
    lf = LaneFinder(geo)
    lanes = None
    for i in range(3):
        lanes, _, _ = lf.process(frame, 0.35, 3.7, t=i * 0.05)
    assert lanes.conf > 0.4
    assert abs(lanes.left.y_at(10) + 1.85) < 0.3 and abs(lanes.right.y_at(10) - 1.85) < 0.3
    assert lanes.right.dashed and not lanes.left.dashed
    assert abs(lanes.offset) < 0.3


# ---------------------------------------------------------------- planner
def test_red_light_stops_at_line():
    pl = Planner()
    w = straight_world(60, signal=SignalInfo(Light.RED, 40.0, 0.9), stop_line_dist=25.0)
    plan = pl.step(w, P, True, now=1.0)
    assert plan.stop_at == pytest.approx(25.0 - P["stop_line_offset_m"])
    assert plan.mode == Mode.STOPPING and plan.target_speed_kph < 60


def test_green_light_cruises():
    pl = Planner()
    w = straight_world(40, signal=SignalInfo(Light.GREEN, 40.0, 0.9), speed_limit_kph=50.0)
    plan = pl.step(w, P, True, now=1.0)
    assert plan.mode == Mode.CRUISE and plan.target_speed_kph == pytest.approx(50 + P["speed_over_limit_kph"])


def test_yellow_too_close_goes_through():
    pl = Planner()
    w = straight_world(60, signal=SignalInfo(Light.YELLOW, 20.0, 0.9), stop_line_dist=6.0)
    plan = pl.step(w, P, True, now=1.0)
    assert plan.stop_at is None


def test_stop_sign_full_stop_then_go():
    pl = Planner()
    w = straight_world(0.0, stop_sign_dist=4.0, stop_line_dist=1.5)
    t = 10.0
    plan = pl.step(w, P, True, now=t)
    assert plan.mode == Mode.STOP_WAIT
    plan = pl.step(w, P, True, now=t + P["stop_sign_wait_s"] + 0.1)
    assert plan.mode != Mode.STOP_WAIT and plan.target_speed_kph > 5


def test_stop_sign_waits_for_cross_traffic():
    pl = Planner()
    car = Obj("car", (0, 0, 1, 1), 0.9, x=12.0, y=-9.0, vx=-5.0, track_id=3)
    w = straight_world(0.0, stop_sign_dist=4.0, stop_line_dist=1.5, objects=[car])
    pl.step(w, P, True, now=10.0)
    plan = pl.step(w, P, True, now=10.0 + P["stop_sign_wait_s"] + 0.1)
    assert plan.mode == Mode.STOP_WAIT and "cross traffic" in plan.reason


def test_follow_slower_car():
    pl = Planner()
    lead = Obj("car", (0, 0, 1, 1), 0.9, x=20.0, y=0.0, vx=-4.0, track_id=1, extra={"half_width": 0.9})
    w = straight_world(50, objects=[lead])
    plan = pl.step(w, P, True, now=1.0)
    assert plan.mode == Mode.FOLLOW and plan.target_speed_kph < 50


def test_pedestrian_stop():
    pl = Planner()
    ped = Obj("person", (0, 0, 1, 1), 0.9, x=18.0, y=0.3, extra={"half_width": 0.4})
    w = straight_world(30, objects=[ped])
    plan = pl.step(w, P, True, now=1.0)
    assert plan.stop_at == pytest.approx(18.0 - P["pedestrian_buffer_m"])


def test_lost_road_faults():
    pl = Planner()
    w = World(speed_kph=30, speed_valid=True)
    pl.step(w, P, True, now=1.0)
    plan = pl.step(w, P, True, now=3.5)
    assert plan.mode == Mode.FAULT and plan.target_speed_kph == 0


def test_outage_is_all_way_stop():
    pl = Planner()
    w = straight_world(0.0, signal=SignalInfo(Light.FLASHING_RED, 15.0, 0.8), stop_line_dist=1.2)
    plan = pl.step(w, P, True, now=5.0)
    assert plan.mode == Mode.STOP_WAIT


# ---------------------------------------------------------------- turns
def test_turn_queue_cancel_switch():
    tm = TurnManager()
    assert "queued" in tm.press("left", 0)
    assert tm.s.pending == "left"
    tm.press("right", 1)
    assert tm.s.pending == "right"
    assert "cancelled" in tm.press("right", 2)
    assert tm.s.pending is None


def test_turn_preview_and_execute():
    tm = TurnManager()
    tm.press("right", 0)
    t = 0.0
    for i in range(10):                                        # a junction seen while driving toward it
        t += 0.05
        w = straight_world(20, stop_line_dist=30.0 - 20 / 3.6 * t)
        tm.update(w, P, 0.05, t, {"left": True, "right": True})
    assert tm.s.preview and tm.s.preview[-1][1] > 5           # path ends to the right
    # drive past the line: cue disappears, dead-reckoning takes over; the right line ends at the junction
    w2 = straight_world(20)
    w2.lanes.right = LaneLine((0, 0, 1.85), 3.0, conf=0.9)
    for _ in range(200):
        t += 0.05
        tm.update(w2, P, 0.05, t, {"left": True, "right": True})
        if tm.s.active:
            break
    assert tm.s.active


def test_turn_needs_confirmed_junction_and_an_opening():
    # one frame of 'crosswalk' paint must not start a turn
    tm = TurnManager()
    tm.press("right", 0)
    tm.update(straight_world(20, crosswalk_dist=2.0), P, 0.05, 0.05, {"left": True, "right": True})
    for i in range(40):
        tm.update(straight_world(20), P, 0.05, 0.1 + i * 0.05, {"left": True, "right": True})
    assert not tm.s.active and not tm.s.preview
    # a confirmed junction, but a solid line keeps going on the right: no turn into the barrier
    tm = TurnManager()
    tm.press("right", 0)
    t = 0.0
    for i in range(10):
        t += 0.05
        tm.update(straight_world(20, stop_line_dist=12.0 - 20 / 3.6 * t), P, 0.05, t, {"left": True, "right": True})
    for i in range(120):
        t += 0.05
        tm.update(straight_world(20), P, 0.05, t, {"left": True, "right": True})
    assert not tm.s.active


def test_lane_change_on_dashed_line():
    tm = TurnManager()
    tm.press("left", 0)
    w = straight_world(60)
    w.lanes.left = LaneLine((0, 0, -1.85), 50, dashed=True, conf=0.9)
    w.lanes.left2 = LaneLine((0, 0, -5.55), 50, conf=0.9)
    tm.update(w, P, 0.05, 0.1, {"left": True, "right": True})
    assert tm.s.lane_change and tm.lateral_shift(3.7) == 0
    tm.s.lc_progress = 0.5
    assert tm.lateral_shift(3.7) == pytest.approx(-1.85)


def test_arc_path_geometry():
    pts = arc_path(10, 9, "right")
    x, y = pts[-1]
    assert x > 15 and y > 20


# ---------------------------------------------------------------- control
def test_steering_sign_and_damping():
    lat = LateralController()
    w = straight_world(40)
    w.lanes.center = (0, 0, 1.0)                     # lane centre is 1 m to our right
    from fsd.world import Plan
    plan = Plan(mode=Mode.CRUISE, path=[(x, 1.0) for x in np.linspace(0, 40, 20)])
    out = [lat.step(w, plan, P, 0.033) for _ in range(30)][-1]
    assert out > 0.05                                 # steer right
    w.yaw_rate = 0.8                                  # already turning hard right
    lat2 = LateralController()
    out2 = [lat2.step(w, plan, P, 0.033) for _ in range(30)][-1]
    assert out2 < out                                 # damping backs off


def test_speed_controller_brakes_for_stop():
    lon = SpeedController()
    from fsd.world import Plan
    w = straight_world(50)
    plan = Plan(mode=Mode.STOPPING, target_speed_kph=10, stop_at=12.0)
    out = [lon.step(w, plan, P, 0.033) for _ in range(20)][-1]
    assert out < -0.2
    plan = Plan(mode=Mode.CRUISE, target_speed_kph=80)
    lon2 = SpeedController()
    assert [lon2.step(w, plan, P, 0.033) for _ in range(20)][-1] > 0.3


def test_vehicle_geometry_roundtrip():
    geo = Geometry.build(Calibration(), 1920, 1080)
    pts = np.array([[10.0, -1.5], [25.0, 2.0], [40.0, 0.0]])
    back = geo.screen_to_vehicle(geo.vehicle_to_screen(pts))
    assert np.allclose(back, pts, atol=1e-2)
    assert geo.horizon_y < 1080 * 0.6
    assert math.isfinite(geo.focal_px)
