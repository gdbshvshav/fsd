"""Tests for the steering model, junction/lead memory, tracker and the small-object finders."""
import math

import cv2
import numpy as np

from fsd.control.steer_model import SteerObserver, max_steer_rad, steer_value_for_curvature
from fsd.perception.signs import find_stop_signs
from fsd.perception.tracking import Tracker
from fsd.planning.junction_memory import JunctionMemory
from fsd.planning.lead_memory import LeadMemory
from fsd.world import Obj


# ---------------------------------------------------------------- steering model
def test_observer_tracks_partial_steering():
    ob = SteerObserver(rise=3.9, ret=6.0)
    dt = 1 / 60
    vals = []
    for _ in range(240):
        k = ob.choose(0.35, dt)
        ob.apply(k, dt)
        vals.append(ob.s)
    assert abs(np.mean(vals[60:]) - 0.35) < 0.05
    assert max(vals[60:]) < 0.5            # doesn't overshoot to full lock


def test_observer_returns_to_centre():
    ob = SteerObserver()
    ob.s = 0.8
    for _ in range(60):
        ob.apply(ob.choose(0.0, 1 / 60), 1 / 60)
    assert abs(ob.s) < 0.08


def test_observer_left_is_mirror_of_right():
    a, b = SteerObserver(), SteerObserver()
    for _ in range(90):
        a.apply(a.choose(0.5, 1 / 60), 1 / 60)
        b.apply(b.choose(-0.5, 1 / 60), 1 / 60)
    assert abs(a.s + b.s) < 1e-6


def test_lock_shrinks_with_speed_and_value_grows():
    assert max_steer_rad(30, 32, 28) < max_steer_rad(5, 32, 28)
    k = 1 / 300                              # a gentle highway bend
    slow = steer_value_for_curvature(k, 5, 2.9, 32, 28)
    fast = steer_value_for_curvature(k, 30, 2.9, 32, 28)
    assert 0 < slow < fast <= 1.0
    assert steer_value_for_curvature(-k, 30, 2.9, 32, 28) == -fast


# ---------------------------------------------------------------- junction memory
def _drive(jm, t0, secs, v, **seen):
    t = t0
    for _ in range(int(secs * 30)):
        t += 1 / 30
        jm.update(t, v, seen.get("stop_line", 999.0), 999.0, seen.get("sign", 999.0),
                  seen.get("light", "none"), 999.0)
    return t


def test_bar_is_dead_reckoned_when_it_leaves_the_picture():
    jm = JunctionMemory()
    t = 0.0
    d = 40.0
    for _ in range(30):                       # 1 s seeing the bar while driving 10 m/s
        t += 1 / 30
        d -= 10 / 30
        jm.update(t, 10.0, d, 999.0, 999.0, "none", 999.0)
    assert abs(jm.bar_distance() - d) < 1.0
    t = _drive(jm, t, 1.0, 10.0)              # 1 s blind
    assert abs(jm.bar_distance() - (d - 10.0)) < 1.0


def test_bar_kept_while_standing_still():
    jm = JunctionMemory()
    jm.update(0.0, 0.0, 3.0, 999.0, 999.0, "none", 999.0)
    jm.update(0.05, 0.0, 3.0, 999.0, 999.0, "none", 999.0)
    _drive(jm, 0.05, 30.0, 0.0)               # 30 s at a red light, bar out of view
    assert jm.bar_distance() < 5.0


def test_far_bar_does_not_replace_near_one():
    jm = JunctionMemory()
    t = 0.0
    for _ in range(10):
        t += 1 / 30
        jm.update(t, 0.0, 12.0, 999.0, 999.0, "none", 999.0)
    for _ in range(10):
        t += 1 / 30
        jm.update(t, 0.0, 45.0, 999.0, 999.0, "none", 999.0)   # the next junction's bar
    assert abs(jm.bar_distance() - 12.0) < 1.0


def test_stop_sign_requires_full_stop_until_done():
    jm = JunctionMemory()
    t = 0.0
    for _ in range(5):
        t += 1 / 30
        jm.update(t, 8.0, 30.0, 999.0, 32.0, "none", 999.0)
    assert jm.full_stop_needed
    t = _drive(jm, t, 0.5, 8.0)               # sign and bar out of view: still needed
    assert jm.full_stop_needed
    jm.mark_stopped()
    assert not jm.full_stop_needed


# ---------------------------------------------------------------- lead memory
def test_hidden_close_car_is_remembered_while_stopped():
    lm = LeadMemory()
    o = Obj("obstacle", (0, 0, 1, 1), 0.6, x=6.0, y=0.0)
    o.vx = -2.0
    t = 0.0
    lm.update(t, 2.0, o)
    for _ in range(300):                      # 10 s, car hidden behind ours, we're stopped
        t += 1 / 30
        got = lm.update(t, 0.0, None, corridor_visible_m=0.0, hidden_m=8.0)
    assert got is not None and got.x < 7.0


def test_far_misread_of_close_car_is_not_believed():
    lm = LeadMemory()
    t = 0.0
    lm.update(t, 3.0, Obj("obstacle", (0, 0, 1, 1), 0.6, x=7.0, y=0.0))
    far = Obj("obstacle", (0, 0, 1, 1), 0.6, x=16.0, y=0.0)
    for _ in range(15):
        t += 1 / 30
        got = lm.update(t, 3.0, far)
    assert got.x < 9.0


def test_lost_far_car_is_forgotten():
    lm = LeadMemory()
    lm.update(0.0, 20.0, Obj("car", (0, 0, 1, 1), 0.6, x=60.0, y=0.0))
    t = 0.0
    for _ in range(40):
        t += 1 / 30
        got = lm.update(t, 20.0, None)
    assert got is None


# ---------------------------------------------------------------- tracker
def test_tracker_closing_speed_is_steady():
    tr = Tracker()
    t = 0.0
    rng = np.random.default_rng(1)
    vx = []
    for i in range(45):
        t += 1 / 30
        x = 40.0 - 10.0 * t + rng.normal(0, 0.6)          # closing at 10 m/s, noisy range
        o = Obj("obstacle", (100, 100, 140, 130), 0.6, x=x, y=0.1)
        tr.update([o], t)
        vx.append(o.vx)
    assert abs(np.mean(vx[-15:]) + 10.0) < 2.0
    assert o.track_id == 1                     # one track all along


# ---------------------------------------------------------------- stop signs
def _octagon(img, cx, cy, r, col=(40, 40, 200)):
    pts = np.array([[cx + r * math.cos(a), cy + r * math.sin(a)]
                    for a in np.linspace(math.pi / 8, 2 * math.pi + math.pi / 8, 9)[:-1]], np.int32)
    cv2.fillPoly(img, [pts], col)


def test_small_far_stop_sign_found():
    img = np.full((512, 960, 3), (230, 180, 100), np.uint8)     # sky
    img[180:] = (90, 100, 105)                                  # road / ground
    cv2.rectangle(img, (700, 170), (701, 200), (150, 150, 150), -1)
    _octagon(img, 700, 168, 3)                                  # ~6 px across: 40+ m away
    found = find_stop_signs(img, 174)
    assert any(abs((o.box[0] + o.box[2]) / 2 - 700) < 5 for o in found)


def test_brake_lights_are_not_a_stop_sign():
    img = np.full((512, 960, 3), (90, 100, 105), np.uint8)
    cv2.rectangle(img, (440, 220), (520, 250), (30, 30, 30), -1)
    cv2.rectangle(img, (445, 228), (455, 236), (30, 30, 255), -1)   # lit (V=255)
    cv2.rectangle(img, (505, 228), (515, 236), (30, 30, 255), -1)
    assert find_stop_signs(img, 174) == []
