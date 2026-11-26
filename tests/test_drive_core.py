"""The new driving core (fsd/drive): each piece on its own, with numbers that matter."""
import math

import numpy as np

from fsd.drive.lane_filter import LaneFilter, filtered
from fsd.drive.lateral import GainTable, LateralLQR, WeaveGuard, dlqr, path_geometry
from fsd.drive.longitudinal import LongitudinalController, SpeedEstimator
from fsd.drive.path_planner import OffsetShaper, edges_path, heading_hold_path, lane_path, quintic
from fsd.drive.side_keeper import SideKeeper
from fsd.drive.speed_planner import SpeedPlanner, path_curvature_profile
from fsd.drive.vehicle_model import (PedalMap, SteerSim, curvature_from_steer, max_curvature,
                                     steer_for_curvature)
from fsd.drive.yaw import YawFusion
from fsd.params import defaults
from fsd.perception.road_model import RoadEdges, side_advice
from fsd.world import LaneLine, Lanes, Obj

P = defaults()


def lanes(c=0.0, b=0.0, a=0.0, conf=0.8, width=3.7, left_col="white", right_col="white", right_dashed=False):
    return Lanes(left=LaneLine((a, b, c - width / 2), 40, left_col, False, conf),
                 right=LaneLine((a, b, c + width / 2), 40, right_col, right_dashed, conf),
                 center=(a, b, c), width_m=width, conf=conf, offset=c, heading=math.atan(b), x_max=40,
                 curvature=2 * a)


# ----------------------------------------------------------------------------- vehicle model
def test_steer_curvature_roundtrip_and_limits():
    for v in (2.0, 15.0, 40.0):
        for s in (-0.8, -0.1, 0.0, 0.3, 0.9):
            k = curvature_from_steer(s, v, P)
            assert abs(steer_for_curvature(k, v, P) - s) < 1e-6
    # full lock gets smaller with speed
    assert max_curvature(40.0, P) < max_curvature(5.0, P) / 2


def test_pedal_map_inverts_and_learns():
    m = PedalMap()
    for v in (3.0, 20.0, 40.0):
        for a in (-5.0, -0.2, 0.5, 2.0):
            ped = m.pedal_for(a, v)
            if -1 < ped < 1:
                assert abs(m.accel_for(ped, v) - a) < 1e-6
    # a car that only manages 1.5 m/s^2 at full W around 20 m/s
    t, v = 0.0, 18.0
    for _ in range(4):
        for _ in range(60):
            m.observe(t, v, 1.0)
            t += 1 / 60
            v += 1.5 / 60
        v = 18.0
        m.observe(t, v, 0.0, valid=False)
    assert abs(m.band(19.0).full - 1.5) < 0.6


def test_steer_sim_time_to():
    s = SteerSim(3.9, 6.0)
    assert abs(s.time_to(0.0, 0.39) - 0.1) < 1e-6
    assert s.time_to(0.5, 0.0) < s.time_to(0.0, 0.5)     # springs back faster than it winds up


# ----------------------------------------------------------------------------- yaw fusion
def test_yaw_ignores_picture_spike():
    f = YawFusion()
    t = 0.0
    for i in range(120):
        spike = -1.1 if 60 <= i < 66 else 0.0          # the picture "sees" 63 deg/s for 0.1 s
        y = f.update(t, 28.0, 0.0, P, spike, None)
        t += 1 / 60
        assert abs(y) < 0.05
    # the model follows the steering
    for _ in range(30):
        y = f.update(t, 20.0, 0.3, P, None, None)
        t += 1 / 60
    assert y > 0.1


# ----------------------------------------------------------------------------- lane filter
def test_lane_filter_smooths_and_predicts():
    f = LaneFilter()
    rng = np.random.default_rng(1)
    t = 0.0
    outs = []
    for i in range(180):
        meas = lanes(c=0.3 + rng.normal(0, 0.08), b=rng.normal(0, 0.01)) if i % 2 == 0 else None
        f.predict(t, 20.0, 0.0)
        if meas is not None:
            f.update(meas, t, key=i)
        outs.append(f.lanes(t).offset)
        t += 1 / 60
    late = np.array(outs[60:])
    assert abs(late.mean() - 0.3) < 0.05 and late.std() < 0.05


def test_lane_filter_rebases_on_lane_change():
    f = LaneFilter()
    f.predict(0.0, 20.0, 0.0)
    f.update(lanes(c=1.6), 0.0, key=1)
    f.predict(1 / 60, 20.0, 0.0)
    f.update(lanes(c=1.6 - 3.7), 1 / 60, key=2)        # detector now reports the lane to our left
    assert f.switched == -1
    assert abs(f.lanes(1 / 60).offset - (1.6 - 3.7)) < 0.3


def test_lane_filter_rejects_single_wild_detection():
    f = LaneFilter()
    t = 0.0
    for i in range(30):
        f.predict(t, 20.0, 0.0)
        f.update(lanes(c=0.0), t, key=i)
        t += 1 / 60
    f.predict(t, 20.0, 0.0)
    f.update(lanes(c=0.0, b=0.4), t, key=99)             # a wildly wrong heading
    assert abs(f.lanes(t).heading) < 0.05


def test_lane_filter_coasts_then_gives_up():
    f = LaneFilter()
    f.predict(0.0, 20.0, 0.0)
    f.update(lanes(c=0.2), 0.0, key=1)
    f.predict(0.8, 20.0, 0.0)
    assert f.lanes(0.8).conf > 0
    f.predict(2.0, 20.0, 0.0)
    assert f.lanes(2.0).conf == 0


def test_filtered_helper_keeps_adjacent_lines():
    f = LaneFilter()
    raw = lanes(c=0.1)
    raw.right2 = LaneLine((0, 0, 5.6), 40)
    out = filtered(raw, f, 0.0, 10.0, 0.0, 1)
    assert out.right2 is not None


# ----------------------------------------------------------------------------- lateral
def test_path_geometry_signs():
    straight_right = [(x, 1.0) for x in np.linspace(0, 30, 16)]      # path 1 m to our right
    g = path_geometry(straight_right, 15.0, 0.1)
    assert g.e < -0.9 and abs(g.psi) < 1e-3                         # we're LEFT of it
    pointing_right = [(x, 0.1 * x) for x in np.linspace(0, 30, 16)]
    g = path_geometry(pointing_right, 15.0, 0.1)
    assert g.psi < -0.05                                             # we point left of it
    arc = [(20 * math.sin(s / 20), 20 * (1 - math.cos(s / 20))) for s in np.linspace(0, 30, 40)]
    g = path_geometry(arc, 10.0, 0.1)
    assert abs(g.k_prev - 1 / 20) < 0.01


def test_lqr_is_stable_at_all_speeds():
    for v in (3.0, 10.0, 25.0, 55.0):
        tau, dt = 0.064, 1 / 60
        A = np.array([[1, v * dt, 0], [0, 1, v * dt], [0, 0, 1 - dt / tau]])
        B = np.array([[0], [0], [dt / tau]])
        K = GainTable().gains(v, tau, dt, 1.0, 0.35, 0.06)
        eig = np.abs(np.linalg.eigvals(A - B @ K))
        assert eig.max() < 1.0


def test_dlqr_scalar():
    K = dlqr(np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]]))
    assert abs(K[0, 0] - 0.618) < 0.01                            # golden ratio solution


def _drive(controller_steps, v, path_fn, seconds=8.0):
    """Tiny kinematic simulation of a car following a path with the LQR (steer value lag)."""
    from fsd.control.steer_model import SteerObserver
    lat = LateralLQR()
    obs = SteerObserver(P["steer_rise_per_s"], P["steer_return_per_s"])
    x = y = h = 0.0
    s_val = 0.0
    dt = 1 / 60
    errs = []
    for i in range(int(seconds / dt)):
        # path in world -> car frame
        wx, wy = path_fn()
        dx, dy = wx - x, wy - y
        ch, sh = math.cos(h), math.sin(h)
        px = ch * dx + sh * dy
        py = -sh * dx + ch * dy
        path = list(zip(px, py))
        target = lat.step(path, v, s_val, 1.0, P, dt, i * dt)
        key = obs.choose(target, dt)            # exactly what the real actuator does
        obs.apply(key, dt)
        s_val = obs.s
        k = curvature_from_steer(s_val, v, P)
        h += v * k * dt
        x += v * math.cos(h) * dt
        y += v * math.sin(h) * dt
        errs.append(abs(lat.geom.e))
    return np.array(errs)


def test_lqr_settles_from_offset_without_weaving():
    xs = np.linspace(-10, 400, 400)

    def path():
        return xs, np.full_like(xs, 1.5)
    e = _drive(None, 25.0, path, 10.0)
    assert e[-60:].max() < 0.15                     # settled
    # never overshoots far past the line
    assert e[200:].max() < 0.5


def test_weave_guard_detects_swinging():
    g = WeaveGuard()
    t = 0.0
    for i in range(600):
        g.update(t, 0.5 * math.sin(2 * math.pi * t / 2.0), True)
        t += 1 / 60
    assert g.weaving and g.soft > 1.5


# ----------------------------------------------------------------------------- speed
def test_speed_planner_slows_before_a_bend_not_in_it():
    sp = SpeedPlanner()
    # straight 80 m then a 40 m radius bend
    path = [(x, 0.0) for x in np.linspace(0, 80, 41)]
    for s in np.linspace(0, 60, 31)[1:]:
        th = s / 40
        path.append((80 + 40 * math.sin(th), 40 * (1 - math.cos(th))))
    v = 30.0
    out = sp.step(v, 120.0, None, path, None, 0.0, 0.0, P, 1 / 60)
    v_bend = math.sqrt(P["curve_lat_accel"] * 40)
    assert out.profile_v[0] < v + 1 and out.profile_v[45] <= v_bend + 0.5
    # several frames: the acceleration request goes negative, smoothly
    a = []
    for _ in range(60):
        a.append(sp.step(v, 120.0, None, path, None, 0.0, 0.0, P, 1 / 60).a_target)
    assert min(a) < -0.5 and max(abs(np.diff(a))) <= 3 * P["jerk_limit"] / 60 + 1e-6


def test_speed_planner_stops_at_point():
    sp = SpeedPlanner()
    path = [(x, 0.0) for x in np.linspace(0, 100, 51)]
    out = None
    for _ in range(120):
        out = sp.step(15.0, 60.0, 20.0, path, None, 0.0, 0.0, P, 1 / 60)
    assert out.a_target < -3.0                      # needs ~5.6 m/s^2
    assert out.profile_v[np.searchsorted(out.profile_s, 22.0)] == 0.0


def test_path_curvature_profile():
    arc = [(25 * math.sin(s / 25), 25 * (1 - math.cos(s / 25))) for s in np.linspace(0, 60, 61)]
    k = path_curvature_profile(arc, np.array([10.0, 30.0]))
    assert np.all(np.abs(k - 0.04) < 0.008)


def test_speed_estimator_ignores_misread_and_fills_gaps():
    e = SpeedEstimator()
    t = 0.0
    e.predict(t, 0.0)
    e.update(t, 20.0, True)
    for i in range(30):
        t += 1 / 60
        e.predict(t, 1.0)
        e.update(t, None, False)                 # no reading for 0.5 s
    assert abs(e.v - 20.5) < 0.2                 # integrated the acceleration
    e.update(t, 70.0, True)                      # "20" misread as "70"
    assert e.v < 25
    for i in range(3):                            # ...unless it keeps saying so
        t += 0.1
        e.predict(t, 0.0)
        e.update(t, 70.0 + i * 0.1, True)
    assert e.v > 60


def test_longitudinal_never_reverses_at_standstill():
    lc = LongitudinalController()
    out = [lc.step(0.0, True, 0.0, -2.0, True, P, 1 / 60, i / 60) for i in range(30)]
    assert max(abs(x) for x in out) < 1e-6


def test_longitudinal_brakes_for_negative_accel():
    lc = LongitudinalController()
    for i in range(30):
        ped = lc.step(20.0, True, 15.0, -4.0, False, P, 1 / 60, i / 60)
    assert ped < -0.3


# ----------------------------------------------------------------------------- path / side
def test_quintic_and_offset_shaper():
    assert quintic(np.array(0.0)) == 0 and abs(quintic(np.array(1.0)) - 1) < 1e-9
    sh = OffsetShaper()
    sh.set_target(3.7, 25.0)
    prev = 0.0
    steps = []
    for _ in range(400):
        sh.advance(25.0 / 60)
        steps.append(sh.now - prev)
        prev = sh.now
    assert abs(sh.now - 3.7) < 1e-6 and max(steps) < 0.1


def test_paths():
    p = lane_path((0.001, 0.01, 0.2), 40.0)
    assert len(p) == 28 and abs(p[0][1] - 0.2) < 1e-9
    e = RoadEdges(True, (0, 0, -4.0), (0, 0, 4.0), 40.0, 8.0, 0.9)
    q = edges_path(e, 2.0)
    assert all(abs(y - 2.0) < 1e-6 for _, y in q)
    hh = heading_hold_path(0.2, 10.0)
    assert hh[-1][1] > 0.5                       # keeps turning right


def test_side_advice_rules():
    # yellow line on our RIGHT: wrong side
    L = lanes(right_col="yellow")
    adv = side_advice(L, RoadEdges())
    assert adv.wrong_side and adv.target_offset > 1.0
    # yellow on the left is correct
    assert not side_advice(lanes(left_col="yellow"), RoadEdges()).wrong_side
    # no lines, wide road: right half
    e = RoadEdges(True, (0, 0, -4.5), (0, 0, 4.5), 40.0, 9.0, 0.9)
    adv = side_advice(Lanes(), e)
    assert adv.two_way and adv.target_offset > 1.5
    # narrow road: middle
    e = RoadEdges(True, (0, 0, -2.0), (0, 0, 2.2), 40.0, 4.2, 0.9)
    adv = side_advice(Lanes(), e)
    assert not adv.two_way and abs(adv.target_offset - 0.1) < 1e-6
    # oncoming car in our lane
    car = Obj("car", (0, 0, 1, 1), 0.9, x=30, y=0.2)
    car.vx = -40
    assert side_advice(lanes(), RoadEdges(), [car]).wrong_side


def test_side_keeper_needs_persistence():
    k = SideKeeper()
    L = lanes(right_col="yellow")
    assert not k.update(L, None, [], P, 0.0, 10.0, False).move_right
    d = k.update(L, None, [], P, 0.7, 10.0, False)
    assert d.wrong_side and d.move_right
    # asks only once every few seconds
    assert not k.update(L, None, [], P, 1.0, 10.0, False).move_right


def test_lane_filter_coasts_across_a_junction():
    f = LaneFilter()
    f.predict(0.0, 10.0, 0.0)
    f.update(lanes(c=0.1, a=0.002), 0.0, key=1)
    f.junction = True
    f.max_age = 4.5
    t = 0.0
    for _ in range(180):                      # 3 s without any lines
        t += 1 / 60
        f.predict(t, 10.0, 0.0)
    out = f.lanes(t)
    assert out.conf >= 0.45 and abs(out.center[0]) < 0.001      # straight on, still confident


def test_supervisor_caps_speed_smoothly_when_drifting():
    from fsd.drive.supervisor import Supervisor
    sup = Supervisor()
    t = 0.0
    caps = []
    for i in range(120):
        e = 0.3 + 0.02 * i                     # error keeps growing ...
        st = sup.update(t, 1 / 60, e, -0.6, False, True, 0.1, 0, False, 60.0, 100.0)  # ... while steering hard left
        caps.append(st.cap_kph)
        t += 1 / 60
    assert caps[-1] < 100 and st.level >= 2 and sup.pop_messages()
    assert max(-np.diff([c for c in caps if c < 900])) <= 15.0 / 60 + 1e-6
    # all fine again -> cap released
    for i in range(1200):
        st = sup.update(t, 1 / 60, 0.0, 0.0, False, True, 0.1, 0, False, 60.0, 80.0)
        t += 1 / 60
    assert st.cap_kph > 900


def test_lane_filter_learns_yaw_bias():
    """Our yaw reads 0.1 rad/s too high (truck steers weaker than the model): the filter learns
    it from the lanes instead of letting the lane heading drift between detections."""
    f = LaneFilter()
    t = 0.0
    for i in range(600):
        f.predict(t, 20.0, 0.1)                 # we THINK we turn right at 0.1 rad/s; truly straight
        if i % 3 == 0:
            f.update(lanes(c=0.0, conf=0.5), t, key=i)
        t += 1 / 60
    assert abs(f.yaw_bias - 0.1) < 0.03
    assert abs(f.lanes(t).heading) < 0.02
