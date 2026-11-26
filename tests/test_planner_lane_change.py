"""Lane changes through the planner: the path must move over smoothly and never jump a lane
twice when the lane under us re-assigns half way."""
import numpy as np

from fsd.params import defaults
from fsd.planning.planner import Planner
from fsd.world import LaneLine, Lanes, Mode, World

P = defaults()


def world(c, t, switch=0, width=3.7):
    w = World(t=t, speed_kph=90.0, speed_valid=True, speed_limit_kph=100)
    w.lanes = Lanes(left=LaneLine((0, 0, c - width / 2), 50, "white", True, 0.8),
                    right=LaneLine((0, 0, c + width / 2), 50, "white", True, 0.8),
                    left2=LaneLine((0, 0, c - 1.5 * width), 50, "white", False, 0.6),
                    right2=LaneLine((0, 0, c + 1.5 * width), 50, "white", False, 0.6),
                    center=(0, 0, c), width_m=width, conf=0.85, offset=c, x_max=50)
    w.lane_switch = switch
    return w


def test_lane_change_path_is_continuous_across_reassignment():
    pl = Planner()
    t = 0.0
    car_y = 0.0                  # where the car is (world), lanes: ours centred at 0, next at +3.7
    lane_centre_world = 0.0
    pl.step(world(0.0, t), P, True, now=t)
    pl.turns.force_lane_change("right", t, True)
    path_y_world = []
    switched = False
    for i in range(400):
        t += 1 / 30
        c = lane_centre_world - car_y
        sw = 0
        if not switched and car_y > 1.85:          # crossed the line: detector reports the new lane
            lane_centre_world = 3.7
            c = lane_centre_world - car_y
            sw = +1
            switched = True
        plan = pl.step(world(c, t, sw), P, True, now=t)
        y0 = plan.path[0][1]
        path_y_world.append(car_y + y0)
        # the "car" follows the path's start smoothly (perfect tracking with lag)
        car_y += (car_y + plan.path[3][1] - car_y) * 0.08
        if plan.mode != Mode.LANE_CHANGE and switched and i > 30:
            break
    yw = np.array(path_y_world)
    assert switched
    assert np.max(np.abs(np.diff(yw))) < 0.3                 # no jump when the lane re-assigned
    assert yw.max() < 3.7 + 0.5                               # never overshoots into a third lane
    assert abs(yw[-1] - 3.7) < 0.3                            # ends in the middle of the new lane
    assert pl.turns.s.lane_change is False


def test_left_turn_waits_for_oncoming_car_and_gets_in_lane():
    from fsd.drive.junction_rules import JunctionRules
    from fsd.planning.turns import TurnState
    from fsd.world import Obj
    jr = JunctionRules()
    ts = TurnState(pending="left", junction_d=8.0)
    w = world(0.0, 0.0)
    car = Obj("car", (0, 0, 1, 1), 0.9, x=40.0, y=-3.7)
    car.vx = -25.0 - 0.0          # coming toward us at 25 m/s while we roll slowly
    w.objects = [car]
    adv = jr.update(w, ts, 0.0, 0.0, 3.7, P)
    assert adv.hold and "oncoming" in adv.message
    car.x = 200.0
    assert not jr.update(w, ts, 0.0, 1.0, 3.7, P).hold
    # a car going OUR way on the left doesn't block a left turn
    car.x, car.vx = 30.0, 0.0
    w.speed_kph = 30.0
    assert not jr.update(w, ts, 30 / 3.6, 2.0, 3.7, P).hold
    # far from the junction with a dashed white line on the left: get in lane
    ts2 = TurnState(pending="left", junction_d=100.0)
    w.objects = []
    adv = jr.update(w, ts2, 20.0, 10.0, 3.7, P)
    assert adv.change_lane == "left"


def test_turn_survives_a_lane_change_to_get_in_lane():
    from fsd.planning.turns import TurnManager
    tm = TurnManager()
    tm.press("left", 0.0) if hasattr(tm, "press") else None
    tm.s.pending, tm.s.junction_d = "left", 90.0
    assert tm.lane_change_for_turn("left", 1.0, True)
    assert tm.s.lane_change and tm.s.pending == "left"
    tm.finish_lane_change()
    assert tm.s.pending == "left" and not tm.s.lane_change and tm.s.junction_d == 90.0
