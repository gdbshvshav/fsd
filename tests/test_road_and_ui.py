"""Road edges (no lane lines needed), steering key dithering, and the animated dashboard."""
import os

import numpy as np

from fsd.control.steer_model import SteerObserver
from fsd.ui.anim import ColorTween, Counter, Pulse, Spring, Toasts, Tween, ease_out_back, ease_out_cubic


def test_road_edges_in_the_simulator():
    from fsd.sim.runner import SimRun
    from fsd.sim.scenarios import SCENARIOS
    r = SimRun(SCENARIOS["highway_straight"])
    for _ in range(20):
        r.step()
    e = r.last_world.road_edges
    s, d = r._car_sd()
    road = r.road
    true_right = road.lanes * road.lane_width + road.shoulder - d
    true_left = -road.shoulder - d
    assert e is not None and e.ok
    yl, yr = e.at(10.0)
    assert abs(yr - true_right) < 0.4 and abs(yl - true_left) < 0.4


def test_dither_holds_tiny_steering_on_average():
    """At motorway speed the steering needed is smaller than one key frame's step;
    the dithered keys must still deliver it on average (no dead band)."""
    obs = SteerObserver(3.9, 6.0)
    vals = []
    for _ in range(600):
        k = obs.choose(0.02, 1 / 60)
        obs.apply(k, 1 / 60)
        vals.append(obs.s)
    assert abs(np.mean(vals[60:]) - 0.02) < 0.004
    # big moves are still direct
    obs = SteerObserver(3.9, 6.0)
    assert obs.choose(0.8, 1 / 60) == 1


def test_springs_and_tweens():
    s = Spring(0.0, 4.0, 1.0)
    for _ in range(120):
        x = s.step(10.0, 1 / 60)
    assert abs(x - 10.0) < 0.05
    # critically damped: no overshoot
    s = Spring(0.0, 4.0, 1.0)
    assert max(s.step(1.0, 1 / 60) for _ in range(200)) <= 1.0 + 1e-6
    t = Tween(0.0, 0.0)
    t.to(5.0)
    assert t.value == 5.0
    assert ease_out_cubic(0.0) == 0 and abs(ease_out_cubic(1.0) - 1) < 1e-9
    assert max(ease_out_back(u / 50) for u in range(51)) > 1.0       # overshoots a bit
    c = ColorTween((0, 0, 0), 0.0)
    c.to((200, 100, 50))
    assert c.value == (200, 100, 50)
    p = Pulse()
    p.set(False)
    assert p.value() <= 1.0
    q = Toasts(2)
    for i in range(4):
        q.push(f"m{i}")
    assert [t.text for t in q.alive()] == ["m3", "m2"]
    q.push("m3")
    assert len(q.alive()) == 2
    assert abs(Counter(0.0, 3.0).step(100.0, 1 / 60)) < 100


def test_dashboard_hud_and_minimap_render():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from types import SimpleNamespace as NS

    import pygame

    from fsd.perception.road_model import RoadEdges
    from fsd.settings import Settings
    from fsd.ui.dashboard import Dashboard
    from fsd.world import LaneLine, Lanes, Light, Mode, Obj, Plan, World

    class App:
        settings = Settings()
        ui_visible = True
        prompt_open = False
        perception = NS(geo=None)
        proc_size = (960, 540)
        want_debug_img = False

        def snapshot(self):
            return self.snap

        def view_frame(self, w, h):
            return np.full((h, w, 3), 90, np.uint8)

        def shutdown(self):
            pass

    app = App()
    d = Dashboard(app, monitor=0, overlay=True)
    w = World(speed_kph=87, speed_valid=True, speed_limit_kph=90)
    w.lanes = Lanes(left=LaneLine((0, 0.01, -1.8), 40, "yellow"), right=LaneLine((0, 0.01, 1.9), 40, "white", True),
                    center=(0.0005, 0.01, 0.05), conf=0.8, x_max=40, width_m=3.7)
    w.road_edges = RoadEdges(True, (0, 0, -6), (0, 0, 5), 40, 11, 0.9)
    w.objects = [Obj("car", (0, 0, 1, 1), 0.9, x=25, y=0.2)]
    w.signal.state = Light.RED
    w.signal.distance = 45
    plans = [Plan(mode=m, target_speed_kph=80, path=[(x, 0.0005 * x * x) for x in range(0, 50, 2)],
                  reason="following", alert=a, turn_pending=t, steer=0.3, pedal=-0.4, stop_at=30)
             for m, a, t in ((Mode.CRUISE, "", None), (Mode.FOLLOW, "Wrong side of the road - moving right", "right"),
                             (Mode.FAULT, "TAKE OVER: road not visible", None))]
    for i in range(45):
        app.snap = NS(world=w if i % 7 else None, plan=plans[i // 15], engaged=i > 3,
                      status={"log": ["12:00:00 FSD engaged"], "fps": 60, "lanes_fps": 30},
                      debug_img=None, training_on=False, training_count=0)
        d.draw(1 / 60)
    assert d.screen.get_at((d.W // 2, 20))[:3] != (0, 0, 0)
    pygame.quit()
