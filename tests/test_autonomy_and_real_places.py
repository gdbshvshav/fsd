"""Turn signals, lane-end merges, road-end turns, and real ER:LC places (high-quality
screenshots of real roads: a 2->1 lane merge at a junction, a 45 zone bend, a hairpin,
and an on-ramp joining the highway)."""
from pathlib import Path

import cv2
import pytest

from fsd.planning.autonomy import Autonomy, Blinker, LaneEndDetector
from fsd.planning.turns import TurnManager
from fsd.world import LaneLine, Lanes, World

ASSETS = Path(__file__).parent / "assets"


# ---------------------------------------------------------------- blinker (Q / E)
def test_blinker_taps_and_cancels_its_own():
    b = Blinker()
    assert b.want("left") == ["Q"] and b.state == "left"
    assert b.want("left") == []                  # already on: no double tap (that would cancel it)
    assert b.want("right") == ["Q", "E"]         # switch sides: old one off, new one on
    assert b.want(None) == ["E"] and b.state is None


def test_blinker_leaves_the_drivers_signal_alone():
    b = Blinker()
    b.driver_pressed("right")                    # you pressed E in game
    assert b.state == "right" and b.owner == "driver"
    assert b.want(None) == []                    # FSD doesn't cancel your blinker
    b.driver_pressed("right")                    # you pressed E again: off
    assert b.state is None


# ---------------------------------------------------------------- lane ends
def _world(left, right, conf=0.8, x_max=50.0):
    w = World()
    w.lanes = Lanes(left=LaneLine(left, x_max), right=LaneLine(right, x_max), conf=conf, x_max=x_max)
    return w


def test_lane_end_detected_and_merge_side():
    d = LaneEndDetector(need_frames=3)
    # right line closes in toward the left line: 3.7 m wide here, ~1 m at 38 m -> merge left
    w = _world((0.0, 0.0, -1.85), (0.0, -0.075, 1.85 + 0.75))
    for _ in range(4):
        st = d.update(w)
    assert st.ending and st.side == "right" and st.merge_to == "left"


def test_parallel_lane_is_not_ending():
    d = LaneEndDetector(need_frames=3)
    w = _world((0.0001, 0.02, -1.85), (0.0001, 0.02, 1.85))          # gentle bend, same width
    for _ in range(6):
        st = d.update(w)
    assert not st.ending


def test_autonomy_queues_merge_with_blinker():
    a = Autonomy()
    a.lane_end.need = 2
    tm = TurnManager()
    w = _world((0.0, 0.0, -1.85), (0.0, -0.075, 2.6))
    p = {"auto_merge": 1.0, "auto_route": 1.0, "turn_timeout_s": 30}
    for k in range(3):
        a.step(w, tm, p, now=k * 0.05)
    assert tm.s.pending == "left"
    assert a.blinker.state == "left" and "Q" in a.blinker.pop_taps()


# ---------------------------------------------------------------- real places
def _run(name, frames=4):
    from fsd.fast.pipeline import FastPerception
    from fsd.fast.scaler import FramePair
    from fsd.params import defaults
    from fsd.settings import Settings
    img = cv2.imread(str(ASSETS / name))
    if img is None:
        pytest.skip(f"{name} missing")
    per = FastPerception(Settings(), use_detector=False, use_ocr=False, live=False)
    w = None
    for i in range(frames):
        pair = FramePair(img, 960, 1920)
        w = per.process(pair.work, defaults(), t=i * 0.05, fine=pair.fine)
    return w


@pytest.mark.parametrize("name", ["real_merge_junction.png", "real_curve_45.png", "real_hairpin.png",
                                  "real_ramp_merge.png"])
def test_real_places_no_phantom_lights_or_stop_signs(name):
    """Autumn leaves, yellow chevron signs and red trees must not stop the car."""
    w = _run(name)
    assert w.signal.state.value in ("none", "unknown")
    assert w.stop_sign_dist > 900


def test_real_ramp_merge_lanes():
    w = _run("real_ramp_merge.png")
    assert w.lanes.conf > 0.5
    assert 3.0 < w.lanes.width_m < 5.5
    assert w.lanes.curvature < 0                 # the ramp bends left


def test_real_red_light_still_red():
    w = _run("erlc_intersection_red.png")
    assert w.signal.state.value == "red"
    assert 10 < w.signal.distance < 60
