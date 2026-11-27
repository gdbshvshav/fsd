"""Lane finding on real ER:LC frames from the user's PC (4K, zoomed-in chase camera).

These two failed before: far specks near the horizon beat the real lines next to the
car, and warm white paint on sunny concrete was thrown away as 'too saturated'.
"""
from pathlib import Path

import cv2
import pytest

from fsd.fast.pipeline import FastPerception
from fsd.params import defaults
from fsd.settings import Settings, quad_problem

DATA = Path(__file__).parent / "data"


def run(name, n=6):
    work = cv2.imread(str(DATA / name))
    per = FastPerception(Settings(), use_detector=False, use_ocr=False, live=False)
    p = defaults()
    w = None
    for i in range(n):
        w = per.process(work, p, t=i / 30)
    return w, per


def test_highway_sees_yellow_and_dashes():
    w, _ = run("erlc_highway.png")
    ln = w.lanes
    assert ln.left is not None and ln.right is not None
    assert ln.left.color == "yellow" and -2.6 < ln.left.coeffs[2] < -1.4
    assert 0.9 < ln.right.coeffs[2] < 2.2
    assert 3.0 < ln.width_m < 4.2 and ln.conf > 0.4


def test_town_picks_own_lane_not_phantom():
    w, _ = run("erlc_town.png")
    ln = w.lanes
    assert ln.left is not None and ln.right is not None
    assert -2.3 < ln.left.coeffs[2] < -1.0
    assert 1.8 < ln.right.coeffs[2] < 3.2           # the solid edge, not a line through our car
    assert 3.0 < ln.width_m < 4.5


@pytest.mark.parametrize("quad,ok", [
    (Settings().calibration.src_quad, True),
    ([[0.6, 0.5], [0.4, 0.5], [0.52, 0.39], [0.49, 0.39]], False),      # crossed
    ([[0.4, 0.3], [0.6, 0.3], [0.52, 0.5], [0.49, 0.5]], False),        # upside down
])
def test_quad_check(quad, ok):
    assert (quad_problem(quad) == "") == ok


def test_dev_view_renders():
    from fsd.ui.devview import render
    w, per = run("erlc_town.png", 3)
    img = cv2.imread(str(DATA / "erlc_town.png"))
    out = render(img, w, per, None, {"fps": 10.0}, out_w=640)
    assert out.shape[1] == 640 and out.shape[0] > 300


def test_ultrawide_warm_light_recalibrates_and_sees_lines():
    # 3495x1651 window, sunset-tinted asphalt, zoomed-out camera: saw nothing before
    w, per = run("erlc_ultrawide.png", 90)
    ln = w.lanes
    assert ln.left is not None and ln.right is not None
    assert -2.4 < ln.left.coeffs[2] < -1.0 and 0.8 < ln.right.coeffs[2] < 2.4
    assert 2.8 < ln.width_m < 4.3 and ln.conf > 0.4
    assert per.bootstrap.applied >= 1


@pytest.mark.parametrize("name,min_conf", [("erlc_hwy_day.png", 0.8), ("erlc_hwy_night.png", 0.5)])
def test_full_window_day_and_night_stay_locked(name, min_conf):
    # full Roblox window from the user's PC; ~14 s so the self-calibration gets its chance and
    # must NOT knock out lanes that are already seen well (it did: 0.91 -> 0.44)
    w, per = run(name, 420)
    ln = w.lanes
    assert ln.left is not None and ln.right is not None
    assert -2.4 < ln.left.coeffs[2] < -1.2 and 1.2 < ln.right.coeffs[2] < 2.4
    assert ln.conf > min_conf
