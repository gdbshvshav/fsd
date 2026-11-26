"""The multi-core vision engine: same answers as the single-process one, results moved to 'now'."""
import time
from pathlib import Path

import cv2
import numpy as np

from fsd.fast.parallel import ParallelPerception, advance_lanes, shift_poly
from fsd.fast.scaler import FramePair, shrink_to
from fsd.params import defaults
from fsd.settings import Settings
from fsd.world import LaneLine, Lanes

SHOT = Path(__file__).parent / "assets" / "erlc_intersection_red.png"


def test_shift_poly_straight_and_turning():
    # a line 1.8 m to the right, straight ahead: driving forward doesn't move it
    assert np.allclose(shift_poly((0.0, 0.0, 1.8), 5.0, 0.0), (0.0, 0.0, 1.8))
    # after turning left by 0.1 rad, a straight line now points right (b grows by 0.1)
    a, b, c = shift_poly((0.0, 0.0, 1.8), 0.0, -0.1)
    assert abs(b - 0.1) < 1e-9
    # on a bend, the part we've driven past drops out: c changes by b*d + a*d^2
    a, b, c = shift_poly((0.01, 0.05, 0.0), 2.0, 0.0)
    assert abs(c - (0.01 * 4 + 0.1)) < 1e-9 and abs(b - (0.04 + 0.05)) < 1e-9


def test_advance_lanes_moves_everything():
    L = Lanes(left=LaneLine((0, 0.02, -1.8), 50), right=LaneLine((0, 0.02, 1.9), 50),
              center=(0, 0.02, 0.05), conf=0.8, x_max=50)
    n = advance_lanes(L, 10.0, 0.0)
    assert abs(n.left.coeffs[2] - (-1.8 + 0.2)) < 1e-9 and n.x_max == 40
    assert abs(n.offset - (0.05 + 0.2)) < 1e-9


def test_scaler_4k_uses_slicing():
    big = np.zeros((2052, 3838, 3), np.uint8)
    pair = FramePair(big, 960, 1920)
    assert pair.fine.shape[1] == 1919 and pair.work.shape[1] in (959, 960)
    assert shrink_to(np.zeros((1080, 1920, 3), np.uint8), 960).shape[1] == 960


def test_parallel_engine_end_to_end():
    img = cv2.imread(str(SHOT))
    per = ParallelPerception(Settings(), use_ocr=False, lane_workers=1)
    try:
        p = defaults()
        w = None
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 25:
            pair = FramePair(img, 960, 1920)
            w = per.process(pair.work, p, t=time.perf_counter(), fine=pair.fine)
            if per.lane_res is not None and w.lanes.conf > 0.2 and w.signal.state.value == "red":
                break
            time.sleep(0.03)
        assert per.error == ""
        assert per.lane_res is not None and w.lanes.conf > 0.2
        assert w.signal.state.value == "red"
    finally:
        per.close()
