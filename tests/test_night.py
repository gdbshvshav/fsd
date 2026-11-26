"""ER:LC at night: the picture is brightened for the lane finder, so lines aren't lost."""
import copy

import numpy as np

from fsd.fast.pipeline import FastPerception
from fsd.fast.scaler import FramePair
from fsd.perception.exposure import AutoExposure
from fsd.sim.runner import SimRun
from fsd.sim.scenarios import SCENARIOS


def _lane_conf(frames, settings, p, dark, cap):
    per = FastPerception(copy.deepcopy(settings), use_detector=False, use_ocr=False, live=False)
    per.exposure.max_gain = cap
    confs = []
    for i, f in enumerate(frames):
        d = (f.astype(np.float32) * dark).astype(np.uint8)
        pair = FramePair(d, 960, 1920)
        confs.append(per.process(pair.work, p, t=i / 30, fine=pair.fine).lanes.conf)
    return float(np.mean(confs[10:])), per.exposure


def test_night_lanes_found_with_auto_exposure():
    r = SimRun(SCENARIOS["highway_straight"])
    frames = []
    while r.t < 1.2:
        frames.append(r.step().copy())
    night, exp = _lane_conf(frames, r.settings, r.p, 0.18, 3.5)
    blind, _ = _lane_conf(frames, r.settings, r.p, 0.18, 1.0)
    assert blind < 0.2 and night > 0.6 and exp.night


def test_daylight_untouched():
    e = AutoExposure()
    img = np.full((100, 100, 3), 120, np.uint8)
    for _ in range(20):
        e.update(img, (10, 10, 90, 90))
    assert e.gain == 1.0 and e.apply(img) is img
