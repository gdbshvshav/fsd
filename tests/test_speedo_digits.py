"""The fast speedometer reader: learns digit shapes from OCR, then reads every frame."""
from pathlib import Path

import cv2
import numpy as np

from fsd.perception.speedo_digits import DigitReader
from fsd.sim.render import Renderer

ASSETS = Path(__file__).parent / "assets"


def _hud(v):
    img = np.full((1024, 1920, 3), (90, 95, 100), np.uint8)
    Renderer._hud(None, img, v)
    return img[int(0.78 * 1024):int(0.95 * 1024), int(0.40 * 1920):int(0.60 * 1920)]


def test_reads_every_speed_after_learning_a_few():
    d = DigitReader()
    for v in (127, 345, 689, 0, 50):
        assert d.learn(_hud(v), str(v))
    for v in range(0, 260):
        got = d.read(_hud(v))
        assert got is not None and got[0] == v


def test_unknown_digit_is_not_guessed():
    d = DigitReader()
    d.learn(_hud(11), "11")
    assert d.read(_hud(18)) is None          # never seen an 8: let OCR answer instead


def _real(name):
    im = cv2.imread(str(ASSETS / name))
    H, W = im.shape[:2]
    return im[int(0.78 * H):int(0.95 * H), int(0.42 * W):int(0.58 * W)]


def test_real_erlc_speedometer_any_resolution():
    d = DigitReader()
    assert d.learn(_real("real_highway_0kmh.png"), "0")       # 2576 px wide screenshot
    assert d.learn(_real("real_ramp_5kmh.png"), "5")
    assert d.read(_real("erlc_intersection_red.png"))[0] == 0  # 1920 px wide screenshot
    assert d.read(_real("real_ramp_5kmh.png"))[0] == 5
