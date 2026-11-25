"""Checks against a REAL ER:LC screenshot (stopped at a red light, chase camera).

Needs the full install (YOLO + OCR); skipped when those aren't installed.
"""
from pathlib import Path

import cv2
import pytest

from fsd.params import defaults
from fsd.settings import Settings
from fsd.world import Light

SHOT = Path(__file__).parent / "assets" / "erlc_intersection_red.png"


@pytest.fixture(scope="module")
def world_and_per():
    pytest.importorskip("ultralytics")
    from fsd.perception import Perception
    s = Settings()
    per = Perception(s, use_ocr=False)
    if not per.detector.ok:
        pytest.skip(per.detector.error)
    img = cv2.imread(str(SHOT))
    w = None
    for i in range(3):
        w = per.process(img, defaults(), t=i * 0.05)
    return w, per


def test_red_lights_read_red(world_and_per):
    w, _ = world_and_per
    assert w.signal.state == Light.RED
    assert 10 < w.signal.distance < 60


def test_own_car_is_not_a_car_ahead(world_and_per):
    w, _ = world_and_per
    cars_ahead = [o for o in w.objects if o.kind == "car" and 0 < o.x < 40 and abs(o.y) < 3]
    assert not cars_ahead


def test_crosswalk_right_ahead(world_and_per):
    w, _ = world_and_per
    assert w.crosswalk_dist < 10


def test_speedometer_text():
    pytest.importorskip("rapidocr_onnxruntime")
    from fsd.perception.ocr import OcrEngine, find_speedo
    e = OcrEngine()
    img = cv2.imread(str(SHOT))
    found = find_speedo(e, img)
    assert found is not None and found[1] == "kph"
    x1, y1, x2, y2 = found[0]
    H, W = img.shape[:2]
    texts = [t for t, _, _ in e.read(img[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)])]
    assert any(t.replace(" ", "").upper().startswith("0KM") for t in texts)
