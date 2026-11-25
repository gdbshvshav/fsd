"""Run perception on a screenshot (or folder) and save an annotated picture.

    python -m fsd.tools.inspect_image shot.png [--no-detector] [--out out.jpg]

Great for checking calibration: lane lines should come out parallel and
straight in the bird's-eye panel on the right.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..params import defaults
from ..perception import Perception
from ..settings import Settings
from ..ui.draw_debug import annotate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-detector", action="store_true")
    ap.add_argument("--quad", default=None, help='JSON [[x,y]*4] fractions: bl, br, tr, tl')
    ap.add_argument("--qw", type=float, default=None, help="trapezoid width in m")
    ap.add_argument("--qd", type=float, default=None, help="trapezoid depth in m")
    ap.add_argument("--no-ego-mask", action="store_true", help="screenshot has no car in it")
    ap.add_argument("--repeat", type=int, default=3, help="run N times (lets smoothing settle)")
    a = ap.parse_args()

    s = Settings.load()
    if a.quad:
        s.calibration.src_quad = json.loads(a.quad)
    if a.qw:
        s.calibration.quad_width_m = a.qw
    if a.qd:
        s.calibration.quad_depth_m = a.qd
    if a.no_ego_mask:
        s.calibration.ego_mask = [0, 0, 0, 0]
    per = Perception(s, use_detector=not a.no_detector, use_ocr=False)
    p = defaults()
    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"can't read {a.image}")
    w = None
    for i in range(a.repeat):
        w = per.process(img, p, t=i * 0.05)
    out = annotate(img, w, per)
    dst = a.out or str(Path(a.image).with_suffix("")) + "_fsd.jpg"
    cv2.imwrite(dst, out)
    ln = w.lanes
    print(f"saved {dst}")
    print(f"lanes conf={ln.conf:.2f} width={ln.width_m:.2f} offset={ln.offset:+.2f} "
          f"heading={np.degrees(ln.heading):+.1f}deg curv={ln.curvature:+.4f}")
    for side in ("left", "right", "left2", "right2"):
        l = getattr(ln, side)
        if l:
            print(f"  {side}: y@6m={l.y_at(6):+.2f} {l.color} {'dashed' if l.dashed else 'solid'} conf={l.conf:.2f} seen to {l.x_max:.0f}m")
    print(f"signal={w.signal.state.value} d={w.signal.distance:.0f} stop_line={w.stop_line_dist:.1f} "
          f"crosswalk={w.crosswalk_dist:.1f} stop_sign={w.stop_sign_dist:.1f}")
    for o in w.objects:
        print(f"  {o.kind:10s} score={o.score:.2f} x={o.x:6.1f} y={o.y:+5.1f} {o.extra}")
    print("timings", {k: f"{v*1000:.1f}ms" for k, v in w.timings.items()})


if __name__ == "__main__":
    main()
