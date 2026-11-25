"""Run the 60 fps vision on a screenshot (or a folder of them) and save a picture
of everything it found.

    python -m fsd.tools.inspect_image shot.png [--out out.jpg] [--yolo]
    python -m fsd.tools.inspect_image folder_of_shots/

Shows: lane lines it's using (thick), other paint lines (thin), the lane centre,
stop bars / crosswalks, traffic lights with their state, obstacles, your car's
exact outline, and timings. Text output lists the numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..fast.pipeline import FastPerception
from ..fast.scaler import FramePair
from ..params import defaults
from ..settings import Settings

STATE_COL = {"red": (60, 60, 255), "yellow": (0, 210, 255), "green": (90, 230, 90)}


def draw(frame: np.ndarray, w, per: FastPerception) -> np.ndarray:
    out = (frame.astype(np.float32) * 0.55).astype(np.uint8)
    geo = per.geo
    res = per.last_screen
    # your car
    if per.ego_shape.mask is not None:
        m = np.zeros(frame.shape[:2], np.uint8)
        per.ego_shape.paste(m, 1.0, 1.0)
        out[m > 0] = frame[m > 0]
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 255), 1, cv2.LINE_AA)
    # paint chains
    if res is not None:
        for pts, ego, yel in res.chains_screen:
            order = np.argsort(pts[:, 1])
            col = (0, 220, 255) if yel else (255, 255, 255)
            cv2.polylines(out, [pts[order].reshape(-1, 1, 2)], False, col if ego else (150, 150, 150),
                          3 if ego else 1, cv2.LINE_AA)
    ln = w.lanes
    if ln.conf > 0:
        xs = np.linspace(1, min(60, max(10, ln.x_max)), 30)
        a, b, c = ln.center
        p = geo.vehicle_to_screen(np.stack([xs, a * xs * xs + b * xs + c], 1))
        p = p[(p[:, 1] > geo.horizon_y) & (p[:, 1] < frame.shape[0])]
        if len(p) > 1:
            cv2.polylines(out, [p.astype(np.int32)], False, (255, 0, 255), 2, cv2.LINE_AA)
    for d, col, name in ((w.stop_line_dist, (0, 0, 255), "STOP"), (w.crosswalk_dist, (0, 180, 255), "CROSSWALK")):
        if d < 80:
            c0 = ln.center[2] if ln.conf > 0 else 0.0
            p = geo.vehicle_to_screen(np.array([[d, c0 - 2], [d, c0 + 2]], np.float32)).astype(int)
            cv2.line(out, tuple(p[0]), tuple(p[1]), col, 3)
            cv2.putText(out, f"{name} {d:.0f}m", tuple(p[1] + [6, 0]), 0, 0.45, col, 1, cv2.LINE_AA)
    for o in w.objects:
        x1, y1, x2, y2 = o.box
        if o.kind == "light":
            col = STATE_COL.get(o.extra.get("state"), (160, 160, 160))
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            continue
        col = (255, 170, 60) if o.kind in ("car", "obstacle") else (60, 170, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{o.kind} {o.x:.0f}m", (x1, max(12, y1 - 4)), 0, 0.45, col, 1, cv2.LINE_AA)
    s = w.signal
    txt = [f"lane conf {ln.conf:.2f} width {ln.width_m:.1f} m offset {ln.offset:+.2f} m curve {ln.curvature:+.4f}",
           f"light {s.state.value} {s.distance:.0f} m   stop line {w.stop_line_dist:.0f}   crosswalk {w.crosswalk_dist:.0f}",
           f"{w.timings.get('total', 0) * 1000:.1f} ms total"]
    for i, t in enumerate(txt):
        cv2.putText(out, t, (8, 18 + 18 * i), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run_one(path: Path, per: FastPerception, out_path: Path | None) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"can't read {path}")
    pair = FramePair(img)
    per.geo = None
    p = defaults()
    w = None
    for i in range(6):                 # a few frames so smoothing and the car outline settle
        w = per.process(pair.work, p, t=i / 30, fine=pair.fine)
    vis = draw(pair.work, w, per)
    dst = out_path or path.with_name(path.stem + "_fsd.jpg")
    cv2.imwrite(str(dst), vis)
    ln = w.lanes
    return {
        "image": path.name, "saved": str(dst),
        "lane_conf": round(ln.conf, 2), "lane_width_m": round(ln.width_m, 2), "offset_m": round(ln.offset, 2),
        "left": ln.left and {"y8": round(ln.left.y_at(8), 2), "color": ln.left.color, "dashed": ln.left.dashed},
        "right": ln.right and {"y8": round(ln.right.y_at(8), 2), "color": ln.right.color, "dashed": ln.right.dashed},
        "light": w.signal.state.value, "light_m": round(min(w.signal.distance, 999), 1),
        "stop_line_m": round(min(w.stop_line_dist, 999), 1), "crosswalk_m": round(min(w.crosswalk_dist, 999), 1),
        "objects": [(o.kind, round(o.x, 1), round(o.y, 1)) for o in w.objects if o.kind != "light"],
        "ms": {k: round(v * 1000, 1) for k, v in w.timings.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="screenshot or folder")
    ap.add_argument("--out", default=None)
    ap.add_argument("--yolo", action="store_true", help="also run YOLO (slow without a GPU)")
    a = ap.parse_args()
    s = Settings.load()
    per = FastPerception(s, use_detector=a.yolo, use_ocr=False, live=False)
    src = Path(a.image)
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")) if src.is_dir() else [src]
    for f in files:
        if f.stem.endswith("_fsd"):
            continue
        info = run_one(f, per, Path(a.out) if (a.out and len(files) == 1) else None)
        print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
