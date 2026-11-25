"""Calibrate the camera view.

    python -m fsd.tools.calibrate            # grabs the game screen after a 5 s countdown
    python -m fsd.tools.calibrate --image shot.png
    python -m fsd.tools.calibrate --manual   # click the lane lines yourself

Park or drive slowly in the MIDDLE of a STRAIGHT lane with the default chase
camera behind the car. Auto mode finds the lane lines; manual mode asks you to
click 4 points: bottom-left, top-left on the left line, then bottom-right,
top-right on the right line.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from ..perception.calibrate import apply, auto_calibrate, from_lines
from ..settings import Settings


def grab_frames(s: Settings, n: int = 20, countdown: int = 5) -> list[np.ndarray]:
    from ..capture import Capture
    for i in range(countdown, 0, -1):
        print(f"  switch to Roblox... {i}")
        time.sleep(1)
    cap = Capture(s.monitor, 30).start()
    frames, last = [], -1
    t0 = time.time()
    while len(frames) < n and time.time() - t0 < 10:
        f, _, seq = cap.latest()
        if f is not None and seq != last:
            frames.append(f.copy())
            last = seq
        time.sleep(0.05)
    cap.stop()
    return frames


def manual(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    pts: list[tuple[int, int]] = []
    H, W = frame.shape[:2]
    scale = min(1.0, 1400 / W)
    view = cv2.resize(frame, None, fx=scale, fy=scale)
    names = ["LEFT line, near the bottom", "LEFT line, far up", "RIGHT line, near the bottom", "RIGHT line, far up"]

    def on_click(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((int(x / scale), int(y / scale)))

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_click)
    while True:
        img = view.copy()
        for p in pts:
            cv2.circle(img, (int(p[0] * scale), int(p[1] * scale)), 6, (0, 255, 0), -1)
        msg = f"Click: {names[len(pts)]}" if len(pts) < 4 else "Enter = save, R = redo, Esc = cancel"
        cv2.putText(img, msg, (20, 40), 0, 1.0, (0, 255, 255), 2)
        cv2.imshow("calibrate", img)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            return None
        if k in (ord("r"), ord("R")):
            pts.clear()
        if k == 13 and len(pts) == 4:
            break
    cv2.destroyAllWindows()

    def line(p, q):
        a = (q[0] - p[0]) / (q[1] - p[1] + 1e-6)
        return np.array([a, p[0] - a * p[1]])
    return line(pts[0], pts[1]), line(pts[2], pts[3])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--manual", action="store_true")
    ap.add_argument("--lane-width", type=float, default=3.7)
    a = ap.parse_args()
    s = Settings.load()
    frames = [cv2.imread(a.image)] if a.image else grab_frames(s)
    if not frames or frames[0] is None:
        raise SystemExit("No image captured.")
    fields, msg = (None, "manual mode") if a.manual else auto_calibrate(frames, s.calibration, a.lane_width)
    print(msg)
    if fields is None:
        print("Opening manual mode...")
        lines = manual(frames[len(frames) // 2])
        if lines is None:
            raise SystemExit("Cancelled.")
        H, W = frames[0].shape[:2]
        fields, msg = from_lines(lines[0], lines[1], W, H, s.calibration, a.lane_width)
        print(msg)
        if fields is None:
            raise SystemExit("Calibration failed.")
    apply(s.calibration, fields)
    s.save()
    cv2.imwrite("data/calibration_frame.png", frames[len(frames) // 2])
    print("Saved to data/settings.json. Check it with:")
    print("  python -m fsd.tools.inspect_image data/calibration_frame.png")


if __name__ == "__main__":
    main()
