"""OpenCV debug drawing: what perception saw, on the camera image + bird's-eye panel."""
from __future__ import annotations

import cv2
import numpy as np

COL = {"red": (40, 40, 240), "yellow": (0, 210, 255), "green": (60, 220, 60), "flashing_red": (40, 40, 240),
       "off": (150, 150, 150), "back": (90, 90, 90), "unknown": (200, 200, 200), "none": (200, 200, 200)}
KIND_COL = {"car": (255, 170, 60), "person": (60, 200, 255), "cone": (0, 140, 255), "stop_sign": (40, 40, 230),
            "speed_sign": (230, 230, 230), "emergency": (255, 60, 255)}


def annotate(img: np.ndarray, w, per, plan=None) -> np.ndarray:
    geo = per.geo
    out = img.copy()
    H, W = out.shape[:2]
    ln = w.lanes

    def draw_poly(coeffs, color, x0=4.0, x1=45.0, thick=4):
        xs = np.linspace(x0, x1, 24)
        a, b, c = coeffs
        pts = geo.vehicle_to_screen(np.stack([xs, a * xs * xs + b * xs + c], 1))
        pts = pts[(pts[:, 1] > geo.horizon_y) & (pts[:, 1] < H)]
        if len(pts) > 1:
            cv2.polylines(out, [pts.astype(np.int32)], False, color, thick, cv2.LINE_AA)

    for side, col in (("left", (255, 120, 0)), ("right", (255, 120, 0)), ("left2", (120, 60, 0)), ("right2", (120, 60, 0))):
        l = getattr(ln, side)
        if l:
            draw_poly(l.coeffs, (0, 220, 255) if l.color == "yellow" else col, x1=min(55, l.x_max))
    if ln.conf > 0:
        draw_poly(ln.center, (255, 255, 255), thick=2)
    if plan is not None and plan.path:
        pts = geo.vehicle_to_screen(np.array(plan.path, dtype=np.float32))
        pts = pts[(pts[:, 1] > geo.horizon_y) & (pts[:, 1] < H)]
        if len(pts) > 1:
            cv2.polylines(out, [pts.astype(np.int32)], False, (255, 80, 200), 6, cv2.LINE_AA)
    for d, col, label in ((w.stop_line_dist, (0, 0, 255), "STOP LINE"), (w.crosswalk_dist, (255, 255, 0), "CROSSWALK")):
        if d < 900:
            p = geo.vehicle_to_screen(np.array([[d, ln.center[2] - 2], [d, ln.center[2] + 2]]))
            cv2.line(out, tuple(p[0].astype(int)), tuple(p[1].astype(int)), col, 4)
            cv2.putText(out, f"{label} {d:.0f}m", tuple(p[1].astype(int) + [8, 0]), 0, 0.6, col, 2)

    for o in w.objects:
        x1, y1, x2, y2 = o.box
        if o.kind == "light":
            st = o.extra.get("state", "unknown")
            col = COL.get(st, (200, 200, 200))
            label = f"{st} {o.x:.0f}m"
        else:
            col = KIND_COL.get(o.kind, (200, 200, 200))
            label = f"{o.kind} {o.x:.0f}m" + (f" {o.vx:+.0f}m/s" if o.kind == "car" else "")
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, label, (x1, max(12, y1 - 4)), 0, 0.5, col, 2)

    s = w.signal
    cv2.putText(out, f"LIGHT: {s.state.value} {s.distance:.0f}m  conf {s.conf:.2f}", (10, 30), 0, 0.8,
                COL.get(s.state.value, (255, 255, 255)), 2)
    cv2.putText(out, f"LANE conf {ln.conf:.2f} off {ln.offset:+.2f}m  speed {w.speed_kph:.0f} kph", (10, 60),
                0, 0.7, (255, 255, 255), 2)
    if plan is not None:
        cv2.putText(out, f"{plan.mode.value.upper()}  target {plan.target_speed_kph:.0f}  {plan.reason}", (10, 90),
                    0, 0.7, (255, 80, 200), 2)
    q = np.array(per.cal.src_quad) * [W, H]
    cv2.polylines(out, [q.astype(np.int32)], True, (0, 255, 0), 1)

    # bird's-eye panel
    dbg = per.lanes.debug
    if dbg.bev is not None:
        bev = dbg.bev.copy()
        m = dbg.mask > 0
        bev[m] = (0, 0, 255)
        for c in dbg.candidates or []:
            xs = np.linspace(4, c.x_max, 20)
            uv = geo.vehicle_to_bev(np.stack([xs, [c.y_at(x) for x in xs]], 1))
            cv2.polylines(bev, [uv.astype(np.int32)], False, (0, 255, 0), 2)
        scale = H / bev.shape[0]
        bev = cv2.resize(bev, (int(bev.shape[1] * scale), H))
        out = np.hstack([out, bev])
    return out
