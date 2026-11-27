"""Developer view: exactly what FSD sees and decides, drawn on the game picture.

Press D on the dashboard to switch to it (D again goes back). The same picture is
what gets sent to Discord with every report (F10 / takeovers / health checks), so
what you see here is what I see when you send a report.

Drawn:
  * paint pixels FSD found (magenta = white paint, orange = yellow paint)
  * every lane-line candidate it linked (thin), the two it chose for YOUR lane (thick)
  * the fitted lane lines it drives on (cyan = left, green = right, white = centre)
  * the calibration trapezoid (green; RED + "BAD CALIBRATION" if it is broken)
  * the horizon (yellow dashed)
  * your car's box (purple) and where it normally sits (purple dashed)
  * road edges (white), stop line / crosswalk bars
  * every detected object with its kind, distance and speed
  * the planned path (pink) and any turn preview (blue)
  * a text panel with the numbers: fps, lane confidence, offset, heading, steering
    command and what the car is doing and why.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

KIND_COL = {"car": (255, 170, 60), "person": (60, 200, 255), "cone": (0, 140, 255), "stop_sign": (40, 40, 230),
            "speed_sign": (230, 230, 230), "emergency": (255, 60, 255), "obstacle": (0, 200, 255)}
LIGHT_COL = {"red": (40, 40, 240), "yellow": (0, 210, 255), "green": (60, 220, 60)}


from ..settings import quad_problem  # noqa: E402


def _dash_line(img, p0, p1, col, thick=1, dash=10):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    n = max(1, int(np.linalg.norm(p1 - p0) / dash))
    for i in range(0, n, 2):
        a = p0 + (p1 - p0) * i / n
        b = p0 + (p1 - p0) * min(n, i + 1) / n
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), col, thick, cv2.LINE_AA)


def _dash_rect(img, box, col, thick=1):
    x1, y1, x2, y2 = box
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        _dash_line(img, a, b, col, thick, 8)


def render(work: np.ndarray, w, per, plan=None, info: dict | None = None, out_w: int | None = None,
           panel: bool = True) -> np.ndarray:
    """work: the ~960 px game picture FSD processed (BGR). Returns a BGR picture of
    width out_w (default: the work picture's own width) with everything drawn on it."""
    info = info or {}
    H0, W0 = work.shape[:2]
    out_w = out_w or W0
    s = out_w / W0
    out_h = int(round(H0 * s))
    base = cv2.resize(work, (out_w, out_h), interpolation=cv2.INTER_LINEAR) if s != 1 else work.copy()
    out = (base.astype(np.float32) * 0.62).astype(np.uint8)        # dim a bit so drawings pop
    th = max(1, int(round(out_w / 640)))
    fs = 0.42 * max(0.8, out_w / 960)
    geo = getattr(per, "geo", None)

    def S(x, y):
        return int(round(x * s)), int(round(y * s))

    def to_px(pts_vehicle):
        if geo is None:
            return np.zeros((0, 2), np.int32)
        p = geo.vehicle_to_screen(np.asarray(pts_vehicle, np.float32))
        keep = np.isfinite(p).all(1) & (p[:, 1] > geo.horizon_y + 1) & (p[:, 1] < H0 + 2) \
            & (p[:, 0] > -W0) & (p[:, 0] < 2 * W0)
        return (p[keep] * s).astype(np.int32)

    def poly(coeffs, x0, x1, col, t, n=30):
        xs = np.linspace(x0, x1, n)
        a, b, c = coeffs
        pts = to_px(np.stack([xs, a * xs * xs + b * xs + c], 1))
        if len(pts) > 1:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, col, t, cv2.LINE_AA)
        return pts

    res = getattr(per, "last_screen", None)
    # 1) paint pixels
    if res is not None and getattr(res, "masks", None) is not None:
        pm = res.masks
        if pm.paint.shape[:2] == (H0, W0):
            paint = cv2.resize(pm.paint, (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
            yel = cv2.resize(pm.yellow, (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0 \
                if getattr(pm, "yellow", None) is not None else np.zeros_like(paint)
            out[paint & ~yel] = (255, 60, 255)
            out[paint & yel] = (0, 150, 255)
    # 2) candidate chains + the two chosen
    if res is not None:
        for pts, ego, yel in getattr(res, "chains_screen", []) or []:
            if len(pts) < 2:
                continue
            p = (pts.astype(np.float32) * s).astype(np.int32)
            p = p[np.argsort(p[:, 1])]
            col = (0, 220, 255) if yel else (200, 200, 200)
            cv2.polylines(out, [p.reshape(-1, 1, 2)], False, col, th * (3 if ego else 1), cv2.LINE_AA)
    # 3) calibration trapezoid + horizon
    cal = getattr(per, "cal", None)
    bad = ""
    if cal is not None:
        bad = quad_problem(cal.src_quad)
        q = (np.array(cal.quad_px(W0, H0), np.float32) * s).astype(np.int32)
        cv2.polylines(out, [q.reshape(-1, 1, 2)], True, (40, 40, 255) if bad else (60, 230, 60), th, cv2.LINE_AA)
    if geo is not None:
        hy = geo.horizon_y * s
        if -out_h < hy < 2 * out_h:
            _dash_line(out, (0, hy), (out_w, hy), (0, 230, 230), th, 14)
            cv2.putText(out, "horizon", (6, int(hy) - 4), 0, fs, (0, 230, 230), 1, cv2.LINE_AA)
    # 4) road edges, lanes, centre
    if w is not None and geo is not None:
        e = getattr(w, "road_edges", None)
        if e is not None and getattr(e, "ok", False):
            poly(e.left, 0, 45, (235, 235, 235), th)
            poly(e.right, 0, 45, (235, 235, 235), th)
        ln = w.lanes
        for side, col in (("left2", (140, 110, 0)), ("right2", (0, 110, 40)), ("left", (255, 220, 0)),
                          ("right", (60, 255, 60))):
            l = getattr(ln, side, None)
            if l is not None:
                pts = poly(l.coeffs, -12, max(6.0, min(70.0, l.x_max)), col, th * (3 if side in ("left", "right") else 2))
                if len(pts):
                    tip = pts[np.argmin(pts[:, 1])]
                    cv2.putText(out, f"{side} {l.coeffs[2]:+.1f}m {l.color[0]}{' dash' if l.dashed else ''} "
                                f"c{l.conf:.2f}", (int(tip[0]) + 4, int(tip[1]) + 10), 0, fs * 0.9, col, 1,
                                cv2.LINE_AA)
        if ln.conf > 0:
            poly(ln.center, -12, 50, (255, 255, 255), th)
        for d, col, label in ((getattr(w, "stop_line_dist", 999), (0, 0, 255), "STOP"),
                              (getattr(w, "crosswalk_dist", 999), (255, 255, 0), "XWALK")):
            if d < 120:
                c = ln.center[2] if ln.conf > 0 else 0.0
                p = to_px([[d, c - 2.5], [d, c + 2.5]])
                if len(p) == 2:
                    cv2.line(out, tuple(p[0]), tuple(p[1]), col, th * 2)
                    cv2.putText(out, f"{label} {d:.0f}m", (int(p[1][0]) + 5, int(p[1][1])), 0, fs, col, 1,
                                cv2.LINE_AA)
    # 5) planned path + turn preview
    if plan is not None and geo is not None:
        if getattr(plan, "path", None):
            pts = to_px(np.array(plan.path, np.float32))
            if len(pts) > 1:
                cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (200, 80, 255), th * 2, cv2.LINE_AA)
        if getattr(plan, "turn_preview", None):
            pts = to_px(np.array(plan.turn_preview, np.float32))
            if len(pts) > 1:
                cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (255, 150, 50), th * 2, cv2.LINE_AA)
    # 6) ego box (now / usual)
    eb = getattr(per, "ego_box", None)
    eh = getattr(per, "ego_home", None)
    if eh is not None:
        _dash_rect(out, (*S(eh[0], eh[1]), *S(eh[2], eh[3])), (255, 0, 200), th)
    if eb is not None:
        a, b = S(eb[0], eb[1])
        c, d = S(eb[2], eb[3])
        cv2.rectangle(out, (a, b), (c, d), (255, 0, 255), th)
        cv2.putText(out, "you", (a + 2, b - 4), 0, fs, (255, 0, 255), 1, cv2.LINE_AA)
    # 7) objects
    if w is not None:
        for o in w.objects:
            x1, y1, x2, y2 = o.box
            if o.kind == "light":
                stt = o.extra.get("state", "?")
                col = LIGHT_COL.get(stt, (180, 180, 180))
                label = f"light {stt} {o.x:.0f}m"
            else:
                col = KIND_COL.get(o.kind, (200, 200, 200))
                label = f"{o.kind} {o.x:.0f}m y{o.y:+.1f}" + (f" {o.vx:+.0f}m/s" if o.kind == "car" else "")
                if o.kind == "speed_sign" and o.extra.get("value"):
                    label += f" ={o.extra['value']}"
            a, b = S(x1, y1)
            c, d = S(x2, y2)
            cv2.rectangle(out, (a, b), (c, d), col, th)
            cv2.putText(out, label, (a, max(10, b - 4)), 0, fs, col, 1, cv2.LINE_AA)
        if plan is not None and getattr(plan, "lead", None) is not None:
            x1, y1, x2, y2 = plan.lead.box
            a, b = S(x1, y1)
            c, d = S(x2, y2)
            cv2.rectangle(out, (a - 3, b - 3), (c + 3, d + 3), (0, 0, 255), th * 2)
            cv2.putText(out, "LEAD", (a, d + 14), 0, fs, (0, 0, 255), 1, cv2.LINE_AA)
    if bad:
        msg = f"BAD CALIBRATION: {bad}  (F4 on a straight lane, or it resets itself)"
        cv2.putText(out, msg, (10, out_h - 12), 0, fs * 1.2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, msg, (10, out_h - 12), 0, fs * 1.2, (60, 60, 255), 1, cv2.LINE_AA)
    if panel:
        out = _with_panel(out, w, plan, info, fs, th)
    return out


def _fmt(v, f="{:.2f}"):
    try:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f.format(v)
        return str(v)
    except Exception:  # noqa: BLE001
        return "?"


def info_lines(w, plan, info: dict) -> list[str]:
    rows: list[str] = []
    if w is not None:
        ln = w.lanes
        rows += [
            f"lanes conf {ln.conf:.2f}  width {ln.width_m:.2f} m  offset {ln.offset:+.2f} m",
            f"heading {math.degrees(getattr(ln, 'heading', 0.0)):+.1f} deg  curv {getattr(ln, 'curvature', 0.0):+.4f}",
            f"L {('%s %+.2f c%.2f' % (ln.left.color, ln.left.coeffs[2], ln.left.conf)) if ln.left else 'none'}   "
            f"R {('%s %+.2f c%.2f' % (ln.right.color, ln.right.coeffs[2], ln.right.conf)) if ln.right else 'none'}",
            f"speed {w.speed_kph:.0f} kph  yaw {math.degrees(getattr(w, 'yaw_rate', 0.0)):+.1f} deg/s",
            f"light {w.signal.state.value} {w.signal.distance:.0f} m  stop {getattr(w, 'stop_line_dist', 999):.0f}  "
            f"objects {len(w.objects)}",
        ]
    if plan is not None:
        rows += [f"{plan.mode.value.upper()}  target {plan.target_speed_kph:.0f} kph",
                 f"why: {plan.reason}"[:90]]
    for k in ("fps", "lanes_hz", "steer", "steer_cmd", "pedal", "engaged", "cal", "ego", "note"):
        if k in info:
            rows.append(f"{k}: {_fmt(info[k])}"[:90])
    return rows


def _with_panel(img, w, plan, info, fs, th):
    rows = info_lines(w, plan, info)
    if not rows:
        return img
    lh = int(22 * fs / 0.42)
    hgt = lh * len(rows) + 10
    wid = int(min(img.shape[1] * 0.62, 640 * fs / 0.42))
    pan = img[4:4 + hgt, 4:4 + wid]
    pan[:] = (pan.astype(np.float32) * 0.35).astype(np.uint8)
    for i, r in enumerate(rows):
        cv2.putText(img, r, (10, 4 + lh * (i + 1) - 4), 0, fs, (240, 240, 240), 1, cv2.LINE_AA)
    return img
