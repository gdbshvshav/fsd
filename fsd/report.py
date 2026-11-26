"""Debug reports you can send me (Claude) - and that go to Discord by themselves.

Press F10 any time something looks wrong (or it happens by itself on a takeover,
a disengage, a bad health check or an error). A zip is written to data/reports/
and posted to your Discord webhook with the developer-view picture on top.

Upload that zip in the chat: it has everything needed to see what went wrong.

  summary.txt        what FSD thought, in plain numbers (read this first)
  now_dev.jpg        developer view of the newest frame (lines, boxes, path, numbers)
  dev_XX.jpg         developer view once a second for the last ~12 s
  raw_XX.jpg         the plain game picture at the same moments
  telemetry.csv      every control step of the last ~30 s: lanes, speed (all sources),
                     steering wanted / sent, pedal, plan mode and reason
  settings.json      your settings incl. camera calibration (no keys - those stay in .env)
  params.json        driving parameters in use
  status.json        engine status (fps per stage, speedometer reader, input device...)
  learned.json       what was learned about your car
  errors.log         last part of the error log

Nothing here touches .env.
"""
from __future__ import annotations

import csv
import io
import json
import math
import threading
import time
import zipfile
from collections import deque
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "data" / "reports"


def _num(v):
    if isinstance(v, (bool, int, str)) or v is None:
        return v
    try:
        f = float(v)
        return round(f, 4) if math.isfinite(f) else None
    except Exception:  # noqa: BLE001
        return str(v)


def telemetry_row(now, w, plan, steer, pedal, actuator, driver=None) -> dict:
    ln = w.lanes
    r = {
        "t": now,
        "lanes_conf": ln.conf, "lane_off": ln.offset, "lane_head_deg": math.degrees(getattr(ln, "heading", 0.0)),
        "lane_curv": getattr(ln, "curvature", 0.0), "lane_w": ln.width_m,
        "L": ln.left.coeffs[2] if ln.left else None, "L_conf": ln.left.conf if ln.left else None,
        "R": ln.right.coeffs[2] if ln.right else None, "R_conf": ln.right.conf if ln.right else None,
        "speed_kph": w.speed_kph, "speed_valid": w.speed_valid,
        "yaw_dps": math.degrees(getattr(w, "yaw_rate", 0.0)),
        "vo_v": getattr(getattr(w, "vo", None), "v", None), "vo_ok": getattr(getattr(w, "vo", None), "ok", None),
        "light": w.signal.state.value, "light_d": w.signal.distance,
        "stop_d": getattr(w, "stop_line_dist", 999), "xwalk_d": getattr(w, "crosswalk_dist", 999),
        "n_obj": len(w.objects),
        "mode": plan.mode.value if plan is not None else "", "target_kph": plan.target_speed_kph if plan else None,
        "reason": (plan.reason if plan is not None else "")[:80],
        "steer_cmd": steer, "pedal_cmd": pedal,
        "steer_game_est": getattr(getattr(actuator, "steer_obs", None), "s", None),
        "keys": " ".join(getattr(actuator, "pressed_log", []) or []),
        "stale": getattr(actuator, "stale", None),
    }
    if driver is not None:
        try:
            st = driver.status()
            for k in ("v_est", "yaw_est", "cam_lag", "visual_speed", "e_look", "psi"):
                if k in st:
                    r["drv_" + k] = st[k]
        except Exception:  # noqa: BLE001
            pass
    return {k: _num(v) for k, v in r.items()}


class Reporter:
    """Keeps the last ~30 s of telemetry and a picture a second, writes report zips."""

    def __init__(self, app):
        self.app = app
        self.rows: deque = deque(maxlen=60 * 30)
        self.shots: deque = deque(maxlen=12)        # (t, raw_jpg, dev_jpg)
        self._last_shot = 0.0
        self._busy = threading.Lock()
        self.last_path: Path | None = None
        self._last_auto = 0.0

    # ---------------------------------------------------------------- recording (control loop)
    def add(self, now, w, plan, steer, pedal) -> None:
        try:
            self.rows.append(telemetry_row(now, w, plan, steer, pedal, self.app.actuator,
                                           getattr(self.app, "driver", None)))
        except Exception:  # noqa: BLE001
            pass
        if now - self._last_shot >= 1.0:
            self._last_shot = now
            work = getattr(self.app, "_last_work", None)
            if work is not None:
                threading.Thread(target=self._shot, args=(now, work.copy(), w, plan), daemon=True).start()

    def _shot(self, now, work, w, plan) -> None:
        try:
            raw = cv2.imencode(".jpg", cv2.resize(work, (640, int(work.shape[0] * 640 / work.shape[1]))),
                               [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            dev = self.dev_image(work, w, plan, width=960)
            devj = cv2.imencode(".jpg", dev, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
            self.shots.append((now, raw, devj))
        except Exception:  # noqa: BLE001
            pass

    def dev_image(self, work, w, plan, width=1280):
        from .ui.devview import render
        return render(work, w, self.app.perception, plan, self.info(), out_w=width)

    def info(self) -> dict:
        a = self.app
        st = {}
        try:
            st = a.perception.status()
        except Exception:  # noqa: BLE001
            pass
        act = a.actuator
        return {
            "fps": float(getattr(a, "fps", 0.0)),
            "lanes_hz": float(st.get("lanes_fps", 0.0) or 0.0),
            "steer": f"wanted {getattr(act, 'steer', 0):+.2f} game {getattr(act.steer_obs, 's', 0):+.2f}",
            "pedal": f"{getattr(act, 'pedal', 0):+.2f} via {getattr(act, 'kind', 'keyboard')}",
            "engaged": bool(getattr(a, "engaged", False)),
            "note": f"speed digits {st.get('speedo_digits', '?')}/10, speed age {st.get('speed_age_ms', -1):.0f} ms"
            if isinstance(st.get("speed_age_ms"), (int, float)) else "",
        }

    # ---------------------------------------------------------------- writing a report
    def report(self, reason: str, note: str = "", send: bool = True, auto: bool = False) -> Path | None:
        """Write the zip now (in the background) and post it to Discord."""
        now = time.time()
        if auto and now - self._last_auto < 20:
            return None
        if auto:
            self._last_auto = now
        if not self._busy.acquire(blocking=False):
            return None
        snap = self.app.snapshot()
        work = getattr(self.app, "_last_work", None)
        rows = list(self.rows)
        shots = list(self.shots)
        threading.Thread(target=self._write, args=(reason, note, send, snap, None if work is None else work.copy(),
                                                   rows, shots), daemon=True).start()
        return None

    def _write(self, reason, note, send, snap, work, rows, shots) -> None:
        try:
            path = self._build(reason, note, snap, work, rows, shots)
            self.last_path = path
            try:
                self.app.learner.say(f"Report saved: data/reports/{path.name} - upload it to Claude in the chat.")
            except Exception:  # noqa: BLE001
                pass
            if send:
                self._post(reason, note, path, snap)
        except Exception:  # noqa: BLE001
            try:
                from .app import log_error
                log_error("report")
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._busy.release()

    def _build(self, reason, note, snap, work, rows, shots) -> Path:
        REPORTS.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe = "".join(c if c.isalnum() else "_" for c in reason)[:30]
        path = REPORTS / f"{stamp}_{safe}.zip"
        a = self.app
        w, plan = snap.world, snap.plan
        summary = self.summary(reason, note, w, plan, rows)
        now_dev = None
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("summary.txt", summary)
            if work is not None and w is not None:
                now_dev = self.dev_image(work, w, plan, width=1280)
                z.writestr("now_dev.jpg", cv2.imencode(".jpg", now_dev, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
                z.writestr("now_raw.png", cv2.imencode(".png", work)[1].tobytes())
            t_end = shots[-1][0] if shots else 0
            for t, raw, dev in shots:
                k = f"{t - t_end:+05.0f}s".replace("+", "p").replace("-", "m")
                z.writestr(f"dev_{k}.jpg", dev)
                z.writestr(f"raw_{k}.jpg", raw)
            if rows:
                buf = io.StringIO()
                keys = list(rows[-1].keys())
                wr = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
                wr.writeheader()
                t0 = rows[-1]["t"] or 0
                for r in rows:
                    r = dict(r)
                    r["t"] = round((r["t"] or 0) - t0, 3)
                    wr.writerow(r)
                z.writestr("telemetry.csv", buf.getvalue())
            try:
                z.writestr("settings.json", json.dumps(asdict(a.settings), indent=1, default=str))
            except Exception:  # noqa: BLE001
                pass
            try:
                z.writestr("params.json", json.dumps(a.store.snapshot(), indent=1, default=str))
            except Exception:  # noqa: BLE001
                pass
            try:
                st = dict(a.perception.status())
                st["input"] = getattr(a.actuator, "kind", "keyboard")
                st["fps"] = getattr(a, "fps", 0.0)
                st["proc_size"] = getattr(a, "proc_size", None)
                st["game_rect"] = getattr(a, "game_rect", None)
                st["ui_rect"] = getattr(a, "ui_rect", None)
                st["ego_box"] = getattr(a.perception, "ego_box", None)
                st["ego_home"] = getattr(a.perception, "ego_home", None)
                st["version"] = _version()
                z.writestr("status.json", json.dumps(st, indent=1, default=str))
            except Exception:  # noqa: BLE001
                pass
            try:
                z.writestr("learned.json", json.dumps(a.driver.learned(), indent=1, default=str))
            except Exception:  # noqa: BLE001
                pass
            err = ROOT / "data" / "logs" / "errors.log"
            if err.exists():
                z.writestr("errors.log", err.read_text(errors="replace")[-20000:])
        if now_dev is not None:
            cv2.imwrite(str(path.with_suffix(".jpg")), now_dev, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return path

    def summary(self, reason, note, w, plan, rows) -> str:
        from .ui.devview import info_lines, quad_problem
        a = self.app
        L = [f"ER:LC FSD report - {reason}", time.strftime("%Y-%m-%d %H:%M:%S"), f"version {_version()}"]
        if note:
            L.append(f"note: {note}")
        L.append(f"input: {getattr(a.actuator, 'kind', 'keyboard')}   fps {getattr(a, 'fps', 0.0):.1f}   "
                 f"engaged {getattr(a, 'engaged', False)}")
        try:
            st = a.perception.status()
            L.append("engine: " + ", ".join(f"{k}={_num(v)}" for k, v in st.items() if k not in ("detector",)))
        except Exception:  # noqa: BLE001
            pass
        try:
            cal = a.settings.calibration
            bad = quad_problem(cal.src_quad)
            L.append(f"calibration: quad {np.round(np.array(cal.src_quad), 3).tolist()} "
                     f"cam_to_bumper {cal.cam_to_bumper_m} -> {'OK' if not bad else 'BAD: ' + bad}")
        except Exception:  # noqa: BLE001
            pass
        L += info_lines(w, plan, self.info())
        if rows:
            # quick stats of the last 10 s: what usually matters
            last = [r for r in rows if (rows[-1]["t"] or 0) - (r["t"] or 0) < 10]
            def col(k):
                return np.array([r[k] for r in last if isinstance(r.get(k), (int, float))], float)
            conf, sp, st_ = col("lanes_conf"), col("speed_kph"), col("steer_cmd")
            ped = col("pedal_cmd")
            if conf.size:
                L.append(f"last 10 s: lanes seen {np.mean(conf > 0.4):.0%} of the time, mean conf {conf.mean():.2f}")
            if sp.size:
                L.append(f"speed {sp.min():.0f}-{sp.max():.0f} kph")
            if st_.size > 2:
                flips = int(np.sum(np.diff(np.sign(np.round(st_, 2))) != 0))
                L.append(f"steering: mean |cmd| {np.abs(st_).mean():.2f}, sign flips {flips} (weaving if high)")
            if ped.size > 2:
                flips = int(np.sum(np.diff(np.sign(np.round(ped, 1))) != 0))
                L.append(f"pedal: gas/brake switches {flips} (speed pumping if high)")
        return "\n".join(L) + "\n"

    def _post(self, reason, note, path: Path, snap) -> None:
        from .notify import ORANGE, discord
        d = discord()
        if not d.ok:
            return
        jpg = path.with_suffix(".jpg")
        files = [jpg] if jpg.exists() else []
        size_mb = path.stat().st_size / 1e6
        if size_mb < 9.5:
            files.append(path)
        w = snap.world
        desc = (note + "\n" if note else "") + \
            f"Zip: `data/reports/{path.name}` ({size_mb:.1f} MB) - download it and upload it to Claude."
        fields = []
        if w is not None:
            fields = [("Lanes", f"conf {w.lanes.conf:.2f}, width {w.lanes.width_m:.1f} m", True),
                      ("Speed", f"{w.speed_kph:.0f} kph", True),
                      ("FPS", f"{getattr(self.app, 'fps', 0.0):.0f}", True)]
        d.embed(f"Report: {reason}", desc, ORANGE, fields, files=files)


def _version() -> str:
    try:
        head = (ROOT / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = ROOT / ".git" / head.split()[1]
            if ref.exists():
                return ref.read_text().strip()[:8]
            packed = ROOT / ".git" / "packed-refs"
            if packed.exists():
                for ln in packed.read_text().splitlines():
                    if ln.endswith(head.split()[1]):
                        return ln[:8]
        return head[:8]
    except Exception:  # noqa: BLE001
        return "?"
