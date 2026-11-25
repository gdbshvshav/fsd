"""Perception pipeline: one call turns a screen frame into a World."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ..settings import Settings
from ..world import Light, Obj, World
from .detector import AsyncDetector, Detector
from .ego_shape import EgoShape
from .geometry import Geometry
from .lanes import LaneFinder
from .lights import SignalTracker, find_light_candidates
from .ocr import OcrWorker
from .signs import STOP_SIGN_H_M, find_speed_sign_crops, find_stop_signs, sign_position
from .tracking import EgoMotion, EmergencyDetector, Tracker, iou

ROOT = Path(__file__).resolve().parents[2]


class Perception:
    def __init__(self, settings: Settings, use_detector: bool = True, use_ocr: bool = True,
                 on_calibrated=None):
        self.s = settings
        self.cal = settings.calibration
        self.geo: Geometry | None = None
        self.lanes: LaneFinder | None = None
        self.signals: SignalTracker | None = None
        self.motion: EgoMotion | None = None
        self.tracker = Tracker()
        self.emerg = EmergencyDetector()
        self.detector = Detector(ROOT, settings.model_path, settings.fallback_model, settings.device) \
            if use_detector else None
        self.ocr = OcrWorker(self.cal, on_calibrated).start() if use_ocr else None
        # live use: detection on its own thread; replays/tests: inline (deterministic)
        self.async_det = AsyncDetector(self.detector) if (use_ocr and self.detector and self.detector.ok) else None
        self._det_seq = 0
        self._det_objs: list[Obj] = []
        self._last_sign_submit = 0.0
        self.ego_box: tuple[int, int, int, int] = (0, 0, 0, 0)
        self.ego_shape = EgoShape()
        self.frame_id = 0
        self.last_objects: list[Obj] = []

    def _setup(self, W: int, H: int) -> None:
        self.geo = Geometry.build(self.cal, W, H)
        self.lanes = LaneFinder(self.geo)
        self.lanes.set_ego_mask(self.cal.ego_mask)
        self.signals = SignalTracker(self.geo, self.cal.cam_to_bumper_m)
        self.motion = EgoMotion(self.geo.focal_px)
        x1, y1, x2, y2 = self.cal.ego_mask
        self.ego_box = (int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))

    def _update_ego(self, frame, t) -> None:
        """Exact outline of our car (no AI), and let the ego box follow that outline."""
        self.ego_shape.update(frame, self.ego_box, t)
        bb = self.ego_shape.bbox
        if bb is None:
            return
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = bb
        cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        sane = 0.3 * W < cx < 0.7 * W and 0.3 * H < cy < 0.9 * H and 0.03 * W < bw < 0.35 * W and 0.03 * H < bh < 0.5 * H
        if not sane:
            return
        m = 0.06
        target = (x1 - m * bw, y1 - m * bh, x2 + m * bw, y2 + m * bh)
        a = 0.3
        self.ego_box = tuple(int((1 - a) * e + a * tv) for e, tv in zip(self.ego_box, target))

    def _is_ego(self, box) -> bool:
        """Our own car: the vehicle sitting in the middle of the chase camera.

        Its box also slowly follows our car (zooming, different vehicles), so a
        truck or a bus is recognised as 'you' too.
        """
        ex1, ey1, ex2, ey2 = self.ego_box
        ew, eh = ex2 - ex1, ey2 - ey1
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        inside = (ex1 - 0.25 * ew < cx < ex2 + 0.25 * ew) and (ey1 - 0.25 * eh < cy < ey2 + 0.25 * eh)
        if not (inside or iou(box, self.ego_box) > 0.3):
            return False
        bw, bh = x2 - x1, y2 - y1
        if 0.4 * ew < bw < 2.5 * ew and 0.4 * eh < bh < 2.5 * eh:
            a = 0.05                                       # follow our car's box slowly
            self.ego_box = tuple(int((1 - a) * e + a * b) for e, b in zip(self.ego_box, box))
        return True

    def status(self) -> dict:
        d = self.detector
        dev = ""
        if d is not None and d.ok:
            dev = f"GPU {d.gpu_name}" if d.gpu_name else "CPU (no GPU found!)"
        return {
            "device": dev,
            "det_ms": self.async_det.ms if self.async_det else 0.0,
            "det_fps": self.async_det.fps if self.async_det else 0.0,
            "detector": ("custom ER:LC model" if self.detector and self.detector.custom else
                         "COCO model" if self.detector and self.detector.ok else
                         (self.detector.error if self.detector else "off")),
            "ocr": self.ocr.status if self.ocr else "off",
        }

    def process(self, frame: np.ndarray, p: dict, t: float | None = None) -> World:
        t = time.perf_counter() if t is None else t
        H, W = frame.shape[:2]
        if self.geo is None or self.geo.W != W or self.geo.H != H:
            self._setup(W, H)
        geo = self.geo
        w = World(t=t, frame_id=self.frame_id)
        self.frame_id += 1
        tm = {}

        t0 = time.perf_counter()
        lanes, stop_line, crosswalk = self.lanes.process(frame, p["lane_smoothing"], p["lane_width_m"], t)
        w.lanes = lanes
        w.stop_line_dist = stop_line if stop_line is not None else 999.0
        w.crosswalk_dist = crosswalk if crosswalk is not None else 999.0
        tm["lanes"] = time.perf_counter() - t0
        self._update_ego(frame, t)

        # ---- objects ----
        t0 = time.perf_counter()
        objs: list[Obj] = []
        det_frame = frame
        fresh = True
        min_score = min(p["light_min_score"], p["object_min_score"])
        if self.async_det is not None:
            self.async_det.submit(frame, min_score)
            res, rframe, rt, seq = self.async_det.latest()
            fresh = seq != self._det_seq
            self._det_seq = seq
            if rframe is not None and rframe.shape == frame.shape and t - rt < 0.6:
                objs = [Obj(o.kind, o.box, o.score, extra=dict(o.extra)) for o in res]
                det_frame = rframe
        elif self.detector is not None and self.detector.ok:
            objs = self.detector.detect(frame, min_score)
        else:
            objs = find_light_candidates(frame, geo.horizon_y) + find_stop_signs(frame, geo.horizon_y)
        tm["detect"] = time.perf_counter() - t0

        lights, others = [], []
        for ob in objs:
            if ob.kind == "light":
                if ob.score >= p["light_min_score"]:
                    lights.append(ob)
                continue
            if ob.score < p["object_min_score"]:
                continue
            if ob.kind in ("car", "emergency") and self._is_ego(ob.box):
                continue  # that's our own car in the chase camera
            if ob.kind in ("car", "person", "cone", "emergency"):
                x1, y1, x2, y2 = ob.box
                gp = geo.ground_point((x1 + x2) / 2, y2)
                if gp is None:
                    continue
                ob.x, ob.y = gp
                if ob.kind == "person":
                    ob.extra["half_width"] = 0.4
                else:
                    corners = geo.screen_to_vehicle(np.array([[x1, y2], [x2, y2]], dtype=np.float32))
                    ob.extra["half_width"] = float(np.clip(abs(corners[1, 1] - corners[0, 1]) / 2, 0.3, 2.0))
            elif ob.kind in ("stop_sign", "speed_sign"):
                ob.x, ob.y = sign_position(geo, ob.box, STOP_SIGN_H_M, self.cal.cam_to_bumper_m)
            others.append(ob)

        if fresh:
            others = self.tracker.update(others, t)
            self._det_objs = others
        else:
            others = self._det_objs                  # same detections as last frame: keep tracks as they are
        lane_cy = lanes.center[2] if lanes.conf > 0 else 0.0
        w.signal = self.signals.update(det_frame, lights, lane_cy, p["flash_window_s"], t) if fresh \
            else self.signals.last_info
        w.objects = others + lights

        # ---- stop signs that face us, on our side ----
        for ob in others:
            if ob.kind == "stop_sign" and 0 < ob.x < 70 and -3.0 < ob.y - lane_cy < 9.0:
                w.stop_sign_dist = min(w.stop_sign_dist, ob.x)

        # ---- speed + speed limit via OCR thread ----
        if self.ocr is not None:
            crops = []
            if t - self._last_sign_submit > 0.35:
                self._last_sign_submit = t
                crops = [c for b, c in find_speed_sign_crops(frame, geo.horizon_y)
                         if (b[0] + b[2]) / 2 > W * 0.5]
                crops += [frame[o.box[1]:o.box[3], o.box[0]:o.box[2]] for o in others if o.kind == "speed_sign"]
            self.ocr.submit(frame, crops)
            w.speed_kph = self.ocr.speed_kph
            w.speed_valid = self.ocr.speed_valid
            w.speed_unit = self.ocr.unit
            if self.ocr.limit_kph is not None:
                w.speed_limit_kph = self.ocr.limit_kph

        # ---- motion, emergency, scene ----
        w.yaw_rate = self.motion.update(frame, geo.horizon_y, t)
        w.emergency_near = self.emerg.update(frame, [o for o in others if o.kind in ("car", "emergency")], t) \
            or any(o.kind == "emergency" and o.x < 60 for o in others)
        small = cv2.resize(frame, (64, 36))
        w.tunnel = float(small[18:, :].mean()) < 55
        w.road_conf = lanes.conf
        w.horizon_y, w.vanish_x = self._vanishing(lanes)
        tm["total"] = sum(tm.values())
        w.timings = tm
        self.last_objects = w.objects
        return w

    def _vanishing(self, lanes) -> tuple[float, float | None]:
        geo = self.geo
        if lanes.conf < 0.3 or lanes.left is None or lanes.right is None:
            return geo.horizon_y, None
        pts = []
        for ln in (lanes.left, lanes.right):
            p = geo.vehicle_to_screen(np.array([[12.0, ln.y_at(12)], [30.0, ln.y_at(30)]]))
            pts.append(p)
        (x1, y1), (x2, y2) = pts[0]
        (x3, y3), (x4, y4) = pts[1]
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-6:
            return geo.horizon_y, None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
        if not (-geo.W < px < 2 * geo.W) or not (0 < py < geo.H):
            return geo.horizon_y, None
        return float(py), float(px)

    def close(self) -> None:
        if self.ocr:
            self.ocr.stop()
        if self.async_det:
            self.async_det.stop()


__all__ = ["Perception", "Light"]
