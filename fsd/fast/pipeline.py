"""FastPerception: the 60 fps, GPU-free perception pipeline.

Per frame (work picture ~960 px wide):
    paint -> lane lines (screen)      fast/paint.py, fast/screen_lanes.py
    stop bars / crosswalks            fast/screen_lanes.py
    your car's exact outline          perception/ego_shape.py (no AI)
    obstacles in your lane            fast/obstacles.py (no AI)
    traffic lights from lit lamps     fast/lamps.py (no AI, every 2nd frame, sharper picture)
    stop signs (red octagons)         perception/signs.py (no AI, every 3rd frame)
    yaw rate, emergency lights        perception/tracking.py
    speed + speed limits              perception/ocr.py (own thread)
    YOLO (optional extra)             perception/detector.py (own thread, never waited for)
    self-calibration                  fast/autocal.py

Same interface as the old Perception, so planner, recorder, replays and the
dashboard all work unchanged.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ..perception.detector import AsyncDetector, Detector
from ..perception.ego_shape import EgoShape
from ..perception.geometry import Geometry
from ..perception.lights import SignalTracker
from ..perception.ocr import OcrWorker
from ..perception.signs import STOP_SIGN_H_M, find_speed_sign_crops, find_stop_signs, sign_position
from ..perception.tracking import EgoMotion, EmergencyDetector, Tracker, iou
from ..settings import Settings
from ..world import Obj, World
from .autocal import AutoCalibrator
from .lamps import find_lamps
from .obstacles import ObstacleFinder
from .screen_lanes import ScreenLaneFinder, ScreenLaneResult

ROOT = Path(__file__).resolve().parents[2]


class Rate:
    """Counts how many times a second something happens."""

    def __init__(self):
        self.n = 0
        self.t0 = time.perf_counter()
        self.value = 0.0

    def tick(self) -> None:
        self.n += 1
        now = time.perf_counter()
        if now - self.t0 >= 1.0:
            self.value = self.n / (now - self.t0)
            self.n, self.t0 = 0, now


class Background:
    """Runs a slow job on the newest frame in its own thread; the main loop only
    ever reads the last finished result, so it never waits."""

    def __init__(self, fn, name: str, min_period: float = 0.0):
        import threading
        self.fn = fn
        self.min_period = min_period
        self.result = None
        self.ms = 0.0
        self.rate = Rate()
        self._args = None
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._stop = threading.Event()
        threading.Thread(target=self._run, name=name, daemon=True).start()

    def submit(self, *args) -> None:
        with self._lock:
            self._args = args
        self._new.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._new.wait(0.5)
            self._new.clear()
            with self._lock:
                args, self._args = self._args, None
            if args is None:
                continue
            t0 = time.perf_counter()
            try:
                self.result = self.fn(*args)
            except Exception:  # noqa: BLE001 - a bad frame must never kill the worker
                pass
            dt = time.perf_counter() - t0
            self.ms = 0.8 * self.ms + 0.2 * dt * 1000
            self.rate.tick()
            if self.min_period > dt:
                time.sleep(self.min_period - dt)

    def stop(self) -> None:
        self._stop.set()
        self._new.set()


class FastPerception:
    def __init__(self, settings: Settings, use_detector: bool = True, use_ocr: bool = True,
                 on_calibrated=None, live: bool | None = None):
        self.s = settings
        self.cal = settings.calibration
        self.on_calibrated = on_calibrated
        self.live = use_ocr if live is None else live
        self.geo: Geometry | None = None
        self.screen: ScreenLaneFinder | None = None
        self.signals: SignalTracker | None = None
        self.motion: EgoMotion | None = None
        self.tracker = Tracker()
        self.emerg = EmergencyDetector()
        self.obstacles = ObstacleFinder()
        self.ego_shape = EgoShape()
        self.ego_box: tuple[int, int, int, int] = (0, 0, 0, 0)
        self.autocal = AutoCalibrator(self.cal)
        self.detector = None
        if use_detector:
            try:
                self.detector = Detector(ROOT, settings.model_path, settings.fallback_model, settings.device)
            except Exception:  # noqa: BLE001 - YOLO is optional
                self.detector = None
        self.async_det = AsyncDetector(self.detector) if (self.live and self.detector and self.detector.ok) else None
        self._det_seq = 0
        self._yolo_objs: list[Obj] = []
        self.ocr = OcrWorker(self.cal, on_calibrated).start() if use_ocr else None
        self.frame_id = 0
        self.rate = Rate()
        self.ms = 0.0
        self._lamps: list[Obj] = []
        self._stops: list[Obj] = []
        self._last_sign_submit = 0.0
        self.last_objects: list[Obj] = []
        self.last_screen: ScreenLaneResult | None = None
        self.messages: list[str] = []
        self._bg_ego = Background(self._ego_job, "ego-outline", 0.12) if self.live else None
        self._bg_objs = Background(self._objects_job, "lamps-signs", 0.04) if self.live else None

    # ------------------------------------------------------------------ setup
    def _setup(self, W: int, H: int) -> None:
        self.geo = Geometry.build(self.cal, W, H)
        self.screen = ScreenLaneFinder(self.geo)
        self.signals = SignalTracker(self.geo, self.cal.cam_to_bumper_m)
        self.motion = EgoMotion(self.geo.focal_px)
        x1, y1, x2, y2 = self.cal.ego_mask
        self.ego_box = (int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))
        self.ego_shape = EgoShape()

    def _rebuild_geometry(self, W: int, H: int) -> None:
        """New calibration: keep trackers/ego, rebuild everything geometric."""
        self.geo = Geometry.build(self.cal, W, H)
        self.screen.set_geometry(self.geo)
        self.signals = SignalTracker(self.geo, self.cal.cam_to_bumper_m)
        self.motion = EgoMotion(self.geo.focal_px)

    # back-compat for code that used the old Perception attributes
    @property
    def lanes(self):
        return self.screen

    def status(self) -> dict:
        d = self.detector
        dev = ""
        if d is not None and d.ok:
            dev = f"GPU {d.gpu_name}" if d.gpu_name else "CPU"
        return {
            "detector": ("custom ER:LC model" if d and d.custom else "COCO model" if d and d.ok else "no-AI vision"),
            "device": dev,
            "det_ms": self.async_det.ms if self.async_det else 0.0,
            "det_fps": self.async_det.fps if self.async_det else 0.0,
            "vision_fps": self.rate.value,
            "lights_fps": self._bg_objs.rate.value if self._bg_objs else 0.0,
            "vision_ms": self.ms,
            "ocr": self.ocr.status if self.ocr else "off",
            "autocal": self.autocal.message,
        }

    # ------------------------------------------------------------------ background jobs
    def _ego_job(self, frame, box, t):
        self.ego_shape.update(frame, box, t)
        return True

    def _objects_job(self, frame, fine, horizon_y):
        H, W = frame.shape[:2]
        src = fine if fine is not None else frame
        k = W / src.shape[1]
        lamps = find_lamps(src, horizon_y / k, scale_to=k)
        stops = find_stop_signs(frame, horizon_y)
        return lamps, stops

    # ------------------------------------------------------------------ ego car
    def _update_ego(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Exact outline of our car (no AI); the ego box follows it. Returns a mask for paint finding."""
        if self._bg_ego is not None:
            self._bg_ego.submit(frame, self.ego_box, t)
        else:
            self.ego_shape.update(frame, self.ego_box, t)
        H, W = frame.shape[:2]
        bb = self.ego_shape.bbox
        if bb is not None:
            x1, y1, x2, y2 = bb
            cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            if 0.3 * W < cx < 0.7 * W and 0.3 * H < cy < 0.9 * H and 0.03 * W < bw < 0.35 * W \
                    and 0.03 * H < bh < 0.5 * H:
                m = 0.06
                target = (x1 - m * bw, y1 - m * bh, x2 + m * bw, y2 + m * bh)
                self.ego_box = tuple(int(0.7 * e + 0.3 * tv) for e, tv in zip(self.ego_box, target))
        mask = np.zeros((H, W), np.uint8)
        if self.ego_shape.mask is not None:
            self.ego_shape.paste(mask, 1.0, 1.0)
            mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
        else:
            x1, y1, x2, y2 = self.ego_box
            mask[max(0, y1):y2, max(0, x1):x2] = 255
        return mask

    def _is_ego(self, box) -> bool:
        ex1, ey1, ex2, ey2 = self.ego_box
        ew, eh = ex2 - ex1, ey2 - ey1
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        inside = (ex1 - 0.25 * ew < cx < ex2 + 0.25 * ew) and (ey1 - 0.25 * eh < cy < ey2 + 0.25 * eh)
        return bool(inside or iou(box, self.ego_box) > 0.3)

    # ------------------------------------------------------------------ main
    def process(self, frame: np.ndarray, p: dict, t: float | None = None, fine: np.ndarray | None = None) -> World:
        t = time.perf_counter() if t is None else t
        t_start = time.perf_counter()
        H, W = frame.shape[:2]
        if self.geo is None or self.geo.W != W or self.geo.H != H:
            self._setup(W, H)
        geo = self.geo
        w = World(t=t, frame_id=self.frame_id)
        self.frame_id += 1
        tm = {}

        # ---- your car ----
        t0 = time.perf_counter()
        ego_mask = self._update_ego(frame, t)
        tm["ego"] = time.perf_counter() - t0

        # ---- lanes, stop bars, crosswalks ----
        t0 = time.perf_counter()
        res = self.screen.process(frame, ego_mask, p["lane_smoothing"], p["lane_width_m"], t)
        self.last_screen = res
        w.lanes = res.lanes
        w.stop_line_dist = res.stop_line if res.stop_line is not None else 999.0
        w.crosswalk_dist = res.crosswalk if res.crosswalk is not None else 999.0
        tm["lanes"] = time.perf_counter() - t0

        # ---- self-calibration ----
        rear = (self.ego_shape.bbox[3] / H) if self.ego_shape.bbox else None
        fields = self.autocal.add(res.ego_pair_screen, W, H, rear, t)
        if fields:
            for k, v in fields.items():
                setattr(self.cal, k, v)
            self._rebuild_geometry(W, H)
            self.messages.append(self.autocal.message)
            if self.on_calibrated:
                try:
                    self.on_calibrated()
                except Exception:  # noqa: BLE001
                    pass

        # ---- objects ----
        t0 = time.perf_counter()
        objs: list[Obj] = []
        # traffic lights from lamps + stop signs: background thread (~20x/s), sharper picture
        if self._bg_objs is not None:
            self._bg_objs.submit(frame, fine, geo.horizon_y)
            if self._bg_objs.result is not None:
                self._lamps, self._stops = self._bg_objs.result
        else:
            self._lamps, self._stops = self._objects_job(frame, fine, geo.horizon_y)
        objs += [Obj(o.kind, o.box, o.score, extra=dict(o.extra)) for o in self._lamps]
        objs += [Obj(o.kind, o.box, o.score, extra=dict(o.extra)) for o in self._stops]
        # YOLO, if it's there: never waited for
        min_score = min(p["light_min_score"], p["object_min_score"])
        if self.async_det is not None:
            self.async_det.submit(frame, min_score)
            r_objs, rframe, rt, seq = self.async_det.latest()
            if rframe is not None and rframe.shape == frame.shape and t - rt < 0.8:
                self._yolo_objs = [Obj(o.kind, o.box, o.score, extra=dict(o.extra)) for o in r_objs]
            elif t - rt >= 0.8:
                self._yolo_objs = []
        elif self.detector is not None and self.detector.ok and not self.live:
            self._yolo_objs = self.detector.detect(frame, min_score)
        for o in self._yolo_objs:
            if o.kind == "light" and any(iou(o.box, l.box) > 0.1 for l in objs if l.kind == "light"):
                continue                       # the lamp finder already has this light
            objs.append(Obj(o.kind, o.box, o.score, extra=dict(o.extra)))
        tm["objects"] = time.perf_counter() - t0

        lights, others = [], []
        for ob in objs:
            if ob.kind == "light":
                if ob.score >= p["light_min_score"]:
                    lights.append(ob)
                continue
            if ob.score < p["object_min_score"]:
                continue
            if ob.kind in ("car", "emergency") and self._is_ego(ob.box):
                continue
            if ob.kind in ("car", "person", "cone", "emergency"):
                x1, y1, x2, y2 = ob.box
                gp = geo.ground_point((x1 + x2) / 2, y2)
                if gp is None:
                    continue
                ob.x, ob.y = gp
                if ob.kind == "person":
                    ob.extra["half_width"] = 0.4
                else:
                    c = geo.screen_to_vehicle(np.array([[x1, y2], [x2, y2]], np.float32))
                    ob.extra["half_width"] = float(np.clip(abs(c[1, 1] - c[0, 1]) / 2, 0.3, 2.0))
            elif ob.kind in ("stop_sign", "speed_sign"):
                ob.x, ob.y = sign_position(geo, ob.box, STOP_SIGN_H_M, self.cal.cam_to_bumper_m)
            others.append(ob)

        # obstacles from free space (no AI): skip if YOLO already sees a car at that spot
        t0 = time.perf_counter()
        free = self.obstacles.find(frame, res.lanes, self.screen, res.masks.wide, ego_mask, p["lane_width_m"])
        for o in free:
            if not any(c.kind in ("car", "person", "cone") and abs(c.x - o.x) < 4 for c in others):
                others.append(o)
        tm["obstacles"] = time.perf_counter() - t0

        others = self.tracker.update(others, t)
        lane_cy = res.lanes.center[2] if res.lanes.conf > 0 else 0.0
        w.signal = self.signals.update(frame, lights, lane_cy, p["flash_window_s"], t)
        w.objects = others + lights

        for ob in others:
            if ob.kind == "stop_sign" and 0 < ob.x < 70 and -3.0 < ob.y - lane_cy < 9.0:
                w.stop_sign_dist = min(w.stop_sign_dist, ob.x)

        # ---- speed + limits (OCR thread) ----
        if self.ocr is not None:
            crops = []
            src = fine if fine is not None else frame
            if t - self._last_sign_submit > 0.35:
                self._last_sign_submit = t
                k = src.shape[1] / W
                crops = [c for b, c in find_speed_sign_crops(src, geo.horizon_y * k) if (b[0] + b[2]) / 2 > src.shape[1] * 0.5]
            self.ocr.submit(src, crops)
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
        w.road_conf = res.lanes.conf
        w.horizon_y = geo.horizon_y
        w.vanish_x = self.screen.vp_x
        tm.update({k: v for k, v in res.timings.items()})
        tm["total"] = time.perf_counter() - t_start
        w.timings = tm
        self.ms = 0.9 * self.ms + 0.1 * tm["total"] * 1000
        self.rate.tick()
        self.last_objects = w.objects
        return w

    def close(self) -> None:
        if self.ocr:
            self.ocr.stop()
        if self.async_det:
            self.async_det.stop()
        for b in (self._bg_ego, self._bg_objs):
            if b is not None:
                b.stop()
