"""60 fps: the vision work spread over several CPU cores (separate processes).

Python runs one thread of Python code at a time (the GIL), so threads alone could
never get lanes + lights + the car outline + steering done 60 times a second. Here
the heavy jobs run in their OWN PROCESSES, each on its own core:

    main process      capture -> shrink -> (shared memory) ; speedometer, trackers,
                      planner, steering - every frame, 60x a second
    lane workers (2)  paint, lane lines, stop bars, crosswalks, obstacles, self-calibration.
                      They take turns (frame 1 -> A, frame 2 -> B, ...), so together
                      they keep up with 60 fps even if one lane pass takes ~25-30 ms.
    object worker     traffic-light lamps, stop signs, and our car's exact outline.

Frames go to the workers through shared memory (no copying 2 MB per frame through a
pipe); only small results come back. Results are a frame or two old by the time
they're used, so the main process moves them forward to "now" with our own speed and
turn rate (the car drove d metres and turned a little since that picture).

If anything about the processes fails (antivirus, very old PC), FSD falls back to the
single-process pipeline automatically.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from dataclasses import replace
from multiprocessing import shared_memory

import cv2
import numpy as np

from ..perception.geometry import Geometry
from ..perception.lights import SignalTracker
from ..perception.signs import STOP_SIGN_H_M, find_speed_sign_crops, sign_position
from ..perception.tracking import EgoMotion, EmergencyDetector, Tracker
from ..settings import Settings
from ..perception.visual_odometry import VisualOdometry, hidden_from_box
from ..world import Lanes, LaneLine, Obj, World
from .pipeline import Rate

SLOTS = 6


# ---------------------------------------------------------------------- shared frames
class FrameRing:
    """A few frame slots in shared memory: work (~960 px) and fine (~1920 px) pictures."""

    def __init__(self, work_shape: tuple, fine_shape: tuple | None, create: bool = True, names=None):
        self.work_shape, self.fine_shape = tuple(work_shape), tuple(fine_shape) if fine_shape else None
        wb = int(np.prod(self.work_shape))
        fb = int(np.prod(self.fine_shape)) if self.fine_shape else 1
        if create:
            self.shm_w = shared_memory.SharedMemory(create=True, size=wb * SLOTS)
            self.shm_f = shared_memory.SharedMemory(create=True, size=fb * SLOTS)
        else:
            self.shm_w = shared_memory.SharedMemory(name=names[0])
            self.shm_f = shared_memory.SharedMemory(name=names[1])
        self.work = np.ndarray((SLOTS,) + self.work_shape, np.uint8, buffer=self.shm_w.buf)
        self.fine = np.ndarray((SLOTS,) + self.fine_shape, np.uint8, buffer=self.shm_f.buf) if self.fine_shape else None
        self.created = create

    @property
    def names(self) -> tuple[str, str]:
        return self.shm_w.name, self.shm_f.name

    def put(self, slot: int, work: np.ndarray, fine: np.ndarray | None) -> None:
        self.work[slot][...] = work
        if self.fine is not None and fine is not None and fine.shape == self.fine_shape:
            self.fine[slot][...] = fine

    def close(self) -> None:
        for s in (self.shm_w, self.shm_f):
            try:
                s.close()
                if self.created:
                    s.unlink()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------- workers
def _worker_main(role: str, settings: Settings, ring_spec: tuple, jobs: mp.Queue, out: mp.Queue) -> None:
    """Runs in its own process. role: 'lanes' or 'objects'."""
    try:
        cv2.setNumThreads(1)                       # one core each; we have several processes
        from .pipeline import FastPerception
        per = FastPerception(settings, use_detector=False, use_ocr=False, live=False)
        ring = FrameRing(ring_spec[0], ring_spec[1], create=False, names=ring_spec[2])
        ego_mask = None
        while True:
            job = jobs.get()
            if job is None:
                break
            kind = job[0]
            if kind == "cal":                      # new calibration from another worker / main
                for k, v in job[1].items():
                    setattr(per.cal, k, v)
                per.geo = None
                continue
            if kind == "ego_mask":
                ego_mask = np.unpackbits(job[1])[:int(np.prod(job[2]))].reshape(job[2]) * 255 \
                    if job[1] is not None else None
                continue
            _, slot, seq, t, p, extra = job
            work = np.array(ring.work[slot])       # private copy: the slot may be reused
            fine = np.array(ring.fine[slot]) if ring.fine is not None else None
            if role == "lanes":
                r = per.lanes_job(work, p, t, ego_mask, extra.get("ego_rear"))
            else:
                if extra.get("ego_box") is not None:
                    per.ego_box = tuple(extra["ego_box"])
                r = per.objects_job(work, fine, t, extra.get("do_ego", False))
            r["seq"] = seq
            r["role"] = role
            out.put(r)
    except Exception as e:  # noqa: BLE001
        try:
            out.put({"role": role, "error": f"{type(e).__name__}: {e}"})
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------- moving results to "now"
def shift_poly(coeffs, d: float, theta: float):
    """Lane polynomial y = a x^2 + b x + c seen `d` metres ago (and before turning theta rad):
    where is it now? x_then = x_now + d; then rotate by -theta (small angle)."""
    a, b, c = coeffs
    return (a, 2 * a * d + b - theta, a * d * d + b * d + c)


def advance_edges(e, d: float, theta: float):
    """Road edges moved forward like the lanes."""
    if e is None or not e.ok:
        return e
    return replace(e, left=shift_poly(e.left, d, theta), right=shift_poly(e.right, d, theta),
                   x_max=max(0.0, e.x_max - d))


def advance_lanes(L: Lanes, d: float, theta: float) -> Lanes:
    if L is None or (abs(d) < 1e-3 and abs(theta) < 1e-4):
        return L

    def mv(line):
        if line is None:
            return None
        return replace(line, coeffs=shift_poly(line.coeffs, d, theta), x_max=max(0.0, line.x_max - d))
    nl = replace(L, left=mv(L.left), right=mv(L.right), left2=mv(L.left2), right2=mv(L.right2),
                 center=shift_poly(L.center, d, theta), x_max=max(0.0, L.x_max - d))
    a, b, c = nl.center
    nl.offset = c
    nl.heading = float(np.arctan(b))
    return nl


class _Screen:
    """What the dashboard reads from `perception.last_screen`."""

    def __init__(self):
        self.chains_screen = []
        self.masks = None


class _Masks:
    def __init__(self, paint):
        self.paint = paint
        self.wide = np.zeros_like(paint)


class _EgoView:
    """What the dashboard reads from `perception.ego_shape` (exact outline of our car)."""

    def __init__(self):
        self.mask = None
        self.region = None
        self.bbox = None

    def paste(self, target_mask: np.ndarray, sx: float, sy: float) -> None:
        if self.mask is None or self.region is None:
            return
        x1, y1, x2, y2 = self.region
        a, b = int(x1 * sx), int(y1 * sy)
        w, h = max(1, int((x2 - x1) * sx)), max(1, int((y2 - y1) * sy))
        m = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_LINEAR)
        H, W = target_mask.shape[:2]
        a2, b2 = min(W, a + w), min(H, b + h)
        if a2 <= max(0, a) or b2 <= max(0, b):
            return
        sub = m[max(0, -b):max(0, -b) + (b2 - max(0, b)), max(0, -a):max(0, -a) + (a2 - max(0, a))]
        target_mask[max(0, b):b2, max(0, a):a2] |= np.where(sub > 127, 255, 0).astype(np.uint8)

    def reset(self) -> None:
        self.mask = self.region = self.bbox = None


class ParallelPerception:
    """Same interface as FastPerception (process(frame, p, t, fine) -> World), but the heavy
    work runs in worker processes."""

    def __init__(self, settings: Settings, lane_workers: int | None = None, on_calibrated=None,
                 use_ocr: bool = True, ego_period_s: float = 0.4, objects_period_s: float = 0.06):
        from ..perception.ocr import OcrWorker
        self.s = settings
        self.cal = settings.calibration
        self.on_calibrated = on_calibrated
        cores = os.cpu_count() or 2
        self.n_lanes = lane_workers if lane_workers else (2 if cores >= 4 else 1)
        self.ego_period = ego_period_s
        self.objects_period = objects_period_s
        self.ocr = OcrWorker(self.cal, on_calibrated).start() if use_ocr else None
        self.geo: Geometry | None = None
        self.signals: SignalTracker | None = None
        self.motion: EgoMotion | None = None
        self.tracker = Tracker()
        self.emerg = EmergencyDetector()
        self.ego_shape = _EgoView()
        self.ego_box = (0, 0, 0, 0)
        self.ego_home = (0, 0, 0, 0)
        self.last_screen = _Screen()
        self.last_objects: list[Obj] = []
        self.messages: list[str] = []
        self.rate = Rate()
        self.lane_rate = Rate()
        self.obj_rate = Rate()
        self.ms = 0.0
        self.lane_ms = 0.0
        self.lag_ms = 0.0
        self.yaw_hint: float | None = None
        self.vo = VisualOdometry()
        self.frame_id = 0
        self.ring: FrameRing | None = None
        self.procs: list[mp.Process] = []
        self.lane_q: list[mp.Queue] = []
        self.obj_q: mp.Queue | None = None
        self.out_q: mp.Queue | None = None
        self.busy_lane: list[bool] = []
        self.busy_obj = False
        self.next_lane = 0
        self.seq = 0
        self.slot = 0
        self.lane_res: dict | None = None
        self.obj_res: dict | None = None
        self._last_obj_submit = 0.0
        self._last_ego_submit = 0.0
        self._last_sign_submit = 0.0
        self.error = ""
        self.speed_ms_hist: list[tuple[float, float]] = []
        self.autocal_message = ""

    # ------------------------------------------------------------------ processes
    def _start(self, work: np.ndarray, fine: np.ndarray | None) -> None:
        ctx = mp.get_context("spawn")
        self.ring = FrameRing(work.shape, fine.shape if fine is not None else None)
        spec = (self.ring.work_shape, self.ring.fine_shape, self.ring.names)
        self.out_q = ctx.Queue()
        for _ in range(self.n_lanes):
            q = ctx.Queue()
            pr = ctx.Process(target=_worker_main, args=("lanes", self.s, spec, q, self.out_q),
                             name="fsd-lanes", daemon=True)
            pr.start()
            self.procs.append(pr)
            self.lane_q.append(q)
            self.busy_lane.append(False)
        self.obj_q = ctx.Queue()
        pr = ctx.Process(target=_worker_main, args=("objects", self.s, spec, self.obj_q, self.out_q),
                         name="fsd-objects", daemon=True)
        pr.start()
        self.procs.append(pr)

    def _setup(self, W: int, H: int) -> None:
        self.geo = Geometry.build(self.cal, W, H)
        self.signals = SignalTracker(self.geo, self.cal.cam_to_bumper_m)
        self.motion = EgoMotion(self.geo.focal_px)
        x1, y1, x2, y2 = self.cal.ego_mask
        self.ego_home = (int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))
        self.ego_box = self.ego_home

    def alive(self) -> bool:
        return bool(self.procs) and all(p.is_alive() for p in self.procs)

    # ------------------------------------------------------------------ results in
    def _drain(self) -> None:
        if self.out_q is None:
            return
        while True:
            try:
                r = self.out_q.get_nowait()
            except queue.Empty:
                break
            if "error" in r:
                self.error = f"{r['role']} worker: {r['error']}"
                self.messages.append(self.error)
                continue
            if r["role"] == "lanes":
                # a worker finished: it's free again
                for i in range(len(self.busy_lane)):
                    if self.busy_lane[i] == r["seq"]:
                        self.busy_lane[i] = False
                if self.lane_res is None or r["seq"] > self.lane_res["seq"]:
                    self.lane_res = r
                    self.lane_rate.tick()
                    self.lane_ms = 0.8 * self.lane_ms + 0.2 * r["timings"]["lanes_total"] * 1000
                if r.get("cal"):
                    self._apply_cal(r["cal"], r.get("cal_msg", ""))
            else:
                self.busy_obj = False
                self.obj_res = r
                self.obj_rate.tick()
                if r.get("ego") is not None:
                    region, mask, bbox = r["ego"]
                    self._ego_update(region, mask, bbox)

    def _apply_cal(self, fields: dict, msg: str) -> None:
        for k, v in fields.items():
            setattr(self.cal, k, v)
        if self.geo is not None:
            W, H = self.geo.W, self.geo.H
            self.geo = None
            self._setup(W, H)
        for q in self.lane_q + ([self.obj_q] if self.obj_q else []):
            q.put(("cal", fields))
        if msg:
            self.messages.append(msg)
            self.autocal_message = msg
        if self.on_calibrated:
            try:
                self.on_calibrated()
            except Exception:  # noqa: BLE001
                pass

    def _ego_update(self, region, mask, bbox) -> None:
        """Same sanity rules as the single-process pipeline: the outline may follow the car
        but never balloon over the road or swallow the car in front."""
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        hx1, hy1, hx2, hy2 = self.ego_home
        hw_, hh_ = hx2 - hx1, hy2 - hy1
        cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        sane = (abs(cx - (hx1 + hx2) / 2) < 0.35 * hw_ and abs(cy - (hy1 + hy2) / 2) < 0.35 * hh_
                and 0.5 * hw_ < bw < 1.5 * hw_ and 0.5 * hh_ < bh < 1.5 * hh_ and y1 > hy1 - 0.12 * hh_)
        if not sane:
            self.ego_shape.reset()
            self.ego_box = self.ego_home
            return
        self.ego_shape.mask, self.ego_shape.region, self.ego_shape.bbox = mask, region, bbox
        m = 0.06
        target = (x1 - m * bw, y1 - m * bh, x2 + m * bw, y2 + m * bh)
        self.ego_box = tuple(int(0.7 * e + 0.3 * tv) for e, tv in zip(self.ego_box, target))
        # tell the lane workers what to ignore
        H, W = self.geo.H, self.geo.W
        full = np.zeros((H, W), np.uint8)
        self.ego_shape.paste(full, 1.0, 1.0)
        lim = np.zeros_like(full)
        mw, mh, mt = int(0.3 * hw_), int(0.3 * hh_), int(0.08 * hh_)
        lim[max(0, hy1 - mt):hy2 + mh, max(0, hx1 - mw):hx2 + mw] = 255
        full &= lim
        cw, ch = int(0.12 * hw_), int(0.10 * hh_)
        full[max(0, hy1 + ch):hy2 - ch, max(0, hx1 + cw):hx2 - cw] = 255
        full = cv2.dilate(full, np.ones((7, 7), np.uint8))
        packed = np.packbits(full > 0, axis=None)
        for q in self.lane_q:
            q.put(("ego_mask", packed, (H, W)))

    # ------------------------------------------------------------------ jobs out
    def _submit(self, work, fine, p, t) -> None:
        self.slot = (self.slot + 1) % SLOTS
        self.seq += 1
        want_obj = not self.busy_obj and t - self._last_obj_submit >= self.objects_period
        # the sharp picture is only needed by the light/sign worker: copy it only then
        self.ring.put(self.slot, work, fine if want_obj else None)
        rear = (self.ego_shape.bbox[3] / work.shape[0]) if self.ego_shape.bbox else None
        # lanes: the next free worker (round robin); if all are busy this frame is skipped for lanes
        for k in range(self.n_lanes):
            i = (self.next_lane + k) % self.n_lanes
            if self.busy_lane[i] is False:
                self.busy_lane[i] = self.seq
                self.lane_q[i].put(("job", self.slot, self.seq, t, p, {"ego_rear": rear}))
                self.next_lane = (i + 1) % self.n_lanes
                break
        if want_obj:
            do_ego = t - self._last_ego_submit >= self.ego_period
            if do_ego:
                self._last_ego_submit = t
            self._last_obj_submit = t
            self.busy_obj = True
            self.obj_q.put(("job", self.slot, self.seq, t, p, {"do_ego": do_ego, "ego_box": self.ego_box}))

    # ------------------------------------------------------------------ main
    def process(self, frame: np.ndarray, p: dict, t: float | None = None, fine: np.ndarray | None = None) -> World:
        t = time.perf_counter() if t is None else t
        t_start = time.perf_counter()
        H, W = frame.shape[:2]
        if self.geo is None or self.geo.W != W or self.geo.H != H:
            self._setup(W, H)
            if self.ring is None:
                self._start(frame, fine)
        self._drain()
        self._submit(frame, fine, p, t)
        geo = self.geo
        w = World(t=t, frame_id=self.frame_id)
        self.frame_id += 1

        # ---- speed first (needed to move the lane result forward to now) ----
        src = fine if fine is not None else frame
        if self.ocr is not None:
            crops = []
            if t - self._last_sign_submit > 0.35:
                self._last_sign_submit = t
                k = src.shape[1] / W
                crops = [c for b, c in find_speed_sign_crops(src, geo.horizon_y * k)
                         if (b[0] + b[2]) / 2 > src.shape[1] * 0.5]
            self.ocr.submit(src, crops)
            self.ocr.fast_read(src)
            w.speed_kph = self.ocr.speed_kph
            w.speed_valid = self.ocr.speed_valid
            w.speed_unit = self.ocr.unit
            if self.ocr.limit_kph is not None:
                w.speed_limit_kph = self.ocr.limit_kph
        w.yaw_rate = self.motion.update(frame, geo.horizon_y, t)
        # the driving core's fused yaw (chassis model) is far more reliable than the picture's
        yaw_shift = self.yaw_hint if self.yaw_hint is not None else w.yaw_rate

        # ---- lanes etc. from the newest worker result, moved forward to now ----
        lr = self.lane_res
        free: list[Obj] = []
        if lr is not None:
            age = max(0.0, t - lr["t"])
            self.lag_ms = 0.8 * self.lag_ms + 0.2 * age * 1000
            v = w.speed_kph / 3.6
            d, theta = v * age, yaw_shift * age
            w.lanes_t = lr["t"]
            if lr.get("edges") is not None:
                w.road_edges = advance_edges(lr["edges"], d, theta)
            w.lanes = advance_lanes(lr["lanes"], d, theta) if age < 0.5 else lr["lanes"]
            if age > 0.5:
                w.lanes = replace(w.lanes, conf=w.lanes.conf * 0.5)
            w.stop_line_dist = (lr["stop"] - d) if lr["stop"] is not None else 999.0
            w.crosswalk_dist = (lr["cross"] - d) if lr["cross"] is not None else 999.0
            w.clear_ahead_m = max(0.0, lr["clear_to"] - d)
            w.extra_hidden_m = lr["hidden_to"]
            w.vanish_x = lr["vp_x"]
            for o in lr["obstacles"]:
                o2 = Obj(o.kind, o.box, o.score, x=o.x - d, y=o.y, extra=dict(o.extra))
                free.append(o2)
            # what the dashboard shows
            self.last_screen.chains_screen = [(c[0].astype(np.int32), c[1], c[2]) for c in lr["chains"]]
            if lr.get("paint") is not None:
                hh, ww = lr["shape"]
                paint = np.unpackbits(lr["paint"])[:hh * ww].reshape(hh, ww) * 255
                self.last_screen.masks = _Masks(paint.astype(np.uint8))
        else:
            w.lanes = Lanes()

        # ---- speed + turn rate from the road moving under us (every frame, ~1 ms) ----
        try:
            hidden = hidden_from_box(geo, self.ego_box, frame.shape[1])
            c = w.lanes.center[2] if w.lanes.conf > 0.4 else 0.0
            w.vo = self.vo.update(frame, geo, t, hidden, c,
                                  v_prior=w.speed_kph / 3.6 if w.speed_valid else None)
        except Exception:  # noqa: BLE001
            w.vo = None

        # ---- objects: lights, signs (worker) + free-space obstacles (lane worker) ----
        lights, others = [], []
        orr = self.obj_res
        if orr is not None and t - orr["t"] < 0.6:
            for o in orr["lamps"]:
                ob = Obj(o.kind, o.box, o.score, extra=dict(o.extra))
                if ob.score >= p["light_min_score"]:
                    lights.append(ob)
            for o in orr["stops"]:
                ob = Obj(o.kind, o.box, o.score, extra=dict(o.extra))
                if ob.score < p["object_min_score"]:
                    continue
                ob.x, ob.y = sign_position(geo, ob.box, STOP_SIGN_H_M, self.cal.cam_to_bumper_m)
                others.append(ob)
        others += free
        others = self.tracker.update(others, t)
        lane_cy = w.lanes.center[2] if w.lanes.conf > 0 else 0.0
        w.signal = self.signals.update(frame, lights, lane_cy, p["flash_window_s"], t)
        w.objects = others + lights
        for ob in others:
            if ob.kind == "stop_sign" and 0 < ob.x < 70 and -3.0 < ob.y - lane_cy < 9.0:
                w.stop_sign_dist = min(w.stop_sign_dist, ob.x)

        # ---- scene ----
        w.emergency_near = self.emerg.update(frame, [o for o in others if o.kind in ("car", "emergency")], t)
        small = cv2.resize(frame, (64, 36))
        w.tunnel = float(small[18:, :].mean()) < 55
        w.road_conf = w.lanes.conf
        w.horizon_y = geo.horizon_y
        tm = dict(lr["timings"]) if lr is not None else {}
        tm["main"] = time.perf_counter() - t_start
        tm["total"] = tm["main"]
        w.timings = tm
        self.ms = 0.9 * self.ms + 0.1 * tm["main"] * 1000
        self.rate.tick()
        self.last_objects = w.objects
        return w

    # ------------------------------------------------------------------ misc (dashboard / app)
    @property
    def screen(self):
        return self.last_screen

    def status(self) -> dict:
        return {
            "detector": f"no-AI vision on {self.n_lanes + 1} extra cores",
            "device": "CPU (parallel)",
            "det_ms": 0.0, "det_fps": 0.0,
            "vision_fps": self.rate.value,
            "lanes_fps": self.lane_rate.value,
            "lights_fps": self.obj_rate.value,
            "vision_ms": self.lane_ms,
            "main_ms": self.ms,
            "lag_ms": self.lag_ms,
            "ocr": self.ocr.status if self.ocr else "off",
            "autocal": self.autocal_message,
            "error": self.error,
        }

    def close(self) -> None:
        if self.ocr:
            self.ocr.stop()
        for q in self.lane_q + ([self.obj_q] if self.obj_q else []):
            try:
                q.put(None)
            except Exception:  # noqa: BLE001
                pass
        for pr in self.procs:
            pr.join(timeout=1.0)
            if pr.is_alive():
                pr.terminate()
        if self.ring is not None:
            self.ring.close()


def make_perception(settings: Settings, on_calibrated=None, use_detector: bool = False, log=print):
    """Parallel engine when the PC has the cores for it, else the single-process pipeline."""
    from .pipeline import FastPerception
    if os.environ.get("FSD_SINGLE_PROCESS") or use_detector:
        return FastPerception(settings, on_calibrated=on_calibrated, use_detector=use_detector)
    try:
        return ParallelPerception(settings, on_calibrated=on_calibrated)
    except Exception as e:  # noqa: BLE001
        log(f"Parallel vision unavailable ({e}); using one process")
        return FastPerception(settings, on_calibrated=on_calibrated, use_detector=use_detector)


__all__ = ["ParallelPerception", "FrameRing", "advance_lanes", "shift_poly", "make_perception", "LaneLine"]
