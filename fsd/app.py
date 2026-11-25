"""ER:LC FSD application: wires capture -> perception -> planner -> controls,
plus hotkeys, recorder, learning and the dashboard.

Threads:
  capture   newest screen frame (dxcam)
  control   perception + planning + control, as fast as frames arrive (~30-60 Hz)
  actuator  W/A/S/D key timing at 60 Hz
  ocr       speedometer / speed-limit reading
  learner   Gemini / Groq / replay tests
  keyhook   your keyboard (F1, Q/E, takeover)
  main      dashboard (pygame)
"""
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .capture import Capture, FileSource
from .control.camera_aim import CameraAim
from .control.controllers import LateralController, SpeedController
from .learning.learner import Learner
from .learning.recorder import Recorder, TrainingRecorder
from .params import ParamStore
from .perception import Perception
from .perception.calibrate import apply as apply_calibration
from .perception.calibrate import auto_calibrate
from .planning.planner import Planner
from .settings import Settings
from .ui.draw_debug import annotate
from .winput import VK, Actuator, InputMonitor, roblox_focused
from .world import Mode, Plan, World


@dataclass
class Snapshot:
    world: World | None = None
    plan: Plan | None = None
    engaged: bool = False
    debug_img: np.ndarray | None = None
    status: dict = field(default_factory=dict)
    training_on: bool = False
    training_count: int = 0


class App:
    def __init__(self, source: str | None = None, dry_run: bool = False):
        self.settings = Settings.load()
        self.store = ParamStore()
        self.store.vehicle = self.settings.vehicle
        self.dry_run = dry_run
        self.capture = FileSource(source) if source else Capture(self.settings.monitor, self.settings.capture_fps)
        self.perception = Perception(self.settings, on_calibrated=self.settings.save)
        self.planner = Planner()
        self.lat = LateralController()
        self.lon = SpeedController()
        self.actuator = Actuator()
        self.camera = CameraAim()
        self.recorder = Recorder()
        self.recorder.on_saved = self._scenario_saved
        self.trainrec = TrainingRecorder()
        self.engaged = False
        self.learner = Learner(self.store, self.settings, lambda: self.engaged, self.camera)
        self._snap = Snapshot()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.prompt_open = False
        self.prompt_since = 0.0
        self.pending_scenario = None
        self.calib_frames: list[np.ndarray] | None = None
        self.fps = 0.0
        self.paused_reason = ""
        self._last_tune = time.perf_counter()
        self._force_camera = False
        handlers = {
            VK[self.settings.engage_key]: self.toggle_engage,
            VK[self.settings.turn_left_key]: lambda: self.turn("left"),
            VK[self.settings.turn_right_key]: lambda: self.turn("right"),
            VK[self.settings.record_key]: self.toggle_training,
            VK[self.settings.camera_key]: self.aim_camera,
            VK["F4"]: self.start_calibration,
            VK["F6"]: self.rollback,
            VK["F7"]: self.open_prompt,
        }
        self.inputs = InputMonitor(handlers, self._driver_key)

    # ------------------------------------------------------------------ hotkeys
    def toggle_engage(self) -> None:
        if self.engaged:
            self.disengage("F1")
            return
        if not self.dry_run and not roblox_focused():
            self.learner.say("Click into Roblox first, then press F1.")
            return
        self.planner.reset()
        self.lat.reset()
        self.lon.reset()
        self.engaged = True
        self.actuator.set_enabled(not self.dry_run)
        self.learner.say("FSD engaged" + (" (dry run: no keys pressed)" if self.dry_run else ""))

    def disengage(self, why: str) -> None:
        if not self.engaged:
            return
        self.engaged = False
        self.actuator.set_enabled(False)
        speed = self._snap.world.speed_kph if self._snap.world else 0.0
        self.learner.say(f"Disengaged ({why})")
        if why == "takeover" or speed > 5:
            self.recorder.trigger(time.perf_counter(), why, {"params_version": self.store.version,
                                                              "vehicle": self.store.vehicle})

    def _driver_key(self, vk: int) -> None:
        # runs inside the keyboard hook: keep it tiny
        if self.engaged:
            threading.Thread(target=self.disengage, args=("takeover",), daemon=True).start()

    def turn(self, direction: str) -> None:
        msg = self.planner.turns.press(direction, time.perf_counter())
        self.learner.say(msg)

    def toggle_training(self) -> None:
        on = self.trainrec.toggle()
        self.learner.say("Training recorder ON - just drive normally" if on else
                         f"Training recorder OFF ({self.trainrec.count} frames). Run label/review/train when ready.")

    def aim_camera(self) -> None:
        self._force_camera = True
        self.learner.say("Re-aiming camera")

    def start_calibration(self) -> None:
        self.calib_frames = []
        self.learner.say("Calibrating: hold a straight line in the middle of your lane...")

    def rollback(self) -> None:
        v = self.store.rollback()
        self.learner.say(f"Rolled back to params v{v}")

    def open_prompt(self) -> None:
        if self.pending_scenario is not None:
            self.prompt_open = True
            self.prompt_since = time.perf_counter()

    def _scenario_saved(self, d) -> None:
        self.pending_scenario = d
        self.prompt_open = True
        self.prompt_since = time.perf_counter()
        self.learner.say(f"Saved last 30 s ({d.name}). Press F7 / click the dashboard to say what went wrong.")

    def submit_note(self, note: str) -> None:
        self.prompt_open = False
        if self.pending_scenario is not None:
            self.learner.diagnose(self.pending_scenario, note)
            self.pending_scenario = None

    # ------------------------------------------------------------------ control loop
    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snap

    def _finish_calibration(self, frames) -> None:
        fields, msg = auto_calibrate(frames, self.settings.calibration, self.store["lane_width_m"])
        if fields:
            apply_calibration(self.settings.calibration, fields)
            self.settings.save()
            self.perception.geo = None           # rebuild geometry with the new calibration
        self.learner.say(msg)

    def _row(self, w: World, plan: Plan, steer: float, pedal: float) -> dict:
        keys = [k for k in ("W", "A", "S", "D", "Q", "E") if self.inputs.is_down(k)]
        return {
            "engaged": self.engaged, "speed": round(w.speed_kph, 2), "speed_valid": w.speed_valid,
            "limit": w.speed_limit_kph, "target": round(plan.target_speed_kph, 2), "mode": plan.mode.value,
            "reason": plan.reason, "steer": round(steer, 3), "pedal": round(pedal, 3),
            "light": w.signal.state.value, "light_d": round(min(w.signal.distance, 999), 1),
            "stop_line": round(min(w.stop_line_dist, 999), 1), "stop_sign": round(min(w.stop_sign_dist, 999), 1),
            "crosswalk": round(min(w.crosswalk_dist, 999), 1), "lane_conf": round(w.lanes.conf, 2),
            "lane_off": round(w.lanes.offset, 2), "lane_head": round(w.lanes.heading, 3),
            "yaw": round(w.yaw_rate, 3), "keys": keys, "turn": plan.turn_pending,
            "emergency": w.emergency_near,
        }

    def _tune_stats(self) -> dict:
        rows = [r for r in self.recorder.rows if r.get("engaged")]
        if len(rows) < 100:
            return {}
        steer = np.array([r["steer"] for r in rows])
        err = np.array([r["target"] - r["speed"] for r in rows])
        modes = {}
        for r in rows:
            modes[r["mode"]] = modes.get(r["mode"], 0) + 1
        return {
            "seconds": round(rows[-1]["t"] - rows[0]["t"], 1),
            "lane_offset_std_m": round(float(np.std([r["lane_off"] for r in rows])), 3),
            "steer_jerk": round(float(np.mean(np.abs(np.diff(steer)))), 4),
            "steer_sign_flips_per_min": round(float(np.sum(np.diff(np.sign(steer)) != 0)) /
                                              max(1, (rows[-1]["t"] - rows[0]["t"]) / 60), 1),
            "speed_error_mean_kph": round(float(np.mean(err)), 2),
            "speed_error_abs_kph": round(float(np.mean(np.abs(err))), 2),
            "hard_brakes": int(np.sum(np.array([r["pedal"] for r in rows]) < -0.7)),
            "mean_lane_conf": round(float(np.mean([r["lane_conf"] for r in rows])), 2),
            "mode_share": {k: round(v / len(rows), 2) for k, v in modes.items()},
        }

    def control_loop(self) -> None:
        last_seq = -1
        last_t = time.perf_counter()
        last_dbg = 0.0
        frames_n, fps_t = 0, time.perf_counter()
        dbg = None
        while not self._stop.is_set():
            frame, stamp, seq = self.capture.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.002)
                continue
            last_seq = seq
            now = time.perf_counter()
            dt = min(0.2, now - last_t)
            last_t = now
            p = self.store.snapshot()

            w = self.perception.process(frame, p, t=now)
            self.actuator.speed_kph = w.speed_kph

            # pause (don't steer) if Roblox lost focus, so keys never go to another app
            focused = self.dry_run or isinstance(self.capture, FileSource) or roblox_focused()
            active = self.engaged and focused
            if self.engaged:
                want = active and not self.dry_run
                if want != self.actuator.enabled:
                    self.actuator.set_enabled(want)
            self.paused_reason = "" if focused or not self.engaged else "Paused: Roblox isn't the active window"

            plan = self.planner.step(w, p, active, now)
            steer = pedal = 0.0
            if active:
                steer = self.lat.step(w, plan, p, dt)
                pedal = self.lon.step(w, plan, p, dt)
                self.actuator.command(steer, pedal)
                if plan.mode == Mode.FAULT and w.speed_kph < 2:
                    self.disengage("lost road")
            if self.paused_reason:
                plan.alert = self.paused_reason

            # camera
            self.camera.update(w, p, active, frame.shape[1], frame.shape[0], force=self._force_camera)
            if self._force_camera:
                self.learner.camera_check(frame)
                self._force_camera = False
            elif active:
                self.learner.camera_check(frame)

            # calibration capture
            if self.calib_frames is not None:
                self.calib_frames.append(frame.copy())
                if len(self.calib_frames) >= 20:
                    frames, self.calib_frames = self.calib_frames, None
                    threading.Thread(target=self._finish_calibration, args=(frames,), daemon=True).start()

            # recording
            self.recorder.add(now, frame, self._row(w, plan, steer, pedal))
            self.trainrec.add(now, frame, w.speed_kph)
            if self.prompt_open and now - self.prompt_since > 120:
                self.submit_note("")
            if active and now - self._last_tune > 60:
                self._last_tune = now
                stats = self._tune_stats()
                if stats:
                    self.learner.tune(stats)

            # debug picture for the dashboard (~7 Hz)
            if now - last_dbg > 0.14:
                last_dbg = now
                small = annotate(frame, w, self.perception, plan)
                dbg = cv2.resize(small, (640, int(small.shape[0] * 640 / small.shape[1])), interpolation=cv2.INTER_AREA)

            frames_n += 1
            if now - fps_t > 1.0:
                self.fps = frames_n / (now - fps_t)
                frames_n, fps_t = 0, now
            st = self.perception.status()
            st.update({"fps": self.fps, "params": self.store.version, "log": list(self.learner.log)})
            if self.learner.busy:
                st["log"] = st["log"] + [self.learner.busy]
            with self._lock:
                self._snap = Snapshot(w, plan, self.engaged, dbg, st, self.trainrec.on, self.trainrec.count)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.capture.start()
        self.actuator.start()
        self.inputs.start()
        self.learner.start()
        if getattr(self.capture, "error", ""):
            self.learner.say(self.capture.error)
        threading.Thread(target=self.control_loop, name="control", daemon=True).start()

    def shutdown(self) -> None:
        self.engaged = False
        self.actuator.stop()
        self._stop.set()
        self.capture.stop()
        self.inputs.stop()
        self.learner.stop()
        self.perception.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="ER:LC FSD")
    ap.add_argument("--replay", help="folder/image to use instead of the screen (testing)")
    ap.add_argument("--dry-run", action="store_true", help="never press keys; just show what it would do")
    ap.add_argument("--headless", action="store_true", help="no dashboard window")
    a = ap.parse_args()
    app = App(a.replay, a.dry_run)
    app.start()
    if a.headless:
        try:
            while True:
                time.sleep(1)
                s = app.snapshot()
                if s.world:
                    print(f"{app.fps:4.0f} fps  {s.plan.mode.value:10s} v={s.world.speed_kph:5.1f} "
                          f"lane={s.world.lanes.conf:.2f} light={s.world.signal.state.value}")
        except KeyboardInterrupt:
            app.shutdown()
        return
    from .ui.dashboard import Dashboard
    Dashboard(app, app.settings.ui_monitor, app.settings.ui_overlay).run()


if __name__ == "__main__":
    main()
