"""Tunable parameters, their safe bounds, and versioned storage.

Every number the driving stack uses lives here. The learning system (Gemini
disengage review, Groq tuning) may only change values listed in SPEC, and only
inside the listed [min, max] range. Every change is saved as a new numbered
version so you can roll back instantly.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PARAMS_DIR = ROOT / "data" / "params"


@dataclass(frozen=True)
class Spec:
    default: float
    lo: float
    hi: float
    doc: str


# name -> Spec. Units: m (metres), s (seconds), kph, px at 1080p, duty 0..1
SPEC: dict[str, Spec] = {
    # ---- speed ----
    "default_speed_limit_kph": Spec(60, 20, 130, "Speed used before any speed-limit sign is read"),
    "speed_over_limit_kph": Spec(3, 0, 15, "How far above the posted limit to cruise"),
    "max_speed_kph": Spec(140, 40, 220, "Hard cap no matter what"),
    "curve_lat_accel": Spec(3.2, 1.5, 7.0, "Max sideways accel in curves (m/s^2); lower = slower in bends"),
    "comfort_decel": Spec(3.0, 1.5, 7.0, "Normal braking (m/s^2) when stopping for lights/signs"),
    "max_decel": Spec(7.0, 4.0, 12.0, "Emergency braking (m/s^2)"),
    "stop_line_offset_m": Spec(1.2, 0.0, 4.0, "Stop this far before the stop line"),
    "stop_sign_fallback_m": Spec(4.0, 1.0, 10.0, "If no stop line is seen, stop this far before the sign"),
    "stop_sign_wait_s": Spec(1.5, 0.5, 5.0, "Full-stop pause at stop signs / flashing red"),
    "yellow_commit_s": Spec(1.6, 0.5, 4.0, "On yellow, go through if we'd reach the line within this time"),
    "light_to_stop_m": Spec(12.0, 4.0, 25.0, "No stop line seen: stop this far before the signal head (ER:LC lights hang over the junction)"),
    # ---- following ----
    "follow_time_gap_s": Spec(1.8, 0.8, 4.0, "Gap to the car ahead in seconds"),
    "follow_min_gap_m": Spec(5.0, 2.0, 15.0, "Standstill gap to the car ahead"),
    "pedestrian_buffer_m": Spec(8.0, 3.0, 20.0, "Stop this far from people in the path"),
    "corridor_half_width_m": Spec(1.6, 1.0, 3.0, "Half-width of the driving corridor for obstacle checks"),
    # ---- lateral control ----
    "lookahead_base_m": Spec(7.0, 3.0, 20.0, "Pure-pursuit lookahead at 0 speed"),
    "lookahead_gain_s": Spec(0.55, 0.2, 1.5, "Extra lookahead per m/s of speed"),
    "lookahead_max_m": Spec(32.0, 12.0, 60.0, "Max lookahead"),
    "steer_gain": Spec(1.15, 0.3, 3.0, "Curvature -> steering duty"),
    "steer_ff_gain": Spec(1.0, 0.0, 2.0, "Feed-forward from road curvature (1 = exactly the bend)"),
    "auto_merge": Spec(1.0, 0.0, 1.0, "1 = merge by itself (with blinker) when our lane ends"),
    "auto_route": Spec(1.0, 0.0, 1.0, "1 = pick a turn by itself when the road ends (T-junction)"),
    "steer_adapt": Spec(1.0, 0.0, 1.0, "1 = learn this car's steering strength while driving"),
    "cam_yaw_lag_s": Spec(0.2, 0.0, 0.8, "How far the game camera lags the car's turning (s)"),
    "steer_damping": Spec(0.35, 0.0, 1.5, "Damps yaw rate to stop overshoot/swerving"),
    "steer_rate_limit": Spec(4.0, 1.0, 12.0, "Max change in steering duty per second"),
    "steer_speed_scale": Spec(0.0, 0.0, 0.06, "Extra gentleness per kph (0 = rely on the steering model)"),
    # ---- the car's keyboard steering (Roblox chassis), see control/steer_model.py ----
    "wheelbase_m": Spec(2.9, 2.2, 5.0, "Distance between axles"),
    "steer_lock_deg": Spec(32.0, 15.0, 45.0, "Full-lock wheel angle at walking speed"),
    "steer_falloff_ms": Spec(28.0, 10.0, 60.0, "Speed (m/s) at which full lock has halved"),
    "steer_rise_per_s": Spec(3.9, 1.0, 10.0, "How fast holding A/D moves the steering (fraction of lock per s)"),
    "steer_return_per_s": Spec(6.0, 1.0, 15.0, "How fast the steering springs back when released"),
    "lane_center_bias_m": Spec(0.0, -0.8, 0.8, "Shift inside the lane (+ = right)"),
    # ---- longitudinal control ----
    "throttle_kp": Spec(0.09, 0.01, 0.4, "Throttle per kph of speed error"),
    "throttle_ki": Spec(0.015, 0.0, 0.1, "Throttle integral"),
    "throttle_ff": Spec(0.006, 0.0, 0.02, "Base throttle per kph to hold speed"),
    "brake_kp": Spec(0.12, 0.02, 0.5, "Brake per kph of overspeed"),
    "coast_band_kph": Spec(2.0, 0.0, 8.0, "Dead-band where we neither throttle nor brake"),
    # ---- turns ----
    "turn_speed_kph": Spec(22, 10, 40, "Speed through intersection turns"),
    "turn_entry_m": Spec(5.0, 0.0, 15.0, "Distance past the stop line before starting to steer (left turns add a lane)"),
    "turn_right_radius_m": Spec(9.0, 5.0, 20.0, "Right-turn arc radius"),
    "turn_left_radius_m": Spec(15.0, 8.0, 30.0, "Left-turn arc radius"),
    "turn_preview_m": Spec(60.0, 20.0, 150.0, "Show the Q/E turn preview this far before the junction"),
    "turn_timeout_s": Spec(90.0, 20.0, 300.0, "Forget a queued Q/E turn after this long"),
    "lane_change_s": Spec(3.0, 1.5, 6.0, "Duration of a lane change"),
    # ---- perception ----
    "lane_min_conf": Spec(0.35, 0.1, 0.9, "Below this lane confidence we slow down"),
    "lane_smoothing": Spec(0.35, 0.05, 0.9, "Lane fit smoothing (higher = steadier, slower to react)"),
    "lane_width_m": Spec(3.7, 2.8, 5.0, "Expected lane width"),
    "light_min_score": Spec(0.35, 0.1, 0.9, "Minimum detector score for traffic lights"),
    "object_min_score": Spec(0.35, 0.1, 0.9, "Minimum detector score for cars/people/cones"),
    "flash_window_s": Spec(2.0, 1.0, 4.0, "Window to decide a red light is flashing (outage)"),
    # ---- camera ----
    "camera_target_horizon": Spec(0.40, 0.25, 0.55, "Where the horizon should sit (fraction of screen height)"),
    "camera_yaw_gain": Spec(0.25, 0.0, 1.0, "How fast the camera re-centres sideways"),
    "camera_pitch_gain": Spec(0.2, 0.0, 1.0, "How fast the camera fixes its tilt"),
}

VEHICLE_SPEC_KEYS = {
    "wheelbase_m", "steer_lock_deg", "steer_falloff_ms", "steer_rise_per_s", "steer_return_per_s",
    "steer_gain", "steer_damping", "steer_ff_gain", "comfort_decel", "max_decel",
    "curve_lat_accel", "throttle_kp", "throttle_ff", "brake_kp", "turn_right_radius_m",
    "turn_left_radius_m", "turn_speed_kph",
}


def defaults() -> dict[str, float]:
    return {k: v.default for k, v in SPEC.items()}


def clamp_changes(changes: dict[str, Any]) -> dict[str, float]:
    """Keep only known params, as floats, inside their bounds."""
    out: dict[str, float] = {}
    for k, v in (changes or {}).items():
        if k not in SPEC:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        s = SPEC[k]
        out[k] = min(max(fv, s.lo), s.hi)
    return out


class ParamStore:
    """Thread-safe params with numbered versions on disk.

    data/params/v0001.json, v0002.json ... and current.txt naming the live one.
    Per-vehicle overrides live in data/params/vehicles.json.
    """

    def __init__(self, directory: Path = PARAMS_DIR):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.vehicle = "default"
        self._values = defaults()
        self._vehicles: dict[str, dict[str, float]] = {}
        self.version = 0
        self._load()

    # ---------- disk ----------
    def _versions(self) -> list[int]:
        out = []
        for p in self.dir.glob("v*.json"):
            try:
                out.append(int(p.stem[1:]))
            except ValueError:
                pass
        return sorted(out)

    def _load(self) -> None:
        cur = self.dir / "current.txt"
        vers = self._versions()
        if not vers:
            self._write_version(defaults(), note="factory defaults")
            return
        v = vers[-1]
        if cur.exists():
            try:
                v = int(cur.read_text().strip())
            except ValueError:
                pass
        data = json.loads((self.dir / f"v{v:04d}.json").read_text())
        merged = defaults()
        merged.update(clamp_changes(data.get("values", {})))
        self._values = merged
        self.version = v
        vf = self.dir / "vehicles.json"
        if vf.exists():
            try:
                raw = json.loads(vf.read_text())
                self._vehicles = {name: clamp_changes(vals) for name, vals in raw.items()}
            except json.JSONDecodeError:
                self._vehicles = {}

    def _write_version(self, values: dict[str, float], note: str, source: str = "manual") -> int:
        v = (self._versions() or [0])[-1] + 1
        payload = {"version": v, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "note": note, "source": source, "values": values}
        tmp = self.dir / f"v{v:04d}.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.dir / f"v{v:04d}.json")
        (self.dir / "current.txt").write_text(str(v))
        self._values = dict(values)
        self.version = v
        return v

    # ---------- api ----------
    def get(self, key: str) -> float:
        with self._lock:
            veh = self._vehicles.get(self.vehicle, {})
            if key in veh:
                return veh[key]
            return self._values[key]

    def __getitem__(self, key: str) -> float:
        return self.get(key)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            out = dict(self._values)
            out.update(self._vehicles.get(self.vehicle, {}))
            return out

    def apply(self, changes: dict[str, Any], note: str, source: str = "manual") -> int:
        """Apply clamped changes as a new version. Returns the new version number."""
        with self._lock:
            clean = clamp_changes(changes)
            vals = dict(self._values)
            vals.update(clean)
            return self._write_version(vals, note, source)

    def set_vehicle_override(self, vehicle: str, changes: dict[str, Any]) -> None:
        with self._lock:
            clean = {k: v for k, v in clamp_changes(changes).items() if k in VEHICLE_SPEC_KEYS}
            self._vehicles.setdefault(vehicle, {}).update(clean)
            (self.dir / "vehicles.json").write_text(json.dumps(self._vehicles, indent=2))

    def rollback(self) -> int:
        """Go back to the version before the current one."""
        with self._lock:
            vers = [v for v in self._versions() if v < self.version]
            if not vers:
                return self.version
            v = vers[-1]
            data = json.loads((self.dir / f"v{v:04d}.json").read_text())
            vals = defaults()
            vals.update(clamp_changes(data.get("values", {})))
            self._values = vals
            self.version = v
            (self.dir / "current.txt").write_text(str(v))
            return v

    def history(self) -> list[dict]:
        out = []
        for v in self._versions():
            try:
                d = json.loads((self.dir / f"v{v:04d}.json").read_text())
                out.append({"version": v, "time": d.get("time"), "note": d.get("note"), "source": d.get("source")})
            except (OSError, json.JSONDecodeError):
                continue
        return out

    def reset_factory(self) -> int:
        return self.apply(defaults(), "factory reset", "manual")


def describe_spec() -> str:
    """Text table of tunables for AI prompts."""
    lines = []
    for k, s in SPEC.items():
        lines.append(f"{k}: default {s.default}, range [{s.lo}, {s.hi}] - {s.doc}")
    return "\n".join(lines)


def copy_tree_safe(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
