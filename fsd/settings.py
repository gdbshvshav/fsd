"""App settings that are not driving parameters: screen, calibration, keys, AI.

Saved in data/settings.json. API keys come from the .env file (never committed).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "data" / "settings.json"
ENV_FILE = ROOT / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    """Minimal .env loader: KEY=value lines, # comments. Does not override real env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


@dataclass
class Calibration:
    # Frame size the calibration was made at. Everything scales from this.
    width: int = 1920
    height: int = 1080
    # Ground trapezoid in the Roblox window (fractions of width/height):
    # bottom-left, bottom-right, top-right, top-left of your lane, from 1 m to 40 m
    # ahead of the front bumper. Defaults are measured from a real ER:LC screenshot
    # (default chase camera); F4 re-measures it for your exact camera.
    src_quad: list = field(default_factory=lambda: [
        [0.4598, 0.5182], [0.5441, 0.5182], [0.5204, 0.3904], [0.4951, 0.3904]])
    # Real-world size of that trapezoid: width (m) at the bottom and depth (m) bottom->top
    quad_width_m: float = 3.7
    quad_depth_m: float = 39.0
    # Distance (m) from the car's front bumper to the bottom edge of the trapezoid
    bumper_to_quad_m: float = 1.0
    # Vertical field of view of the game camera (deg). Roblox's default Camera.FieldOfView
    # is 70 and it is VERTICAL. Used for sign/light distances and auto-calibration.
    vfov_deg: float = 70.0
    # Distance from the game camera to the car's front bumper (ER:LC's chase cam is high and far back)
    cam_to_bumper_m: float = 15.75
    # Units printed on ER:LC speed limit signs (US-style map -> mph)
    sign_unit: str = "mph"
    # Part of the screen covered by YOUR car (x1, y1, x2, y2 fractions). In ER:LC's chase
    # camera the car sits in the middle of the screen, not at the bottom. Detections there
    # are ignored, and the box follows your car automatically while driving.
    ego_mask: list = field(default_factory=lambda: [0.455, 0.47, 0.545, 0.615])
    # Speedometer region (fractions x1,y1,x2,y2). None = auto-find on first drive.
    speedo_roi: list | None = None
    speed_unit: str = "auto"   # "auto" | "kph" | "mph"


@dataclass
class Settings:
    monitor: int = 0                # which monitor ER:LC is on (0 = primary)
    capture_fps: int = 60
    engage_key: str = "F1"
    turn_left_key: str = "Q"
    turn_right_key: str = "E"
    record_key: str = "F2"          # training recorder on/off
    camera_key: str = "F3"          # re-aim camera now
    vehicle: str = "default"
    ui_monitor: int = 1             # dashboard fills this monitor if you have 2 screens
    ui_overlay: bool = False        # True = always use the small floating window, even with 2 screens
    ui_on_top: bool = True          # stay above Roblox
    ui_opacity: float = 0.92        # floating window see-through amount (0.3 - 1.0)
    ui_size: list = field(default_factory=lambda: [800, 450])   # floating window size (px)
    ui_pos: list | None = None      # floating window position; None = top-right corner
    model_path: str = "models/erlc.pt"   # trained ER:LC model (made by training mode)
    fallback_model: str = "yolo11s.pt"   # COCO model: cars, people, lights, stop signs
    device: str = "auto"            # "auto" | "cuda" | "cpu"
    gemini_model: str = "gemini-2.5-flash"
    groq_model: str = "llama-3.3-70b-versatile"
    ai_camera_check_s: float = 3.0
    calibration: Calibration = field(default_factory=Calibration)

    # ------------------------------------------------------------------
    @property
    def gemini_key(self) -> str:
        return os.environ.get("GEMINI_API_KEY", "")

    @property
    def groq_key(self) -> str:
        return os.environ.get("GROQ_API_KEY", "")

    @classmethod
    def load(cls) -> "Settings":
        load_env()
        s = cls()
        if SETTINGS_FILE.exists():
            try:
                raw = json.loads(SETTINGS_FILE.read_text())
                cal = raw.pop("calibration", {}) or {}
                for k, v in raw.items():
                    if hasattr(s, k):
                        setattr(s, k, v)
                for k, v in cal.items():
                    if hasattr(s.calibration, k):
                        setattr(s.calibration, k, v)
            except (json.JSONDecodeError, OSError):
                pass
        s.gemini_model = os.environ.get("GEMINI_MODEL", s.gemini_model)
        s.groq_model = os.environ.get("GROQ_MODEL", s.groq_model)
        return s

    def save(self) -> None:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.replace(tmp, SETTINGS_FILE)
