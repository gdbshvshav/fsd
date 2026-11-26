"""Start-up health check: is everything FSD needs actually working on THIS PC?

About 15 s after start (once the vision has had time to settle) it checks, and
reports in the dashboard log and on Discord, in plain words:

  * frames per second of the control loop (60 is the goal; under 30 = slow PC / capture)
  * lane detection rate (how many lane results per second the vision workers deliver)
  * whether the speedometer has been found and is being read (and how: digit reader /
    OCR / road motion)
  * whether lane lines are being seen at all (a car on the road, camera not moved)
  * whether the calibration looks like it matches the camera (horizon where expected)
  * night mode on/off

Each line says what to do if it's wrong. It's run again whenever you ask (F12 in the
app re-runs it), and the result is also kept for the Discord disengage log.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HealthReport:
    ok: bool = True
    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def add(self, good: bool, text: str, fix: str = "") -> None:
        self.lines.append(("OK  " if good else "FIX ") + text)
        if not good:
            self.ok = False
            self.problems.append(text + (f" - {fix}" if fix else ""))


def check(status: dict, world, fps: float, seconds_running: float) -> HealthReport:
    r = HealthReport()
    # control loop
    if fps >= 50:
        r.add(True, f"control loop {fps:.0f} fps")
    elif fps >= 30:
        r.add(True, f"control loop {fps:.0f} fps (60 is the goal - close other programs if you can)")
    else:
        r.add(False, f"control loop only {fps:.0f} fps",
              "close the browser / OBS, set Roblox graphics lower, or run with fewer screens")
    lanes_fps = status.get("lanes_fps")
    if lanes_fps is not None:
        r.add(lanes_fps >= 15, f"lane detection {lanes_fps:.0f}/s",
              "the vision workers are starved: close heavy programs")
    # speedometer
    if world is not None and world.speed_valid:
        how = "road motion" if status.get("visual_speed") else "speedometer"
        r.add(True, f"speed read ({world.speed_kph:.0f} km/h, via {how})")
    else:
        ocr = status.get("ocr", "")
        r.add(False, "speed not read yet" + (f" ({ocr})" if ocr else ""),
              "drive a little so the speedometer shows a number; keep the ER:LC HUD visible")
    # lanes
    if world is not None:
        conf = world.lanes.conf
        r.add(conf > 0.3 or seconds_running < 10, f"lane lines {'seen' if conf > 0.3 else 'not seen'} "
              f"(confidence {conf:.2f})",
              "be on a road with lines; if you are, press F4 to recalibrate the camera")
        # calibration sanity: the lane should be roughly in front of us
        if conf > 0.4:
            off = world.lanes.offset
            r.add(abs(off) < 2.5, f"lane centre {off:+.1f} m from the car",
                  "the camera may have moved: press F4 to recalibrate")
    night = status.get("night_gain", 1.0)
    if night and night > 1.6:
        r.lines.append(f"INFO night mode: picture brightened x{night:.1f}")
    kind = status.get("road_kind")
    if kind and kind != "unknown":
        r.lines.append(f"INFO road type: {kind}")
    return r
