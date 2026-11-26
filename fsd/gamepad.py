"""Drive ER:LC with a virtual Xbox controller instead of W/A/S/D.

Why: a key is either fully on or fully off, so holding the wheel at 12% or the throttle
at 30% meant flicking keys 60 times a second. The car felt twitchy and the speed
pumped (gas, brake, gas...). A controller is analog:

    left stick X  -> steering   (-1 full left .. +1 full right, any value in between)
    right trigger -> throttle   (0..1)
    left trigger  -> brake / reverse (0..1)

It uses the free ViGEmBus driver through the `vgamepad` package (pip install vgamepad;
on first install it runs the ViGEmBus installer - click through it once). If the
driver isn't there, FSD falls back to the keyboard by itself and says so.

Roblox only listens to one kind of input at a time: when you touch the keyboard it
switches to keyboard, when the controller moves it switches back. While FSD drives we
re-send the controller state every tick, so it stays in charge; your W/A/S/D still
take over immediately (that is how a takeover is detected anyway).
"""
from __future__ import annotations

import threading
import time

from .winput import Actuator


def gamepad_available() -> tuple[bool, str]:
    try:
        import vgamepad as vg  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"vgamepad not installed ({type(e).__name__})"
    try:
        pad = vg.VX360Gamepad()
        pad.reset()
        pad.update()
        del pad
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, f"ViGEmBus driver missing or blocked ({e})"


class GamepadActuator(Actuator):
    """Same interface as the keyboard Actuator (command(steer, pedal), set_enabled, tick...)."""

    kind = "controller"

    def __init__(self, hz: float = 120.0, pad=None):
        super().__init__(hz)
        if pad is None:
            import vgamepad as vg
            pad = vg.VX360Gamepad()
        self.pad = pad
        self.out_steer = 0.0          # what the stick is at
        self.out_throttle = 0.0
        self.out_brake = 0.0
        self._pad_lock = threading.Lock()
        # Roblox ignores the first few percent of stick travel (dead zone, ~0.1-0.2 depending on the
        # chassis). Map 0 -> 0 and anything else past the dead zone so small corrections still count.
        self.stick_dead = 0.12
        self.trigger_dead = 0.05
        self.steer_obs.dither = False

    # ------------------------------------------------------------- mapping
    def _stick(self, s: float) -> float:
        if abs(s) < 1e-3:
            return 0.0
        d = self.stick_dead
        return (1 if s > 0 else -1) * min(1.0, d + (1 - d) * abs(s))

    def _trig(self, v: float) -> float:
        if v < 1e-3:
            return 0.0
        d = self.trigger_dead
        return min(1.0, d + (1 - d) * v)

    def _send(self, steer: float, throttle: float, brake: float) -> None:
        with self._pad_lock:
            try:
                self.pad.left_joystick_float(x_value_float=float(self._stick(steer)), y_value_float=0.0)
                self.pad.right_trigger_float(value_float=float(self._trig(throttle)))
                self.pad.left_trigger_float(value_float=float(self._trig(brake)))
                self.pad.update()
            except Exception:  # noqa: BLE001 - never crash the drive loop
                pass
        self.out_steer, self.out_throttle, self.out_brake = steer, throttle, brake

    def release_all(self) -> None:
        try:
            self._send(0.0, 0.0, 0.0)
        except Exception:  # noqa: BLE001
            pass
        super().release_all()

    # ------------------------------------------------------------- the 120 Hz loop
    def tick(self) -> tuple[int, int]:
        with self._lock:
            if not self.enabled:
                return 0, 0
            self.stale = time.perf_counter() - self.last_cmd > self.max_age
            if self.stale:
                s_cmd, pedal = 0.0, 0.0
            else:
                s_cmd, pedal = self.steer, self.pedal
            if pedal < 0 and self.speed_kph < 1.5 and not self.allow_reverse:
                pedal = 0.0
            # the game's own steering value follows the stick at the chassis' turn rate - same model
            # as with keys, but now the target is any value, not just -1/0/+1
            self.steer_obs.s = _approach(self.steer_obs.s, s_cmd, self.steer_obs.rise, self.steer_obs.ret, self.dt)
        self._send(s_cmd, max(0.0, pedal), max(0.0, -pedal))
        self.pressed_log = [f"pad {s_cmd:+.2f} {pedal:+.2f}"]
        return (1 if s_cmd > 0.05 else -1 if s_cmd < -0.05 else 0), (1 if pedal > 0.05 else -1 if pedal < -0.05 else 0)


def _approach(s: float, target: float, rise: float, ret: float, dt: float) -> float:
    """Roblox chassis steering: moves toward the input at `rise` (away from centre) or
    max(rise, ret) (back toward / through centre)."""
    if abs(target) > abs(s) and target * s >= 0:
        rate = rise
    else:
        rate = max(rise, ret)
    step = rate * dt
    if target > s:
        return min(target, s + step)
    return max(target, s - step)


def _offer_driver_install(log) -> None:
    """vgamepad ships the ViGEmBus installer; open it once (you click through it, Windows asks
    for permission) so the controller works after the next start."""
    import os
    from pathlib import Path
    flag = Path("data") / "vigem_installer_opened"
    if os.name != "nt" or flag.exists():
        return
    try:
        import vgamepad
        base = Path(vgamepad.__file__).parent
        msi = next(iter(sorted(base.rglob("ViGEmBusSetup_x64.msi"))), None)
        if msi is None:
            return
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text("1")
        os.startfile(str(msi))  # type: ignore[attr-defined]
        log("Opened the controller driver installer (ViGEmBus). Click through it, then restart FSD "
            "for smooth controller driving.")
    except Exception:  # noqa: BLE001
        pass


def make_actuator(mode: str = "auto", log=print) -> Actuator:
    """mode: 'auto' (controller if it works, else keyboard) | 'controller' | 'keyboard'."""
    if mode in ("auto", "controller"):
        ok, why = gamepad_available()
        if ok:
            log("Driving with a virtual Xbox controller (analog steering, gas and brake).")
            return GamepadActuator()
        log(f"Controller not available: {why}. Using W/A/S/D for now.")
        if "driver" in why:
            _offer_driver_install(log)
    return Actuator()
