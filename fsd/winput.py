"""Windows keyboard/mouse input.

* Output: SendInput with hardware scan codes (what Roblox reads), plus relative
  mouse moves, right-button hold and wheel for camera control.
* Input: a low-level keyboard hook that tells *your* key presses apart from the
  ones this program injects (LLKHF_INJECTED). That is how F1/Q/E work and how any
  W/A/S/D press from you instantly takes over.
* Actuator: turns smooth commands (-1..1) into key presses with sigma-delta
  modulation at the game's frame rate, so "30% left" really averages 30%.

On non-Windows machines everything becomes a harmless stub (used by the tests).
"""
from __future__ import annotations

import atexit
import ctypes
import sys
import threading
import time
from collections.abc import Callable

IS_WIN = sys.platform == "win32"

# Virtual-key codes
VK = {
    "F1": 0x70, "F2": 0x71, "F3": 0x72, "F4": 0x73, "F5": 0x74, "F6": 0x75, "F7": 0x76,
    "F8": 0x77, "F9": 0x78, "F10": 0x79, "F11": 0x7A, "F12": 0x7B,
    "Q": 0x51, "E": 0x45, "W": 0x57, "A": 0x41, "S": 0x53, "D": 0x44,
    "SPACE": 0x20, "UP": 0x26, "DOWN": 0x28, "LEFT": 0x25, "RIGHT": 0x27,
    "ESC": 0x1B, "ENTER": 0x0D,
}
# Set-1 scan codes for keys we press
SCAN = {"W": 0x11, "A": 0x1E, "S": 0x1F, "D": 0x20, "SPACE": 0x39, "Q": 0x10, "E": 0x12}

DRIVE_KEYS = {VK["W"], VK["A"], VK["S"], VK["D"], VK["UP"], VK["DOWN"], VK["LEFT"], VK["RIGHT"], VK["SPACE"]}

if IS_WIN:
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    ULONG_PTR = ctypes.c_size_t

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _U)]

    class KBDLLHOOKSTRUCT(ctypes.Structure):
        _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
    KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE = 0x0002, 0x0008
    MOUSEEVENTF_MOVE, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, MOUSEEVENTF_WHEEL = 0x0001, 0x0008, 0x0010, 0x0800
    WH_KEYBOARD_LL, WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP = 13, 0x0100, 0x0101, 0x0104, 0x0105
    LLKHF_INJECTED = 0x10
    WM_QUIT = 0x0012

    LowLevelProc = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
    user32.SetWindowsHookExW.argtypes = [ctypes.c_int, LowLevelProc, wintypes.HINSTANCE, wintypes.DWORD]
    user32.SetWindowsHookExW.restype = wintypes.HHOOK
    user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
    user32.CallNextHookEx.restype = ctypes.c_ssize_t
    user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE

    try:  # 1 ms timer resolution so our 60 Hz key timing is accurate
        ctypes.WinDLL("winmm").timeBeginPeriod(1)
    except OSError:
        pass
    try:  # real pixel coordinates on 4K / scaled displays (otherwise window rects are off)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # noqa: BLE001
        try:
            user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass

    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsIconic.argtypes = [wintypes.HWND]
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _title(hwnd) -> str:
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value

    def roblox_client_rect() -> tuple[int, int, int, int] | None:
        """Screen rectangle (x, y, w, h) of the Roblox game area, or None."""
        found = []

        def cb(hwnd, _):
            if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd) and _title(hwnd) == "Roblox":
                found.append(hwnd)
            return True
        user32.EnumWindows(WNDENUMPROC(cb), 0)
        if not found:
            return None
        hwnd = found[0]
        r = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(r))
        pt = wintypes.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(pt))
        w, h = r.right - r.left, r.bottom - r.top
        if w < 200 or h < 200:
            return None
        return pt.x, pt.y, w, h

    def window_rect(hwnd) -> tuple[int, int, int, int] | None:
        r = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
            return None
        return r.left, r.top, r.right - r.left, r.bottom - r.top

    def _send(*inputs: "INPUT") -> None:
        arr = (INPUT * len(inputs))(*inputs)
        user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))

    def key_down(name: str) -> None:
        _send(INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(0, SCAN[name], KEYEVENTF_SCANCODE, 0, 0)))

    def key_up(name: str) -> None:
        _send(INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(0, SCAN[name], KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, 0)))

    def mouse_move(dx: int, dy: int) -> None:
        _send(INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, 0)))

    def mouse_right(down: bool) -> None:
        flag = MOUSEEVENTF_RIGHTDOWN if down else MOUSEEVENTF_RIGHTUP
        _send(INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(0, 0, 0, flag, 0, 0)))

    def mouse_wheel(clicks: int) -> None:
        _send(INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(0, 0, ctypes.c_uint32(int(clicks) * 120).value,
                                                    MOUSEEVENTF_WHEEL, 0, 0)))

    def foreground_title() -> str:
        return _title(user32.GetForegroundWindow())
else:  # ---------------- stubs for Linux/macOS (tests, replays) ----------------
    LOG: list[tuple] = []

    def key_down(name: str) -> None:
        LOG.append(("down", name))

    def key_up(name: str) -> None:
        LOG.append(("up", name))

    def mouse_move(dx: int, dy: int) -> None:
        LOG.append(("move", dx, dy))

    def mouse_right(down: bool) -> None:
        LOG.append(("rmb", down))

    def mouse_wheel(clicks: int) -> None:
        LOG.append(("wheel", clicks))

    def foreground_title() -> str:
        return "Roblox"

    def roblox_client_rect():
        return None

    def window_rect(hwnd):
        return None


def roblox_focused() -> bool:
    return "roblox" in foreground_title().lower()


# ---------------------------------------------------------------------------
class InputMonitor:
    """Watches the physical keyboard. Calls handlers for key-down events.

    handlers: {vk_code: callback}. `on_drive_key` fires for any physical
    W/A/S/D/arrows/space press (takeover). Injected keys are ignored.
    """

    def __init__(self, handlers: dict[int, Callable[[], None]], on_drive_key: Callable[[int], None]):
        self.handlers = handlers
        self.on_drive_key = on_drive_key
        self.physical_down: set[int] = set()
        self._thread_id = 0
        self._thread: threading.Thread | None = None
        self._proc = None

    def start(self) -> "InputMonitor":
        if IS_WIN:
            self._thread = threading.Thread(target=self._run, name="keyhook", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        if IS_WIN and self._thread_id:
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)

    def is_down(self, name: str) -> bool:
        return VK[name] in self.physical_down

    def feed(self, vk: int, down: bool, threaded: bool = False) -> None:
        """Process one physical key event (also used by tests).

        Key state is updated right here; a hotkey handler may run on a helper
        thread so the OS hook returns quickly.
        """
        if not down:
            self.physical_down.discard(vk)
            return
        first = vk not in self.physical_down
        self.physical_down.add(vk)
        if vk in DRIVE_KEYS:
            self.on_drive_key(vk)
        if first and vk in self.handlers:
            if threaded:
                threading.Thread(target=self._call, args=(vk,), daemon=True).start()
            else:
                self._call(vk)

    def _call(self, vk: int) -> None:
        try:
            self.handlers[vk]()
        except Exception as e:  # noqa: BLE001 - never kill the hook
            print("hotkey handler error:", e)

    def _run(self) -> None:
        self._thread_id = kernel32.GetCurrentThreadId()

        def proc(n_code, w_param, l_param):
            if n_code == 0:
                kb = ctypes.cast(l_param, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                if not (kb.flags & LLKHF_INJECTED):
                    if w_param in (WM_KEYDOWN, WM_SYSKEYDOWN):
                        # Hook callbacks must return fast; hotkey work runs on its own thread.
                        self.feed(kb.vkCode, True, threaded=True)
                    elif w_param in (WM_KEYUP, WM_SYSKEYUP):
                        self.feed(kb.vkCode, False)
            return user32.CallNextHookEx(None, n_code, w_param, l_param)

        self._proc = LowLevelProc(proc)
        hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._proc, kernel32.GetModuleHandleW(None), 0)
        if not hook:
            print("Could not install keyboard hook; run as the same user as Roblox.")
            return
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        user32.UnhookWindowsHookEx(hook)


# ---------------------------------------------------------------------------
class Actuator:
    """Converts steer/pedal commands into W/A/S/D presses at the game's frame rate.

    steer: -1 (full left) .. +1 (full right)
    pedal: -1 (full brake) .. +1 (full throttle)
    Sigma-delta: every tick (1/60 s) a key is held if the running error says so,
    so the average key-down fraction equals the command. Pulses are never shorter
    than one game frame, which Roblox would miss.
    """

    def __init__(self, hz: float = 60.0):
        self.dt = 1.0 / hz
        self.steer = 0.0
        self.pedal = 0.0
        self.enabled = False
        self.allow_reverse = False
        self.speed_kph = 0.0
        self._acc_s = 0.0
        self._acc_p = 0.0
        self._held: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.pressed_log: list[str] = []  # for the recorder: keys held this tick
        self.last_cmd = 0.0
        self.max_age = 0.3                # no fresh command for this long -> release W/A/S/D
        self.stale = False
        from .control.steer_model import SteerObserver
        self.steer_obs = SteerObserver()
        atexit.register(self.release_all)

    def start(self) -> "Actuator":
        threading.Thread(target=self._run, name="actuator", daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self.release_all()

    ret_scale = 1.0          # learned per vehicle by drive/steer_check.py

    def set_steer_model(self, rise: float, ret: float) -> None:
        self.steer_obs.rise, self.steer_obs.ret = rise, ret * self.ret_scale

    def command(self, steer: float, pedal: float) -> None:
        """steer: wanted Roblox steering value (-1 full left .. +1 full right); pedal: -1 brake .. +1 throttle."""
        with self._lock:
            self.last_cmd = time.perf_counter()
            self.steer = max(-1.0, min(1.0, steer))
            self.pedal = max(-1.0, min(1.0, pedal))

    def set_enabled(self, on: bool) -> None:
        with self._lock:
            self.enabled = on
            self.last_cmd = time.perf_counter()
            if not on:
                self.steer = self.pedal = 0.0
                self._acc_s = self._acc_p = 0.0
                self.steer_obs.reset()
        if not on:
            self.release_all()

    def release_all(self) -> None:
        for k in list(self._held):
            key_up(k)
        self._held.clear()

    def tap(self, key: str, hold_s: float = 0.06) -> None:
        """Press and release a key once (turn signals Q / E) without blocking the caller."""
        def run():
            key_down(key)
            time.sleep(hold_s)
            key_up(key)
        threading.Thread(target=run, daemon=True).start()

    def _set(self, key: str, down: bool) -> None:
        if down and key not in self._held:
            key_down(key)
            self._held.add(key)
        elif not down and key in self._held:
            key_up(key)
            self._held.discard(key)

    @staticmethod
    def _sd(acc: float, cmd: float) -> tuple[float, int]:
        """One sigma-delta step. Returns (new_acc, -1/0/+1)."""
        if abs(cmd) >= 0.97:
            return 0.0, (1 if cmd > 0 else -1)
        acc += cmd
        if acc >= 0.5:
            return acc - 1.0, 1
        if acc <= -0.5:
            return acc + 1.0, -1
        return acc, 0

    def tick(self) -> tuple[int, int]:
        with self._lock:
            if not self.enabled:
                return 0, 0
            # Safety: if the brain stops sending commands (slow PC, hiccup), never
            # keep holding throttle or steering - coast until fresh commands arrive.
            self.stale = time.perf_counter() - self.last_cmd > self.max_age
            if self.stale:
                self._acc_p = 0.0
                s_cmd, pedal = 0.0, 0.0
            else:
                s_cmd, pedal = self.steer, self.pedal
            # steering: choose the key that brings the (estimated) Roblox steering value to s_cmd
            s = self.steer_obs.choose(s_cmd, self.dt)
            self.steer_obs.apply(s, self.dt)
            # never press S at a standstill (that would reverse) unless allowed
            if pedal < 0 and self.speed_kph < 1.5 and not self.allow_reverse:
                pedal = 0.0
            self._acc_p, p = self._sd(self._acc_p, pedal)
        self._set("A", s < 0)
        self._set("D", s > 0)
        self._set("W", p > 0)
        self._set("S", p < 0)
        self.pressed_log = sorted(self._held)
        return s, p

    def _run(self) -> None:
        nxt = time.perf_counter()
        while not self._stop.is_set():
            self.tick()
            nxt += self.dt
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()


class CameraMover:
    """Rotates/zooms the Roblox camera: right-drag to rotate, wheel to zoom."""

    def __init__(self):
        self._busy = threading.Lock()

    def rotate(self, dx: float, dy: float, steps: int = 6, step_s: float = 0.012) -> None:
        if abs(dx) < 1 and abs(dy) < 1:
            return
        if not self._busy.acquire(blocking=False):
            return
        try:
            mouse_right(True)
            time.sleep(0.02)
            for _ in range(steps):
                mouse_move(round(dx / steps), round(dy / steps))
                time.sleep(step_s)
            time.sleep(0.01)
        finally:
            mouse_right(False)
            self._busy.release()

    def zoom(self, clicks: int) -> None:
        for _ in range(abs(clicks)):
            mouse_wheel(1 if clicks > 0 else -1)
            time.sleep(0.03)
