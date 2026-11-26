"""Small animation toolkit for the dashboard: springs, tweens, easing, fades.

Everything on screen moves through one of these so nothing ever pops:

* Spring      - physically damped follow (speed needle, steering wheel, cars).
* Tween       - timed transition with an easing curve (banners, toasts, colours).
* Pulse       - smooth on/off breathing (blinkers, warnings) instead of a hard blink.
* ColorTween  - fades between colours.
* Toasts      - messages that slide in, stay, and slide out, newest on top.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------- easing curves
def ease_out_cubic(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 1 - (1 - u) ** 3


def ease_in_out_cubic(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 4 * u * u * u if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2


def ease_out_back(u: float, s: float = 1.4) -> float:
    u = min(1.0, max(0.0, u)) - 1
    return 1 + (s + 1) * u * u * u + s * u * u


def ease_out_expo(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 if u >= 1 else 1 - 2 ** (-10 * u)


EASINGS = {"cubic": ease_out_cubic, "inout": ease_in_out_cubic, "back": ease_out_back, "expo": ease_out_expo,
           "linear": lambda u: min(1.0, max(0.0, u))}


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def lerp_color(a, b, u: float) -> tuple:
    return tuple(int(round(lerp(x, y, u))) for x, y in zip(a, b))


# ----------------------------------------------------------------------------- spring
class Spring:
    """Critically damped (or slightly bouncy) spring toward a target. Frame-rate independent."""

    def __init__(self, value: float = 0.0, freq: float = 4.0, damping: float = 1.0):
        self.x = float(value)
        self.v = 0.0
        self.freq = freq
        self.damping = damping
        self.target = float(value)

    def step(self, target: float | None, dt: float) -> float:
        if target is not None:
            self.target = float(target)
        dt = min(dt, 0.1)
        w = 2 * math.pi * self.freq
        z = self.damping
        # semi-implicit Euler in small substeps (stable for any dt we see)
        n = max(1, int(dt / 0.008) + 1)
        h = dt / n
        for _ in range(n):
            a = w * w * (self.target - self.x) - 2 * z * w * self.v
            self.v += a * h
            self.x += self.v * h
        return self.x

    def snap(self, value: float) -> None:
        self.x = self.target = float(value)
        self.v = 0.0


class Spring2:
    """Two springs (x, y) - for positions."""

    def __init__(self, x: float = 0.0, y: float = 0.0, freq: float = 4.0, damping: float = 1.0):
        self.a = Spring(x, freq, damping)
        self.b = Spring(y, freq, damping)

    def step(self, x: float, y: float, dt: float) -> tuple[float, float]:
        return self.a.step(x, dt), self.b.step(y, dt)


# ----------------------------------------------------------------------------- tween
class Tween:
    def __init__(self, value: float = 0.0, duration: float = 0.3, easing: str = "cubic"):
        self.start = float(value)
        self.end = float(value)
        self.t0 = 0.0
        self.duration = duration
        self.easing = EASINGS.get(easing, ease_out_cubic)
        self.now = time.perf_counter

    def to(self, value: float, duration: float | None = None) -> None:
        value = float(value)
        if abs(value - self.end) < 1e-9:
            return
        self.start = self.value
        self.end = value
        self.t0 = self.now()
        if duration is not None:
            self.duration = duration

    @property
    def value(self) -> float:
        if self.duration <= 0:
            return self.end
        u = (self.now() - self.t0) / self.duration
        if u >= 1:
            return self.end
        return lerp(self.start, self.end, self.easing(u))

    @property
    def done(self) -> bool:
        return self.now() - self.t0 >= self.duration


class ColorTween:
    def __init__(self, color=(0, 0, 0), duration: float = 0.35):
        self.a = tuple(color)
        self.b = tuple(color)
        self.t = Tween(1.0, duration, "inout")

    def to(self, color) -> None:
        color = tuple(color)
        if color == self.b:
            return
        self.a = self.value
        self.b = color
        self.t.start, self.t.end = 0.0, 0.0
        self.t.to(1.0)

    @property
    def value(self) -> tuple:
        return lerp_color(self.a, self.b, self.t.value)


class Pulse:
    """Smooth 0..1 breathing at `hz` while on; fades out when switched off."""

    def __init__(self, hz: float = 1.5):
        self.hz = hz
        self.on = False
        self.level = Tween(0.0, 0.25, "cubic")
        self.phase0 = time.perf_counter()

    def set(self, on: bool) -> None:
        if on and not self.on:
            self.phase0 = time.perf_counter()
        self.on = on
        self.level.to(1.0 if on else 0.0)

    def value(self) -> float:
        ph = (time.perf_counter() - self.phase0) * self.hz
        wave = 0.5 + 0.5 * math.cos(2 * math.pi * ph)     # starts bright
        return self.level.value * (0.25 + 0.75 * wave)


# ----------------------------------------------------------------------------- toasts
@dataclass
class Toast:
    text: str
    color: tuple
    born: float
    life: float = 4.0
    slide: Tween = field(default_factory=lambda: Tween(0.0, 0.35, "back"))

    def alpha(self, now: float) -> float:
        age = now - self.born
        if age < 0.2:
            return age / 0.2
        if age > self.life - 0.4:
            return max(0.0, (self.life - age) / 0.4)
        return 1.0


class Toasts:
    def __init__(self, max_n: int = 3):
        self.items: list[Toast] = []
        self.max_n = max_n
        self._last_text = ""
        self._last_t = 0.0

    def push(self, text: str, color=(56, 132, 255), life: float = 4.0) -> None:
        now = time.perf_counter()
        if not text or (text == self._last_text and now - self._last_t < 6.0):
            return
        self._last_text, self._last_t = text, now
        t = Toast(text, tuple(color), now, life)
        t.slide.to(1.0)
        self.items.insert(0, t)
        del self.items[self.max_n:]

    def alive(self) -> list[Toast]:
        now = time.perf_counter()
        self.items = [t for t in self.items if now - t.born < t.life]
        return self.items


class Counter:
    """Animated number (e.g. speed digits roll smoothly to the new value)."""

    def __init__(self, value: float = 0.0, freq: float = 3.0):
        self.s = Spring(value, freq, 1.0)

    def step(self, target: float, dt: float) -> float:
        return self.s.step(target, dt)
