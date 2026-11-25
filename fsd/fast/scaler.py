"""Cheap frame shrinking.

A 4K frame is 25 MB; resizing it properly costs 20-40 ms on a CPU. Taking every
2nd or 4th pixel is a free NumPy view and good enough for lane paint and lamps.
We keep two sizes:

    work  (~960 wide)   lanes, obstacles, car outline   -> fast
    fine  (~1920 wide)  traffic-light lamps, speedometer -> small things stay readable
"""
from __future__ import annotations

import cv2
import numpy as np


def shrink_to(frame: np.ndarray, width: int) -> np.ndarray:
    """Shrink to about `width` pixels wide, using slicing when it's a clean factor."""
    H, W = frame.shape[:2]
    if W <= width:
        return frame
    k = W // width
    if k >= 2 and W / k >= width * 0.92:
        out = frame[::k, ::k]
        if out.shape[1] <= width * 1.1:
            return out
        frame = out
        H, W = frame.shape[:2]
    if W > width:
        frame = cv2.resize(frame, (width, max(1, int(round(H * width / W)))), interpolation=cv2.INTER_LINEAR)
    return frame


class FramePair:
    """The two working sizes of one captured frame."""

    __slots__ = ("work", "fine", "full_w", "full_h")

    def __init__(self, frame: np.ndarray, work_w: int = 960, fine_w: int = 1920):
        self.full_h, self.full_w = frame.shape[:2]
        self.fine = shrink_to(frame, fine_w)
        self.work = shrink_to(self.fine, work_w)
        if not self.work.flags["C_CONTIGUOUS"]:
            self.work = np.ascontiguousarray(self.work)

    @property
    def fine_scale(self) -> float:
        """fine pixels per work pixel"""
        return self.fine.shape[1] / self.work.shape[1]
