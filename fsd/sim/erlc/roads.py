"""Road paint pulled out of the map parts: exact ground truth for lane keeping.

In the ER:LC map, lane paint is made of thin flat Asphalt parts (0.7 studs wide)
coloured white (light grey) or yellow; crosswalk blocks and stop bars are wider
white blocks. From those we can tell, at any spot, exactly where the lane lines
to our left and right are - which is how a drive is scored (were we centred?
did we touch the yellow line?), and how test drives are started in a lane.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mapdata import ErlcMap


@dataclass
class Paint:
    cx: np.ndarray        # centre x (studs)
    cz: np.ndarray        # centre z
    y: np.ndarray         # top surface height
    ux: np.ndarray        # unit direction along the paint (x)
    uz: np.ndarray        # (z)
    half_len: np.ndarray  # along
    half_w: np.ndarray    # across
    yellow: np.ndarray    # bool

    def __len__(self) -> int:
        return len(self.cx)


def extract_paint(emap: ErlcMap) -> tuple[Paint, Paint]:
    """(lines, blocks): lane lines (thin, long) and wide paint (crosswalk blocks, stop bars)."""
    P = emap.parts
    R = P.rot
    up = np.abs(R[:, 1, 1]) > 0.97
    vert = np.where(up, P.size[:, 1], 99.0)
    c = P.color
    asph = P.mat_is("Asphalt") | P.mat_is("SmoothPlastic") | P.mat_is("Plastic")
    white = (c.min(1) > 0.5) & (c.max(1) - c.min(1) < 0.12)
    yellow = (c[:, 0] > 0.55) & (c[:, 1] > 0.4) & (c[:, 2] < 0.3) & (c[:, 0] - c[:, 2] > 0.35)
    paint = up & (vert <= 0.25) & asph & (white | yellow) & (P.transp < 0.5)
    sx, sz = P.size[:, 0], P.size[:, 2]
    along_x = sx >= sz
    ax = np.where(along_x[:, None], R[:, :, 0], R[:, :, 2])     # world vector of the long axis
    L = np.maximum(sx, sz) / 2
    Wd = np.minimum(sx, sz) / 2
    n = np.hypot(ax[:, 0], ax[:, 2]) + 1e-9
    ux, uz = ax[:, 0] / n, ax[:, 2] / n
    top = P.pos[:, 1] + vert / 2

    def make(sel):
        return Paint(P.pos[sel, 0], P.pos[sel, 2], top[sel], ux[sel], uz[sel], L[sel], Wd[sel], yellow[sel])

    is_line = paint & (Wd <= 0.6) & (L >= 1.5)
    is_block = paint & (Wd > 0.6) & (Wd <= 6.0)
    return make(is_line), make(is_block)


class RoadTruth:
    """Where are the lane lines around a point?"""

    def __init__(self, emap: ErlcMap, cell: float = 32.0):
        self.map = emap
        self.lines, self.blocks = extract_paint(emap)
        self.cell = cell
        self._grid: dict[tuple[int, int], list[int]] = {}
        L = self.lines
        for i in range(len(L)):
            # register the segment in every cell it passes through (sampled every half cell)
            n = max(1, int(L.half_len[i] * 2 / (cell / 2)))
            for t in np.linspace(-L.half_len[i], L.half_len[i], n + 1):
                k = (int((L.cx[i] + t * L.ux[i]) // cell), int((L.cz[i] + t * L.uz[i]) // cell))
                self._grid.setdefault(k, [])
                if not self._grid[k] or self._grid[k][-1] != i:
                    self._grid[k].append(i)

    def _cand(self, x: float, z: float, r: float) -> np.ndarray:
        c = self.cell
        out = set()
        for i in range(int((x - r) // c), int((x + r) // c) + 1):
            for j in range(int((z - r) // c), int((z + r) // c) + 1):
                out.update(self._grid.get((i, j), ()))
        return np.fromiter(out, int) if out else np.zeros(0, int)

    def around(self, x: float, z: float, hx: float, hz: float, reach: float = 16.0, gap: float = 12.0):
        """Lane lines beside the point (x, z) for a car heading (hx, hz) (unit, studs).

        Returns a list of (offset, yellow) sorted by offset: signed distance in studs from the
        point to each line that runs roughly parallel to us and is level with us (+ = to our right).
        A dashed line counts across its gaps (`gap` studs past each dash)."""
        idx = self._cand(x, z, reach + gap)
        if not len(idx):
            return []
        L = self.lines
        par = L.ux[idx] * hx + L.uz[idx] * hz
        idx = idx[np.abs(par) > 0.85]
        if not len(idx):
            return []
        dx, dz = x - L.cx[idx], z - L.cz[idx]
        along = dx * L.ux[idx] + dz * L.uz[idx]
        inside = np.abs(along) <= L.half_len[idx] + gap
        idx, dx, dz = idx[inside], dx[inside], dz[inside]
        # right of the car = heading rotated clockwise (seen from above, x east / z south): (-hz, hx)
        rx, rz = -hz, hx
        off = -(dx * rx + dz * rz)          # line position relative to us along our right vector
        # remove the along-line part so slanted lines still give their perpendicular offset
        keep = np.abs(off) <= reach
        out = sorted(zip(off[keep].tolist(), L.yellow[idx[keep]].tolist()))
        # merge duplicates (dashes of the same line)
        merged: list[tuple[float, bool]] = []
        for o, y in out:
            if merged and abs(o - merged[-1][0]) < 1.0:
                continue
            merged.append((o, y))
        return merged

    def lane_at(self, x: float, z: float, hx: float, hz: float) -> tuple[float, float, bool, bool] | None:
        """(left_line_offset, right_line_offset, left_is_yellow, right_is_yellow) of the lane we're in."""
        lines = self.around(x, z, hx, hz)
        left = [l for l in lines if l[0] < 0]
        right = [l for l in lines if l[0] > 0]
        if not left or not right:
            return None
        lo, ly = max(left)
        ro, ry = min(right)
        if not (5.0 < ro - lo < 22.0):
            return None
        return lo, ro, ly, ry

    def beside_yellow(self, x: float, z: float, hx: float, hz: float, search: float = 60.0,
                      lane_half: float = 7.5) -> tuple[float, float]:
        """A point in the first lane to the right of the nearest yellow centre line that runs
        along our heading (right-hand traffic: the yellow line is on our left)."""
        L = self.lines
        idx = self._cand(x, z, search)
        idx = idx[L.yellow[idx]] if len(idx) else idx
        if not len(idx):
            return x, z
        par = np.abs(L.ux[idx] * hx + L.uz[idx] * hz) > 0.85
        idx = idx[par]
        if not len(idx):
            return x, z
        d = np.hypot(L.cx[idx] - x, L.cz[idx] - z)
        i = idx[int(np.argmin(d))]
        # project onto that line, then step to the right
        t = (x - L.cx[i]) * L.ux[i] + (z - L.cz[i]) * L.uz[i]
        t = float(np.clip(t, -L.half_len[i], L.half_len[i]))
        px, pz = L.cx[i] + t * L.ux[i], L.cz[i] + t * L.uz[i]
        return px + (-hz) * lane_half, pz + hx * lane_half

    def approach(self, cx: float, cz: float, dist: float = 150.0, prefer: tuple[float, float] | None = None):
        """A spawn (x, z, hx, hz) `dist` studs before the point (cx, cz) on a road that leads to it,
        in the right-hand lane, heading toward it. prefer: rough heading wanted (unit x, z)."""
        L = self.lines
        idx = self._cand(cx, cz, dist + 60)
        best, best_score = None, -1e9
        for i in idx:
            if not L.yellow[i]:
                continue
            for sgn in (1.0, -1.0):
                hx, hz = L.ux[i] * sgn, L.uz[i] * sgn
                # walk back from the junction along this line's direction
                px, pz = cx - hx * dist, cz - hz * dist
                # the line must pass near that spawn and point at the junction
                along = (px - L.cx[i]) * L.ux[i] + (pz - L.cz[i]) * L.uz[i]
                if abs(along) > L.half_len[i] + 15:
                    continue
                perp = abs(-(px - L.cx[i]) * L.uz[i] + (pz - L.cz[i]) * L.ux[i])
                if perp > 20:
                    continue
                sc = -perp
                if prefer is not None:
                    sc += 30.0 * (hx * prefer[0] + hz * prefer[1])
                if sc > best_score:
                    x, z = self.beside_yellow(px, pz, hx, hz)
                    if self.lane_at(x, z, hx, hz) is not None:
                        best, best_score = (x, z, hx, hz), sc
        return best
