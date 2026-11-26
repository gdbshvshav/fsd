"""The real ER:LC traffic lights in the map, switched red / yellow / green.

In the map export a signal head is a yellow housing mesh with a black backplate and
three lamp meshes stacked in it (red on top, yellow, green at the bottom). The export
is a frozen snapshot, so here we find every head, work out which way it faces, group
heads into junctions, and run a light cycle - the two directions of each junction
take turns, like in game. The renderer is told every frame which lamps are lit.

Also used to score a drive: entering a junction while the heads facing us are red.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .meshes import MapMeshes, MeshIndex

LIT = {"red": (1.0, 0.12, 0.08), "yellow": (1.0, 0.78, 0.12), "green": (0.12, 1.0, 0.35)}
OFF = {"red": (0.20, 0.05, 0.05), "yellow": (0.22, 0.18, 0.04), "green": (0.04, 0.18, 0.08)}


@dataclass
class Head:
    housing: int                  # mesh instance index
    lamps: dict[str, int]         # colour -> mesh instance index
    pos: np.ndarray               # housing centre (studs)
    face: np.ndarray              # horizontal unit vector the lamps shine toward
    junction: int = -1
    phase: int = 0                # which direction group of its junction


@dataclass
class Junction:
    centre: np.ndarray
    radius: float
    heads: list[int] = field(default_factory=list)
    axis: float = 0.0             # angle of phase 0's facing (radians, mod pi)
    offset_s: float = 0.0         # cycle offset so junctions aren't all in step


@dataclass
class Plan:
    green_s: float = 12.0
    yellow_s: float = 3.0
    all_red_s: float = 2.0
    mode: str = "cycle"           # cycle | flashing | off | fixed
    fixed: str = "red"            # for mode == fixed (heads facing the route's start)

    @property
    def period(self) -> float:
        return 2 * (self.green_s + self.yellow_s + self.all_red_s)

    def state(self, phase: int, t: float) -> str:
        if self.mode == "off":
            return "off"
        if self.mode == "flashing":
            return "red" if (t % 1.0) < 0.5 else "off"
        half = self.green_s + self.yellow_s + self.all_red_s
        u = (t - phase * half) % self.period
        if u < self.green_s:
            return "green"
        if u < self.green_s + self.yellow_s:
            return "yellow"
        return "red"


def find_heads(mm: MapMeshes) -> list[Head]:
    I = mm.inst
    c = np.clip(I.color, 0, 1)
    s = np.sort(I.size, 1)
    yellow = (c[:, 0] > 0.7) & (c[:, 1] > 0.5) & (c[:, 2] < 0.3)
    housing = yellow & (np.abs(s[:, 0] - 1.4) < 0.25) & (np.abs(s[:, 2] - 3.9) < 0.5) & (s[:, 1] < 1.8)
    idx = MeshIndex(mm)
    lamp_geom = None
    heads = []
    for h in np.nonzero(housing)[0]:
        near = [j for j in idx.near(I.pos[h, 0], I.pos[h, 2], 2.5)
                if j != h and np.linalg.norm(I.pos[j] - I.pos[h]) < 2.6 and np.sort(I.size[j])[0] < 0.5
                and abs(np.sort(I.size[j])[2] - 1.15) < 0.35]
        if lamp_geom is None and len(near) >= 3:
            from collections import Counter
            lamp_geom = Counter(int(I.geom[j]) for j in near).most_common(1)[0][0]
        if lamp_geom is not None:
            near = [j for j in near if I.geom[j] == lamp_geom] or near
        if len(near) < 3:
            continue
        near = sorted(near, key=lambda j: -I.pos[j, 1])[:3]
        lamps = {"red": near[0], "yellow": near[1], "green": near[2]}
        d = I.pos[near[1]] - I.pos[h]
        f = np.array([d[0], 0.0, d[2]])
        if np.linalg.norm(f) < 1e-3:
            continue
        heads.append(Head(int(h), lamps, I.pos[h].copy(), f / np.linalg.norm(f)))
    return heads


def group_junctions(heads: list[Head], link: float = 90.0) -> list[Junction]:
    n = len(heads)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    P = np.array([h.pos for h in heads]) if heads else np.zeros((0, 3))
    for i in range(n):
        d = np.hypot(P[:, 0] - P[i, 0], P[:, 2] - P[i, 2])
        for j in np.nonzero(d < link)[0]:
            parent[find(int(j))] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    juncs = []
    rng = np.random.default_rng(7)
    for k, members in enumerate(groups.values()):
        pts = P[members]
        centre = pts.mean(0)
        radius = float(max(15.0, np.max(np.hypot(pts[:, 0] - centre[0], pts[:, 2] - centre[2])) * 0.85))
        ang = np.array([math.atan2(heads[i].face[2], heads[i].face[0]) % math.pi for i in members])
        # dominant direction (mod 90 deg): average of doubled... take the first head's axis
        axis = float(ang[0])
        J = Junction(centre, radius, members, axis, float(rng.uniform(0, 30)))
        for i in members:
            heads[i].junction = k
            a = math.atan2(heads[i].face[2], heads[i].face[0]) % math.pi
            diff = abs((a - axis + math.pi / 2) % math.pi - math.pi / 2)
            heads[i].phase = 0 if diff < math.pi / 4 else 1
        juncs.append(J)
    return juncs


class Signals:
    def __init__(self, mm: MapMeshes, plan: Plan | None = None):
        self.mm = mm
        self.heads = find_heads(mm)
        self.junctions = group_junctions(self.heads)
        self.plan = plan or Plan()
        self.forced: dict[int, str] = {}       # junction index -> fixed state for its phase facing us

    def state_of(self, head: Head, t: float) -> str:
        J = self.junctions[head.junction]
        if head.junction in self.forced:
            want, phase_us = self.forced[head.junction]
            if want == "flashing":
                return "red" if (t % 1.0) < 0.5 else "off"
            other = {"red": "green", "green": "red", "yellow": "red", "off": "off"}[want]
            return want if head.phase == phase_us else other
        return self.plan.state(head.phase, t + J.offset_s)

    def colours(self, t: float) -> dict[int, tuple]:
        """mesh instance index -> (rgb, glow) for every lamp."""
        out = {}
        for h in self.heads:
            st = self.state_of(h, t)
            for col, i in h.lamps.items():
                if st == col:
                    out[i] = (LIT[col], 1.0)
                else:
                    out[i] = (OFF[col], 0.0)
        return out

    def facing_us(self, j: int, x: float, z: float, hx: float, hz: float) -> list[Head]:
        """Heads of junction j that shine toward a car at (x, z) heading (hx, hz)."""
        return [self.heads[i] for i in self.junctions[j].heads
                if self.heads[i].face[0] * -hx + self.heads[i].face[2] * -hz > 0.8]

    def nearest_ahead(self, x: float, z: float, hx: float, hz: float, max_d: float = 400.0) -> int | None:
        best, bd = None, max_d
        for k, J in enumerate(self.junctions):
            dx, dz = J.centre[0] - x, J.centre[2] - z
            ahead = dx * hx + dz * hz
            side = abs(-dx * hz + dz * hx)
            if J.radius * 0.6 < ahead < bd and side < J.radius + 10:
                if self.facing_us(k, x, z, hx, hz):
                    best, bd = k, ahead
        return best

    def force_for_route(self, x: float, z: float, hx: float, hz: float, state: str) -> int | None:
        """Make the first junction ahead show `state` to us (and the opposite to cross traffic)."""
        j = self.nearest_ahead(x, z, hx, hz)
        if j is None:
            return None
        faces = self.facing_us(j, x, z, hx, hz)
        self.forced[j] = (state, faces[0].phase)
        return j
