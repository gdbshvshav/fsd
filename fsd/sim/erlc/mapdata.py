"""The Liberty County map: parts + terrain, downloaded once and cached.

Units: everything here is in Roblox STUDS (the map's own units) with Roblox axes
(x east, y up, z south). 1 stud = 0.28 m; drive.py converts to metres for FSD.

Sources, in order:
1. data/erlc_map/parts.npz + terrain.bin   (cache written by a previous download)
2. download from erlc-tools.com            (needs internet, done once)
3. a packed dev copy (parts_*.bin, 32-byte records) if you point ERLC_MAP_DIR at it
"""
from __future__ import annotations

import gzip
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STUD_M = 0.28
BASE_URL = "https://erlc-tools.com/map-3d"
ROOT = Path(__file__).resolve().parents[3]
CACHE = Path(os.environ.get("ERLC_MAP_DIR", ROOT / "data" / "erlc_map"))

SHAPE_BOX, SHAPE_WEDGE, SHAPE_CORNER, SHAPE_CYL_X, SHAPE_BALL, SHAPE_CYL_Y, SHAPE_BALL2 = range(7)

TERRAIN_MATERIALS = ["Plastic", "SmoothPlastic", "Neon", "Wood", "WoodPlanks", "Marble", "Basalt", "Slate",
                     "CrackedLava", "Concrete", "Limestone", "Granite", "Pavement", "Brick", "Pebble",
                     "Cobblestone", "Rock", "Sandstone", "CorrodedMetal", "DiamondPlate", "Foil", "Metal",
                     "Grass", "LeafyGrass", "Sand", "Fabric", "Snow", "Mud", "Ground", "Asphalt", "Salt",
                     "Ice", "Glacier", "Glass", "ForceField", "Air", "Water"]

# terrain colours in game (from the map export), sRGB 0..1
TERRAIN_COLORS = {
    "Grass": (0.282, 0.298, 0.216), "LeafyGrass": (0.255, 0.290, 0.224), "Rock": (0.471, 0.439, 0.404),
    "Basalt": (0.569, 0.518, 0.490), "Sand": (0.600, 0.588, 0.486), "Sandstone": (0.314, 0.302, 0.278),
    "Cobblestone": (0.384, 0.310, 0.267), "Ground": (0.553, 0.420, 0.325), "Limestone": (0.420, 0.357, 0.294),
    "Mud": (0.514, 0.424, 0.349), "Slate": (0.388, 0.384, 0.376), "Asphalt": (0.314, 0.329, 0.329),
    "WoodPlanks": (0.675, 0.580, 0.424), "Concrete": (0.498, 0.400, 0.247),
}


@dataclass
class Parts:
    pos: np.ndarray        # (N,3) studs
    rot: np.ndarray        # (N,3,3) rotation matrices
    size: np.ndarray       # (N,3) studs
    color: np.ndarray      # (N,3) 0..1 sRGB
    mat: np.ndarray        # (N,) material index into `materials`
    transp: np.ndarray     # (N,) 0..1
    shape: np.ndarray      # (N,)
    materials: list[str]

    def __len__(self) -> int:
        return len(self.pos)

    def mat_is(self, name: str) -> np.ndarray:
        return self.mat == (self.materials.index(name) if name in self.materials else -1)


@dataclass
class Terrain:
    x0: float
    z0: float
    step: float
    height: np.ndarray     # (nz, nx) studs
    water: np.ndarray      # (nz, nx) bool
    mat: np.ndarray        # (nz, nx) terrain material code (index+1, 0 = none)

    def height_at(self, x: np.ndarray | float, z: np.ndarray | float) -> np.ndarray:
        fx = (np.asarray(x, float) - self.x0) / self.step
        fz = (np.asarray(z, float) - self.z0) / self.step
        nz, nx = self.height.shape
        ix = np.clip(np.floor(fx).astype(int), 0, nx - 2)
        iz = np.clip(np.floor(fz).astype(int), 0, nz - 2)
        ax = np.clip(fx - ix, 0, 1)
        az = np.clip(fz - iz, 0, 1)
        h = self.height
        return ((h[iz, ix] * (1 - ax) + h[iz, ix + 1] * ax) * (1 - az) +
                (h[iz + 1, ix] * (1 - ax) + h[iz + 1, ix + 1] * ax) * az)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# ---------------------------------------------------------------- decoding
def parts_from_export(buf: bytes, stride: int, materials: list[str]) -> Parts:
    """The site's own format: float32 records of `stride` values."""
    a = np.frombuffer(buf, "<f4").reshape(-1, stride)
    return Parts(a[:, 0:3].astype(np.float64), quat_to_rot(a[:, 3:7].astype(np.float64)),
                 np.maximum(a[:, 7:10].astype(np.float64), 0.01), np.clip(a[:, 10:13], 0, 1).astype(np.float32),
                 a[:, 13].astype(np.int16), np.clip(a[:, 14], 0, 1).astype(np.float32),
                 a[:, 15].astype(np.int8), [m.split("\n")[0].strip() for m in materials])


PACKED = np.dtype([("pos", "<f4", 3), ("q", "i1", 4), ("size", "<u2", 3), ("col", "u1", 3), ("mat", "u1"),
                   ("transp", "u1"), ("shape", "u1"), ("var", "u1"), ("pad", "u1", 3)])


def parts_from_packed(buf: bytes, materials: list[str]) -> Parts:
    """Compact 32-byte records (dev copy)."""
    P = np.frombuffer(buf, PACKED)
    return Parts(P["pos"].astype(np.float64), quat_to_rot(P["q"].astype(np.float64) / 127.0),
                 np.maximum(P["size"] / 20.0, 0.01), P["col"].astype(np.float32) / 255.0,
                 P["mat"].astype(np.int16), P["transp"].astype(np.float32) / 255.0,
                 P["shape"].astype(np.int8), materials)


def terrain_from_bin(buf: bytes, x0: float, z0: float, step: float, nx: int, nz: int) -> Terrain:
    S = nx * nz
    h = np.frombuffer(buf, "<u2", S).reshape(nz, nx).astype(np.float32) / 100.0 - 64.0
    water = np.frombuffer(buf, "<u2", S, offset=S * 2).reshape(nz, nx) > 0
    mat = np.frombuffer(buf, "u1", S, offset=S * 4).reshape(nz, nx)
    return Terrain(x0, z0, step, h, water, mat)


def _ungzip(b: bytes) -> bytes:
    return gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ERLC-FSD-sim/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


# ---------------------------------------------------------------- the map
class ErlcMap:
    def __init__(self, parts: Parts, terrain: Terrain | None, meshes=None):
        self.parts = parts
        self.terrain = terrain
        self.meshes = meshes              # meshes.MapMeshes or None
        self.mesh_index = None
        if meshes is not None:
            from .meshes import MeshIndex
            self.mesh_index = MeshIndex(meshes)
        self._grid: dict[tuple[int, int], np.ndarray] | None = None
        self.cell = 64.0

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, season_parts: str = "fall2026", season_shapes: str = "summer", log=print,
             with_meshes: bool = True) -> "ErlcMap":
        CACHE.mkdir(parents=True, exist_ok=True)
        npz = CACHE / f"parts_{season_parts}.npz"
        tbin = CACHE / f"terrain_{season_shapes}.bin"
        tmeta = CACHE / "terrain.json"
        if npz.exists():
            z = np.load(npz, allow_pickle=False)
            parts = Parts(z["pos"], z["rot"], z["size"], z["color"], z["mat"], z["transp"], z["shape"],
                          json.loads(str(z["materials"])))
        else:
            parts = cls._fetch_parts(season_parts, log)
            np.savez_compressed(npz, pos=parts.pos, rot=parts.rot, size=parts.size, color=parts.color,
                                mat=parts.mat, transp=parts.transp, shape=parts.shape,
                                materials=json.dumps(parts.materials))
        terrain = None
        if tbin.exists() and tmeta.exists():
            m = json.loads(tmeta.read_text())
            terrain = terrain_from_bin(tbin.read_bytes(), m["x0"], m["z0"], m["step"], m["nx"], m["nz"])
        else:
            try:
                meta = json.loads(_get(f"{BASE_URL}/{season_shapes}/meta.json"))
                t = meta["terrain"]
                raw = _ungzip(_get(f"{BASE_URL}/{season_shapes}/terrain.bin.gz?v={meta.get('build', 0)}"))
                tbin.write_bytes(raw)
                tmeta.write_text(json.dumps(t))
                terrain = terrain_from_bin(raw, t["x0"], t["z0"], t["step"], t["nx"], t["nz"])
            except Exception as e:  # noqa: BLE001
                log(f"terrain not available ({e}); flat ground will be used")
        meshes = None
        if with_meshes:
            from .meshes import MapMeshes
            meshes = MapMeshes.load(season_parts, season_shapes, log)
        return cls(parts, terrain, meshes)

    @staticmethod
    def _fetch_parts(season: str, log) -> Parts:
        # a packed dev copy next to the cache?
        packed = CACHE / f"parts_{season}.bin"
        if packed.exists():
            mats = json.loads((CACHE / "meta.json").read_text())["materials"]
            return parts_from_packed(packed.read_bytes(), mats)
        log(f"Downloading the Liberty County map ({season}) from erlc-tools.com - once, ~2 MB ...")
        meta = json.loads(_get(f"{BASE_URL}/{season}/meta.json"))
        build = meta.get("ownBuild") or meta.get("build", 0)
        raw = _ungzip(_get(f"{BASE_URL}/{season}/parts.bin.gz?v={build}"))
        return parts_from_export(raw, meta["parts"]["stride"], meta["parts"]["materials"])

    # ------------------------------------------------------------ lookups
    def _build_grid(self) -> None:
        c = self.cell
        keys = np.floor(self.parts.pos[:, [0, 2]] / c).astype(np.int64)
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        k = keys[order]
        brk = np.flatnonzero(np.any(np.diff(k, axis=0) != 0, axis=1)) + 1
        starts = np.r_[0, brk]
        ends = np.r_[brk, len(k)]
        self._grid = {(int(k[s, 0]), int(k[s, 1])): order[s:e] for s, e in zip(starts, ends)}
        # big parts can stick out of their cell: remember the largest half-extent
        self._max_half = float(np.max(np.linalg.norm(self.parts.size, axis=1)) / 2)

    def near(self, x: float, z: float, radius: float) -> np.ndarray:
        """Indices of parts whose centre is within `radius` (+ their own size) of (x, z)."""
        if self._grid is None:
            self._build_grid()
        c = self.cell
        r = radius + min(self._max_half, 400.0)
        i0, i1 = int(np.floor((x - r) / c)), int(np.floor((x + r) / c))
        j0, j1 = int(np.floor((z - r) / c)), int(np.floor((z + r) / c))
        out = [self._grid[(i, j)] for i in range(i0, i1 + 1) for j in range(j0, j1 + 1) if (i, j) in self._grid]
        if not out:
            return np.zeros(0, int)
        idx = np.concatenate(out)
        P = self.parts
        half = np.linalg.norm(P.size[idx], axis=1) / 2
        d = np.hypot(P.pos[idx, 0] - x, P.pos[idx, 2] - z)
        return idx[d < radius + half]

    def ground_at(self, x: float, z: float, y_hint: float | None = None, radius: float = 3.0) -> float:
        """Height (studs) of the drivable surface under (x, z): the top of the highest flat part
        below y_hint + 4 that covers the point, else the terrain."""
        P = self.parts
        idx = self.near(x, z, radius)
        best = None
        lim = (y_hint + 4.0) if y_hint is not None else 1e9
        if len(idx):
            R = P.rot[idx]
            up = R[:, :, 1]                                   # local y axis in world
            flat = np.abs(up[:, 1]) > 0.97
            for i, r, u in zip(idx[flat], R[flat], up[flat]):
                if P.transp[i] > 0.9 or P.shape[i] not in (SHAPE_BOX,):
                    continue
                rel = np.array([x, 0.0, z]) - P.pos[i]
                lx, lz = rel @ r[:, 0], rel @ r[:, 2]
                if abs(lx) > P.size[i, 0] / 2 or abs(lz) > P.size[i, 2] / 2:
                    continue
                top = P.pos[i, 1] + abs(r[1, 1]) * P.size[i, 1] / 2
                if top <= lim and (best is None or top > best):
                    best = top
        if self.mesh_index is not None:
            ms = self.mesh_index.surface_below(x, z, lim if lim < 1e8 else 2000.0)
            if ms is not None and (best is None or ms > best):
                best = ms
        ter = float(self.terrain.height_at(x, z)) if self.terrain is not None else 0.0
        if best is None or (ter > best and ter <= lim):
            return ter
        return float(best)

    def on_road(self, x: float, z: float, y: float) -> bool:
        """Is (x, z) on a road surface (asphalt / concrete slab or road mesh) near height y?"""
        P = self.parts
        idx = self.near(x, z, 2.0)
        road_mats = [P.materials.index(m) for m in ("Asphalt", "Concrete") if m in P.materials]
        for i in idx:
            if P.mat[i] not in road_mats or abs(P.rot[i][1, 1]) < 0.97 or max(P.size[i, 0], P.size[i, 2]) < 8:
                continue
            r = P.rot[i]
            rel = np.array([x, 0.0, z]) - P.pos[i]
            if abs(rel @ r[:, 0]) <= P.size[i, 0] / 2 and abs(rel @ r[:, 2]) <= P.size[i, 2] / 2 \
                    and abs(P.pos[i, 1] - y) < 4:
                return True
        if self.mesh_index is not None:
            s = self.mesh_index.surface_below(x, z, y + 4.0, only=("Asphalt", "Concrete"))
            return s is not None and abs(s - y) < 4
        return False
