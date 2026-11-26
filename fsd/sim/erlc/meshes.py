"""Mesh parts of the map (unions / MeshParts): many town roads, sidewalks, signals,
lamp posts, trees and building details are meshes, not plain blocks.

Format (from the map export):
* meshes.json  - list of geometries (`geoms`): vertex/index counts and byte ranges,
                 plus the instance record stride and material names
* geoms.bin    - every geometry, compressed with meshoptimizer (vertex: int16 position
                 in -0.5..0.5 of the part size, int8 normal, rgb bytes, half-float uv)
* meshinst.bin - float32 records: position, quaternion, size, colour (r < 0 = keep vertex
                 colours), material, transparency, geometry index, texture, ...

Decoding needs the `meshoptimizer` package (pip install meshoptimizer).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mapdata import BASE_URL, CACHE, _get, _ungzip, quat_to_rot

# sRGB byte -> linear (what the site uses for vertex colours); we keep sRGB for our shader
_SRGB = np.arange(256, dtype=np.float32) / 255.0


@dataclass
class Geom:
    pos: np.ndarray          # (V,3) float32 in -0.5..0.5
    norm: np.ndarray         # (V,3) float32
    color: np.ndarray | None  # (V,3) 0..1 or None
    index: np.ndarray        # (I,) uint32


@dataclass
class MeshInstances:
    pos: np.ndarray          # (N,3)
    rot: np.ndarray          # (N,3,3)
    size: np.ndarray         # (N,3)
    color: np.ndarray        # (N,3)  (negative r = use the mesh's own vertex colours)
    mat: np.ndarray          # (N,)
    transp: np.ndarray       # (N,)
    geom: np.ndarray         # (N,)
    materials: list[str]
    tex: np.ndarray | None = None   # (N,) texture index or -1

    def __len__(self) -> int:
        return len(self.pos)

    def mat_is(self, name: str) -> np.ndarray:
        return self.mat == (self.materials.index(name) if name in self.materials else -1)


def decode_geoms(meta: dict, blob: bytes) -> list[Geom | None]:
    import meshoptimizer as mo
    vs = meta["vstride"]
    out: list[Geom | None] = []
    for g in meta["geoms"]:
        try:
            off, v, vb = g["off"], g["v"], g["vb"]
            # (the wrapper's dtype option under-allocates; decode as float32 rows and view the bytes)
            raw = mo.decode_vertex_buffer(v, vs, blob[off:off + vb])
            raw = np.ascontiguousarray(raw).view(np.uint8).reshape(v, vs)
            p = raw[:, 0:6].copy().view("<i2").astype(np.float32) / 65534.0
            n = raw[:, 8:11].copy().view("i1").astype(np.float32) / 127.0
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-6
            col = _SRGB[raw[:, 12:15]] if g.get("col") else None
            ioff = off + vb + (-vb & 3)
            idx = mo.decode_index_buffer(g["i"], 4, blob[ioff:ioff + g["ib"]])
            out.append(Geom(p, n, col, np.asarray(idx, np.uint32).ravel()))
        except Exception:  # noqa: BLE001 - one bad geometry shouldn't stop the map
            out.append(None)
    return out


def instances_from_bin(buf: bytes, meta: dict) -> MeshInstances:
    h = meta["stride"]
    r = np.frombuffer(buf, "<f4").reshape(-1, h).astype(np.float64)
    return MeshInstances(r[:, 0:3], quat_to_rot(r[:, 3:7]), np.maximum(r[:, 7:10], 0.01),
                         r[:, 10:13].astype(np.float32), r[:, 13].astype(np.int16),
                         np.clip(r[:, 14], 0, 1).astype(np.float32), r[:, 15].astype(np.int32),
                         list(meta["materials"]), r[:, 16].astype(np.int32) if h > 16 else None)


def texture_means(meta: dict, season_parts: str, season_shapes: str, log=print) -> list | None:
    """Average colour of every mesh texture (sRGB 0..255 + % opaque), cached. Without the
    textures, textured meshes (signs, trees, building fronts) would all come out white."""
    path = CACHE / f"texmean_{season_parts}.json"
    if path.exists():
        return json.loads(path.read_text())
    import cv2
    out = []
    log("Measuring map texture colours (once) ...")
    for t in meta.get("textures", []):
        base = f"{BASE_URL}/{season_parts if t.get('own') else season_shapes}/tex/{t['id']}.webp"
        try:
            buf = np.frombuffer(_get(base), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            small = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
            if small.ndim == 3 and small.shape[2] == 4:
                ok = small[:, :, 3] >= 100
                px = small[:, :, :3][ok]
            else:
                ok = np.ones(small.shape[:2], bool)
                px = small.reshape(-1, small.shape[2] if small.ndim == 3 else 1)[:, :3]
            m = px.mean(0)[::-1] if len(px) else np.zeros(3)
            out.append([int(m[0]), int(m[1]), int(m[2]), int(100 * ok.mean())])
        except Exception:  # noqa: BLE001
            out.append(None)
    path.write_text(json.dumps(out))
    return out


class MapMeshes:
    def __init__(self, meta: dict, geoms: list[Geom | None], inst: MeshInstances, texmean: list | None = None):
        self.meta, self.geoms, self.inst = meta, geoms, inst
        self.texmean = texmean

    @classmethod
    def load(cls, season_parts: str = "fall2026", season_shapes: str = "summer", log=print) -> "MapMeshes | None":
        import importlib.util
        if importlib.util.find_spec("meshoptimizer") is None:
            log("meshoptimizer not installed - map meshes (some roads, signals, trees) skipped")
            return None
        CACHE.mkdir(parents=True, exist_ok=True)
        mj = CACHE / f"meshes_{season_parts}.json"
        gb = CACHE / f"geoms_{season_shapes}.bin"
        ib = CACHE / f"meshinst_{season_parts}.bin"
        try:
            if not mj.exists():
                mj.write_bytes(_get(f"{BASE_URL}/{season_parts}/meshes.json"))
            meta = json.loads(mj.read_text())
            if not gb.exists():
                log("Downloading map meshes (~15 MB, once) ...")
                gb.write_bytes(_ungzip(_get(f"{BASE_URL}/{season_shapes}/geoms.bin.gz?v={meta.get('build', 0)}")))
            if not ib.exists():
                b = meta.get("ownBuild") or meta.get("build", 0)
                ib.write_bytes(_ungzip(_get(f"{BASE_URL}/{season_parts}/meshinst.bin.gz?v={b}")))
        except Exception as e:  # noqa: BLE001
            log(f"map meshes not available ({e})")
            return None
        geoms = decode_geoms(meta, gb.read_bytes())
        try:
            tm = texture_means(meta, season_parts, season_shapes, log)
        except Exception as e:  # noqa: BLE001
            log(f"texture colours not available ({e})")
            tm = None
        return cls(meta, geoms, instances_from_bin(ib.read_bytes(), meta), tm)


def save_path() -> Path:
    return CACHE


ROAD_MATERIALS = ("Asphalt", "Concrete", "Pavement", "Slate", "ERLC_ConcreteBlock", "SmoothPlastic", "Plastic",
                  "Rough Plastic", "Granite", "Cobblestone", "Brick", "Sand", "Mud", "Grass")


class MeshIndex:
    """Spatial lookup of mesh instances + 'what surface is under this point' for driving."""

    def __init__(self, mm: MapMeshes, cell: float = 64.0):
        self.mm = mm
        self.cell = cell
        I = mm.inst
        keys = np.floor(I.pos[:, [0, 2]] / cell).astype(np.int64)
        self.grid: dict[tuple[int, int], list[int]] = {}
        for i, (a, b) in enumerate(keys):
            self.grid.setdefault((int(a), int(b)), []).append(i)
        self.half = np.linalg.norm(I.size, axis=1) / 2
        self.max_half = float(min(400.0, self.half.max()))
        road = np.zeros(len(I), bool)
        for m in ROAD_MATERIALS:
            road |= I.mat_is(m)
        self.road = road & (I.transp < 0.9)
        self.inv_rot = np.transpose(I.rot, (0, 2, 1))

    def near(self, x: float, z: float, radius: float) -> np.ndarray:
        c = self.cell
        r = radius + self.max_half
        out = []
        for i in range(int(np.floor((x - r) / c)), int(np.floor((x + r) / c)) + 1):
            for j in range(int(np.floor((z - r) / c)), int(np.floor((z + r) / c)) + 1):
                out.extend(self.grid.get((i, j), ()))
        if not out:
            return np.zeros(0, int)
        idx = np.asarray(out)
        I = self.mm.inst
        d = np.hypot(I.pos[idx, 0] - x, I.pos[idx, 2] - z)
        return idx[d < radius + self.half[idx]]

    def surface_below(self, x: float, z: float, y_top: float, only: tuple[str, ...] | None = None) -> float | None:
        """Highest road-like mesh surface at (x, z) that is below y_top (studs), or None.
        `only`: restrict to these material names."""
        I = self.mm.inst
        best = None
        allowed = None
        if only:
            allowed = np.zeros(len(I), bool)
            for m in only:
                allowed |= I.mat_is(m)
        for i in self.near(x, z, 1.0):
            if not self.road[i] or (allowed is not None and not allowed[i]):
                continue
            g = self.mm.geoms[I.geom[i]]
            if g is None:
                continue
            # ray straight down in the mesh's unit space
            Rinv = self.inv_rot[i]
            o = (Rinv @ (np.array([x, y_top, z]) - I.pos[i])) / I.size[i]
            dvec = (Rinv @ np.array([0.0, -1.0, 0.0])) / I.size[i]
            tri = g.pos[g.index.reshape(-1, 3)]
            v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
            e1, e2 = v1 - v0, v2 - v0
            pvec = np.cross(dvec, e2)
            det = np.einsum("ij,ij->i", e1, pvec)
            ok = np.abs(det) > 1e-12
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            tvec = o - v0
            u = np.einsum("ij,ij->i", tvec, pvec) * inv
            q = np.cross(tvec, e1)
            v = (q @ dvec) * inv
            t = np.einsum("ij,ij->i", e2, q) * inv
            hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
            if not hit.any():
                continue
            tt = t[hit].min()
            hy = y_top - tt              # dvec has unit world length in y (scaled back by size)
            if best is None or hy > best:
                best = hy
        return best
