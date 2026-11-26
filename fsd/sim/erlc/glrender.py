"""OpenGL renderer of the Liberty County map from ER:LC's chase camera.

Uses moderngl: on your PC it runs on the graphics card (any GPU, or the built-in
one); on a machine without one it falls back to Mesa's software renderer, which
is slower but gives the same picture. Everything is drawn as instanced shapes
(blocks, wedges, cylinders, balls) like Roblox parts, with:

* sun + sky light, Roblox-style blue-tinted shadows (shadow map),
* Neon parts glowing (street lamps, signal lamps), signal lamps switchable,
* the terrain height field coloured by material,
* distance haze, our own car, and the ER:LC HUD speedometer on top.

Coordinates are studs, Roblox axes (x east, y up, z south).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .mapdata import (SHAPE_BALL, SHAPE_BALL2, SHAPE_BOX, SHAPE_CORNER, SHAPE_CYL_X, SHAPE_CYL_Y, SHAPE_WEDGE,
                      TERRAIN_COLORS, TERRAIN_MATERIALS, ErlcMap)

# how much lighter a material looks in game than its raw colour (from real screenshots)
MATERIAL_GAIN = {"Asphalt": 2.1, "Concrete": 1.05, "Pavement": 1.1, "Grass": 1.5}
# terrain as it looks in game (the raw terrain colours are much greyer than the textures)
TERRAIN_LOOK = {"Grass": (0.36, 0.52, 0.24), "LeafyGrass": (0.32, 0.47, 0.22), "Ground": (0.50, 0.42, 0.33),
                "Mud": (0.45, 0.38, 0.30), "Rock": (0.47, 0.44, 0.40), "Sand": (0.66, 0.62, 0.50),
                "Limestone": (0.62, 0.57, 0.50), "Basalt": (0.36, 0.34, 0.33), "Sandstone": (0.58, 0.48, 0.38)}
MESH_BASE = 1000                                 # batch keys >= this are map meshes
SUN_DIR = np.array([0.35, 0.82, 0.45])          # towards the sun
SUN_DIR = SUN_DIR / np.linalg.norm(SUN_DIR)
SKY_TOP = np.array([0.36, 0.60, 0.86])
SKY_HORIZON = np.array([0.72, 0.82, 0.92])


# ---------------------------------------------------------------- meshes (unit size, centred)
def _box():
    f = []
    for axis in range(3):
        for sgn in (-1, 1):
            n = np.zeros(3)
            n[axis] = sgn
            u = np.zeros(3)
            v = np.zeros(3)
            u[(axis + 1) % 3] = 0.5
            v[(axis + 2) % 3] = 0.5
            c = n * 0.5
            q = [c - u - v, c + u - v, c + u + v, c - u + v]
            if sgn < 0:
                q = q[::-1]
            f += [(q[0], n), (q[1], n), (q[2], n), (q[0], n), (q[2], n), (q[3], n)]
    return f


def _tri_list(tris):
    out = []
    for a, b, c in tris:
        a, b, c = map(np.array, (a, b, c))
        n = np.cross(b - a, c - a)
        n = n / (np.linalg.norm(n) + 1e-12)
        out += [(a, n), (b, n), (c, n)]
    return out


def _wedge():
    # Roblox WedgePart: full bottom and back (+z) faces, slope rising from front-bottom to back-top
    h = 0.5
    p = {"bfl": (-h, -h, -h), "bfr": (h, -h, -h), "bbl": (-h, -h, h), "bbr": (h, -h, h),
         "tbl": (-h, h, h), "tbr": (h, h, h)}
    T = [(p["bfl"], p["bbr"], p["bfr"]), (p["bfl"], p["bbl"], p["bbr"]),            # bottom
         (p["bbl"], p["tbr"], p["bbr"]), (p["bbl"], p["tbl"], p["tbr"]),            # back
         (p["bfl"], p["bfr"], p["tbr"]), (p["bfl"], p["tbr"], p["tbl"]),            # slope
         (p["bfl"], p["tbl"], p["bbl"]), (p["bfr"], p["bbr"], p["tbr"])]            # sides
    return _tri_list(T)


def _corner():
    h = 0.5
    a, b, c, d = (-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)
    top = (h, h, h)
    return _tri_list([(a, c, b), (a, d, c), (a, b, top), (b, c, top), (c, d, top), (d, a, top)])


def _cyl(axis: int, seg: int = 14):
    out = []
    ang = np.linspace(0, 2 * math.pi, seg + 1)
    for i in range(seg):
        a0, a1 = ang[i], ang[i + 1]

        def P(a, t):
            v = np.zeros(3)
            v[axis] = t
            v[(axis + 1) % 3] = 0.5 * math.cos(a)
            v[(axis + 2) % 3] = 0.5 * math.sin(a)
            return v

        def N(a):
            v = np.zeros(3)
            v[(axis + 1) % 3] = math.cos(a)
            v[(axis + 2) % 3] = math.sin(a)
            return v
        p00, p01, p10, p11 = P(a0, -0.5), P(a0, 0.5), P(a1, -0.5), P(a1, 0.5)
        out += [(p00, N(a0)), (p11, N(a1)), (p01, N(a0)), (p00, N(a0)), (p10, N(a1)), (p11, N(a1))]
        for t, sgn in ((0.5, 1), (-0.5, -1)):
            c = np.zeros(3)
            c[axis] = t
            n = np.zeros(3)
            n[axis] = sgn
            tri = [c, P(a0, t), P(a1, t)] if sgn > 0 else [c, P(a1, t), P(a0, t)]
            out += [(tri[0], n), (tri[1], n), (tri[2], n)]
    return out


def _ball(seg: int = 12):
    out = []
    for i in range(seg):
        t0, t1 = math.pi * i / seg, math.pi * (i + 1) / seg
        for j in range(seg * 2):
            p0, p1 = 2 * math.pi * j / (seg * 2), 2 * math.pi * (j + 1) / (seg * 2)

            def S(t, p):
                return np.array([math.sin(t) * math.cos(p), math.cos(t), math.sin(t) * math.sin(p)])
            a, b, c, d = S(t0, p0), S(t0, p1), S(t1, p1), S(t1, p0)
            for v in (a, c, b, a, d, c):
                out.append((v * 0.5, v))
    return out


def _pack(tris) -> np.ndarray:
    """pos, normal, vertex colour (white) - 9 floats per vertex."""
    return np.array([np.r_[p, n, 1.0, 1.0, 1.0] for p, n in tris], np.float32)


VFMT = ("3f 3f 3f", "in_pos", "in_norm", "in_vcol")
IFMT = ("4f 4f 4f 4f/i", "m0", "m1", "m2", "m3")


SHAPE_MESH = {SHAPE_BOX: _box, SHAPE_WEDGE: _wedge, SHAPE_CORNER: _corner, SHAPE_CYL_X: lambda: _cyl(0),
              SHAPE_BALL: _ball, SHAPE_CYL_Y: lambda: _cyl(1), SHAPE_BALL2: _ball}

# ---------------------------------------------------------------- shaders
VS = """
#version 330
uniform mat4 viewproj;
uniform mat4 lightvp;
in vec3 in_pos; in vec3 in_norm; in vec3 in_vcol;
in vec4 m0; in vec4 m1; in vec4 m2; in vec4 m3;
in vec4 col;              // rgb + flags (1 = glow)
out vec3 v_norm; out vec3 v_col; out float v_glow; out vec3 v_world; out vec4 v_light;
void main() {
    mat4 M = mat4(m0, m1, m2, m3);
    vec4 w = M * vec4(in_pos, 1.0);
    mat3 N = mat3(M);
    v_norm = normalize(transpose(inverse(N)) * in_norm);
    v_col = col.rgb * in_vcol; v_glow = col.a; v_world = w.xyz;
    v_light = lightvp * w;
    gl_Position = viewproj * w;
}
"""

FS = """
#version 330
uniform vec3 sun; uniform vec3 campos; uniform vec3 skyh; uniform float haze;
uniform sampler2D shadowmap; uniform int use_shadow;
in vec3 v_norm; in vec3 v_col; in float v_glow; in vec3 v_world; in vec4 v_light;
out vec4 frag;
float lit() {
    if (use_shadow == 0) return 1.0;
    vec3 p = v_light.xyz / v_light.w * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0) return 1.0;
    float s = 0.0;
    vec2 texel = 1.0 / vec2(textureSize(shadowmap, 0));
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
        float d = texture(shadowmap, p.xy + vec2(i, j) * texel).r;
        s += (p.z - 0.0015 > d) ? 0.0 : 1.0;
    }
    return s / 9.0;
}
void main() {
    if (v_glow > 0.5) { frag = vec4(v_col, 1.0); return; }  // Neon: full bright, unshaded
    vec3 n = normalize(v_norm);
    float ndl = max(dot(n, sun), 0.0);
    float L = lit() * ndl;
    // Roblox-ish, matched to real ER:LC screenshots: warm sun + strong BLUE sky fill,
    // so shade is dark blue (sunlit town asphalt ~80 grey, in shadow ~(40,42,60))
    vec3 sky_fill = vec3(0.47, 0.50, 0.62) * (0.75 + 0.25 * n.y);
    vec3 light = vec3(1.0, 0.98, 0.94) * 0.95 * L + sky_fill;
    // Roblox tone-maps: bright sunlit colours roll off instead of clipping (a sunlit yellow
    // signal housing is ~V 225 in game, never pure 255 like a lit lamp)
    vec3 outc = vec3(1.0) - exp(-1.6 * v_col * light);
    float dist = length(v_world - campos);
    float f = clamp((dist - 600.0) / haze, 0.0, 0.85);
    frag = vec4(mix(outc, skyh, f), 1.0);
}
"""

SHADOW_VS = """
#version 330
uniform mat4 lightvp;
in vec3 in_pos; in vec3 in_norm;
in vec4 m0; in vec4 m1; in vec4 m2; in vec4 m3; in vec4 col;
void main() { gl_Position = lightvp * mat4(m0, m1, m2, m3) * vec4(in_pos, 1.0); }
"""
SHADOW_FS = """
#version 330
void main() { }
"""


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def perspective(vfov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(vfov_deg) / 2)
    M = np.zeros((4, 4))
    M[0, 0], M[1, 1] = f / aspect, f
    M[2, 2], M[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    M[3, 2] = -1
    return M


def ortho(half: float, near: float, far: float) -> np.ndarray:
    M = np.eye(4)
    M[0, 0] = M[1, 1] = 1.0 / half
    M[2, 2] = -2.0 / (far - near)
    M[2, 3] = -(far + near) / (far - near)
    return M


def instance_matrices(pos, rot, size) -> np.ndarray:
    """Column-major 4x4 per instance (as 4 vec4 columns) = translate * rotate * scale."""
    n = len(pos)
    M = np.zeros((n, 4, 4), np.float32)
    M[:, :3, :3] = rot * size[:, None, :]
    M[:, :3, 3] = pos
    M[:, 3, 3] = 1
    return np.ascontiguousarray(M.transpose(0, 2, 1).reshape(n, 16))


class GLRenderer:
    def __init__(self, emap: ErlcMap, width: int = 1920, height: int = 1024, vfov: float = 70.0,
                 view_radius: float = 1400.0, shadows: bool = True):
        import moderngl
        self.map = emap
        self.W, self.H, self.vfov = width, height, vfov
        self.view_radius = view_radius
        self.shadows = shadows
        try:
            self.ctx = moderngl.create_standalone_context()
        except Exception:  # noqa: BLE001
            self.ctx = moderngl.create_standalone_context(backend="egl")
        ctx = self.ctx
        self.renderer_name = ctx.info.get("GL_RENDERER", "?")
        self.prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.sprog = ctx.program(vertex_shader=SHADOW_VS, fragment_shader=SHADOW_FS)
        self.fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((width, height))],
                                   depth_attachment=ctx.depth_renderbuffer((width, height)))
        self.smap_size = 2048
        self.smap = ctx.depth_texture((self.smap_size, self.smap_size))
        self.smap.compare_func = ""
        self.sfbo = ctx.framebuffer(depth_attachment=self.smap)
        self.shape_vbo = {k: ctx.buffer(_pack(f())) for k, f in SHAPE_MESH.items()}
        self.shape_n = {k: len(_pack(f())) for k, f in SHAPE_MESH.items()}
        P = emap.parts
        self.inst_all = instance_matrices(P.pos, P.rot, P.size)
        glow = P.mat_is("Neon").astype(np.float32)
        col = P.color.copy()
        # Roblox material textures make some materials look much lighter than their colour value
        for name, gain in MATERIAL_GAIN.items():
            sel = P.mat_is(name)
            col[sel] = np.clip(col[sel] * gain, 0, 1)
        self.col_all = np.c_[col, glow].astype(np.float32)
        self.visible_part = P.transp < 0.95
        self.color_override: dict[int, tuple] = {}     # part index -> rgb (signals)
        self._mesh_gl: dict[int, tuple] = {}
        self.mesh_color_override: dict[int, tuple] = {}    # mesh instance -> (rgb, glow): signal lamps
        if emap.meshes is not None:
            I = emap.meshes.inst
            self.mesh_inst_all = instance_matrices(I.pos, I.rot, I.size)
            mc = np.where(I.color[:, :1] < 0, 1.0, I.color).astype(np.float32)
            for name, gain in MATERIAL_GAIN.items():
                sel = I.mat_is(name)
                mc[sel] = mc[sel] * gain          # may exceed 1 for vertex-coloured meshes: the shader clamps
            tm = emap.meshes.texmean
            if I.tex is not None and tm:
                # no textures in our renderer: a textured mesh gets its texture's average colour
                # (times its own colour), so STOP signs come out red, leaves green or autumn red...
                lut = np.ones((len(tm) + 1, 3), np.float32)
                for k, v in enumerate(tm):
                    if v:
                        c = np.array(v[:3], np.float32) / 255.0
                        if c[0] > 1.35 * c[1] and c[0] > 1.35 * c[2] and c[0] > 0.3:
                            c = np.array([0.78, 0.07, 0.07], np.float32)   # red signs: red, not pink
                        lut[k] = c
                ti = np.where((I.tex >= 0) & (I.tex < len(tm)), I.tex, len(tm))
                textured = ti < len(tm)
                mc[textured] = mc[textured] * lut[ti[textured]] * 1.25
            self.mesh_col_all = np.c_[mc, I.mat_is("Neon").astype(np.float32)].astype(np.float32)
        self.extra: list[tuple[int, np.ndarray, np.ndarray]] = []   # per frame (shape, mats, cols)
        self._terrain_vao = None
        self._terrain_center = None

    # ------------------------------------------------------------ terrain around a point
    def _build_terrain(self, cx: float, cz: float, half: float = 2200.0) -> None:
        T = self.map.terrain
        if T is None:
            return
        nz, nx = T.height.shape
        i0 = int(max(0, (cx - half - T.x0) // T.step))
        i1 = int(min(nx - 1, (cx + half - T.x0) // T.step))
        j0 = int(max(0, (cz - half - T.z0) // T.step))
        j1 = int(min(nz - 1, (cz + half - T.z0) // T.step))
        if i1 - i0 < 2 or j1 - j0 < 2:
            return
        h = T.height[j0:j1 + 1, i0:i1 + 1].copy()
        mat = T.mat[j0:j1 + 1, i0:i1 + 1]
        h = self._carve_roads(h, T, i0, j0, cx, cz, half)
        xs = T.x0 + np.arange(i0, i1 + 1) * T.step
        zs = T.z0 + np.arange(j0, j1 + 1) * T.step
        X, Z = np.meshgrid(xs, zs)
        gy, gx = np.gradient(h, T.step)
        nrm = np.stack([-gx, np.ones_like(h), -gy], -1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        colors = np.zeros(h.shape + (3,), np.float32)
        colors[:] = TERRAIN_LOOK["Grass"]
        for code in np.unique(mat):
            if code == 0 or code > len(TERRAIN_MATERIALS):
                continue
            name = TERRAIN_MATERIALS[code - 1]
            if name in TERRAIN_LOOK or name in TERRAIN_COLORS:
                colors[mat == code] = TERRAIN_LOOK.get(name, TERRAIN_COLORS.get(name))
        h = h - 0.6          # keep the ground just under road slabs (8-stud grid can poke through)
        colors[T.water[j0:j1 + 1, i0:i1 + 1]] = (0.18, 0.33, 0.42)
        # one big "instance" with identity matrix: positions are world already
        V = np.c_[X.ravel(), h.ravel(), Z.ravel(), nrm.reshape(-1, 3), colors.reshape(-1, 3)].astype(np.float32)
        w = i1 - i0 + 1
        idx = []
        for j in range(j1 - j0):
            r0 = j * w
            r1 = (j + 1) * w
            a = np.arange(w - 1)
            idx.append(np.stack([r0 + a, r1 + a, r0 + a + 1, r0 + a + 1, r1 + a, r1 + a + 1], 1).ravel())
        idx = np.concatenate(idx).astype(np.int32)
        # expand to non-indexed with per-vertex colour via a second attribute stream (use instance col = white)
        verts = V[idx]
        self._terrain_vbo = self.ctx.buffer(verts.tobytes())
        self._terrain_col = self.ctx.buffer(np.array([1, 1, 1, 0], np.float32).tobytes())
        ident = np.eye(4, dtype=np.float32).T.reshape(16)
        self._terrain_inst = self.ctx.buffer(ident.tobytes())
        self._terrain_n = len(verts)
        self._terrain_center = (cx, cz)
        self._terrain_half = half

    def _carve_roads(self, h: np.ndarray, T, i0: int, j0: int, cx: float, cz: float, half: float) -> np.ndarray:
        """The terrain export is a height field, so where a road runs through a tunnel or a cut
        the hill would cover it. Push the ground below every road surface instead."""
        road_top = np.full(h.shape, np.inf, np.float32)

        def stamp(pos, rot, size, top):
            # footprint (oriented rectangle) -> grid cells
            corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.5
            ax_x, ax_z = rot[[0, 2], 0], rot[[0, 2], 2]
            pts = pos[[0, 2]] + corners[:, :1] * size[0] * ax_x + corners[:, 1:] * size[2] * ax_z
            g = np.stack([(pts[:, 0] - T.x0) / T.step - i0, (pts[:, 1] - T.z0) / T.step - j0], 1)
            mask = np.zeros(h.shape, np.uint8)
            cv2.fillConvexPoly(mask, np.round(g).astype(np.int32), 1)
            if mask.any():
                sel = mask.astype(bool)
                road_top[sel] = np.minimum(road_top[sel], top)

        P = self.map.parts
        idx = self.map.near(cx, cz, half)
        roadish = (P.mat_is("Asphalt") | P.mat_is("Concrete"))[idx] & (np.abs(P.rot[idx, 1, 1]) > 0.97) \
            & (np.maximum(P.size[idx, 0], P.size[idx, 2]) > 10)
        for i in idx[roadish]:
            stamp(P.pos[i], P.rot[i], P.size[i], P.pos[i, 1] + P.size[i, 1] / 2)
        mi = self.map.mesh_index
        if mi is not None:
            I = self.map.meshes.inst
            midx = mi.near(cx, cz, half)
            sel = midx[(I.mat_is("Asphalt")[midx]) & (np.maximum(I.size[midx, 0], I.size[midx, 2]) > 10)]
            for i in sel:
                stamp(I.pos[i], I.rot[i], I.size[i], I.pos[i, 1] - abs(I.rot[i][1, 1]) * I.size[i, 1] / 2 + 0.3)
        if np.isfinite(road_top).any():
            fin = np.isfinite(road_top)
            grown = cv2.erode(np.where(fin, road_top, 1e6).astype(np.float32), np.ones((3, 3), np.uint8))
            fin2 = grown < 1e5
            cap = np.where(fin2, grown - 0.8, np.inf)
            over = h > cap + 2.0                 # only where the hill would cover the road
            h = np.where(over, cap, h)
        return h

    # ------------------------------------------------------------ meshes
    def _geom_buffers(self, key: int):
        """(vbo, ibo or None) for a part shape (key < MESH_BASE) or a map mesh (MESH_BASE + geom)."""
        if key < MESH_BASE:
            return self.shape_vbo[key], None
        if key not in self._mesh_gl:
            g = self.map.meshes.geoms[key - MESH_BASE]
            vc = g.color if g.color is not None else np.ones_like(g.pos)
            v = np.c_[g.pos, g.norm, vc].astype(np.float32)
            self._mesh_gl[key] = (self.ctx.buffer(v.tobytes()), self.ctx.buffer(g.index.astype(np.uint32).tobytes()))
        return self._mesh_gl[key]

    def _mesh_batches(self, cam_pos: np.ndarray, fwd: np.ndarray) -> list:
        mi = self.map.mesh_index
        if mi is None:
            return []
        idx = mi.near(cam_pos[0], cam_pos[2], self.view_radius)
        if not len(idx):
            return []
        I = self.map.meshes.inst
        rel = I.pos[idx] - cam_pos
        half = mi.half[idx]
        dist = np.linalg.norm(rel, axis=1)
        keep = (rel @ fwd > -half - 40.0) & ~((half < 1.5) & (dist > 450)) & (I.transp[idx] < 0.95)
        keep &= np.array([self.map.meshes.geoms[g] is not None for g in I.geom[idx]])
        idx = idx[keep]
        if not len(idx):
            return []
        mats = self.mesh_inst_all[idx]
        cols = self.mesh_col_all[idx]
        if self.mesh_color_override:
            ov = self.mesh_color_override
            for k, i in enumerate(idx):
                v = ov.get(int(i))
                if v is not None:
                    cols[k, :3], cols[k, 3] = v
        geo = I.geom[idx]
        out = []
        order = np.argsort(geo, kind="stable")
        geo_s = geo[order]
        cuts = np.flatnonzero(np.r_[True, geo_s[1:] != geo_s[:-1], True])
        for a, b in zip(cuts[:-1], cuts[1:]):
            sel = order[a:b]
            out.append((MESH_BASE + int(geo_s[a]), mats[sel], cols[sel]))
        return out

    # ------------------------------------------------------------ helpers
    def _select(self, cam_pos: np.ndarray, fwd: np.ndarray) -> np.ndarray:
        idx = self.map.near(cam_pos[0], cam_pos[2], self.view_radius)
        if not len(idx):
            return idx
        P = self.map.parts
        rel = P.pos[idx] - cam_pos
        half = np.linalg.norm(P.size[idx], axis=1) / 2
        ahead = rel @ fwd > -half - 40.0
        tiny = (half < 1.0) & (np.linalg.norm(rel, axis=1) > 450)       # props too small to see far away
        keep = ahead & ~tiny & self.visible_part[idx]
        return idx[keep]

    # ------------------------------------------------------------ render
    def render(self, cam_pos, cam_target, extra_instances=None) -> np.ndarray:
        """extra_instances: list of (shape, pos(N,3), rot(N,3,3), size(N,3), rgba(N,4)) - our car etc."""
        ctx = self.ctx
        cam_pos = np.asarray(cam_pos, float)
        cam_target = np.asarray(cam_target, float)
        fwd = cam_target - cam_pos
        fwd /= np.linalg.norm(fwd)
        if self.map.terrain is not None and (self._terrain_center is None or
                                             max(abs(cam_pos[0] - self._terrain_center[0]),
                                                 abs(cam_pos[2] - self._terrain_center[1])) > self._terrain_half * 0.5):
            self._build_terrain(cam_pos[0], cam_pos[2])
        idx = self._select(cam_pos, fwd)
        P = self.map.parts
        batches = []
        if len(idx):
            cols = self.col_all[idx].copy()
            if self.color_override:
                for k, i in enumerate(idx):
                    if int(i) in self.color_override:
                        cols[k, :3], cols[k, 3] = self.color_override[int(i)]
            shapes = P.shape[idx]
            mats = self.inst_all[idx]
            for sh in np.unique(shapes):
                sel = shapes == sh
                batches.append((int(sh), mats[sel], cols[sel]))
        for sh, pos, rot, size, rgba in (extra_instances or []):
            batches.append((sh, instance_matrices(pos, rot, size), np.asarray(rgba, np.float32)))

        V = look_at(cam_pos, cam_target)
        Pm = perspective(self.vfov, self.W / self.H, 0.5, 5000.0)
        vp = (Pm @ V).astype(np.float32)
        # sun "camera" for shadows, centred a bit ahead of us
        centre = cam_pos + fwd * 180.0
        centre[1] = cam_target[1]
        L = look_at(centre + SUN_DIR * 1500.0, centre, up=(0.0, 0.0, 1.0) if abs(SUN_DIR[1]) > 0.99 else (0, 1, 0))
        lvp = (ortho(320.0, 10.0, 3200.0) @ L).astype(np.float32)

        batches += self._mesh_batches(cam_pos, fwd)
        bufs = []
        for sh, mats, cols in batches:
            if (sh not in self.shape_vbo and sh < MESH_BASE) or not len(mats):
                continue
            ib = ctx.buffer(np.ascontiguousarray(mats, np.float32).tobytes())
            cb = ctx.buffer(np.ascontiguousarray(cols, np.float32).tobytes())
            bufs.append((sh, ib, cb, len(mats)))

        # shadow pass
        if self.shadows:
            self.sfbo.use()
            self.sfbo.clear(depth=1.0)
            ctx.enable(ctx.DEPTH_TEST)
            ctx.disable(ctx.CULL_FACE)
            self.sprog["lightvp"].write(lvp.T.tobytes())
            for sh, ib, cb, n in bufs:
                vbo, ibo = self._geom_buffers(sh)
                vao = ctx.vertex_array(self.sprog, [(vbo, *VFMT), (ib, *IFMT)], ibo, skip_errors=True)
                vao.render(instances=n)
                vao.release()

        # main pass
        self.fbo.use()
        ctx.viewport = (0, 0, self.W, self.H)
        self.fbo.clear(*SKY_HORIZON, 1.0, depth=1.0)
        ctx.enable(ctx.DEPTH_TEST)
        self._sky()
        pr = self.prog
        pr["viewproj"].write(vp.T.tobytes())
        pr["lightvp"].write(lvp.T.tobytes())
        pr["sun"].value = tuple(SUN_DIR)
        pr["campos"].value = tuple(cam_pos)
        pr["skyh"].value = tuple(SKY_HORIZON)
        pr["haze"].value = 2600.0
        pr["use_shadow"].value = 1 if self.shadows else 0
        self.smap.use(location=0)
        pr["shadowmap"].value = 0
        if self.map.terrain is not None and self._terrain_center is not None:
            vao = ctx.vertex_array(pr, [(self._terrain_vbo, *VFMT), (self._terrain_inst, *IFMT),
                                        (self._terrain_col, "4f/i", "col")], skip_errors=True)
            vao.render(instances=1)
            vao.release()
        for sh, ib, cb, n in bufs:
            vbo, ibo = self._geom_buffers(sh)
            vao = ctx.vertex_array(pr, [(vbo, *VFMT), (ib, *IFMT), (cb, "4f/i", "col")], ibo, skip_errors=True)
            vao.render(instances=n)
            vao.release()
        for _, ib, cb, _ in bufs:
            ib.release()
            cb.release()
        data = self.fbo.read(components=3, alignment=1)
        img = np.frombuffer(data, np.uint8).reshape(self.H, self.W, 3)[::-1]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _sky(self) -> None:
        # clear already gives the horizon colour; a gradient is added after read-back for speed
        pass


def sky_gradient(img: np.ndarray, horizon_y: float) -> np.ndarray:
    """Blend a Roblox-like blue gradient into pixels that are still the clear colour."""
    H = img.shape[0]
    clear = np.array(SKY_HORIZON[::-1] * 255, np.int16)
    mask = (np.abs(img.astype(np.int16) - clear).sum(-1) < 6)
    t = np.clip((horizon_y - np.arange(H)) / max(1.0, horizon_y), 0, 1)[:, None]
    grad = (SKY_HORIZON[::-1] * (1 - t) + SKY_TOP[::-1] * t) * 255
    out = img.copy()
    ys, xs = np.nonzero(mask)
    out[ys, xs] = grad[ys, 0, None] if False else grad[ys]
    return out
