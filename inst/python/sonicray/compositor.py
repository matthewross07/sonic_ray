# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
"""GpuCompositor: FrameSpec -> uint8 RGB frame, entirely on the Apple GPU.

Per frame the CPU only builds a FrameSpec (cheap, in the story) and the spark primitive list (numpy); every pixel operation runs as a
Metal kernel on layers that are memory-mapped straight into GPU buffers:

    layer sums -> satellite scan -> river halo -> tone map -> sharpen -> bloom (3 blurs) -> day/night sky (histogram percentile on GPU)
    -> sun orb -> spark rasteriser -> firework screen-blend -> beacons -> fade -> dither -> uint8

One GpuCompositor owns one Metal context; use one per worker process (spawn, never fork).
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass

import numpy as np

from .framespec import FrameSpec
from .gpu import GPU, SonicrayError
from .sparks import PRIM_DTYPE, SparkSystem

SHIMMER = (0.56, 0.46, 1.00)
WHITE = (1.00, 1.00, 1.00)
MAX_PRIMS = 600_000
MAX_TILE_ENTRIES = 3_000_000


@dataclass
class Look:
    sharpen: float = 0.0            # unsharp-mask amount (0.65 for the final Poudre film)
    dither: bool = False            # triangular dither before 8-bit quantisation (hides banding in dark gradients)
    beacon_tint: tuple = (0.20, 0.50, 1.00)
    spark_slow: float = 1.5         # time-stretch used by the spark model (see sparks.py)
    # spark colours of planes 0, 1, 2 (default: crimson, algae green, royal violet).  Plane 3 = shimmer, 4 = white core.
    spark_colors: tuple = ((1.00, 0.22, 0.18), (0.20, 0.96, 0.32), (0.46, 0.20, 1.00))


class LayerSet:
    """Lazily memory-maps `<dir>/<name>.f32` into GPU buffers; also gives CPU (memmap) access."""

    def __init__(self, gpu: GPU, directory: str):
        self.gpu, self.dir = gpu, directory
        self._h = {}

    def path(self, name):
        return os.path.join(self.dir, name + ".f32")

    def has(self, name):
        return name is not None and os.path.exists(self.path(name))

    def handle(self, name):
        if name not in self._h:
            if not self.has(name):
                raise SonicrayError(f"layer '{name}' not found in {self.dir} (expected {name}.f32)")
            self._h[name] = self.gpu.mmap(self.path(name))
        return self._h[name]

    def cpu(self, name, ch):
        if not self.has(name):
            return None
        return np.memmap(self.path(name), dtype=np.float32, mode="r", shape=(self.gpu.H, self.gpu.W, ch))


class GpuCompositor:
    def __init__(self, layers_dir: str, look: Look = None):
        self.look = look or Look()
        meta = json.load(open(os.path.join(layers_dir, "scene_meta.json")))
        self.meta = meta
        self.W, self.H, self.cam = int(meta["width"]), int(meta["height"]), meta["cam"]
        W, H = self.W, self.H
        self.g = g = GPU(W, H)
        self.layers = LayerSet(g, layers_dir)
        L = self.layers
        for need in ("amb", "mask", "riv"):
            if not L.has(need):
                raise SonicrayError(f"{layers_dir} is missing required layer {need}.f32 (see framespec.py for the layer contract)")

        mask = np.array(L.cpu("mask", 1)[..., 0])
        self.mask = mask
        m_eff = np.clip(mask, 0.0, 1.0).astype(np.float32)
        self.h_meff = g.alloc1(); g.write(self.h_meff, m_eff)
        # geometry the stories need (sun path, horizon band, atmospheric perspective)
        yy = np.arange(H, dtype=np.float32)[:, None]
        rows = np.where(mask.sum(1) > 0)[0]
        self.y_top = float(rows.min()) if len(rows) else 0.31 * H
        self.y_bot = float(rows.max()) if len(rows) else float(H - 1)
        far = np.clip(1.0 - (yy - self.y_top) / max(self.y_bot - self.y_top, 1.0), 0.0, 1.0) ** 2.1
        self.h_aerial = g.alloc1(); g.write(self.h_aerial, (far * m_eff).astype(np.float32))
        self.ox0, self.ox1 = 0.36 * W, 0.54 * W
        ys = np.where(mask[:, int(self.ox1)] > 0.5)[0]
        self.y_edge = float(ys.min()) if len(ys) else 0.32 * H
        self.orb_r = 0.015 * H
        self.orb_y_top = -0.03 * H
        self.orb_y_set = self.y_edge + 0.018 * H

        # satellite-scan coordinates (optional)
        self.h_uu = self.h_vv = None
        if L.has("uv_u") and L.has("uv_v"):
            gam = lambda c: np.power(np.clip(c, 0, 1), 1 / 2.2)           # 8-bit textures come back as value^2.2: decode
            U, V = L.cpu("uv_u", 3), L.cpu("uv_v", 3)
            self.h_uu, self.h_vv = g.alloc1(), g.alloc1()
            g.write(self.h_uu, (gam(U[..., 0]) + gam(U[..., 1]) / 255.0).astype(np.float32))
            g.write(self.h_vv, (gam(V[..., 0]) + gam(V[..., 1]) / 255.0).astype(np.float32))

        a3, a1 = g.alloc3, g.alloc1
        self.lin, self.bl, self.terrain, self.tblur, self.bright = a3(), a3(), a3(), a3(), a3()
        self.b1, self.b2, self.b3 = a3(), a3(), a3()
        self.ov, self.ob1, self.ob2, self.sky_buf = a3(), a3(), a3(), a3()
        self.lum, self.g1, self.g2 = a1(), a1(), a1()
        self.out = g.alloc_bytes(W * H * 3)
        self.bmax, self.omax, self.p90, self.hor = g.alloc_bytes(16), g.alloc_bytes(16), g.alloc_bytes(16), g.alloc_bytes(16)
        self.hist_nb = 8192
        self.hist = g.alloc_bytes(self.hist_nb * 4)
        self.sparks = SparkSystem(W, H, self.cam, self.look.spark_slow)
        self.h_prims = g.alloc_bytes(MAX_PRIMS * PRIM_DTYPE.itemsize)
        self.h_toff = g.alloc_bytes(((-(-W // 32)) * (-(-H // 32)) + 1) * 4)
        self.h_tidx = g.alloc_bytes(MAX_TILE_ENTRIES * 4)

    # ------------------------------------------------------------------------------------------------------ one frame
    def render(self, spec: FrameSpec) -> bytes:
        g, W, H, L, look = self.g, self.W, self.H, self.layers, self.look
        s = W / 480.0
        h = L.handle
        terms = [(h(t.layer), t.mode, t.weight) for t in spec.terms]
        bea_terms = [(h(t.layer), t.mode, t.weight) for t in spec.beacon_terms]

        # ---- CPU: spark primitives (numpy), uploaded straight into shared memory
        use_ov = 0
        ntx = -(-W // 32)
        if len(spec.bursts):
            prims = self.sparks.primitives(spec.t, spec.bursts)
            if len(prims):
                if len(prims) > MAX_PRIMS:
                    raise SonicrayError(f"{len(prims)} spark primitives in one frame exceeds MAX_PRIMS={MAX_PRIMS}")
                off, idx, ntx = self.sparks.bin_tiles(prims)
                if len(idx) > MAX_TILE_ENTRIES:
                    raise SonicrayError("too many spark/tile entries in one frame; reduce n_sparks")
                g.write(self.h_prims, prims); g.write(self.h_toff, off); g.write(self.h_tidx, idx)
                use_ov = 1

        sky, orb = spec.sky, spec.orb
        day = 1 if sky is not None else 0
        use_stars = 1 if (spec.star_amount > 0 and L.has("stars")) else 0
        dummy = h("amb")
        sa = h(sky.layer_a) if day else dummy
        sb = h(sky.layer_b) if day else dummy
        stars = h("stars") if use_stars else dummy
        z3 = (0.0, 0.0, 0.0)
        skyp = struct.pack("4i8f12f2fi", W, H, day, use_stars,
                           sky.f if day else 0.0, spec.t, spec.star_amount, self.y_edge, self.ox1,
                           sky.dawn_band if day else 0.0, sky.set_band if day else 0.0, sky.sky_exp if day else 0.0,
                           *(sky.zenith if day else z3), *(sky.horizon_tint if day else z3), *(sky.venus if day else z3), *(sky.sunset if day else z3),
                           -24.0, 40.0 / self.hist_nb, self.hist_nb)

        beacons = []
        for b in spec.beacons[:24]:
            beacons += [b.x, b.y, b.beam, b.pulse]
        nb = len(beacons) // 4

        g.zero(self.bmax, 16); g.zero(self.omax, 16)
        if day:
            g.zero(self.hist, self.hist_nb * 4)
        d2 = struct.pack("2i", W, H)
        g.begin()
        g.axpy(self.lin, terms, init=True)
        sat = spec.sat
        if sat is not None and sat.events and self.h_uu is not None:
            ev = sat.events[:4]
            ts = [e[0] for e in ev] + [0.0] * (4 - len(ev)); uc = [e[1] for e in ev] + [0.0] * (4 - len(ev))
            second = L.has(sat.second_layer)
            g.op("sat_add", [self.lin, dummy, h(sat.second_layer) if second else dummy, self.h_uu, self.h_vv, h("riv")],
                 struct.pack("4i4f4f4f3f", W, H, len(ev), 1 if second else 0, sat.t, sat.half_width, sat.duration, sat.tau, *ts, *uc, *sat.color))
        if bea_terms:
            g.axpy(self.bl, bea_terms, init=True)
            g.axpy(self.lin, [(self.bl, 0, tuple(float(x) for x in look.beacon_tint))], init=False)
            g.op("lum_mean", [self.bl, self.lum], d2)
            g.blur(self.lum, self.g1, 12 * s, 1)
            g.blur(self.lum, self.g2, 30 * s, 1)
            g.op("halo_add", [self.lin, self.g1, self.g2, h("riv")],
                 struct.pack("2i7f", W, H, 7.5, 14.0, 1.15, 0.45, 0.12 * 0.476, 0.44 * 0.476, 0.92 * 0.476))
        g.op("tonemap", [self.lin, self.h_meff, self.terrain, self.bright, self.bmax], d2)
        if look.sharpen > 0:
            g.blur(self.terrain, self.tblur, max(1.2 * W / 1920.0, 0.5), 3)
            g.op("sharpen_comb", [self.terrain, self.tblur], struct.pack("2if", W, H, look.sharpen))
        g.blur(self.bright, self.b1, 2.5 * s, 3)
        g.blur(self.bright, self.b2, 8.5 * s, 3)
        g.blur(self.bright, self.b3, 22 * s, 3)
        # sky: percentile on the GPU, then render
        if day:
            g.op("sky_hist", [sa, sb, self.hist], skyp, (W + 3) // 4, (H + 3) // 4)
            g.op("sky_pctl", [self.hist, self.p90], skyp, 1, 1)
        g.op("sky_hor", [sa, sb, self.p90, self.hor], skyp, 1, 1)
        g.op("sky_render", [sa, sb, self.p90, stars, self.sky_buf], skyp)
        if use_ov:
            cols = [float(x) for c in (*look.spark_colors, SHIMMER, WHITE) for x in c]
            g.op("spark_raster", [self.h_prims, self.h_toff, self.h_tidx, self.ov, self.omax], struct.pack("3i15f", W, H, ntx, *cols))
            g.blur(self.ov, self.ob1, 0.85 * s, 3)
            g.blur(self.ov, self.ob2, 2.6 * s, 3)
        orbp = [0.0] * 9
        orb_mode = 0
        if orb is not None:
            orb_mode = orb.mode
            orbp = [orb.x, orb.y, orb.radius, orb.visibility, orb.dusk, orb.amount, *orb.color]
        bc = beacons + [0.0] * (96 - len(beacons))
        g.op("final_frame", [self.terrain, self.h_meff, self.sky_buf, self.hor, self.h_aerial, self.b1, self.b2, self.b3,
                             self.ov, self.ob1, self.ob2, self.bmax, self.omax, self.out],
             struct.pack("6iIi3f9f96f", W, H, orb_mode, 1 if spec.aerial > 0 else 0, use_ov, 1 if look.dither else 0, spec.seed & 0x7FFFFFFF, nb,
                         spec.aerial, spec.fade, W / 640.0, *orbp, *bc))
        g.end()
        return g.read_bytes(self.out, W * H * 3)
