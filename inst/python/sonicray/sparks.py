"""Firework sparks: CPU-side *geometry* (vectorised numpy over every spark of every burst in the frame), GPU-side *pixels*.

The reference compositor drew each spark segment with PIL (thousands of Python-level draw calls per frame). Here the simulation is the
same maths -- same seeds, same ballistic drag/gravity, same curved trails -- but evaluated for all sparks at once, emitted as a list of
primitives (thick lines + discs on five planes), sorted into draw order, and binned into 32x32-pixel tiles for the `spark_raster`
Metal kernel. Drawing is at 2x resolution (SS = 2) with a 2x2 box filter, like the original.
"""
from __future__ import annotations

import math

import numpy as np

from .framespec import PLANE_SHIMMER, PLANE_WHITE, Burst

SS = 2
M_PTS = 9                          # points along each spark's trail (head .. tail)
TILE = 32

PRIM_DTYPE = np.dtype([("a", "f4"), ("b", "f4"), ("c", "f4"), ("d", "f4"), ("w", "f4"), ("fill", "f4"), ("plane", "i4"), ("kind", "i4")])


class SparkSystem:
    def __init__(self, W, H, cam, slow=1.5):
        self.W, self.H, self.slow = int(W), int(H), float(slow)
        self.px = self.W / 480.0
        lf = np.asarray(cam["lookfrom"], dtype=np.float64)
        f = np.asarray(cam["lookat"], dtype=np.float64) - lf
        f /= np.linalg.norm(f)
        r = np.cross(f, [0.0, 1.0, 0.0]); r /= np.linalg.norm(r)
        self.lf, self.f, self.r, self.u = lf, f, r, np.cross(r, f)
        self.tf = math.tan(cam["fov"] * math.pi / 360.0)
        # ultra-fine filament widths (sub-pixel at 1080p after the 2x downsample)
        self.w_tail = max(1, int(round(0.28 * SS * self.px)))
        self.w_mid = max(1, int(round(0.42 * SS * self.px)))
        self.w_head = max(2, int(round(0.58 * SS * self.px)))
        self._static = {}

    # ------------------------------------------------------------------ projection (world -> 2x-resolution screen)
    def _proj(self, P):
        P = np.asarray(P, dtype=np.float64) - self.lf
        dot = lambda v: P[:, 0] * v[0] + P[:, 1] * v[1] + P[:, 2] * v[2]          # explicit (BLAS would spin threads)
        xc, yc, zc = dot(self.r), dot(self.u), np.maximum(dot(self.f), 1e-3)
        sx = (xc / (zc * self.tf * self.W / self.H) * 0.5 + 0.5) * (self.W * SS)
        sy = (0.5 - yc / (zc * self.tf) * 0.5) * (self.H * SS)
        return sx, sy

    # ------------------------------------------------------------------ per-burst random sparks (cached: same uid -> same look)
    def _spark_static(self, b: Burst):
        s = self._static.get(b.uid)
        if s is not None:
            return s
        rng = np.random.RandomState(b.uid * 13 + 7)
        N = b.n_sparks
        idx = np.arange(N, dtype=np.float32) + 0.5
        phi = np.arccos(1.0 - 2.0 * idx / N)
        theta = math.pi * (1.0 + 5.0 ** 0.5) * idx
        v = np.stack([np.sin(phi) * np.cos(theta), np.cos(phi), np.sin(phi) * np.sin(theta)], axis=1)
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)).astype(np.float32))
        v = (v[:, :, None] * Q[None, :, :]).sum(1) + rng.normal(scale=0.11, size=(N, 3)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        u_rad = rng.uniform(0.0, 1.0, size=(N, 1)).astype(np.float32)
        sp = (b.v_max * (0.12 + 0.90 * (u_rad ** 0.52)) * (0.94 + 0.04 * (b.uid % 4))).reshape(-1)
        life = rng.uniform(b.life[0], b.life[1], size=N).astype(np.float32) * self.slow
        phase = rng.uniform(0.0, 2.0 * math.pi, size=N).astype(np.float32)
        s = (v.astype(np.float64), sp.astype(np.float64), life.astype(np.float64), phase.astype(np.float64))
        self._static[b.uid] = s
        return s

    # ------------------------------------------------------------------ one frame of primitives
    def primitives(self, t, bursts):
        """-> structured array of primitives in draw order (plane-wise painter's algorithm)."""
        px, slow = self.px, self.slow
        P = {k: [] for k in ("a", "b", "c", "d", "w", "fill", "plane", "kind", "ord")}

        def add(a, b, c, d, w, fill, plane, kind, order):
            n = len(np.atleast_1d(a))
            P["a"].append(np.broadcast_to(np.asarray(a, np.float64), (n,))); P["b"].append(np.broadcast_to(np.asarray(b, np.float64), (n,)))
            P["c"].append(np.broadcast_to(np.asarray(c, np.float64), (n,))); P["d"].append(np.broadcast_to(np.asarray(d, np.float64), (n,)))
            P["w"].append(np.broadcast_to(np.asarray(w, np.float64), (n,))); P["fill"].append(np.broadcast_to(np.asarray(fill, np.float64), (n,)))
            P["plane"].append(np.broadcast_to(np.asarray(plane, np.int64), (n,))); P["kind"].append(np.broadcast_to(np.asarray(kind, np.int64), (n,)))
            P["ord"].append(np.broadcast_to(np.asarray(order, np.int64), (n,)))

        def line(x0, y0, x1, y1, w, fill, plane, order):
            """PIL-style thick line: endpoints truncated to integers, even widths sit half a pixel to the +side."""
            w = np.asarray(w, np.float64)
            off = np.where(w % 2 == 0, 0.5, 0.0)
            add(np.trunc(x0) + off, np.trunc(y0) + off, np.trunc(x1) + off, np.trunc(y1) + off, w, fill, plane, 0, order)

        ranks = {}
        sp_rows = []                       # gathered per-spark rows over all bursts: filled below
        for rank, b in enumerate(bursts):
            dt = t - b.t0
            base = rank * 10_000_000
            ranks[b.uid] = base
            if dt < 0:                                   # ---- rising rocket
                f = 1.0 + dt / b.rise
                if f <= 0:
                    continue
                n_rseg = 5
                fs = np.array([max(f - 0.28 * (k / n_rseg), 0.0) for k in range(n_rseg + 1)])
                ys = b.launch_y + (fs ** 0.82) * (b.origin[1] - b.launch_y)
                sx, sy = self._proj(np.stack([np.full_like(fs, b.origin[0]), ys, np.full_like(fs, b.origin[2])], axis=1))
                for k in range(n_rseg):
                    seg_a = min(255, int(220 * f * b.gain * ((1.0 - k / n_rseg) ** 1.4)))
                    seg_w = self.w_head if k == 0 else (self.w_mid if k < 3 else self.w_tail)
                    line(sx[k + 1], sy[k + 1], sx[k], sy[k], seg_w, seg_a, b.plane, base + 2 * k)
                    line(sx[k + 1], sy[k + 1], sx[k], sy[k], self.w_tail, int(seg_a * 0.42), PLANE_SHIMMER, base + 2 * k + 1)
                rh = 1.15 * SS * px
                add(sx[0] - rh, sy[0] - rh, sx[0] + rh, sy[0] + rh, 0, min(255, int(245 * f * b.gain)), PLANE_WHITE, 1, base + 20)
                continue
            v, sp, life, phase = self._spark_static(b)
            alive = np.where(dt <= life)[0]
            if len(alive):
                sp_rows.append((rank, b, dt, alive, v[alive], sp[alive], life[alive], phase[alive]))
            # compact, jewel-like central flash (drawn after this burst's sparks)
            a_fl = math.exp(-dt / 0.075)
            if a_fl > 0.03:
                hx, hy = self._proj(np.array([b.origin]))
                r = (3.2 + 6.5 * (1.0 - a_fl)) * SS * px
                for j in range(5):
                    rj = r * (1.0 - 0.18 * j) * 0.5
                    plane = PLANE_WHITE if j >= 3 else b.plane
                    add(hx[0] - rj, hy[0] - rj, hx[0] + rj, hy[0] + rj, 0, min(255, int(210 * a_fl * b.gain * (0.10 + 0.15 * j))), plane, 1, base + 9_000_000 + j)

        if sp_rows:                                       # ---- all sparks of all bursts at once
            dts = np.concatenate([np.full(len(r[3]), r[2]) for r in sp_rows])
            gains = np.concatenate([np.full(len(r[3]), r[1].gain) for r in sp_rows])
            trail = np.concatenate([np.full(len(r[3]), r[1].trail_dt) for r in sp_rows])
            planes = np.concatenate([np.full(len(r[3]), r[1].plane, np.int64) for r in sp_rows])
            origin = np.concatenate([np.tile(np.asarray(r[1].origin, np.float64), (len(r[3]), 1)) for r in sp_rows])
            rank_base = np.concatenate([np.full(len(r[3]), r[0] * 10_000_000, np.int64) for r in sp_rows])
            sidx = np.concatenate([r[3] for r in sp_rows]).astype(np.int64)
            v = np.concatenate([r[4] for r in sp_rows]); sp = np.concatenate([r[5] for r in sp_rows])
            life = np.concatenate([r[6] for r in sp_rows]); phase = np.concatenate([r[7] for r in sp_rows])

            fracs = np.linspace(0.0, 1.0, M_PTS)[None, :]
            t_trail = np.maximum(dts[:, None] - fracs * np.minimum(trail, dts * 0.90)[:, None], 0.0)
            tp = t_trail / slow
            drag = (1.0 - np.exp(-2.30 * tp)) / 2.30
            q = v[:, None, :] * (sp[:, None, None] * drag[:, :, None])
            P3 = np.empty_like(q)
            P3[:, :, 0] = origin[:, None, 0] + q[:, :, 0]
            P3[:, :, 1] = origin[:, None, 1] + q[:, :, 1] - 0.5 * 1.38 * (tp ** 2)
            P3[:, :, 2] = origin[:, None, 2] + q[:, :, 2]
            sx, sy = self._proj(P3.reshape(-1, 3))
            sx, sy = sx.reshape(-1, M_PTS), sy.reshape(-1, M_PTS)

            age = dts / life
            base_a = np.maximum(0.0, 1.0 - age) ** 1.35
            glitter = np.where(age > 0.55, 0.72 + 0.28 * np.sin(dts * 36.0 + phase), 1.0)
            ak = np.minimum(np.clip(base_a * glitter * gains, 0.0, 1.15), 1.0)
            keep = ak > 0.02
            order0 = rank_base + sidx * 100
            early = dts < 0.055 * slow
            for seg in range(M_PTS - 1, 0, -1):
                head_w = (1.0 - (seg - 0.5) / (M_PTS - 1)) ** 1.65
                fill = np.trunc(255.0 * ak * head_w)
                ok = keep & (fill > 3)
                if not ok.any():
                    continue
                w_seg = self.w_head if seg == 1 else (self.w_mid if seg <= 3 else self.w_tail)
                o = order0 + (M_PTS - 1 - seg) * 3
                x0, y0, x1, y1 = sx[:, seg], sy[:, seg], sx[:, seg - 1], sy[:, seg - 1]
                m = ok & early & (seg <= 2)
                if m.any():
                    line(x0[m], y0[m], x1[m], y1[m], w_seg, np.trunc(fill[m] * 0.75), PLANE_WHITE, o[m])
                line(x0[ok], y0[ok], x1[ok], y1[ok], w_seg, fill[ok], planes[ok], o[ok] + 1)
                if seg >= 4:
                    m = ok & ~early
                    if m.any():
                        line(x0[m], y0[m], x1[m], y1[m], self.w_tail, np.trunc(fill[m] * 0.28), PLANE_SHIMMER, o[m] + 2)
            hx, hy = sx[:, 0], sy[:, 0]
            rs = 0.58 * SS * px * (0.75 + 0.30 * ak)
            rc = 0.28 * SS * px * (0.75 + 0.30 * ak)
            add(hx[keep] - rs[keep], hy[keep] - rs[keep], hx[keep] + rs[keep], hy[keep] + rs[keep], 0, np.trunc(255.0 * ak[keep]), planes[keep], 1, order0[keep] + 30)
            add(hx[keep] - rc[keep], hy[keep] - rc[keep], hx[keep] + rc[keep], hy[keep] + rc[keep], 0, np.trunc(225.0 * ak[keep] ** 0.9), PLANE_WHITE, 1, order0[keep] + 31)

        if not P["a"]:
            return np.zeros(0, PRIM_DTYPE)
        cat = {k: np.concatenate(v) for k, v in P.items()}
        order = np.argsort(cat["ord"], kind="stable")
        out = np.empty(len(order), PRIM_DTYPE)
        for k in ("a", "b", "c", "d", "w", "fill"):
            out[k] = cat[k][order]
        out["plane"] = cat["plane"][order]; out["kind"] = cat["kind"][order]
        return out

    # ------------------------------------------------------------------ binning into 32x32 tiles (final-resolution pixels)
    def bin_tiles(self, prims):
        """-> (tile_offsets uint32 [ntiles+1], tile_prim_indices uint32). Draw order is preserved inside every tile."""
        ntx, nty = -(-self.W // TILE), -(-self.H // TILE)
        if len(prims) == 0:
            return np.zeros(ntx * nty + 1, np.uint32), np.zeros(1, np.uint32), ntx
        a, b, c, d, w = (prims[k].astype(np.float64) for k in ("a", "b", "c", "d", "w"))
        pad = np.where(prims["kind"] == 0, w * 0.5 + 1.0, 1.5)
        x0, x1 = (np.minimum(a, c) - pad) / SS, (np.maximum(a, c) + pad) / SS
        y0, y1 = (np.minimum(b, d) - pad) / SS, (np.maximum(b, d) + pad) / SS
        tx0 = np.floor(x0 / TILE).astype(np.int64); tx1 = np.floor(x1 / TILE).astype(np.int64)
        ty0 = np.floor(y0 / TILE).astype(np.int64); ty1 = np.floor(y1 / TILE).astype(np.int64)
        vis = (tx1 >= 0) & (tx0 < ntx) & (ty1 >= 0) & (ty0 < nty)
        tx0, tx1, ty0, ty1 = (np.clip(x, 0, m - 1) for x, m in ((tx0, ntx), (tx1, ntx), (ty0, nty), (ty1, nty)))
        wt, ht = tx1 - tx0 + 1, ty1 - ty0 + 1
        counts = np.where(vis, wt * ht, 0)
        total = int(counts.sum())
        if total == 0:
            return np.zeros(ntx * nty + 1, np.uint32), np.zeros(1, np.uint32), ntx
        rep = np.repeat(np.arange(len(prims)), counts)
        start = np.cumsum(counts) - counts
        local = np.arange(total) - np.repeat(start, counts)
        wr = wt[rep]
        tile = (ty0[rep] + local // wr) * ntx + (tx0[rep] + local % wr)
        o = np.argsort(tile, kind="stable")
        off = np.zeros(ntx * nty + 1, np.uint32)
        off[1:] = np.cumsum(np.bincount(tile, minlength=ntx * nty))
        return off, rep[o].astype(np.uint32), ntx
