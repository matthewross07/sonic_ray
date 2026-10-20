#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
"""Per-kernel GPU cost in ms (each kernel repeated 20x inside ONE command buffer, so sync overhead is excluded)."""
import os, struct, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inst", "python"))
from sonicray.cli import _make

ap_layers = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.cache/sonicray_poudre/layers_1920x1080")
c, story = _make(ap_layers, "poudre", 0.65, True)
g, W, H = c.g, c.W, c.H
REP = 20
spec = story.frame_spec(21.3); c.render(spec)

def timeit(name, fn):
    g.begin(); fn(); g.end()                                 # warm
    t0 = time.time(); g.begin()
    for _ in range(REP): fn()
    g.end()
    print(f"  {name:34s} {1000 * (time.time() - t0) / REP:7.2f} ms")

h = c.layers.handle
ids = [n for n in ("amb",) + tuple(f"fwk_toc_{s['site']}" for s in story.sites)[:7]]
terms8 = [(h(n), 0, (0.1, 0.1, 0.1)) for n in ids]
timeit("axpy8 (8 layers, one batch)", lambda: g.axpy(c.lin, terms8, init=True))
d2 = struct.pack("2i", W, H)
timeit("tonemap", lambda: g.op("tonemap", [c.lin, c.h_meff, c.terrain, c.bright, c.bmax], d2))
timeit("blur direct sigma=3.4, 3ch (ov)", lambda: g.blur(c.ov, c.ob1, 3.4, 3))
timeit("blur direct sigma=1.2, 3ch (sharpen)", lambda: g.blur(c.terrain, c.tblur, 1.2, 3))
timeit("blur big sigma=10 (box/blur/up) 3ch", lambda: g.blur(c.bright, c.b1, 10, 3))
timeit("blur big sigma=88 3ch", lambda: g.blur(c.bright, c.b3, 88, 3))
timeit("blur big sigma=48 1ch (halo)", lambda: g.blur(c.lum, c.g1, 48, 1))
timeit("box_down only (k=3, 3ch)", lambda: g.op("box_down", [c.bright, g.S1], struct.pack("5i", W, H, W // 3, H // 3, 3), W // 3, H // 3))
timeit("bilinear_up only (3ch)", lambda: g.op("bilinear_up", [g.S1, c.b1], struct.pack("5i", W // 3, H // 3, W, H, 3)))
timeit("lum_mean", lambda: g.op("lum_mean", [c.bl, c.lum], d2))
sky = story.frame_spec(109.0).sky
skyp = struct.pack("4i8f12f2fi", W, H, 1, 1, sky.f, 109.0, 0.0, c.y_edge, c.ox1, 0.0, 0.0, sky.sky_exp, *sky.zenith, *sky.horizon_tint, *sky.venus, *sky.sunset, -24.0, 40.0 / c.hist_nb, c.hist_nb)
sa, sb = h(sky.layer_a), h(sky.layer_b)
timeit("sky_hist", lambda: g.op("sky_hist", [sa, sb, c.hist], skyp, (W + 3) // 4, (H + 3) // 4))
timeit("sky_render", lambda: g.op("sky_render", [sa, sb, c.p90, h("stars"), c.sky_buf], skyp))
sat = story.frame_spec(17.5).sat
timeit("sat_add (3 events)", lambda: g.op("sat_add", [c.lin, h("amb"), h("amb"), c.h_uu, c.h_vv, h("riv")], struct.pack("4i4f4f4f3f", W, H, 3, 0, 17.5, .18, 4., 1.76, 15.5, 22.5, 29.5, 0, .22, .5, .78, 0, .84, .93, 1.0)))
prims = c.sparks.primitives(21.3, spec.bursts); off, idx, ntx = c.sparks.bin_tiles(prims)
g.write(c.h_prims, prims); g.write(c.h_toff, off); g.write(c.h_tidx, idx)
cols = [float(x) for col in ((1, .2, .2), (.2, 1, .3), (.5, .2, 1), (.6, .5, 1), (1, 1, 1)) for x in col]
timeit(f"spark_raster ({len(prims)} prims)", lambda: g.op("spark_raster", [c.h_prims, c.h_toff, c.h_tidx, c.ov, c.omax], struct.pack("3i15f", W, H, ntx, *cols)))
fin = struct.pack("6iIi3f9f96f", W, H, 0, 0, 1, 1, 5, 0, 0.0, 1.0, W / 640.0, *([0.0] * 9), *([0.0] * 96))
timeit("final_frame (no beacons)", lambda: g.op("final_frame", [c.terrain, c.h_meff, c.sky_buf, c.hor, c.h_aerial, c.b1, c.b2, c.b3, c.ov, c.ob1, c.ob2, c.bmax, c.omax, c.out], fin))
