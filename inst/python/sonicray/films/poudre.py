"""Cache la Poudre v2 story (2:10, no on-screen text): t -> FrameSpec.

Ported from the original CPU compositor (visuals/poudre_3d_v2/02_compose_video.py) so that this package is self-contained.  All the
creative decisions live here -- colours, timing, seasonal hydrology, sun path -- and none of the pixel work:

    TOC (crimson)  Chl-a (green)  Alkalinity (royal blue-purple) fireworks = real Water Quality Portal sample sites, monthly sonde displays
    18 sonde 'lighthouses' ignite up-canyon from 46 s; Landsat-style scans sweep L->M->R every 7 s; a full solar day plays 94-123 s.

Copy this file as the starting point for a new story; keep the FrameSpec contract (framespec.py) and leave compositor.py alone.
"""
from __future__ import annotations

import math

import numpy as np

from ..framespec import Beacon, Burst, FrameSpec, Orb, PLANE_OF, SatScan, SkyDay, Term

# ----------------------------------------------------------------------------------------------- palette
BCOL = {"toc": (1.00, 0.14, 0.12), "chla": (0.20, 0.96, 0.32), "alk": (0.36, 0.14, 1.00)}   # terrain glow of each parameter
RIVER_GLOW = (0.04, 0.30, 0.65)

# ----------------------------------------------------------------------------------------------- timeline (seconds)
SLOW = 1.5
CAND_WEIGHT = 0.056
CELL_WEIGHT = 0.62 * CAND_WEIGHT
CELL_SPARK = 0.52
PEAK = 1.5
MONTH = 7.0
SUN_START, SUN_END = 94.0, 123.0
DW_FIRST = 20.5
FIRE_TIMES = [DW_FIRST + MONTH * m for m in range(40) if DW_FIRST + MONTH * m < SUN_START - 6.0]
SAT_FIRST, SAT_DUR, SAT_TAU = 15.5, 4.0, 1.76
SAT_TIMES = [SAT_FIRST + MONTH * k for k in range(40) if SAT_FIRST + MONTH * k < SUN_START - 8.0]
SAT_CENTERS, SAT_HALF = (0.22, 0.50, 0.78), 0.18
ERA2_START = 46.0
LH_IGNITE = {"pman": ERA2_START + 1.0, "pbr": ERA2_START + 1.0 + 0.7 * SLOW * 1.4, "archery": ERA2_START + 1.0 + 2 * 0.7 * SLOW * 1.4}
ERA3_START = 72.0
ERA3_STEP = 0.38
HOLD_END, FADE = 128.0, 2.0
TOTAL = HOLD_END + FADE
KEYFRAMES = [(6.5, "sparse WQP intro blips"), (17.5, "Landsat pass 1"), (21.3, "first monthly sonde display"), (38.0, "full 8 Hz groove"),
             (56.0, "Era 2 lighthouses + display"), (82.0, "all 18 sondes lit"), (99.0, "sunrise"), (109.0, "solar noon"), (120.5, "sunset")]


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def seasonal_mod(t, param, z_km=0.0):
    """~28 s hydrological year: TOC peaks in spring melt, Chl-a in late summer, alkalinity ramps up through summer."""
    s = ((t - 15.0 - 0.035 * z_km) / 28.0) % 1.0

    def bump(center, width):
        d = abs(((s - center + 0.5) % 1.0) - 0.5)
        u = max(0.0, 1.0 - d / width)
        return u * u * (3.0 - 2.0 * u)

    spring_melt, late_summer = bump(0.22, 0.24), bump(0.74, 0.26)
    s_alk = (s - 0.22) % 1.0
    alk_ramp = smooth(s_alk / 0.72) if s_alk < 0.72 else smooth((1.0 - s_alk) / 0.28)
    if param == "toc":
        val = 0.52 + 0.96 * spring_melt - 0.14 * late_summer
    elif param == "chla":
        val = 0.44 + 1.04 * late_summer - 0.16 * spring_melt
    else:
        val = 0.42 + 1.06 * alk_ramp
    return float(np.clip(val + 0.07 * math.sin(1.15 * t + 0.21 * z_km), 0.40, 1.52))


def active_steps_for_bar(bar_idx):
    """The opening starts slow and 1-to-1 synced with the score, then builds to a 12-note 0.125 s groove."""
    if bar_idx < 3:
        return {0, 8}
    if bar_idx < 6:
        return {0, 4, 8, 12}
    if bar_idx < 10:
        return {0, 2, 4, 8, 10, 12}
    if bar_idx < 16:
        return {0, 2, 4, 6, 8, 10, 12, 14}
    return {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14}


def param_for_step(bar_idx, s16):
    if bar_idx < 3:
        return "toc" if s16 == 0 else ("chla" if (bar_idx % 2 == 0) else "alk")
    if bar_idx < 6:
        return "toc" if s16 in (0, 8) else ("chla" if s16 == 4 else "alk")
    if bar_idx < 16:
        return "toc" if s16 in (0, 8) else ("alk" if s16 in (4, 12) else "chla")
    if (s16 % 4) == 0:
        return "toc"
    return "chla" if (s16 % 2) == 1 else "alk"


class PoudreStory:
    total = TOTAL
    keyframes = KEYFRAMES

    def __init__(self, comp):
        """comp: a GpuCompositor (gives layer presence, geometry and the scene metadata)."""
        self.c = comp
        meta = comp.meta
        L = comp.layers
        self.W, self.H = comp.W, comp.H
        self.sites = meta["sites"]
        self.by = {s["site"]: s for s in self.sites}
        self.cells = {c["id"]: c for c in meta["cells"]}
        self.sunp = meta["sun_path"]
        self.sun_mins = [q["min"] for q in self.sunp]
        self.sun_els = [q["el"] for q in self.sunp]
        self.fwk = {(s["site"], b["color"]): (f"fwk_{b['color']}_{s['site']}" if L.has(f"fwk_{b['color']}_{s['site']}") else None)
                    for s in self.sites for b in s["bursts"]}
        self.bea = {s["site"]: ("bea_" + s["site"] if L.has("bea_" + s["site"]) else None) for s in self.sites}
        self.sun = [("sun_%d" % k) if L.has("sun_%d" % k) else None for k in range(len(self.sunp))]
        self.sky_layers = [("sky_%d" % k) if L.has("sky_%d" % k) else None for k in range(len(self.sunp))]
        self.fwk_cell = {cid: ("fwk_cell_" + cid if L.has("fwk_cell_" + cid) else None) for cid in self.cells}
        self.rest = [s["site"] for s in sorted(self.sites, key=lambda s: -s["z"]) if not s["lighthouse"]]
        self.all = [s["site"] for s in sorted(self.sites, key=lambda s: -s["z"])]
        self.site_scale = {}
        for s in self.sites:
            neigh = sum(math.exp(-((s["x"] - q["x"]) ** 2 + (s["z"] - q["z"]) ** 2) / (2.0 * 7.5 ** 2)) for q in self.sites)
            self.site_scale[s["site"]] = float((1.0 / (neigh ** 0.72)) * 1.42)
        self.cell_events = self._cell_schedule()

    # ------------------------------------------------------------------------------------------------ schedule
    def _cell_schedule(self):
        """Real WQP monitoring cells upstream of riverbluffs, slow and sparse from 3.5 s, building into the 8 Hz groove."""
        ids = sorted(self.cells)
        mk = lambda key, k: np.array([math.sqrt(max(self.cells[c].get(key, 0) * k, 0)) + 0.15 for c in ids], float)
        p = {"toc": mk("n_toc", 1.0), "chla": mk("n_chla", 1.0), "alk": mk("n_alk", 0.25)}
        for k in p:
            p[k] /= p[k].sum()
        rng = np.random.RandomState(2026)
        ev, uid, last = [], 10_000, []
        for i in range(int((SUN_START - 6.0 - 3.5) / 0.125)):
            bar, s16 = i // 16, i % 16
            if s16 not in active_steps_for_bar(bar):
                continue
            col = param_for_step(bar, s16)
            for _ in range(6):
                cid = ids[int(rng.choice(len(ids), p=p[col]))]
                if cid not in last[-4:]:
                    break
            last.append(cid)
            ev.append(("cell:" + cid, col, round(3.5 + i * 0.125, 4), uid)); uid += 1
        return ev

    def _bursts_at(self, t):
        fw, uid = [], 0
        for t0 in FIRE_TIMES:
            for s_idx, n in enumerate(self.all):
                for k, c in enumerate(("toc", "chla", "alk")):
                    fw.append((n, c, t0 + 0.015 * s_idx + 0.125 * k, uid)); uid += 1
        fw += self.cell_events
        has = lambda f: (self.fwk_cell[f[0][5:]] if f[0].startswith("cell:") else self.fwk[(f[0], f[1])]) is not None
        return [f for f in fw if -0.75 < t - f[2] < 6.6 and has(f)]

    # ------------------------------------------------------------------------------------------------ one frame
    def frame_spec(self, t):
        c = self.c
        fw = self._bursts_at(t)
        w_fwk = {}
        for (n, col, t0, _) in fw:
            dt = t - t0
            if dt < 0:
                continue
            cell = n.startswith("cell:")
            z_km = self.cells[n[5:]]["z"] if cell else self.by[n]["z"]
            sm = seasonal_mod(t0, col, z_km)
            if cell:
                w = (1 - math.exp(-dt / 0.025)) * math.exp(-dt / 0.22) * 1.15 * sm
            else:
                w = (1 - math.exp(-dt / (0.035 * SLOW))) * math.exp(-dt / (0.24 * SLOW * PEAK)) * 0.72 * sm
            w_fwk[(n, col)] = w_fwk.get((n, col), 0.0) + w

        ign = dict(LH_IGNITE)
        for k, n in enumerate(self.rest):
            ign[n] = ERA3_START + 0.5 + ERA3_STEP * k
        on, w_bea = {}, {}
        for n, t0 in ign.items():
            dt = t - t0
            if dt < 0 or self.bea[n] is None:
                continue
            ramp = 1 - math.exp(-dt / (0.28 * SLOW))
            flare = 1.0 + 1.8 * math.exp(-dt / (0.55 * SLOW))
            boost = 1.0 + 0.45 * smooth((t - ERA3_START) / 4.0)
            on[n] = ramp
            w_bea[n] = ramp * flare * boost * (1.0 + 0.05 * math.sin(2 * math.pi * 0.4 * t + self.by[n]["x"]))

        p = smooth((t - SUN_START) / (SUN_END - SUN_START))
        fade = 1.0 if t <= HOLD_END else max(0.0, 1 - (t - HOLD_END) / FADE)
        dusk = smooth((t - SUN_END) / max(HOLD_END - SUN_END, 1.0)) if t > SUN_END else 0.0
        spec = FrameSpec(t=t, fade=fade, seed=int(t * 1000) % (2 ** 31))

        # ---- terrain light
        amb_lvl = 0.88 * smooth(p * 1.8) * (1.0 - 0.42 * smooth(max(0.0, (p - 0.75) / 0.25)) - 0.25 * dusk)
        spec.terms.append(Term("amb", (amb_lvl,) * 3))
        sky = None
        if p > 0:
            m = self.sun_mins[-1] * p
            k = int(np.clip(np.searchsorted(self.sun_mins, m, side="right") - 1, 0, len(self.sun_mins) - 2))
            f = float(np.clip((m - self.sun_mins[k]) / (self.sun_mins[k + 1] - self.sun_mins[k]), 0, 1))
            el_now = self.sun_els[k] * (1 - f) + self.sun_els[k + 1] * f
            hi_sun = min(max(el_now - 2.0, 0.0) / 28.0, 1.0)
            sunset_w = smooth(max(0.0, (p - 0.68) / 0.32))
            low_col = np.array([1.0, 0.60, 0.36]) * (1.0 - sunset_w) + np.array([1.0, 0.46, 0.24]) * sunset_w
            warm = low_col * (1.0 - hi_sun ** 0.65) + np.array([1.0, 0.96, 0.88]) * (hi_sun ** 0.65)
            a_, b_ = self.sun[k], self.sun[k + 1]
            sun_gain = smooth(p * 6.0) * (0.66 - 0.14 * sunset_w) * (1.0 - 0.55 * dusk)
            if a_ is not None:
                spec.terms.append(Term(a_, tuple(warm * (sun_gain * (1 - f)))))
            if b_ is not None:
                spec.terms.append(Term(b_, tuple(warm * (sun_gain * f))))
            spec.terms.append(Term("riv", tuple(np.array(RIVER_GLOW) * (smooth(p * 2.5) * (1.0 - 0.35 * dusk))), mode=1))
            # ---- sky + sun
            if self.sky_layers[k] is not None and self.sky_layers[k + 1] is not None:
                sky = SkyDay(self.sky_layers[k], self.sky_layers[k + 1], f,
                             sky_exp=0.44 * smooth(p * 2.6) * (1.0 - 0.38 * dusk),
                             zenith=tuple(np.array([0.36, 0.60, 1.15]) * (1.0 - 0.35 * sunset_w) + np.array([0.24, 0.36, 0.78]) * (0.35 * sunset_w)),
                             horizon_tint=tuple(np.array([1.05, 0.92, 0.82]) * (1.0 - sunset_w) + np.array([1.22, 0.74, 0.48]) * sunset_w),
                             dawn_band=smooth(p * 6.0) * max(0.0, 1.0 - p * 4.2),
                             set_band=smooth(max(0.0, (p - 0.68) / 0.32)) * (1.0 - 0.30 * dusk))
            spec.aerial = (0.05 + 0.03 * smooth(max(0.0, (p - 0.70) / 0.30))) * smooth(p * 2.2)
            if p < 0.42:
                noon_w = smooth(max(0.0, (p - 0.24) / 0.18))
                if noon_w > 0:
                    spec.orb = Orb(1, c.ox0, -0.08 * self.H, amount=0.28 * noon_w)
            else:
                w_desc = float(np.clip((p - 0.42) / 0.58, 0.0, 1.0))
                sun_col = np.array([1.0, 0.94, 0.78]) * (1.0 - w_desc ** 1.2) + np.array([1.0, 0.42, 0.14]) * (w_desc ** 1.2)
                spec.orb = Orb(2, c.ox0 + (c.ox1 - c.ox0) * (w_desc ** 0.85),
                               c.orb_y_top + (c.orb_y_set - c.orb_y_top) * (w_desc ** 0.92) + 0.025 * self.H * dusk,
                               radius=c.orb_r * (1.0 + 0.12 * w_desc), visibility=smooth(min(w_desc * 4.5, 1.0)) * (1.0 - 0.75 * dusk),
                               dusk=dusk, color=tuple(sun_col))
        spec.sky = sky
        star_fade = max(0.0, 1.0 - smooth(p * 4.0)) + 0.45 * dusk
        spec.star_amount = 0.75 * star_fade if star_fade > 0.01 else 0.0

        for (n, col), w in w_fwk.items():
            if w > 1e-3:
                layer = self.fwk_cell[n[5:]] if n.startswith("cell:") else self.fwk[(n, col)]
                if layer is not None:
                    wt = CELL_WEIGHT if n.startswith("cell:") else (CAND_WEIGHT * 0.65 * self.site_scale.get(n, 1.0))
                    spec.terms.append(Term(layer, tuple(np.array(BCOL[col]) * (w * wt))))
        for n, w in w_bea.items():
            if self.bea[n] is not None:
                wv = w * 0.175 * self.site_scale.get(n, 1.0)
                spec.beacon_terms.append(Term(self.bea[n], (wv,) * 3))

        # ---- satellite scans
        events = [(ts, SAT_CENTERS[kk % len(SAT_CENTERS)]) for kk, ts in enumerate(SAT_TIMES) if ts - 0.2 < t < ts + SAT_DUR + 7 * SAT_TAU]
        if events:
            spec.sat = SatScan(t, events, second_layer=self.sun[1] if len(self.sun) > 1 else None,
                               half_width=SAT_HALF, duration=SAT_DUR, tau=SAT_TAU)

        # ---- beacon sprites
        scale = self.W / 640.0
        for n, on_val in on.items():
            if on_val <= 1e-3:
                continue
            si = self.by[n]
            px, py = float(si["px"]), float(si["py"])
            if px < -12 * scale or px > self.W + 12 * scale or py < -12 * scale or py > self.H + 12 * scale:
                continue
            w_b = float(w_bea.get(n, on_val))
            flare = min(max(w_b / max(on_val, 1e-3), 0.85), 1.45)
            pulse = 0.70 * float(on_val) * (0.95 + 0.05 * math.sin(2 * math.pi * 0.4 * t + si["x"])) * (0.82 + 0.18 * flare)
            spec.beacons.append(Beacon(px, py, max(float(si["beam"]), 8.0 * scale), pulse))

        # ---- sparks
        bursts = []
        for (n, col, t0, uid) in fw:
            if n.startswith("cell:"):
                cc = self.cells[n[5:]]
                sm = seasonal_mod(t0, col, cc["z"])
                bursts.append(Burst(uid, PLANE_OF[col], t0, (cc["x"], cc["y"], cc["z"]), cc["y0"], max(24, int(round(68 * (0.55 + 0.45 * sm)))),
                                    CELL_SPARK * 1.15 * (0.50 + 0.56 * sm), 4.5 * (0.58 + 0.42 * sm), 0.28, (0.72, 1.30), 0.36 * SLOW, cell=True))
            else:
                s = self.by[n]
                b = next(b for b in s["bursts"] if b["color"] == col)
                dens = min(max(self.site_scale.get(n, 1.0) ** 0.28, 0.74), 1.10)
                sm = seasonal_mod(t0, col, b["z"])
                bursts.append(Burst(uid, PLANE_OF[col], t0, (b["x"], b["y"], b["z"]), s["y"], max(48, int(round(128 * (0.55 + 0.45 * sm)))),
                                    0.92 * dens * (0.48 + 0.58 * sm), 5.6 * dens * (0.58 + 0.42 * sm), 0.45 * SLOW, (1.05, 1.95), 0.44 * SLOW))
        spec.bursts = bursts
        return spec
