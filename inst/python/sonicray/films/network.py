# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
"""Generic sensor-network story: beacons for sensors, fireworks for samples.  DATA-DRIVEN -- this is the story to adapt for real data.

Inputs, all in the layers directory:
    scene_meta.json   scene geometry written by the renderer (sonicray R package):  width, height, cam, sites[...], cells = [], sun_path = []
    story.json        {"duration_s": 16, "parameters": {"param_a": {"plane": 0, "glow": [1,.14,.12]}, ...}, "spark_colors": [[r,g,b] x3]}
    sensors.csv       site,online_s                    when each sensor's beacon lights up (seconds in the film)
    events.csv        site,parameter,time_s,value      one firework per sample; value in [0, 1] sets size / brightness
Layer files:  amb, mask, riv, [stars], bea_<site>, fwk_<parameter>_<site>   (see framespec.py for the contract)

The bundled demo (R: sr_demo()) generates all of the above from SYNTHETIC data.  To use real sensors replace sensors.csv / events.csv
(and render the layers for your own terrain); nothing in this file needs to change.  Up to 3 parameters (planes 0-2) are supported.
"""
from __future__ import annotations

import csv
import json
import math
import os

from ..framespec import Beacon, Burst, FrameSpec, Term

SLOW = 1.5
PEAK = 1.5


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class NetworkStory:
    def __init__(self, comp):
        self.c = comp
        d = comp.layers.dir
        self.W, self.H = comp.W, comp.H
        cfg = json.load(open(os.path.join(d, "story.json")))
        self.cfg = cfg
        self.total = float(cfg.get("duration_s", 16.0))
        self.params = cfg["parameters"]                                  # name -> {plane, glow}
        if "spark_colors" in cfg:
            comp.look.spark_colors = tuple(tuple(c) for c in cfg["spark_colors"])
        self.sites = {s["site"]: s for s in comp.meta["sites"]}
        L = comp.layers
        self.online = {}
        with open(os.path.join(d, "sensors.csv")) as f:
            for r in csv.DictReader(f):
                if r["site"] in self.sites and L.has("bea_" + r["site"]):
                    self.online[r["site"]] = float(r["online_s"])
        self.events = []
        with open(os.path.join(d, "events.csv")) as f:
            for uid, r in enumerate(csv.DictReader(f)):
                if r["site"] in self.sites and r["parameter"] in self.params:
                    self.events.append((r["site"], r["parameter"], float(r["time_s"]), min(max(float(r["value"]), 0.0), 1.0), uid))
        self.events.sort(key=lambda e: e[2])
        # sensors close together would double-light the terrain: scale each by the local sensor density (7.5 km kernel)
        self.site_scale = {}
        for n, s in self.sites.items():
            neigh = sum(math.exp(-((s["x"] - q["x"]) ** 2 + (s["z"] - q["z"]) ** 2) / (2.0 * 7.5 ** 2)) for q in self.sites.values())
            self.site_scale[n] = float((1.0 / (neigh ** 0.72)) * 1.42)
        self.keyframes = [(self.total * f, lab) for f, lab in ((0.15, "first sensors online"), (0.4, "samples begin"), (0.7, "network active"), (0.92, "finale"))]

    def frame_spec(self, t):
        L = self.c.layers
        spec = FrameSpec(t=t, seed=int(t * 1000) % (2 ** 31))
        spec.fade = min(1.0, t / 0.6) * (1.0 if t <= self.total - 1.5 else max(0.0, (self.total - t) / 1.5))
        spec.terms.append(Term("amb", (1.0, 1.0, 1.0)))
        spec.star_amount = 0.75 if L.has("stars") else 0.0
        scale = self.W / 640.0
        for n, t_on in self.online.items():                              # sensor beacons
            dt = t - t_on
            if dt < 0:
                continue
            ramp = 1 - math.exp(-dt / (0.28 * SLOW))
            flare = 1.0 + 1.8 * math.exp(-dt / (0.55 * SLOW))
            w_b = ramp * flare * 1.2 * (1.0 + 0.05 * math.sin(2 * math.pi * 0.4 * t + self.sites[n]["x"]))
            spec.beacon_terms.append(Term("bea_" + n, (w_b * 0.175 * self.site_scale[n],) * 3))
            s = self.sites[n]
            px, py = float(s["px"]), float(s["py"])
            if -12 * scale <= px <= self.W + 12 * scale and -12 * scale <= py <= self.H + 12 * scale:
                flare_mult = min(max(w_b / max(ramp, 1e-3), 0.85), 1.45)
                pulse = 0.70 * ramp * (0.95 + 0.05 * math.sin(2 * math.pi * 0.4 * t + s["x"])) * (0.82 + 0.18 * flare_mult)
                spec.beacons.append(Beacon(px, py, max(float(s["beam"]), 8.0 * scale), pulse))
        glow = {}
        for (n, par, t0, v, uid) in self.events:                          # sample fireworks
            dt = t - t0
            if not (-0.75 < dt < 6.6):
                continue
            s = self.sites[n]
            pc = self.params[par]
            m = 0.55 + 0.9 * v                                           # value -> size / brightness (0.55 .. 1.45)
            b = next((b for b in s["bursts"] if b["color"] == par), None) or s["bursts"][0]
            dens = min(max(self.site_scale.get(n, 1.0) ** 0.28, 0.74), 1.10)
            spec.bursts.append(Burst(uid, int(pc["plane"]), t0, (b["x"], b["y"], b["z"]), s["y"], max(48, int(round(128 * (0.55 + 0.45 * m)))),
                                     0.92 * dens * (0.48 + 0.58 * m), 5.6 * dens * (0.58 + 0.42 * m), 0.45 * SLOW, (1.05, 1.95), 0.44 * SLOW))
            if dt >= 0 and L.has(f"fwk_{par}_{n}"):
                w = (1 - math.exp(-dt / (0.035 * SLOW))) * math.exp(-dt / (0.24 * SLOW * PEAK)) * 0.72 * m
                glow.setdefault((par, n), 0.0)
                glow[(par, n)] += w
        for (par, n), w in glow.items():
            if w > 1e-3:
                wt = 0.056 * 0.65 * self.site_scale.get(n, 1.0)
                spec.terms.append(Term(f"fwk_{par}_{n}", tuple(float(g) * w * wt for g in self.params[par]["glow"])))
        return spec
