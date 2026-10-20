"""Command line:  python3 -m sonicray <doctor|selftest|compose|still|bench> ...   (or the `sonicray` wrapper script in inst/bin).

Every command prints plain, grep-able lines; `doctor --json` prints one JSON object so an AI assistant can read the machine state.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from . import __version__

_C = None          # per-process compositor/story (workers)
_STORY = None
_FPS = 30


# --------------------------------------------------------------------------------------------------------- doctor / selftest
def _check(name, ok, detail="", fix=""):
    return {"check": name, "ok": bool(ok), "detail": detail, "fix": fix}


def run_doctor():
    out = []
    mac = sys.platform == "darwin"
    out.append(_check("macOS", mac, platform.platform(), "sonicray's GPU path needs a Mac (Apple Silicon recommended)."))
    arm = platform.machine() == "arm64"
    out.append(_check("Apple Silicon", arm, platform.machine(), "Intel Macs may work for the compositor but are untested; ray tracing needs Apple Silicon."))
    out.append(_check("python >= 3.9", sys.version_info >= (3, 9), sys.version.split()[0], "brew install python"))
    try:
        import numpy
        out.append(_check("numpy", True, numpy.__version__))
    except Exception:
        out.append(_check("numpy", False, "", "python3 -m pip install numpy"))
    try:
        import PIL
        out.append(_check("Pillow", True, PIL.__version__))
    except Exception:
        out.append(_check("Pillow", False, "", "python3 -m pip install pillow"))
    ff = shutil.which("ffmpeg")
    out.append(_check("ffmpeg", ff, ff or "", "brew install ffmpeg"))
    cx = shutil.which("clang++")
    out.append(_check("clang++ (Xcode Command Line Tools)", cx, cx or "", "xcode-select --install"))
    gpu_ok, detail, fix = False, "", ""
    try:
        from .gpu import build_native
        lib = build_native()
        out.append(_check("native Metal bridge builds", True, lib))
        gpu_ok, detail = _selftest_frame()
    except Exception as e:                                         # noqa: BLE001
        detail, fix = str(e), "run `sonicray doctor` again after fixing the items above"
    out.append(_check("GPU compositor selftest", gpu_ok, detail, fix))
    return out


def _selftest_frame():
    """Compose one tiny synthetic frame (terrain + a firework + a beacon) on the GPU and check it is neither blank nor saturated."""
    from .compositor import GpuCompositor, Look
    from .framespec import Beacon, Burst, FrameSpec, PLANE_TOC, Term
    W, H = 256, 144
    with tempfile.TemporaryDirectory() as d:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        mask = (yy > 0.45 * H).astype(np.float32)
        rgb = lambda v: np.repeat((v * 0.5)[..., None], 3, -1).astype(np.float32)
        rgb(np.ones((H, W))).tofile(os.path.join(d, "amb.f32"))
        mask.tofile(os.path.join(d, "mask.f32"))
        np.zeros((H, W, 3), np.float32).tofile(os.path.join(d, "riv.f32"))
        rgb(np.exp(-((xx - 128) ** 2 + (yy - 100) ** 2) / 800.0) * 4).tofile(os.path.join(d, "fwk_a.f32"))
        json.dump({"width": W, "height": H, "cam": {"lookfrom": [0, 15.8, 70], "lookat": [0, 1.5, -2], "fov": 22.8}, "sites": [], "cells": [],
                   "sun_path": []}, open(os.path.join(d, "scene_meta.json"), "w"))
        c = GpuCompositor(d, Look(dither=True))
        spec = FrameSpec(t=1.0, seed=7)
        spec.terms = [Term("amb", (0.3, 0.3, 0.3)), Term("fwk_a", (1.0, 0.4, 0.2))]
        spec.bursts = [Burst(1, PLANE_TOC, 0.5, (0, 8, 0), 1.0, 48, 0.9, 5.0, 0.45, (1.05, 1.95), 0.66)]
        spec.beacons = [Beacon(128, 100, 20, 0.8)]
        img = np.frombuffer(c.render(spec), np.uint8).reshape(H, W, 3)
        mean, mx = float(img.mean()), int(img.max())
        ok = 3.0 < mean < 200.0 and mx > 100
        return ok, f"{c.g.name}; 256x144 test frame mean={mean:.1f} max={mx}" + ("" if ok else "  (frame looks blank or saturated)")


def cmd_doctor(a):
    res = run_doctor()
    if a.json:
        print(json.dumps({"sonicray": __version__, "all_ok": all(r["ok"] for r in res), "checks": res}, indent=2))
    else:
        for r in res:
            print(("  ok   " if r["ok"] else "  FAIL ") + r["check"] + (f"   [{r['detail']}]" if r["detail"] else "") + (f"\n         fix: {r['fix']}" if not r["ok"] and r["fix"] else ""))
        print("sonicray " + __version__ + ": " + ("everything needed for GPU compositing is in place" if all(r["ok"] for r in res) else "fix the FAIL items above"))
    return 0 if all(r["ok"] for r in res) else 1


# --------------------------------------------------------------------------------------------------------- rendering
def _make(layers, story, sharpen, dither):
    sharpen = 0.65 if sharpen is None else sharpen
    from .compositor import GpuCompositor, Look
    from .films import STORIES
    if story not in STORIES:
        raise SystemExit(f"unknown story '{story}'. Available: {', '.join(STORIES)}")
    c = GpuCompositor(layers, Look(sharpen=sharpen, dither=dither))
    return c, STORIES[story](c)


def _init_worker(layers, story, sharpen, dither, fps):
    global _C, _STORY, _FPS
    _C, _STORY = _make(layers, story, sharpen, dither)
    _FPS = fps


def _render(i):
    return _C.render(_STORY.frame_spec(i / _FPS))


def _apply_quality(a):
    from .intake import PRESETS
    if getattr(a, "quality", None):
        pr = PRESETS[a.quality]
        if a.fps is None: a.fps = pr["fps"]
        if a.crf is None: a.crf = pr["crf"]
        if a.preset is None: a.preset = pr["encode_preset"]
        if a.sharpen is None: a.sharpen = pr["sharpen"]
        if not a.dither: a.dither = pr["dither"]
    a.fps = a.fps or 30; a.crf = 13 if a.crf is None else a.crf; a.preset = a.preset or "slow"; a.sharpen = 0.65 if a.sharpen is None else a.sharpen
    return a


def cmd_compose(a):
    import multiprocessing as mp
    a = _apply_quality(a)
    meta = json.load(open(os.path.join(a.layers, "scene_meta.json")))
    W, H = meta["width"], meta["height"]
    from .films import STORIES
    total = getattr(STORIES[a.story], "total", None)
    if total is None:                                              # data-driven stories keep their length in story.json
        total = float(json.load(open(os.path.join(a.layers, "story.json")))["duration_s"])
    f0 = int(round(a.t0 * a.fps)); f1 = int(round((a.t1 if a.t1 is not None else total) * a.fps)); nfr = f1 - f0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    silent = a.out if not a.audio else a.out + ".silent.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-",
                           "-c:v", "libx264", "-preset", a.preset, "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", silent],
                          stdin=subprocess.PIPE)
    t_start = time.time()
    # spawn, never fork: forked children that touch Metal crash
    with mp.get_context("spawn").Pool(a.procs, initializer=_init_worker, initargs=(a.layers, a.story, a.sharpen, a.dither, a.fps)) as pool:
        for i, buf in enumerate(pool.imap(_render, range(f0, f1), chunksize=2)):
            ff.stdin.write(buf)
            if i % 120 == 0:
                print(f"frame {i}/{nfr}  ({time.time() - t_start:.0f} s)", flush=True)
    ff.stdin.close(); ff.wait()
    if a.audio:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-ss", str(a.audio_offset), "-i", a.audio, "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-t", str((f1 - f0) / a.fps), "-movflags", "+faststart", a.out], check=True)
        os.remove(silent)
    print(f"wrote {a.out} in {time.time() - t_start:.0f} s")


def cmd_still(a):
    from PIL import Image
    c, story = _make(a.layers, a.story, a.sharpen, a.dither)
    os.makedirs(a.out_dir, exist_ok=True)
    for t in a.times:
        p = os.path.join(a.out_dir, f"{a.story}_{t:07.2f}.png")
        Image.frombytes("RGB", (c.W, c.H), c.render(story.frame_spec(t))).save(p)
        print(p)


def cmd_sheet(a):
    """Quick sheet: the story's keyframes as one contact sheet (the first thing to look at after any design change)."""
    from PIL import Image
    c, story = _make(a.layers, a.story, a.sharpen, a.dither)
    tiles = [Image.frombytes("RGB", (c.W, c.H), c.render(story.frame_spec(t))).resize((a.tile, int(a.tile * c.H / c.W))) for t, _ in story.keyframes]
    cols = 3; rows = -(-len(tiles) // cols); th = tiles[0].height
    sheet = Image.new("RGB", (cols * a.tile + (cols + 1) * 8, rows * th + (rows + 1) * 8), (18, 22, 32))
    for i, im in enumerate(tiles):
        sheet.paste(im, (8 + (i % cols) * (a.tile + 8), 8 + (i // cols) * (th + 8)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    sheet.save(a.out)
    print(a.out)


def cmd_bench(a):
    c, story = _make(a.layers, a.story, a.sharpen, a.dither)
    for t, label in story.keyframes:
        spec = story.frame_spec(t)
        c.render(spec)
        t1 = time.time()
        for _ in range(5):
            c.render(story.frame_spec(t))
        print(f"  t={t:6.1f}s  {1000 * (time.time() - t1) / 5:7.1f} ms/frame   {label}")


def cmd_intake(a):
    from . import intake
    if a.json:
        print(json.dumps({"questions": intake.QUESTIONS, "presets": intake.PRESETS}, indent=2))
    else:
        print(intake.as_text())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sonicray", description="GPU compositing for light-layer films (Apple Silicon / Metal).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("doctor", help="check this Mac and print exact fixes"); s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_doctor)

    def common(p):
        p.add_argument("--layers", required=True, help="directory of *.f32 light layers + scene_meta.json")
        p.add_argument("--story", default="poudre")
        p.add_argument("--sharpen", type=float, default=None, help="unsharp amount (default 0.65, or from --quality)")
        p.add_argument("--dither", action="store_true")

    s = sub.add_parser("intake", help="the questions to ask the user before rendering (defaults + costs)"); s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_intake)
    s = sub.add_parser("compose", help="render a film"); common(s)
    s.add_argument("--out", required=True); s.add_argument("--fps", type=int, default=None); s.add_argument("--crf", type=int, default=None)
    s.add_argument("--quality", choices=["draft", "review", "final"], default=None, help="preset for fps/crf/x264 preset/sharpen/dither (flags override)")
    s.add_argument("--preset", default=None); s.add_argument("--t0", type=float, default=0.0); s.add_argument("--t1", type=float, default=None)
    s.add_argument("--procs", type=int, default=max(1, min(10, (os.cpu_count() or 4) - 4)))
    s.add_argument("--audio", default=None); s.add_argument("--audio-offset", type=float, default=0.0)
    s.set_defaults(fn=cmd_compose)
    s = sub.add_parser("still", help="render single frames as PNG"); common(s)
    s.add_argument("--times", type=float, nargs="+", required=True); s.add_argument("--out-dir", default="."); s.set_defaults(fn=cmd_still)
    s = sub.add_parser("sheet", help="contact sheet of the story's keyframes"); common(s)
    s.add_argument("--out", default="quick_sheet.png"); s.add_argument("--tile", type=int, default=640); s.set_defaults(fn=cmd_sheet)
    s = sub.add_parser("bench", help="ms per frame at the story's keyframes"); common(s); s.set_defaults(fn=cmd_bench)
    a = ap.parse_args(argv)
    sys.exit(a.fn(a) or 0)
