#!/usr/bin/env python3
"""Parity check: sonicray GPU compositor vs the original CPU compositor (visuals/poudre_3d_v2/02_compose_video.py) on the same layers.

  python3 tools/compare_reference.py --layers LAYERS --ref /path/to/02_compose_video.py --times 21.3 56 82 99 109
Prints mean/p99.9/max absolute difference in 8-bit levels per frame, and writes side-by-side PNGs (reference | GPU | 8x difference).
Dithering is off on both sides so the numbers measure the pipeline, not the random generator.
"""
import argparse, importlib.util, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inst", "python"))
from sonicray.cli import _make  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layers", required=True); ap.add_argument("--ref", required=True)
ap.add_argument("--times", type=float, nargs="+", required=True); ap.add_argument("--sharpen", type=float, default=0.65)
ap.add_argument("--out-dir", default="/tmp/sonicray_compare")
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("poudre_ref", a.ref); ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
ref.SHARPEN, ref.DITHER = a.sharpen, False
cpu = ref.Scene(a.layers)
c, story = _make(a.layers, "poudre", a.sharpen, False)
os.makedirs(a.out_dir, exist_ok=True)
for t in a.times:
    im_c = np.asarray(cpu.frame(t).convert("RGB"), np.int16)
    im_g = np.frombuffer(c.render(story.frame_spec(t)), np.uint8).reshape(c.H, c.W, 3).astype(np.int16)
    d = np.abs(im_c - im_g)
    print(f"  t={t:6.1f}s  mean|diff|={d.mean():.3f}/255  p99.9={np.percentile(d, 99.9):.0f}  max={d.max()}  >8 levels: {100 * (d.max(-1) > 8).mean():.3f}% of pixels")
    Image.fromarray(np.concatenate([im_c, im_g, np.clip(d * 8, 0, 255)], 1).astype(np.uint8)).resize((c.W, c.H // 3)).save(os.path.join(a.out_dir, f"compare_{t:06.2f}.png"))
