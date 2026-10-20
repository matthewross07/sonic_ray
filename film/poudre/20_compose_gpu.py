#!/usr/bin/env python3
"""
20_compose_gpu.py -- GPU compositor for the Cache la Poudre v2 film (drop-in for poudre_3d_v2/02_compose_video.py).

Same story, same layers, same maths -- but the per-pixel work (layer sums, satellite scan, halo, tone map, sharpen, bloom,
sky/terrain blend, firework overlay, beacons, dither, uint8 quantise) runs as Metal kernels with the layer files memory-mapped
straight into GPU buffers.  The CPU only does the sparse, branchy work, in parallel worker processes:
state()/timeline, firework spark drawing (PIL), and the sun-era sky.  The reference module is imported unchanged and reused for
all of that, so the two compositors cannot drift apart.

  python3 20_compose_gpu.py --layers LAYERS --out out.mp4 --fps 30 --crf 13 --preset slow --dither --sharpen 0.65 --procs 12
  python3 20_compose_gpu.py --layers LAYERS --compare 6.5 21.3 56 82 99 109 120.5      # GPU vs reference compositor, per-frame error
  python3 20_compose_gpu.py --layers LAYERS --bench                                      # ms/frame for a few story moments
"""
import argparse, ctypes, importlib.util, inspect, json, math, os, struct, subprocess, sys, textwrap, time
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "compositor"))
from gpucomp import GPU  # noqa: E402

DEFAULT_REF = ("/Users/mrvr/Library/CloudStorage/GoogleDrive-matthew.r.v.ross@gmail.com/My Drive/OpenCurrent/Presentations/"
               "In-Situ Pitch Meeting/visuals/poudre_3d_v2/02_compose_video.py")


def load_ref(path):
    spec = importlib.util.spec_from_file_location("poudre_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REF = None


class GpuScene:
    """Built lazily by make_scene() because it subclasses the reference Scene."""


def make_scene_class(ref):
    class _GpuScene(ref.Scene):
        def __init__(self, layer_dir, sharpen=0.0, dither=False):
            meta = json.load(open(os.path.join(layer_dir, "scene_meta.json")))
            self.W, self.H = meta["width"], meta["height"]
            W, H = self.W, self.H
            self.cam, self.fwk_h, self.sites = meta["cam"], meta["fwk_h"], meta["sites"]
            self.by = {s["site"]: s for s in self.sites}
            self.sharpen, self.dither = sharpen, dither
            self.g = g = GPU(W, H)
            path = lambda n: os.path.join(layer_dir, n + ".f32")
            mm = lambda n: g.mmap(path(n)) if os.path.exists(path(n)) else None
            cpu = lambda n, ch: np.memmap(path(n), dtype=np.float32, mode="r", shape=(H, W, ch)) if os.path.exists(path(n)) else None

            self.amb, self.riv = mm("amb"), mm("riv")
            mask = np.array(cpu("mask", 1)[..., 0])
            self.mask = mask
            self.m_eff = np.clip(mask, 0.0, 1.0).astype(np.float32)
            self.h_meff = g.alloc1(); g.write(self.h_meff, self.m_eff)
            self.fwk = {(s["site"], b["color"]): mm(f"fwk_{b['color']}_{s['site']}") for s in self.sites for b in s["bursts"]}
            self.bea = {s["site"]: mm("bea_" + s["site"]) for s in self.sites}
            self.sunp = meta["sun_path"]
            self.sun = [mm("sun_%d" % k) for k in range(len(self.sunp))]
            self.sky_layers = [cpu("sky_%d" % k, 3) for k in range(len(self.sunp))]   # CPU sky() reads these (sun era only)
            self.stars_layer = cpu("stars", 3)
            self.h_stars = mm("stars")
            self.cells = {c["id"]: c for c in meta["cells"]}
            self.fwk_cell = {cid: mm("fwk_cell_" + cid) for cid in self.cells}
            gam = lambda c: np.power(np.clip(c, 0, 1), 1 / 2.2)
            U, V = cpu("uv_u", 3), cpu("uv_v", 3)
            self.uu = self.vv = None
            if U is not None and V is not None:
                self.h_uu, self.h_vv = g.alloc1(), g.alloc1()
                g.write(self.h_uu, (gam(U[..., 0]) + gam(U[..., 1]) / 255.0).astype(np.float32))
                g.write(self.h_vv, (gam(V[..., 0]) + gam(V[..., 1]) / 255.0).astype(np.float32))
                self.uu = True
            self.lh = [s["site"] for s in self.sites if s["lighthouse"]]
            self.rest = [s["site"] for s in sorted(self.sites, key=lambda s: -s["z"]) if not s["lighthouse"]]
            self.all = [s["site"] for s in sorted(self.sites, key=lambda s: -s["z"])]
            self.site_scale = {}
            for s in self.sites:
                d2 = [((s["x"] - q["x"]) ** 2 + (s["z"] - q["z"]) ** 2) for q in self.sites]
                neigh = sum(math.exp(-dd / (2.0 * 7.5 ** 2)) for dd in d2)
                self.site_scale[s["site"]] = float((1.0 / (neigh ** 0.72)) * 1.42)
            self.cell_events = self._cell_schedule()
            yy, xx = np.mgrid[0:H, 0:W]
            self.yy, self.xx = yy.astype(np.float32), xx.astype(np.float32)
            ys_mask = np.where(mask.sum(1) > 0)[0]
            self.y_top = float(ys_mask.min()) if len(ys_mask) else 0.31 * H
            self.y_bot = float(ys_mask.max()) if len(ys_mask) else float(H - 1)
            far = np.clip(1.0 - (self.yy - self.y_top) / max(self.y_bot - self.y_top, 1.0), 0.0, 1.0) ** 2.1
            self.h_aerial = g.alloc1(); g.write(self.h_aerial, (far * self.m_eff).astype(np.float32))
            self.ox0, self.ox1 = 0.36 * W, 0.54 * W
            col = mask[:, int(self.ox1)]
            ys = np.where(col > 0.5)[0]
            self.y_edge = float(ys.min()) if len(ys) else 0.32 * H
            self.orb_r = 0.015 * H
            self.orb_y_top = -0.03 * H
            self.orb_y_set = self.y_edge + 0.018 * H
            self.sun_mins = [q["min"] for q in self.sunp]
            self.sun_els = [q["el"] for q in self.sunp]
            # working buffers
            a3, a1 = g.alloc3, g.alloc1
            self.lin, self.bl, self.terrain, self.tblur, self.bright = a3(), a3(), a3(), a3(), a3()
            self.b1, self.b2, self.b3 = a3(), a3(), a3()
            self.ov, self.ob1, self.ob2 = a3(), a3(), a3()
            self.sky_buf, self.orb_core, self.orb_glow = a3(), a3(), a3()
            self.lum, self.g1, self.g2 = a1(), a1(), a1()
            self.out = g.alloc_bytes(W * H * 3)
            self.planes = [g.alloc_bytes(4 * W * H) for _ in range(5)]
            self.bmax, self.omax = g.alloc_bytes(16), g.alloc_bytes(16)

        # ------------------------------------------------------------------------------------------ one frame
        def render(self, t):
            g, W, H = self.g, self.W, self.H
            smooth = ref.smooth
            st = self.state(t)
            p = st["sun_p"]
            dusk = smooth((t - ref.SUN_END) / max(ref.HOLD_END - ref.SUN_END, 1.0)) if t > ref.SUN_END else 0.0
            amb_lvl = 0.88 * smooth(p * 1.8) * (1.0 - 0.42 * smooth(max(0.0, (p - 0.75) / 0.25)) - 0.25 * dusk)
            terms = [(self.amb, 0, (amb_lvl,) * 3)]
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
                if a_ is not None or b_ is not None:
                    sun_gain = smooth(p * 6.0) * (0.66 - 0.14 * sunset_w) * (1.0 - 0.55 * dusk)
                    if a_ is not None: terms.append((a_, 0, tuple(warm * (sun_gain * (1 - f)))))
                    if b_ is not None: terms.append((b_, 0, tuple(warm * (sun_gain * f))))
                terms.append((self.riv, 1, tuple(np.array([0.04, 0.30, 0.65]) * (smooth(p * 2.5) * (1.0 - 0.35 * dusk)))))
            for (n, c), w in st["w_fwk"].items():
                if w > 1e-3:
                    h = self.fwk_cell[n[5:]] if n.startswith("cell:") else self.fwk[(n, c)]
                    if h is not None:
                        wt = ref.CELL_WEIGHT if n.startswith("cell:") else (ref.CAND_WEIGHT * 0.65 * self.site_scale.get(n, 1.0))
                        terms.append((h, 0, tuple(ref.BCOL[c] * (w * wt))))
            bea_terms = [(self.bea[n], 0, (w * 0.175 * self.site_scale.get(n, 1.0),) * 3)
                         for n, w in st["w_bea"].items() if self.bea[n] is not None]
            sat_ev = []
            if self.uu is not None:
                for kk, ts in enumerate(ref.SAT_TIMES):
                    if ts - 0.2 < t < ts + ref.SAT_DUR + 7 * ref.SAT_TAU:
                        sat_ev.append((ts, ref.SAT_CENTERS[kk % len(ref.SAT_CENTERS)]))
            s = W / 480.0

            # ---- CPU: sun-era sky and firework spark planes (written straight into shared buffers, before the GPU runs)
            has_orb = use_atm = 0
            atm, hor = 0.0, (0.45, 0.55, 0.72)
            if p > 0:
                sky, orb_core, orb_glow, hor_col = self.sky(p, t, dusk)
                g.write(self.sky_buf, sky)
                if orb_core is not orb_glow or orb_core.any():
                    g.write(self.orb_core, orb_core); g.write(self.orb_glow, orb_glow); has_orb = 1
                atm = (0.05 + 0.03 * smooth(max(0.0, (p - 0.70) / 0.30))) * smooth(p * 2.2)
                use_atm, hor = 1, tuple(float(x) for x in hor_col)
            ov_active = [0] * 5
            if st["fireworks"]:
                L = self.overlay_planes(t, st)
                for kidx, key in enumerate(("toc", "chla", "alk", "shimmer", "white")):
                    im = L[key]
                    if im.getbbox() is not None:
                        data = im.tobytes()
                        ctypes.memmove(g.ptr(self.planes[kidx]), data, len(data))
                        ov_active[kidx] = 1
            use_ov = 1 if any(ov_active) else 0
            beacons = []
            scale = W / 640.0
            for n, on_val in st["on"].items():
                if on_val <= 1e-3:
                    continue
                si = self.by[n]
                px, py = float(si["px"]), float(si["py"])
                bh = max(float(si["beam"]), 8.0 * scale)
                if px < -12 * scale or px > W + 12 * scale or py < -12 * scale or py > H + 12 * scale:
                    continue
                w_b = float(st["w_bea"].get(n, on_val))
                flare = min(max(w_b / max(on_val, 1e-3), 0.85), 1.45)
                pulse = 0.70 * float(on_val) * (0.95 + 0.05 * math.sin(2 * math.pi * 0.4 * t + si["x"])) * (0.82 + 0.18 * flare)
                beacons += [px, py, bh, pulse]
            nb = min(len(beacons) // 4, 24)

            g.zero(self.bmax, 16); g.zero(self.omax, 16)
            d2 = struct.pack("2i", W, H)
            g.begin()
            g.axpy(self.lin, terms, init=True)
            if sat_ev:
                sev = sat_ev[:4]
                ts = [e[0] for e in sev] + [0.0] * (4 - len(sev)); uc = [e[1] for e in sev] + [0.0] * (4 - len(sev))
                has_s1 = 1 if (len(self.sun) > 1 and self.sun[1] is not None) else 0
                g.op("sat_add", [self.lin, self.amb, self.sun[1] if has_s1 else self.amb, self.h_uu, self.h_vv, self.riv],
                     struct.pack("4i4f4f4f3f", W, H, len(sev), has_s1, t, ref.SAT_HALF,
                                 ref.SAT_DUR, ref.SAT_TAU, *ts, *uc, *[float(x) for x in ref.SAT]))
            if bea_terms:
                g.axpy(self.bl, bea_terms, init=True)
                g.axpy(self.lin, [(self.bl, 0, tuple(float(x) for x in ref.BLUE))], init=False)
                g.op("lum_mean", [self.bl, self.lum], d2)
                g.blur(self.lum, self.g1, 12 * s, 1)
                g.blur(self.lum, self.g2, 30 * s, 1)
                g.op("halo_add", [self.lin, self.g1, self.g2, self.riv],
                     struct.pack("2i7f", W, H, 7.5, 14.0, 1.15, 0.45, 0.12 * 0.476, 0.44 * 0.476, 0.92 * 0.476))
            g.op("tonemap", [self.lin, self.h_meff, self.terrain, self.bright, self.bmax], d2)
            if self.sharpen > 0:
                g.blur(self.terrain, self.tblur, max(1.2 * W / 1920.0, 0.5), 3)
                g.op("sharpen_comb", [self.terrain, self.tblur], struct.pack("2if", W, H, self.sharpen))
            g.blur(self.bright, self.b1, 2.5 * s, 3)
            g.blur(self.bright, self.b2, 8.5 * s, 3)
            g.blur(self.bright, self.b3, 22 * s, 3)
            if p <= 0:
                g.op("sky_stars", [self.h_stars if self.h_stars is not None else self.amb, self.sky_buf],
                     struct.pack("2i2f", W, H, t, 0.75 if self.h_stars is not None else 0.0))
            if use_ov:
                cols = [ref.SCOL["toc"], ref.SCOL["chla"], ref.SCOL["alk"], np.array([0.56, 0.46, 1.00]), ref.WHITE]
                g.op("ov_from_planes", self.planes + [self.ov, self.omax],
                     struct.pack("2i5i15f", W, H, *ov_active, *[float(x) for c in cols for x in c]))
                g.blur(self.ov, self.ob1, 0.85 * s, 3)
                g.blur(self.ov, self.ob2, 2.6 * s, 3)
            fade = st["fade"]
            seed = int(t * 1000) % (2 ** 31)
            bc = beacons[:96] + [0.0] * (96 - len(beacons[:96]))
            g.op("final_frame", [self.terrain, self.h_meff, self.sky_buf, self.orb_core, self.orb_glow, self.h_aerial,
                                 self.b1, self.b2, self.b3, self.ov, self.ob1, self.ob2, self.bmax, self.omax, self.out],
                 struct.pack("6iIi6f96f", W, H, has_orb, use_atm, use_ov, 1 if self.dither else 0, seed, nb,
                             atm, *hor, fade, scale, *bc))
            g.end()
            return g.read_bytes(self.out, W * H * 3)

    # overlay(): reuse the reference spark drawing verbatim, but stop before its CPU colourise/downsample tail --
    # the five 8-bit planes go to the GPU instead.
    src = textwrap.dedent(inspect.getsource(ref.Scene.overlay))
    cut = src.index("    out = np.zeros((self.H, self.W, 3), np.float32)")
    src = src[:cut] + "    return L\n"
    src = src.replace("def overlay(", "def overlay_planes(", 1)
    exec(src, ref.__dict__)
    _GpuScene.overlay_planes = ref.__dict__["overlay_planes"]
    return _GpuScene


# ---------------------------------------------------------------------------------------------- multiprocessing workers
_SC = None
_FPS = 30


def _init_worker(layers, ref_path, sharpen, dither, fps):
    global _SC, _FPS, REF
    REF = load_ref(ref_path)
    _SC = make_scene_class(REF)(layers, sharpen, dither)
    _FPS = fps


def _render(i):
    return _SC.render(i / _FPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--ref", default=DEFAULT_REF, help="path to poudre_3d_v2/02_compose_video.py (reused for timeline/sky/sparks)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--crf", type=int, default=13)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--dither", action="store_true")
    ap.add_argument("--sharpen", type=float, default=0.0)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--stills", nargs="*", type=float)
    ap.add_argument("--stills-dir", default=".")
    ap.add_argument("--compare", nargs="*", type=float)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()

    if a.compare is not None or a.bench or a.stills is not None:
        ref = load_ref(a.ref)
        Cls = make_scene_class(ref)
        t_load = time.time(); sc = Cls(a.layers, a.sharpen, a.dither); print(f"GPU scene ready in {time.time() - t_load:.1f} s on {sc.g.name}")
        if a.stills is not None:
            os.makedirs(a.stills_dir, exist_ok=True)
            for t in a.stills:
                Image.frombytes("RGB", (sc.W, sc.H), sc.render(t)).save(os.path.join(a.stills_dir, f"gpu_still_{t:06.2f}.png"))
            return
        if a.bench:
            for t in (1.0, 6.5, 17.5, 21.3, 38.0, 56.0, 82.0, 99.0, 109.0, 120.5):
                sc.render(t); n = 5; t1 = time.time()
                for _ in range(n): sc.render(t)
                print(f"  t={t:6.1f}s  {1000 * (time.time() - t1) / n:7.1f} ms/frame")
            return
        ref.SHARPEN, ref.DITHER = a.sharpen, False
        cpu = ref.Scene(a.layers)
        sc.dither = False
        os.makedirs(a.stills_dir, exist_ok=True)
        for t in a.compare:
            im_c = np.asarray(cpu.frame(t).convert("RGB"), np.int16)
            im_g = np.frombuffer(sc.render(t), np.uint8).reshape(sc.H, sc.W, 3).astype(np.int16)
            d = np.abs(im_c - im_g)
            print(f"  t={t:6.1f}s  mean|diff|={d.mean():.3f}/255  p99.9={np.percentile(d, 99.9):.0f}  max={d.max()}  "
                  f">8 levels: {100 * (d.max(-1) > 8).mean():.3f}% of pixels")
            Image.fromarray(np.concatenate([im_c, im_g, np.clip(d * 8, 0, 255)], 1).astype(np.uint8)).resize((sc.W, sc.H // 3)).save(
                os.path.join(a.stills_dir, f"compare_{t:06.2f}.png"))
        return

    # ---- full render: workers each own a Metal context; ordered frames stream into ffmpeg
    import multiprocessing as mp
    meta = json.load(open(os.path.join(a.layers, "scene_meta.json")))
    W, H = meta["width"], meta["height"]
    ref = load_ref(a.ref)
    total = ref.TOTAL
    f0 = int(round(a.t0 * a.fps)); f1 = int(round((a.t1 if a.t1 is not None else total) * a.fps)); nfr = f1 - f0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps),
                           "-i", "-", "-c:v", "libx264", "-preset", a.preset, "-crf", str(a.crf), "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    t_start = time.time()
    with mp.get_context("spawn").Pool(a.procs, initializer=_init_worker, initargs=(a.layers, a.ref, a.sharpen, a.dither, a.fps)) as pool:
        for i, buf in enumerate(pool.imap(_render, range(f0, f1), chunksize=2)):
            ff.stdin.write(buf)
            if i % 120 == 0:
                print(f"frame {i}/{nfr}  ({time.time() - t_start:.0f} s)", flush=True)
    ff.stdin.close(); ff.wait()
    print(f"wrote {a.out} in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
