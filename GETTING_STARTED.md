# Getting started with `sonicray`

Everything you need to build it, test it, and understand how it speeds up the rayshader/rayrender workflow with Metal and OIDN.
(Credit first: this project is an attempt to follow the work of **Tyler Morgan-Wall** — see [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).)

## 1. Exactly what to install

Requirements: a Mac with **Apple Silicon** and **macOS 14+** (tested on macOS 15.7, Apple M4 Pro).

| What | How | Needed for | Notes |
|---|---|---|---|
| Xcode **Command Line Tools** | `xcode-select --install` | compiling the Obj-C++/Metal bridge | **Full Xcode.app is NOT required**: shaders compile at runtime inside the OS |
| **R ≥ 4.3** | https://cran.r-project.org (tested 4.5.2) | everything | |
| **Homebrew** | https://brew.sh | the denoiser | |
| **Intel Open Image Denoise** | `brew install open-image-denoise` | OIDN denoising (2.5.1; Metal GPU backend) | pulls `hwloc` and `tbb`; `cmake`/`ninja` are **not** needed (pre-built bottle) |
| R packages | `install.packages(c("Rcpp", "terra", "sf", "jsonlite", "png"))` | the package | `devtools`/`pkgbuild` optional |
| `ffmpeg` | `brew install ffmpeg` | encoding films | |
| Python 3 + `numpy`, `Pillow`, `scipy` | `pip3 install numpy pillow scipy` | the film compositor (`poudre_3d_v2/02_compose_video.py`) | |
| `rayrender` (+ `rayvertex`) | `install.packages("rayrender")` | **only** for parity tests against the CPU renderer | not needed to render on the GPU |
| Sonic Pi 5 | https://sonic-pi.net | film soundtracks | optional |

Optional later: **Xcode.app** (≈ 35 GB) gives the Metal GPU debugger/frame capture and the offline shader compiler; useful when optimising kernels.

## 2. Build and verify

```bash
git clone git@github.com:matthewross07/sonic_ray.git && cd sonic_ray
R CMD INSTALL .                               # → compiles src/*.cpp as Objective-C++ (-ObjC++) against Metal.framework
Rscript -e 'library(sonicray); str(sr_device_info())'
Rscript tools/test_plane_light.R              # GPU vs analytic vs rayrender (needs rayrender)
Rscript tools/test_denoise.R                  # OIDN error reduction
```
`sr_device_info()` should report your GPU, `supports_raytracing = TRUE`, `unified_memory = TRUE`, and `oidn_available = TRUE` after the brew step.
If `oidn_available` is `FALSE` after installing OIDN, rebuild (`R CMD INSTALL .`): `src/Makevars` checks that the OIDN header really exists.

## 3. How `sonicray` speeds up rayshader / rayrender (the ideas that matter)

1. **A persistent hardware BVH.** rayrender rebuilds its CPU acceleration structure and reloads textures on *every* `render_scene()` call
   (≈ 2–3 s each on a 2–4 M-triangle terrain). `sonicray` uploads the mesh once (`sr_upload_mesh`, ≈ 1.5 s) into a Metal
   `MTLAccelerationStructure` that lives in unified memory for hundreds of renders.
2. **Metal hardware ray tracing** (`metal::raytracing::intersector`) for primary, shadow and bounce rays on the GPU.
3. **Many lights, one path.** Each pixel's camera/bounce path is traced once and *all* lights in a batch (up to 32) get next-event estimation
   along it — exact RGB per light, no colour multiplexing tricks (rayrender needed 3 lights per render via R/G/B emitters, with 5–10 % bias).
4. **Exact sphere-light irradiance.** `E = π L sin²α cosθ` (with horizon clipping, after Lagarde & de Rousiers) × a stochastic
   visibility ray — noise-free except in penumbrae; this matched the closed form on a plane to 0.1 %.
5. **First-hit AOVs + OIDN.** The ambient pass also emits albedo/normal/uv/mask; OIDN uses albedo+normal to denoise cheap low-spp layers.
6. **Same file format as the CPU pipelines** (`*.f32`, row-major RGB), so existing compositors keep working.

## 4. Measured results (Apple M4 Pro; Cache la Poudre v2, 60 m mesh = 3.8 M triangles, 1920×1080)

| Task | CPU (rayrender) | GPU (sonicray) |
|---|---|---|
| Upload scene / build acceleration structure | ≈ 2.4 s **per render call** | 1.3–1.6 s **once** |
| 3 light layers, 256 vs 160 spp | 72 s (160 spp adaptive) | 0.45 s (256 spp) |
| Ambient layer @ 64 spp | — | 0.10 s |
| All 162 layer files of the film | **39.3 min** (file timestamps) | **64.5 s** (512 spp lights, 1024 spp ambient) |

The CPU time is the wall-clock span of the reference layers' file timestamps and includes SS=2 supersampling; the GPU numbers use pixel-jittered
AA instead. The compositor (Python) is **not yet GPU-accelerated**, so end-to-end speed-up is limited by it — see the roadmap.

## 5. Known gaps (be skeptical of these until fixed)

- Light layers differ from rayrender along shadow edges and in a mild foreground band (median pixel ratio ≈ 1.0 ± 0.1; 8.7 % of lit pixels differ by > 2× in
  the burst test, mostly thin filaments at shadow terminators). Cause not fully understood — possibly shading- vs geometric-normal handling.
- Indirect (bounce) light adds only ~1–2 % here; rayrender's shadows are visibly filled in. Needs investigation.
- Textures are uploaded as 32-bit float from R doubles (≈ 0.7 GB for 4608×4856): works on 48 GB, wasteful. An 8-bit upload path is a TODO.
- `sr_timeline`, `sr_seasonal_curve`, `sr_write_sonic_pi`, `sr_compose_video` are skeletons; the GPU compositor kernel has no host code.
- Single device (the default GPU), no multi-GPU; sphere lights only (no area/mesh lights, no volumes, no glossy materials).

## 6. Roadmap

1. **GPU compositor** (largest remaining cost): weighted layer sum + bloom + tone map + sharpen in Metal, layers resident in unified memory.
2. Close the remaining parity gaps (shadow terminator handling, bounce fill) and add a parity test to CI.
3. 8-bit texture upload; reuse the OIDN device across layers.
4. Port the fireworks/satellite overlay to the GPU or vectorise it.
5. A `backend = "metal"` hook for rayrender/rayshader — *to be offered to Tyler Morgan-Wall, on his terms* (see rayshader issue #328).
