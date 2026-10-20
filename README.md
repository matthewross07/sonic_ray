# `sonicray` — Metal (Apple GPU) path tracing for R, for the rayverse-style light-layer workflow

> ### Inspired by — and an attempt to follow in the footsteps of — Tyler Morgan-Wall
>
> **`sonicray` exists because of [Tyler Morgan-Wall](https://github.com/tylermorganwall)'s work.** His R packages
> [**rayshader**](https://github.com/tylermorganwall/rayshader), [**rayrender**](https://github.com/tylermorganwall/rayrender)
> ([docs](https://www.rayrender.net/)) and **rayvertex** are what made beautiful 3D, path-traced maps and films possible from R, and every idea
> here (light layers, path-traced terrain, lights as first-class objects, the pipe-friendly workflow) comes from using and admiring them.
> This project is an **attempt to bring that same capability to Apple Silicon GPUs (Metal)** — to get the same beautiful results far faster —
> and the author is *super inspired* by all he has done.
>
> **What this is not:** it is **not** affiliated with or endorsed by Tyler Morgan-Wall, and it is **not** a copy of his code. `sonicray` is an
> independent, from-scratch implementation (new Metal shaders and a new R/C++ bridge) that aims to **reproduce the capabilities and the
> workflow** of rayrender/rayshader; rayrender is licensed GPL-3 and none of its source is included here. If you want the real thing — the
> full-featured, battle-tested CPU renderer — use **rayrender and rayshader**. See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).

Repo: `matthewross07/sonic_ray` · R package: **`sonicray`** (R package names cannot contain `_`) · License: MIT · macOS / Apple Silicon only.

## What it does today

Renders **light layers** — one path-traced image per light — for large terrain scenes on the Apple GPU using Metal 3 hardware ray tracing,
and (optionally) denoises them with Intel Open Image Denoise. Layers are written as raw `float32` images that a compositor sums:
`frame(t) = Σ weight_i(t) · colour_i · layer_i` (light is linear). Our CPU/rayrender film pipelines (`denver_3d_v2`, `poudre_3d_v2`)
spend most of their time rendering these layers; `sonicray` renders the same layers on the GPU.

## Status — what is verified, and what is not

| Piece | Status | Evidence |
|---|---|---|
| Metal runtime shader compile (Command Line Tools only, no Xcode) | **verified** | both `.metal` files compile on an M4 Pro |
| Scene upload: 3.8 M triangles + 4608×4856 texture → hardware BVH | **verified** | 1.3–1.6 s |
| Identical geometry/camera/texture to the rayrender scene | **verified** | pixel coverage 74.2 % vs 74.2 %; albedo ratio 1.001 (corr 0.997) |
| Ambient (sky) layer | **verified** | within 2.5–4 % of rayrender, corr 0.99; 1080p @ 64 spp in 0.10 s |
| Sphere-light physics | **verified** | flat plane + sphere light: GPU 13.880 vs closed form 13.889 (rayrender 13.77) |
| Sun layers | **verified** | energy GPU/CPU 1.00, corr 0.98 |
| Near sphere lights (fireworks, beacons) | **close** | median pixel ratio 0.90–1.04; larger differences along shadow edges |
| Multi-bounce indirect light | implemented | adds only ~1–2 % on this terrain; shadow fill differs from rayrender |
| Intel OIDN denoising | **verified** | 8 spp + OIDN has 3.2× lower error than raw 8 spp (vs a 512 spp reference), mean unchanged |
| Whole Poudre film's layers on GPU (162 files) | **verified** | 64.5 s vs 39.3 min on CPU (≈ 36×) |
| Unchanged CPU compositor on GPU layers | **verified (sheet)** | 9-panel sheet visually matches the reference film |
| GPU compositor (`compositor/`, driver `film/poudre/20_compose_gpu.py`) | **verified** | Poudre film composited in **161 s** vs 941 s for the CPU compositor (5.8×); 13 story moments differ from the CPU compositor by ≤ 3/255 (mean < 0.01); whole film 45–51 dB PSNR vs the CPU-composited film |
| End to end (layers + composite), Poudre v2, 1080p30 | **measured** | ≈ 3.8 min vs ≈ 49–55 min on CPU (≈ 13–15×), excluding one-off scene prep and the 3 s audio mux |
| `sr_timeline`, `sr_seasonal_curve`, `sr_write_sonic_pi`, `sr_compose_video` | **skeletons** | small R stubs, not used by the film pipeline |
| `rayrender::render_scene(backend = "metal")` hook | not started | |

All numbers are from one machine (Apple M4 Pro, 14 CPU cores, 20 GPU cores, 48 GB). See [GETTING_STARTED.md](GETTING_STARTED.md) for
installation, the tests that produced these numbers, and the roadmap.

## Quick start

```bash
brew install open-image-denoise            # optional; enables OIDN denoising (pulls hwloc + tbb)
R CMD INSTALL .                             # builds the Objective-C++/Metal bridge with the Command Line Tools
Rscript tools/test_plane_light.R            # GPU vs closed-form vs rayrender on a plane + sphere light
Rscript tools/test_denoise.R                # does OIDN reduce error?
```

```r
library(sonicray)
sr_device_info()                                            # Apple GPU, ray tracing support, unified memory, OIDN
mesh <- sr_mesh_from_heightfield(H, xr = c(-47, 46), zr = c(-46, 52), y0 = min(H), vexag = 2.25)   # H = elevation matrix (m)
gpu  <- sr_upload_mesh(mesh, albedo = png::readPNG("albedo.png")[, , 1:3]^2.2)                      # persistent GPU BVH
lights <- data.frame(name = "burst_1", x = 6, y = 6, z = 16, r = 0.55, intensity = 2400)
layers <- sr_render_light_layers(gpu, lights, lookfrom = c(0, 15.8, 70), lookat = c(0, 1.5, -2), fov = 22.8,
                                 width = 1920L, height = 1080L, light_spp = 256L, max_bounces = 4L, denoise = TRUE)
```

Film-specific bridge scripts (Cache la Poudre) live in [`film/poudre/`](film/poudre/).
