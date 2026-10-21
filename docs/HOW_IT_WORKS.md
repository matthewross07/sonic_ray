# How it works, what we measured, what is missing

## Pipeline
```
terrain mesh + albedo ──> Metal acceleration structure (built once, kept in GPU memory)
   one path-traced layer per light (sphere light: exact illuminance x shadow ray; up to 32 lights per pass)
   ──> layers/*.f32  (amb, mask, riv, bea_*, fwk_*, ...)
   ──> story: t -> FrameSpec (weights, sparks, sky, beacons)                       [C++, a few ms per frame]
   ──> GPU compositor: layer sums, scans, halo, tone map, sharpen, bloom, sky, sparks, beacons, dither, uint8   [Metal]
   ──> ffmpeg (x264 or Apple VideoToolbox), fed through a pipe
```
Everything after the R call that starts a render happens in one C++ process. Worker threads prepare and render frames (default 3 at a time, each with its own GPU buffers);
the main thread writes the frames to ffmpeg in order. The compositor maps layer files straight into GPU buffers (no copy). It simulates sparks on the CPU as geometry and rasterises them on the GPU in 16x16 tiles.
Terrain passes skip the sky rows above the terrain's bounding box. A GPU histogram computes the day-sky percentile. The package compiles its Metal code at run time, so you need only the Command Line Tools.
The stories (`src/comp_stories.cpp`) hold every creative decision; the compositor (`src/comp_engine.cpp`, `inst/metal/sonicray_comp.metal`) holds none.

**Read the CPU column fairly.** It measures *my* earlier workflow built on rayrender (one render per light, 2x supersampling, adaptive sampling, scene rebuilt on each call), on one scene. It does not benchmark rayrender, a general-purpose renderer with far more features than this package.

## Measurements (Apple M4 Pro, macOS 15.7; Cache la Poudre film: 1920x1080, 3.8 M triangles, 146 light layers, 3,900 frames)
| Stage | CPU workflow | sonicray |
|---|---|---|
| Light layers | 39.3 min | 18 s at 128 samples per pixel (24.5 s including scene setup; 58 s of tracing at 512) |
| Composite, x264 `slow` crf 13 | 941 s (CPU compositor); 95 s (my earlier Python version of the GPU compositor) | **61-75 s** (varies run to run) |
| Composite, Apple hardware HEVC | n/a | **40-46 s** (about 85-98 frames per second, about 21 s of CPU in total) |
| End to end, including scene setup | about 49-55 min | **about 65-100 s** |

Synthetic demo (18 s film, including data and layers): draft (960x540) 2.9 s, review (1280x720) 5.8 s, final (1920x1080, x264 slow) 17 s.

**Parity.** The C++ compositor matches my earlier Python implementation on 11 Poudre frames and 7 demo frames at 66 dB PSNR or better (mean difference under 0.002/255), including the firework frames.
That Python version matched the original CPU compositor to within 1/255 on sky and sun frames and within 0.3-0.6% total light on firework frames (spark edges differ by up to a pixel).
Layers vs rayrender: ambient and sun within a few percent; near sphere lights median ratio 0.9-1.04; shadow edges differ.

## What I tried that did not help (so you do not repeat it)
- **Tile culling of light layers.** About 25% of every frame is sky, which is zero in every terrain layer, and beacon layers are about 45-55% empty. Fireworks are not sparse: their tails reach the whole terrain, so only 3-6% more tiles are skippable. I kept the one exact, general part (crop terrain passes to the terrain's bounding box).
- **Light culling in the path tracer** (skip the shadow ray when a pixel-light pair contributes less than a threshold). At a threshold costing 0.17% error it was only 10% faster, because these lights reach far. Removed.
- **Parallel segment encoding.** A single x264 `slow` stream already keeps about 10 cores busy, so splitting the film into segments gained nothing (8.0 s vs 8.0-8.1 s for 20 s of film).
- **Denoising with Intel OIDN.** I tested it and removed it from the package. At 32 samples per pixel and up, raw layers were more accurate than denoised ones (OIDN leaves about 3% error); it helps only below that.
- **More worker threads.** With x264 the CPU is saturated, so 1, 3 or 5 threads all take 61-66 s. With the hardware encoder 3 and 5 threads both take about 40 s (GPU-bound).

## What did help
- **Fewer samples.** Layer time scales linearly with samples. Composites from 128 spp layers match the 512 spp ones at 46 dB PSNR or better (mean difference 0.15/255), and 64 spp at 44 dB or better. Over the whole Poudre film (3,900 frames) the 128 spp layers give a worst frame of 45.1 dB and a mean of 48.7 dB against the 512 spp layers. The presets use 32 / 128 / 256 spp for draft / review / final.
- **Hardware encoding for draft and review.** At quality 90, HEVC hardware encoding scored 53.9 dB against the source frames (x264 `slow` crf 13: 51.3 dB) in 2.9 s instead of 8.0 s per 20 s of film, but the file is about 3x bigger (118 MB vs 38 MB per 20 s). Use x264 when file size matters.
- **One process.** Moving from several worker processes to threads cut system CPU time from about 78 s to 26 s per film (per-process page faults and 24 GB of frames through pipes).

## Known gaps
- Sphere lights only; one GPU; up to 3 spark parameters; no glossy materials. Water is a flat dielectric volume with absorption and single scattering (`docs/WATER.md`, only for films where water is the subject).
- Treat rayrender as the reference: our path tracer differs from it along shadow edges and in indirect fill; bounces add only 1-2% on the Poudre terrain.
- Layers still pass through disk (`.f32` files). Keeping them in GPU memory from the path tracer into the compositor, and overlapping rendering with compositing, would save a few seconds; I have not built it.
- Custom stories are C++ today (copy `NetworkStory` in `src/comp_stories.cpp`). An R-function story interface is not built.
- Music is optional and outside the package (see MUSIC_WITH_SONIC_PI.md), and I keep the real-data Poudre inputs outside this repository.
- We tested on one machine (Apple M4 Pro, macOS 15.7). We did not test other macOS versions or other Apple chips.
