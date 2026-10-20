# How it works, what we measured, what is missing

## Pipeline
```
terrain mesh + albedo ──> Metal acceleration structure (built once, kept in GPU memory)
   one path-traced layer per light (sphere light: exact illuminance x shadow ray; up to 32 lights per pass; optional OIDN denoise)
   ──> layers/*.f32  (amb, mask, riv, bea_*, fwk_*, ...)
   ──> story: t -> FrameSpec (weights, sparks, sky, beacons)        [CPU, a few ms]
   ──> GPU compositor: layer sums, scans, halo, tone map, sharpen, bloom, sky, sparks, beacons, dither, uint8   [Metal]
   ──> ffmpeg (x264)
```
The compositor maps layers straight into GPU buffers (no copy). It simulates sparks on the CPU as geometry (vectorised numpy) and rasterises them on the GPU in 32x32 tiles.
A GPU histogram computes the day-sky percentile. The package compiles its native Metal code at run time (`newLibraryWithSource`), so you need only the Command Line Tools.

**Read the CPU column fairly.** It measures *my* earlier workflow built on rayrender (one render per light, 2x supersampling, adaptive sampling, scene rebuilt on each call), on one scene. It does not benchmark rayrender, a general-purpose renderer with far more features than this package.

## Measurements (Apple M4 Pro, macOS 15.7, Poudre film: 1920x1080, 3.8 M triangles, 162 layers, 3,900 frames)
| | CPU | GPU |
|---|---|---|
| Upload scene / acceleration structure | ~2.4 s per render call | 1.3-1.6 s once |
| All layers | 39.3 min | 64.5 s |
| Composite | 941 s (CPU compositor, 8 procs, partly contended) | 95 s (x264 slow) / 81 s (x264 fast) |
| Per frame (compositor, one process) | 1-3 s | 10-80 ms |

Parity with the CPU compositor on the same layers: sky/sun frames within 1/255; firework frames within 0.3-0.6% total light, with about 3% of pixels differing by more than 8/255 at spark edges.
Whole film vs the earlier CPU-composited film of the same layers: mean 49.2 dB PSNR over 3,900 frames, worst frame 38.5 dB (a firework frame; differences are spark edges).
Layers vs rayrender: ambient and sun within a few percent; near sphere lights median ratio 0.9-1.04; shadow edges differ.
Synthetic demo (draft 960x540, 18 s): layers 7.8 s, composite 2.3 s.

Lessons worth knowing: (1) numpy's BLAS spins one thread per core even for 3x3 products, and with several workers that cost 4x throughput until we pinned it to one thread;
(2) forked processes that touch Metal crash, so use `spawn`; (3) after those fixes, x264 sets the floor, not the GPU.

## Known gaps
- Sphere lights only; one GPU; up to 3 spark parameters; no volumetrics or glossy materials.
- Treat rayrender as the reference: our path tracer differs from it along shadow edges and in indirect fill; bounces add only 1-2% on the Poudre terrain.
- Music is optional and outside the package (see MUSIC_WITH_SONIC_PI.md), and I keep the real-data Poudre inputs outside this repository.
- We tested on one machine. We did not test a clean Mac without the Python packages, other macOS versions, or other Apple chips.
- `sr_demo(quality = "final")` renders at 1080p, but we have not timed it.
