# How it works, what was measured, what is missing

## Pipeline
```
terrain mesh + albedo ──> Metal acceleration structure (built once, kept in GPU memory)
   one path-traced layer per light (sphere light: exact illuminance x shadow ray; up to 32 lights per pass; optional OIDN denoise)
   ──> layers/*.f32  (amb, mask, riv, bea_*, fwk_*, ...)
   ──> story: t -> FrameSpec (weights, sparks, sky, beacons)        [CPU, a few ms]
   ──> GPU compositor: layer sums, scans, halo, tone map, sharpen, bloom, sky, sparks, beacons, dither, uint8   [Metal]
   ──> ffmpeg (x264)
```
Layers are mapped straight into GPU buffers (no copy). Sparks are simulated on the CPU as geometry (vectorised numpy), rasterised on the GPU in 32x32 tiles.
The day sky percentile is a GPU histogram. Native Metal code is compiled at run time (`newLibraryWithSource`), so only the Command Line Tools are needed.

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

Lessons worth knowing: (1) numpy's BLAS spins one thread per core even for 3x3 products; with several workers this cost 4x throughput until pinned to one thread;
(2) forked processes that touch Metal crash, use `spawn`; (3) after this, x264 is the floor, not the GPU.

## Known gaps
- Sphere lights only; one GPU; up to 3 spark parameters; no volumetrics or glossy materials.
- The path tracer and rayrender differ along shadow edges and in indirect fill; bounces add only 1-2% on the Poudre terrain.
- Music is optional and outside the package (see MUSIC_WITH_SONIC_PI.md); the real-data Poudre inputs live outside this repository.
- Tested on one machine. Not tested: clean Mac without Python packages, other macOS versions, other Apple chips.
- `sr_demo(quality = "final")` renders at 1080p but was not timed.
