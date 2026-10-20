# AGENTS.md: briefing for AI assistants driving sonicray

Read this before you touch the code or render anything. I wrote it so that you and the user spend their time on the film, not on re-deciding the pipeline.
(`CLAUDE.md` and `GEMINI.md` import this file.)

## What this is
`sonicray` makes films of 3D terrain lit by many individual lights (sensors, samples). Light is linear, so **a frame is a weighted sum of "light layers"**:
render each light once on the GPU (R package, `sr_render_light_layers`), then a GPU compositor (`sonicray` CLI, Python + Metal) re-times, re-colours and encodes in seconds.
Colours, timing and length are cheap to change; terrain, camera and lights are not (they need new layers). Apple Silicon Mac only.

## 1. Before you render, ASK THE USER
Do not pick design values silently. Ask these (run `sr_intake()` or `sonicray intake` to print them with defaults and costs; `--json` for machine-readable). Accept "defaults" as an answer.

1. **Data**: synthetic demo or real? If real: which sensors/samples, what coordinates, what time range maps to the film?
2. **Purpose and length**: slide, web, large screen? How many seconds?
3. **Quality**: `draft` (960x540) -> `review` (1280x720) -> `final` (1920x1080). Start at draft.
4. **Colours**: a colour for each of up to 3 measured parameters, sensor beacons, terrain glow. Colour-blind-safe needed?
5. **Look**: night only (default) or also sunrise/sky? Stars? On-screen text? (default: **no text**, the presenter narrates)
6. **Speed vs quality**: fast encode (x264 `fast`, ~2x quicker, ~0.3 dB lower) or slow (final)?
7. **Audio**: none, or a .wav to mux (with offset)? Music is optional and outside the package: see `docs/MUSIC_WITH_SONIC_PI.md`.
8. **Output**: folder and a NEW file name (never overwrite an approved film).

Show the user a **quick sheet** (`sonicray sheet`) or a draft before any slow render, and get approval before `final`.

## 2. Decisions I already settled (do not relitigate)
- Sphere lights only; exact analytic illuminance x stochastic shadow ray; up to 32 lights per GPU pass; OIDN denoise is optional and on by default when installed.
- Layers are `float32` RGB `.f32` files, row-major, one file per light. Do not change the format.
- Compositing happens on the GPU; the CPU only builds a `FrameSpec` per frame (`inst/python/sonicray/framespec.py`). **To change the look of a film, edit or write a story** (`films/*.py`), not the compositor.
- Worker processes use `spawn`, never `fork` (forked Metal processes crash). numpy BLAS is pinned to 1 thread (package does it on import).
- x264 `slow` crf 13 is the "final" encode; VideoToolbox is draft-only (about 4 dB worse).
- 8-bit textures decode as value^2.2 (sRGB). Decode position maps with ^(1/2.2).
- No text is ever drawn into a film. Synthetic data must be labelled synthetic wherever it is shown.

## 3. Workflow
```r
library(sonicray); sr_doctor()                 # fix anything it prints, then continue
sr_demo("out", "draft")                        # synthetic end-to-end check (about a minute)
```
```bash
inst/bin/sonicray sheet   --layers LAYERS --story network --out sheet.png        # look at this first
inst/bin/sonicray compose --layers LAYERS --story network --out NEW.mp4 --quality draft
inst/bin/sonicray compose --layers LAYERS --story network --out NEW_final.mp4 --quality final --audio score.wav --audio-offset 0.64
```
Changing colours, timing, sensor on-times, event data or length: edit `story.json` / `sensors.csv` / `events.csv` and **re-compose only** (seconds).
Changing terrain, camera, lights or resolution: **re-render layers** (minutes at most).

## 4. Where things are
| | |
|---|---|
| `R/` | scene upload, GPU path tracing (`sr_render_light_layers`), demo, doctor |
| `src/`, `inst/metal/sonicray_pathtracer.metal` | Metal ray tracing bridge (Rcpp + Objective-C++) |
| `inst/python/sonicray/` | GPU compositor: `compositor.py` (engine), `native/*.metal` (kernels), `sparks.py`, `framespec.py`, `films/` (stories), `cli.py` |
| `inst/bin/sonicray` | command line |
| `docs/` | data guide, how it works |
| `tools/` | parity check against the CPU compositor, kernel profiler |

## 5. Traps that cost the most time
- **Layers are baked**: terrain, texture, camera and light positions live inside the layers. Decide them before a final render.
- A killed render can leave a truncated `.f32` that looks done. Delete it if in doubt.
- Spark rasterisation is not bit-identical to the CPU reference (about 3% of pixels differ at spark edges; total light within 0.6%). Sky matches to 1/255.
- A tiny number of very bright pixels is normal (firefly clamp is 50).
- Report what was measured vs estimated; say plainly what you did not verify.
- Do not push, publish or overwrite existing outputs without the user's explicit approval.

## 6. Credits
I am inspired by Tyler Morgan-Wall's rayshader and rayrender (I am not affiliated with him and share no code). Claude wrote most of this project under my direction: see `AI_ATTRIBUTION.md`, and keep the attribution if you extend the project.
