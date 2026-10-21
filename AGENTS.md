# AGENTS.md: briefing for AI assistants driving sonicray

Read this before you touch the code or render anything. I wrote it so that you and the user spend their time on the film, not on re-deciding the pipeline.
(`CLAUDE.md` and `GEMINI.md` import this file.)

## What this is
`sonicray` makes films of 3D terrain lit by many individual lights (sensors, samples). Light is linear, so **a frame is a weighted sum of "light layers"**:
render each light once on the GPU (R package, `sr_render_light_layers`), then a GPU compositor (`sr_compose`, C++ + Metal, one process) re-times, re-colours and encodes in seconds.
Colours, timing and length are cheap to change; terrain, camera and lights are not (they need new layers). Apple Silicon Mac only.

## 1. Before you render, ASK THE USER
Do not pick design values silently. Ask these (run `sr_intake()` to print them with defaults and costs; `sr_intake(json = TRUE)` for machine-readable). Accept "defaults" as an answer.

1. **Data**: synthetic demo or real? If real: which sensors/samples, what coordinates, what time range maps to the film?
2. **Purpose and length**: slide, web, large screen? How many seconds?
3. **Quality**: `draft` (960x540) -> `review` (1280x720) -> `final` (1920x1080). Start at draft.
4. **Colours**: a colour for each of up to 3 measured parameters, sensor beacons, terrain glow. Colour-blind-safe needed?
5. **Look**: night only (default) or also sunrise/sky? Stars? On-screen text? (default: **no text**, the presenter narrates)
6. **Speed vs file size**: Apple hardware encoder (fastest, frees the CPU, files about 3x bigger) or x264 (small files)?
7. **Audio**: none, or a .wav to mux (with offset)? Music is optional and outside the package: see `docs/MUSIC_WITH_SONIC_PI.md`.
8. **Output**: folder and a NEW file name (never overwrite an approved film).

Show the user a **quick sheet** (`sr_sheet()`) or a draft before any slow render, and get approval before `final`.

## 2. Decisions I already settled (do not relitigate)
- Sphere lights only; exact analytic illuminance x stochastic shadow ray; up to 32 lights per GPU pass.
- Layers are `float32` RGB `.f32` files, row-major, one file per light. Do not change the format.
- Compositing happens on the GPU in one C++ process; the CPU only builds a `FrameSpec` per frame (`src/comp_types.h`). **To change the look of a film, edit or write a story** (`src/comp_stories.cpp`), not the compositor (`src/comp_engine.cpp`).
- Presets (`sr_presets()`): draft 32 samples per pixel, review 128, final 256. There is no denoiser: raw layers beat denoised ones from about 32 spp up (see `docs/HOW_IT_WORKS.md`).
- Encoding: Apple hardware HEVC for draft and review (about 40 s for the Poudre film, big files); x264 `slow` crf 13 for final (about 60 s, small files).
- 8-bit textures decode as value^2.2 (sRGB). Decode position maps with ^(1/2.2).
- No text is ever drawn into a film. Synthetic data must be labelled synthetic wherever it is shown.

## 3. Workflow
```r
library(sonicray); sr_doctor()                 # fix anything it prints, then continue
sr_demo("out", "draft")                        # synthetic end-to-end check (a few seconds)
```
```r
sr_sheet("LAYERS", "sheet.png", story = "network")                        # look at this first
sr_compose("LAYERS", "NEW.mp4", story = "network", quality = "draft")
sr_compose("LAYERS", "NEW_final.mp4", story = "network", quality = "final", audio = "score.wav", audio_offset = 0.64)
```
Changing colours, timing, sensor on-times, event data or length: edit `story.json` / `sensors.csv` / `events.csv` and **re-compose only** (seconds).
Changing terrain, camera, lights or resolution: **re-render layers** (about 25 s for the Poudre film). `R/demo.R` is the complete recipe for your own scene: mesh, GPU upload, one light per beacon and per site and parameter, `riv` and `stars` layers, and the story files.
Changing the narrative beyond what the data files control (sun, scans, seasons): copy `NetworkStory` or `PoudreStory` in `src/comp_stories.cpp`, register it in `make_story()` and reinstall. Stories are C++ today; an R-function story interface is not built.

## 4. Where things are
| | |
|---|---|
| `R/` | scene upload, GPU path tracing (`sr_render_light_layers`), demo |
| `src/`, `inst/metal/sonicray_pathtracer.metal` | Metal ray tracing bridge (Rcpp + Objective-C++) |
| `src/comp_*.cpp`, `inst/metal/sonicray_comp.metal` | GPU compositor: engine, kernels, sparks, stories, the threaded film pipeline |
| `R/compose.R` | `sr_compose`, `sr_still`, `sr_sheet`, `sr_doctor`, `sr_intake`, `sr_presets` |
| `docs/` | data guide, how it works |
| `tools/` | `make_rd.R` (help pages), `compare_pngs.R` (frame parity), `smoke_test.R` |

## 5. Traps that cost the most time
- **Layers are baked**: terrain, texture, camera and light positions live inside the layers. Decide them before a final render.
- A killed render can leave a truncated `.f32` that looks done. Delete it if in doubt.
- Spark rasterisation is not bit-identical to the original CPU compositor, which is not in this repository (about 3% of pixels differ at spark edges; total light within 0.6%). Sky matches to 1/255.
- A tiny number of very bright pixels is normal (the firefly clamp is 50).
- More worker threads rarely help: x264 saturates the CPU, and the hardware encoder is GPU-bound. The default (3) is fine; each thread holds about 0.5 GB of GPU buffers.
- Hardware-encoded files are about 3x bigger than x264 files; use x264 for the version you share or archive.
- Do not look for a denoiser option: there is none (see section 2).
- Report what was measured vs estimated; say plainly what you did not verify.
- Do not push, publish or overwrite existing outputs without the user's explicit approval.

## 6. Credits
I am inspired by Tyler Morgan-Wall's rayshader and rayrender (I am not affiliated with him and share no code). Claude wrote most of this project under my direction: see `AI_ATTRIBUTION.md`, and keep the attribution if you extend the project.
