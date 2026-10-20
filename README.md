# sonicray

**Metal GPU path tracing and compositing of light-layer films, for Apple Silicon Macs.**
Render each light once on the GPU, then re-time, re-colour and re-cut the film in seconds.

> **Inspired by Tyler Morgan-Wall.** This project is an attempt to follow in the footsteps of [Tyler Morgan-Wall](https://github.com/tylermorganwall)'s
> [rayshader](https://github.com/tylermorganwall/rayshader) and [rayrender](https://github.com/tylermorganwall/rayrender), which made path-traced 3D maps and films
> possible from R. We are *super inspired* by all he has done. It is **not affiliated with him** and contains **none of his code**; it is an independent implementation of the
> same *idea* for Apple GPUs. If you want the full-featured CPU renderer, use rayrender. See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).
>
> **Written with AI assistance** (Anthropic Claude Sonnet 5.5 in Claude Code, directed by Matthew Ross). See [AI_ATTRIBUTION.md](AI_ATTRIBUTION.md).

## Install (Apple Silicon Mac, macOS 14+)

```bash
xcode-select --install                      # compiler (skip if already installed)
brew install ffmpeg open-image-denoise      # video encoder; optional denoiser
python3 -m pip install numpy pillow
```
```r
install.packages("remotes")                 # then, once the repo is public:
remotes::install_github("matthewross07/sonic_ray")
```
Or run `./install.sh` from a clone, which does all of the above and then `sr_doctor()`.

## Try it (about a minute, no data needed)

```r
library(sonicray)
sr_doctor()                    # checks the Mac and prints the exact fix for anything missing
sr_demo("my_demo")             # SYNTHETIC data -> GPU light layers -> GPU compositor -> my_demo/demo_draft.mp4
```
**The demo data are synthetic**: an invented river valley, invented sensors and invented "samples". They show the pipeline and file formats, not real measurements.
To use real sensors and a real elevation model, see [docs/BRING_YOUR_OWN_DATA.md](docs/BRING_YOUR_OWN_DATA.md).

<p align="center"><img src="docs/media/demo_synthetic.webp" width="800" alt="Synthetic demo: sensors light up along an invented river valley and fire a firework for each sample"></p>
<p align="center"><sub><b>Synthetic data.</b> Invented terrain, sensors and samples, made by <code>sr_demo()</code> (about 20 s on an M4 Pro at review quality).
<a href="docs/media/demo_synthetic.mp4">Full-quality mp4</a> · <a href="docs/media/demo_quick_sheet.png">quick sheet</a></sub></p>

## A real-data example

The full **Cache la Poudre** film (130 s, 1080p30) was made with this package from real data: real elevation, 18 real sonde sites and real Water Quality Portal samples.
**[Watch it on Google Drive](https://drive.google.com/file/d/1ZK00RkxaJZzP5XXc6u0OaxfVcurk3Hm7/view?usp=sharing)** (the soundtrack is an optional Sonic Pi score made separately, see below). The input data are not included in this repository;
the demo above is synthetic so that anyone can run it.

## How fast

Cache la Poudre film, 1080p30, 130 s, Apple M4 Pro (real data; the film that motivated this package):

| Stage | CPU (rayrender + Python) | sonicray (GPU) |
|---|---|---|
| 162 light layers | 39.3 min | **64.5 s** |
| Composite 3,900 frames | 15.7 min (measured, partly contended; ~10 min unloaded) | **95 s** (81 s with x264 `fast`) |
| End to end | ~49-55 min | **~2.7 min (about 18-20x)** |

*The CPU numbers are for our specific workflow (one render per light, 2x supersampling, adaptive sampling, scene rebuilt on every call) and are not a general benchmark of rayrender, which is a general-purpose renderer.*

The synthetic demo at draft quality (960x540, 18 s): layers 7.8 s, composite 2.3 s. Method, caveats and parity numbers: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Music (optional)

Soundtracks are **not part of the package**. The Poudre film's score was made with [Sonic Pi](https://sonic-pi.net) (free, separate app) by asking an AI assistant to write a script locked to
the film's event times; the finished `.wav` is muxed in with `sonicray compose --audio score.wav`. See [docs/MUSIC_WITH_SONIC_PI.md](docs/MUSIC_WITH_SONIC_PI.md) for the recipe and a prompt to give your assistant.

## Using it with an AI assistant

This package is built to be driven by AI assistants. Point yours at **[AGENTS.md](AGENTS.md)** first: it lists the questions to ask you (colours, quality, speed, length, audio),
the design decisions that are already settled (so you do not spend time re-debating them), and the traps that cost the most time.

## Documentation

| | |
|---|---|
| [AGENTS.md](AGENTS.md) | the briefing for AI assistants (start here if you are one) |
| [docs/BRING_YOUR_OWN_DATA.md](docs/BRING_YOUR_OWN_DATA.md) | synthetic demo vs real data; file formats |
| [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) | design, measurements, known gaps |
| [docs/MUSIC_WITH_SONIC_PI.md](docs/MUSIC_WITH_SONIC_PI.md) | optional: add a synced soundtrack |
| [AI_ATTRIBUTION.md](AI_ATTRIBUTION.md) | what AI did, what was verified |
| [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) | credits |

GPL-3 licence (the same licence as rayshader and rayrender); third-party components are listed in [NOTICE](NOTICE). macOS on Apple Silicon only.
