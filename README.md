# sonicray

**Metal GPU path tracing and compositing of light-layer films, for Apple Silicon Macs.**
Render each light once on the GPU, then re-time, re-colour and re-cut the film in seconds.

> **I am inspired by Tyler Morgan-Wall.** This project is my attempt to follow in the footsteps of [Tyler Morgan-Wall](https://github.com/tylermorganwall)'s
> [rayshader](https://github.com/tylermorganwall/rayshader) and [rayrender](https://github.com/tylermorganwall/rayrender), which made path-traced 3D maps and films
> possible from R. I am *super inspired* by all he has done. I am **not affiliated with him** and this repository contains **none of his code**: it is an independent implementation of the
> same *idea* for Apple GPUs. If you want the full-featured CPU renderer, use rayrender. See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).
>
> **I built this with AI.** I directed Anthropic's Claude Sonnet 5.5 (in Claude Code, effort high), and Claude wrote most of the code and docs. See [AI_ATTRIBUTION.md](AI_ATTRIBUTION.md).

## Install (Apple Silicon Mac, macOS 14+, R 4.3 or newer)

```bash
xcode-select --install                      # compiler (skip if already installed)
brew install ffmpeg                         # video encoder
```
```r
install.packages("remotes")
remotes::install_github("matthewross07/sonic_ray")
```
Or run `./install.sh` from a clone, which does all of the above and then `sr_doctor()`.

## Try it (a few seconds, no data needed)

```r
library(sonicray)
sr_doctor()                    # checks the Mac and prints the exact fix for anything missing
sr_demo("my_demo")             # SYNTHETIC data -> GPU light layers -> GPU compositor -> my_demo/demo_draft.mp4 (about 3 s; open it)
```
**The demo data are synthetic.** I invented the river valley, the sensors and the "samples". They show the pipeline and file formats, not real measurements.
To use real sensors and a real elevation model, see [docs/BRING_YOUR_OWN_DATA.md](docs/BRING_YOUR_OWN_DATA.md).

<p align="center"><img src="docs/media/demo_synthetic.webp" width="800" alt="Synthetic demo: sensors light up along an invented river valley and fire a firework for each sample"></p>
<p align="center"><sub><b>Synthetic data.</b> Invented terrain, sensors and samples from <code>sr_demo()</code> (about 5 s on an M4 Pro at review quality).
<a href="docs/media/demo_synthetic.mp4">Full-quality mp4</a> · <a href="docs/media/demo_quick_sheet.png">quick sheet</a></sub></p>

## A real-data example

I made the full **Cache la Poudre** film (130 s, 1080p30) with this package from real data: real elevation, 18 real sonde sites and real Water Quality Portal samples.
**[Watch it on Google Drive](https://drive.google.com/file/d/1ZK00RkxaJZzP5XXc6u0OaxfVcurk3Hm7/view?usp=sharing).** I made the soundtrack separately with Sonic Pi (see below).
I do not include the input data here, so I built the synthetic demo above for anyone to run.

## How fast

I measured the Cache la Poudre film (1080p30, 130 s, real data, the film that motivated this package) on an Apple M4 Pro:

| Stage | CPU workflow (rayrender + Python) | sonicray |
|---|---|---|
| 146 light layers | 39.3 min | **18 s** (128 samples per pixel; 58 s at 512) |
| Composite 3,900 frames | 15.7 min (measured, partly contended; about 10 min unloaded) | **61-75 s** with x264, **40-46 s** with the Apple hardware encoder |
| End to end | about 49-55 min | **about 65-100 s (30-50x)** |

*The CPU numbers measure my own earlier workflow (one render per light, 2x supersampling, adaptive sampling, scene rebuilt on every call). They do not benchmark rayrender, a general-purpose renderer.*

The synthetic demo takes 2.9 s end to end at draft quality (960x540), 5.8 s at review (1280x720) and 17 s at final (1920x1080). Everything runs in one C++ process, in R and C++ only: no Python and no third-party libraries beyond Apple's own.
Method, what I tried that did not help, and parity numbers: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Water (only when the water is the subject)

For films about the water itself (clarity, turbidity, objects seen through it), the path tracer can render a physically based water volume: Fresnel reflection, refraction, Beer-Lambert absorption and scattering.
**Rivers and lakes that are only map features do not need it**: leave it off and use the river mask, as the Poudre film does. See [docs/WATER.md](docs/WATER.md).

## Music (optional)

Soundtracks are **not part of the package**. I made the Poudre film's score with [Sonic Pi](https://sonic-pi.net) (a free, separate app): I asked an AI assistant to write a script locked to
the film's event times, recorded it, and muxed the `.wav` in with `sr_compose(..., audio = "score.wav")`. [docs/MUSIC_WITH_SONIC_PI.md](docs/MUSIC_WITH_SONIC_PI.md) has the recipe and a prompt you can give your assistant.

## Using it with an AI assistant

I built this package for AI assistants to drive. Point yours at **[AGENTS.md](AGENTS.md)** first. It tells the assistant which questions to ask you (colours, quality, speed, length, audio),
which design decisions I already settled (so you do not spend time re-debating them), and which traps cost the most time.

## Documentation

| | |
|---|---|
| [AGENTS.md](AGENTS.md) | the briefing for AI assistants (start here if you are one) |
| [docs/BRING_YOUR_OWN_DATA.md](docs/BRING_YOUR_OWN_DATA.md) | synthetic demo vs real data; file formats |
| [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) | design, measurements, known gaps |
| [docs/WATER.md](docs/WATER.md) | 3D water rendering: when to use it (only when water is the subject), options, limits |
| [docs/MUSIC_WITH_SONIC_PI.md](docs/MUSIC_WITH_SONIC_PI.md) | optional: add a synced soundtrack |
| [AI_ATTRIBUTION.md](AI_ATTRIBUTION.md) | what the AI did, what I did, what we verified |
| [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) | credits |

I license this under GPL-3, the same licence as rayshader and rayrender. [NOTICE](NOTICE) lists third-party components. macOS on Apple Silicon only.
