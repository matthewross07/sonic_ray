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

## How fast

Cache la Poudre film, 1080p30, 130 s, Apple M4 Pro (real data; the film that motivated this package):

| Stage | CPU (rayrender + Python) | sonicray (GPU) |
|---|---|---|
| 162 light layers | 39.3 min | **64.5 s** |
| Composite 3,900 frames | 15.7 min (measured, partly contended; ~10 min unloaded) | **95 s** (81 s with x264 `fast`) |
| End to end | ~49-55 min | **~2.7 min (about 18-20x)** |

The synthetic demo at draft quality (960x540, 18 s): layers 7.8 s, composite 2.3 s. Method, caveats and parity numbers: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Using it with an AI assistant

This package is built to be driven by AI assistants. Point yours at **[AGENTS.md](AGENTS.md)** first: it lists the questions to ask you (colours, quality, speed, length, audio),
the design decisions that are already settled (so you do not spend time re-debating them), and the traps that cost the most time.

## Documentation

| | |
|---|---|
| [AGENTS.md](AGENTS.md) | the briefing for AI assistants (start here if you are one) |
| [docs/BRING_YOUR_OWN_DATA.md](docs/BRING_YOUR_OWN_DATA.md) | synthetic demo vs real data; file formats |
| [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) | design, measurements, known gaps |
| [AI_ATTRIBUTION.md](AI_ATTRIBUTION.md) | what AI did, what was verified |
| [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) | credits |

MIT licence. macOS on Apple Silicon only.
