# Optional: add a soundtrack with Sonic Pi

**Music is an add-on, not part of the package.** sonicray makes the picture; it needs no audio tools. If you want background music that is locked to the film, the approach that worked for the
Cache la Poudre film is:

1. Install [Sonic Pi](https://sonic-pi.net) (free, separate app) if you want to record audio from code.
2. Give your AI assistant the film's event data (`events.csv`, `sensors.csv`, `story.json`) and ask it to write a Sonic Pi script whose notes land on those times.
3. Record the script to a `.wav` in Sonic Pi, then mux it in:
   ```bash
   inst/bin/sonicray compose --layers LAYERS --story network --out film.mp4 --quality final --audio score.wav --audio-offset 0.64
   ```
   `--audio-offset` is how many seconds of lead-in to trim from the start of the recording (Sonic Pi usually starts a little late; find your own value by listening for the first note).

## A prompt that works

> Here are `events.csv` (site, parameter, time_s, value) and `sensors.csv` (site, online_s) for a 18 s film. Write a Sonic Pi 5 script, exactly 18 seconds long at 120 bpm (use a 0.125 s grid),
> that plays a soft pad throughout, a rising tone as each sensor comes online, and one short note per event: pitch by parameter (param_a low, param_b mid, param_c high),
> loudness by `value`. Quantise each event to the nearest 0.125 s and print the quantised times so I can check them against the film. Keep it calm; no drums.

Ask the assistant to **print the times it used**; if the film and the music drift, re-quantise the film events to the same grid in `events.csv` and re-compose (seconds).

## Should this become part of the package?

Our suggestion, not built yet: keep Sonic Pi out of the core (it is a separate app with its own recording step), and add a small optional piece later: a `sonicray timeline --json` command that
exports the film's beats, and a checker that listens for the first note and measures the offset. Until then the recipe above is enough.
