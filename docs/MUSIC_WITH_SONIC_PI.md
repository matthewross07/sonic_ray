# Optional: add a soundtrack with Sonic Pi

**Music is an add-on, not part of the package.** sonicray makes the picture and needs no audio tools. If you want background music locked to the film, here is how I did it for the
Cache la Poudre film:

1. Install [Sonic Pi](https://sonic-pi.net) (a free, separate app) so you can record audio from code.
2. Give your AI assistant the film's event data (`events.csv`, `sensors.csv`, `story.json`) and ask it to write a Sonic Pi script whose notes land on those times.
3. Record the script to a `.wav` in Sonic Pi, then mux it in:
   ```r
   sr_compose("LAYERS", "film.mp4", story = "network", quality = "final", audio = "score.wav", audio_offset = 0.64)
   ```
   `audio_offset` trims that many seconds of lead-in from the start of the recording. Sonic Pi usually starts a little late, so find your own value by listening for the first note.

## A prompt that works

> Here are `events.csv` (site, parameter, time_s, value) and `sensors.csv` (site, online_s) for an 18 s film. Write a Sonic Pi 5 script, exactly 18 seconds long at 120 bpm (use a 0.125 s grid),
> that plays a soft pad throughout, a rising tone as each sensor comes online, and one short note per event: pitch by parameter (param_a low, param_b mid, param_c high),
> loudness by `value`. Quantise each event to the nearest 0.125 s and print the quantised times so I can check them against the film. Keep it calm; no drums.

Ask the assistant to **print the times it used**. If the film and the music drift apart, re-quantise the film events to the same grid in `events.csv` and re-compose, which takes seconds.

## Should this become part of the package?

I suggest not yet, and not in the core: Sonic Pi is a separate app with its own recording step. I would add a small optional piece later: a `sonicray timeline --json` command that
exports the film's beats, and a checker that listens for the first note and measures the offset. I have not built either. Until then, the recipe above is enough.
