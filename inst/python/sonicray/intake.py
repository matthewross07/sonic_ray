"""The questions an AI assistant must put to the user BEFORE rendering, with defaults and measured costs.

`sonicray intake` prints them (add --json for machine-readable).  The point: design decisions (colours, quality, speed, length, text,
audio) belong to the human; every question has a recommended default so the user can answer "defaults" and move on.
Timings are for an Apple M4 Pro (Poudre 1080p30, 130 s film); other sizes are pixel-count estimates and say so.
"""

PRESETS = {
    "draft":  {"resolution": [960, 540],   "fps": 24, "light_spp": 32,  "amb_spp": 64,   "encode_preset": "veryfast", "crf": 20, "sharpen": 0.0,  "dither": False,
               "use": "iterating on colours, timing and framing; look at the quick sheet first"},
    "review": {"resolution": [1280, 720],  "fps": 30, "light_spp": 128, "amb_spp": 256,  "encode_preset": "fast",     "crf": 16, "sharpen": 0.5,  "dither": True,
               "use": "share with collaborators; close to final"},
    "final":  {"resolution": [1920, 1080], "fps": 30, "light_spp": 512, "amb_spp": 1024, "encode_preset": "slow",     "crf": 13, "sharpen": 0.65, "dither": True,
               "use": "the version you present from; only render after the user approves a review"},
}

QUESTIONS = [
    {"id": "data", "ask": "Is this real data or the synthetic demo? If real: which sensors/samples, what coordinates, what time range maps to the film?",
     "default": "synthetic demo", "why": "decides the scene, sensors.csv and events.csv; never present synthetic output as real measurements"},
    {"id": "purpose", "ask": "Where will this be shown (slide deck, web, social, large screen) and for how long?",
     "default": "slide deck, 15-20 s demo / 130 s full film", "why": "sets resolution, fps and length; short loops need less quality"},
    {"id": "quality", "ask": "Which quality preset: draft, review or final?", "options": list(PRESETS), "default": "draft first, then review, then final only after approval",
     "why": "draft is ~10x cheaper than final; most design mistakes are visible in a draft or in `sonicray sheet`"},
    {"id": "colors", "ask": "Colours for each measured parameter (up to 3), sensor beacons and terrain glow? Colour-blind-safe palette needed?",
     "default": "crimson / algae green / royal violet for parameters, ice-blue beacons", "why": "palette lives in story.json (parameters[*].glow, spark_colors); changing it needs only a re-composite, not new layers"},
    {"id": "look", "ask": "Night only, or add sunrise / sky? Starfield? Any text overlay?", "default": "night, stars, NO on-screen text (the presenter narrates)",
     "why": "sun/sky need extra layers (re-render); text is never drawn into films by default"},
    {"id": "speed", "ask": "Fastest turnaround or best encode quality?", "options": ["fast (x264 fast, ~2x quicker, ~0.3 dB lower)", "slow (x264 slow, final)"],
     "default": "fast for review, slow for final", "why": "compositing is GPU-fast; the video encode is the remaining floor"},
    {"id": "audio", "ask": "Add a soundtrack? Provide a .wav (and the offset in seconds) or skip.", "default": "no audio", "why": "muxed after compositing with --audio"},
    {"id": "output", "ask": "Output folder and file name?", "default": "a NEW file name; never overwrite an existing approved film", "why": "approved outputs are read-only by convention"},
]


def as_text():
    out = ["Ask the user these before rendering (accept 'defaults' as an answer):", ""]
    for i, q in enumerate(QUESTIONS, 1):
        out.append(f"{i}. [{q['id']}] {q['ask']}")
        if "options" in q:
            out.append("     options: " + "; ".join(q["options"]))
        out.append(f"     default: {q['default']}")
        out.append(f"     why: {q['why']}")
    out += ["", "Quality presets:"]
    for k, v in PRESETS.items():
        out.append(f"  {k:7s} {v['resolution'][0]}x{v['resolution'][1]} @ {v['fps']} fps, light_spp {v['light_spp']}, amb_spp {v['amb_spp']}, "
                   f"x264 {v['encode_preset']} crf {v['crf']}  -- {v['use']}")
    return "\n".join(out)
