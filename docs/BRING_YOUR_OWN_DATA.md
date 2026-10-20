# Bring your own data

## What is real and what is synthetic
- **The bundled demo (`sr_demo()`) is entirely SYNTHETIC.** The terrain, river, 12 sensors and ~90 samples (`param_a/b/c`) are invented by `sr_demo_data()` from a random seed.
  They exist to show the pipeline and the file formats. Do not present demo output as real measurements.
- **The Cache la Poudre film is the real-data case study** the package grew from (real elevation model, real sonde sites, real Water Quality Portal samples).
  Its code is included as a story (`inst/python/sonicray/films/poudre.py`) and a layer script (`film/poudre/`), but its data are **not** shipped.

## Making your own film from real sensors
You need four things. Everything except step 1 is already automated.

1. **Terrain** (elevation matrix `H` in metres, plus `xr`, `zr` extents in km). From a DEM raster you can build it with `terra`; the demo uses `sr_demo_data()` instead.
   Pick a camera (`lookfrom`, `lookat`, `fov`) and check it with a draft render.
2. **Sensors** at scene coordinates `x, z` (km) with height `y` on the terrain. Convert lon/lat with a local projection (km east/north of an origin, then rotate to your camera yaw).
3. **Light layers**: one per beacon (`bea_<site>`) and per site and parameter (`fwk_<parameter>_<site>`), rendered with `sr_render_light_layers()`; see `R/demo.R` for the exact light table.
   Also write `amb`, `mask` (terrain=1) and `riv` (river mask) as in `R/demo.R`.
4. **Story data** in the layers directory:

| file | columns / keys | meaning |
|---|---|---|
| `sensors.csv` | `site, online_s` | when each sensor's beacon lights up (seconds into the film) |
| `events.csv` | `site, parameter, time_s, value` | one firework per sample; `value` in 0..1 sets its size and brightness |
| `story.json` | `duration_s`, `parameters` {name: {plane 0-2, glow [r,g,b]}}, optional `spark_colors` | length and the colour of each parameter |
| `scene_meta.json` | `width, height, cam, sites[]` (each with `x,y,z`, screen position `px,py`, `beam`, `bursts[]`) | geometry; written by `sr_write_story_files()` |

Map real values to `value` yourself (for example scale each parameter to 0..1 over its range, and map real dates to film seconds). Then:
```bash
inst/bin/sonicray sheet   --layers LAYERS --story network --out sheet.png
inst/bin/sonicray compose --layers LAYERS --story network --out my_film.mp4 --quality review
```
For a different narrative (sun, satellite scans, seasons), copy `films/poudre.py` or `films/network.py` and register it in `films/__init__.py`; the `FrameSpec` contract is documented in `framespec.py`.

Limits today: up to 3 parameters (spark planes 0-2); sphere lights; the helper that converts lon/lat and DEMs to a scene is left to you (about 20 lines of `terra`).
