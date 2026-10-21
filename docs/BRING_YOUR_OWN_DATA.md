# Bring your own data

## What is real and what is synthetic
- **The bundled demo (`sr_demo()`) is entirely SYNTHETIC.** `sr_demo_data()` invents the terrain, the river, 12 sensors and about 90 samples (`param_a/b/c`) from a random seed.
  They show the pipeline and the file formats. Do not present demo output as real measurements.
- **The Cache la Poudre film is my real-data case study** ([watch it](https://drive.google.com/file/d/1ZK00RkxaJZzP5XXc6u0OaxfVcurk3Hm7/view?usp=sharing)), the film this package grew from. It uses a real elevation model, real sonde sites and real Water Quality Portal samples.
  The repository includes its story (`story = "poudre"` in `src/comp_stories.cpp`) and a layer script (`film/poudre/`) but **not** its data.

## Make your own film from real sensors
You need four things. The package automates everything except the first.

1. **Terrain**: an elevation matrix `H` in metres, plus the `xr` and `zr` extents in km. You can build it from a DEM raster with `terra`; the demo uses `sr_demo_data()` instead.
   Pick a camera (`lookfrom`, `lookat`, `fov`) and check it with a draft render.
2. **Sensors** at scene coordinates `x, z` (km) with height `y` on the terrain. Convert lon/lat with a local projection: km east and north of an origin, then rotate to your camera yaw.
3. **Light layers**: one per beacon (`bea_<site>`) and one per site and parameter (`fwk_<parameter>_<site>`). Render them with `sr_render_light_layers()`; `R/demo.R` shows the exact light table.
   Also write `amb`, `mask` (terrain = 1, one channel) and `riv` (river mask: red = highlighted river, green = all water) the way `R/demo.R` does.
4. **Story data** in the layers directory:

| file | columns / keys | meaning |
|---|---|---|
| `sensors.csv` | `site, online_s` | when each sensor's beacon lights up (seconds into the film) |
| `events.csv` | `site, parameter, time_s, value` | one firework per sample; `value` in 0..1 sets its size and brightness |
| `story.json` | `duration_s`, `parameters` {name: {plane 0-2, glow [r,g,b]}}, optional `spark_colors` | length and the colour of each parameter |
| `scene_meta.json` | `width, height, cam, sites[]` (each with `x,y,z`, screen position `px,py`, `beam`, `bursts[]`) | geometry; `sonicray:::sr_write_story_files()` writes it |

Map your real values to `value` yourself (for example scale each parameter to 0..1 over its range) and map your real dates to film seconds. Then run:
```r
sr_sheet("LAYERS", "sheet.png", story = "network")                       # look at this first
sr_compose("LAYERS", "my_film.mp4", story = "network", quality = "review")
```
For a different narrative (sun, satellite scans, seasons), copy `NetworkStory` or `PoudreStory` in `src/comp_stories.cpp`, register your copy in `make_story()`, and reinstall. A story turns a time into a `FrameSpec` (`src/comp_types.h`); the compositor needs no changes.

Limits today: up to 3 parameters (spark planes 0-2) and sphere lights only. I leave the helper that converts lon/lat and DEMs into a scene to you; it takes about 20 lines of `terra`.
