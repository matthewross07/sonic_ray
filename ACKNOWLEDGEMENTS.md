# Acknowledgements — standing on Tyler Morgan-Wall's shoulders

**This whole project is an attempt to copy, in spirit and in capability, the work of [Tyler Morgan-Wall](https://github.com/tylermorganwall),
and the author is super inspired by everything he has built.**

## What he made that this builds on
- **[rayshader](https://github.com/tylermorganwall/rayshader)** — 2D and 3D data visualisation in R (hillshading, `plot_3d()`, `render_highquality()`).
- **[rayrender](https://github.com/tylermorganwall/rayrender)** ([docs](https://www.rayrender.net/)) — a path tracer for R: scenes built from
  primitives and meshes, emissive lights, materials, OIDN denoising, a pipe-friendly `add_object()` / `render_scene()` interface.
- **rayvertex** — mesh construction used throughout our CPU pipeline.

Our 3D environmental-data films (Denver Water and Cache la Poudre) were first rendered entirely with rayrender/rayshader. Every design idea in
`sonicray` — rendering each light as its own *layer* and recombining them linearly, invisible emissive spheres as lights, adaptive sampling,
image-textured terrain, denoising — comes straight out of that experience. `sonicray` asks one question: *how fast could this be on a GPU?*

## What `sonicray` is, and is not
- It **is** an independent attempt to reproduce rayrender-style results on Apple Silicon using Metal 3 hardware ray tracing and Intel Open Image Denoise.
- It is **not** affiliated with, endorsed by, or a replacement for rayrender/rayshader, and it contains **no rayrender source code**
  (rayrender is GPL-3; `sonicray` is MIT and written from scratch). rayrender remains the reference: we validate our images against its output.
- The GPU request that started the conversation is [rayshader issue #328](https://github.com/tylermorganwall/rayshader/issues/328). If this work
  turns out to be useful, the hope is to share it back with the rayverse (for example as a backend), on Tyler's terms.

## Other thanks
- **Intel Open Image Denoise** (Apache-2.0) — denoising. · **Apple Metal** — hardware ray tracing on Apple GPUs. · **Rcpp** — R/C++ glue.
- **Sonic Pi** (Sam Aaron) — the generative soundtracks that sync to our films.
