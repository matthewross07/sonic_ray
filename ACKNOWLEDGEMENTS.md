# Acknowledgements: standing on Tyler Morgan-Wall's shoulders

**This whole project is my attempt to copy, in spirit and in capability, the work of [Tyler Morgan-Wall](https://github.com/tylermorganwall).
I am super inspired by everything he has built.**

## What he made that this builds on
- **[rayshader](https://github.com/tylermorganwall/rayshader)**: 2D and 3D data visualisation in R (hillshading, `plot_3d()`, `render_highquality()`).
- **[rayrender](https://github.com/tylermorganwall/rayrender)** ([docs](https://www.rayrender.net/)): a path tracer for R with scenes built from
  primitives and meshes, emissive lights, materials, OIDN denoising, and a pipe-friendly `add_object()` / `render_scene()` interface.
- **rayvertex**: mesh construction that my CPU pipeline uses throughout.

I first rendered my 3D environmental-data films (Denver Water and Cache la Poudre) entirely with rayrender and rayshader. Every design idea in
`sonicray` comes straight out of that experience: rendering each light as its own *layer* and recombining the layers linearly, invisible emissive spheres as lights, adaptive sampling,
image-textured terrain, denoising. `sonicray` asks one question: *how fast could this run on a GPU?*

## What `sonicray` is, and is not
- It **is** my independent attempt to reproduce rayrender-style results on Apple Silicon with Metal 3 hardware ray tracing and Intel Open Image Denoise.
- It **is not** affiliated with, endorsed by, or a replacement for rayrender or rayshader, and it contains **no rayrender source code**.
  I license `sonicray` under GPL-3 like rayrender, but Claude wrote it from scratch. rayrender stays the reference: we validate our images against its output.
- I have long wanted to do this, and AI assistance finally made it possible. If this work proves useful, I hope to share it back with the rayverse
  (for example as a backend) on Tyler's terms. Sharing the same licence (GPL-3) keeps that simple.

## Other thanks
- **Intel Open Image Denoise** (Apache-2.0) for denoising, **Apple Metal** for hardware ray tracing on Apple GPUs, and **Rcpp** for the R/C++ glue.
- **Sonic Pi** (Sam Aaron) for the optional soundtracks on my films. It is not part of this package; see docs/MUSIC_WITH_SONIC_PI.md.
