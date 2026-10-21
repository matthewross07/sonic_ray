\name{sr_upload_mesh}
\alias{sr_upload_mesh}
\title{Upload a mesh + an albedo image to a persistent Metal acceleration structure}
\description{Upload a mesh + an albedo image to a persistent Metal acceleration structure}
\usage{sr_upload_mesh(mesh, albedo, water = NULL)}
\arguments{
\item{mesh}{output of [sr_mesh_from_heightfield()] (optionally extended with [sr_mesh_append()] for submerged objects).}
\item{albedo}{numeric array (h x w x 3) of *linear* albedo in [0, 1]. For rayrender parity pass `png::readPNG(f)^2.2` (rayrender decodes 8-bit image textures with a ~2.2 gamma).}
\item{water}{optional list that makes a 3D water volume. **Use it only when the water itself is the subject** (clarity, turbidity, submerged objects); for rivers and lakes that are just map features keep `water = NULL` and use the albedo and river mask (see `docs/WATER.md`). Fields: `level` (world y of the surface, required), `ior` (1.333), `roughness` (capillary-wave tilt, 0), `cutaway` (TRUE: a box bounded by the terrain with glass side walls, FALSE: water everywhere below `level`), `absorption` (RGB per unit; or `kd` to use one value for all three), `scattering` (RGB per unit), `metres_per_unit` (scene units to metres, 1: coefficients are taken per scene unit; for terrain scenes with 1 unit = 1 km and vertical exaggeration v use `1000 / v` and give coefficients per metre). With `water = NULL` (the default) the water code is compiled out and costs nothing.}
}
\value{a `sonicray_gpu_scene`.}
