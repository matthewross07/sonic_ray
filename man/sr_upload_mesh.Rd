\name{sr_upload_mesh}
\alias{sr_upload_mesh}
\title{Upload a mesh + an albedo image to a persistent Metal acceleration structure}
\description{Upload a mesh + an albedo image to a persistent Metal acceleration structure}
\usage{sr_upload_mesh(mesh, albedo)}
\arguments{
\item{mesh}{output of [sr_mesh_from_heightfield()].}
\item{albedo}{numeric array (h x w x 3) of *linear* albedo in [0, 1]. For rayrender parity pass `png::readPNG(f)^2.2` (rayrender decodes 8-bit image textures with a ~2.2 gamma).}
}
