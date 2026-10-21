\name{sr_mesh_from_heightfield}
\alias{sr_mesh_from_heightfield}
\title{Build a triangle mesh from a heightfield matrix (same vertex/uv/triangle layout as the CPU rayrender pipelines)}
\description{Row 1 of `H` is the first z (north-most / far) row. Vertices are ordered column-major, uv = ((col-0.5)/nc, 1-(row-0.5)/nr), and each grid cell becomes two triangles with the winding used by `poudre_3d_v2/lib_scene.R`, so that a GPU render and a rayrender render of the same scene see identical geometry.}
\usage{sr_mesh_from_heightfield(H, xr, zr, y0 = min(H), vexag = 1)}
\arguments{
\item{H}{numeric matrix (nr x nc) of elevations in metres.}
\item{xr,zr}{length-2 ranges of the x and z axes (km).}
\item{y0}{elevation (m) mapped to y = 0.}
\item{vexag}{vertical exaggeration.}
}
