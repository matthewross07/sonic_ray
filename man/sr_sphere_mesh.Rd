\name{sr_sphere_mesh}
\alias{sr_sphere_mesh}
\title{A sphere mesh whose every vertex samples one point of the albedo texture}
\description{A sphere mesh whose every vertex samples one point of the albedo texture}
\usage{sr_sphere_mesh(center, radius, uv = c(0.5, 0.5), n_lat = 16L, n_lon = 24L)}
\arguments{
\item{center}{length-3 centre in scene units.}
\item{radius}{radius in scene units.}
\item{uv}{texture coordinate (u, v) whose texel gives the colour.}
\item{n_lat,n_lon}{sphere resolution.}
}
\value{a mesh for [sr_mesh_append()].}
