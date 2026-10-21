\name{sr_sample_texture}
\alias{sr_sample_texture}
\title{Bilinear sample of a matrix at texture coordinates (v = 1 is the top row, like the mesh uvs)}
\description{Bilinear sample of a matrix at texture coordinates (v = 1 is the top row, like the mesh uvs)}
\usage{sr_sample_texture(m, u, v)}
\arguments{
\item{m}{numeric matrix (rows = image rows).}
\item{u,v}{texture coordinates in 0..1 (vectors of equal length).}
}
\value{numeric vector of the sampled values.}
