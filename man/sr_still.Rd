\name{sr_still}
\alias{sr_still}
\title{Render single frames of a film as PNG files (fast way to check colours, framing and timing)}
\description{Render single frames of a film as PNG files (fast way to check colours, framing and timing)}
\usage{sr_still(layers, times, out_dir = ".", story = "network", sharpen = 0.65, dither = FALSE)}
\arguments{
\item{layers}{directory with `scene_meta.json` and the `*.f32` light layers.}
\item{times}{times in seconds.}
\item{out_dir}{folder for the PNGs.}
\item{story}{"network" or "poudre".}
\item{sharpen}{unsharp amount.}
\item{dither}{TRUE to dither before 8-bit quantisation.}
}
\value{invisibly, the PNG paths.}
