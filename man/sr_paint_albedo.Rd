\name{sr_paint_albedo}
\alias{sr_paint_albedo}
\title{Paint a small square of colour into the albedo texture (so an appended object has a colour)}
\description{Paint a small square of colour into the albedo texture (so an appended object has a colour)}
\usage{sr_paint_albedo(albedo, uv, rgb, half_px = 3L)}
\arguments{
\item{albedo}{numeric array (h x w x 3, linear).}
\item{uv}{texture coordinate (u, v) to paint at (v = 1 is the top row).}
\item{rgb}{linear colour.}
\item{half_px}{half-size of the painted square in pixels.}
}
\value{the modified albedo array.}
