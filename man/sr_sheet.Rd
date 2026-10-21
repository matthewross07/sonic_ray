\name{sr_sheet}
\alias{sr_sheet}
\title{Contact sheet of a story's key moments (look at this before any slow render)}
\description{Contact sheet of a story's key moments (look at this before any slow render)}
\usage{sr_sheet(layers, out = "quick_sheet.png", story = "network", times = NULL, sharpen = 0.65)}
\arguments{
\item{layers}{directory with scene_meta.json and the *.f32 light layers.}
\item{story}{"network" or "poudre".}
\item{sharpen}{unsharp amount.}
\item{out}{the PNG to write.}
\item{times}{key moments in seconds (default: five moments spread over the film).}
}
