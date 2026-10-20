\name{sr_demo}
\alias{sr_demo}
\title{Run the whole pipeline on SYNTHETIC data (about a minute at draft quality)}
\description{Invented terrain + invented sensors and samples -> GPU path-traced light layers -> GPU compositor -> mp4. The data are synthetic; the point is to show the pipeline and the file formats. For real data see `docs/BRING_YOUR_OWN_DATA.md`.}
\usage{sr_demo(out_dir = "sonicray_demo", quality = c("draft", "review", "final"), compose = TRUE, seed = 7L)}
\arguments{
\item{out_dir}{where layers and the film go (created).}
\item{quality}{"draft" (960x540), "review" (1280x720) or "final" (1920x1080).}
\item{compose}{set FALSE to stop after the light layers.}
\item{seed}{random seed of the synthetic data.}
}
\value{invisibly, the path of the mp4 (or the layers directory when `compose = FALSE`).}
