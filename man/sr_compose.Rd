\name{sr_compose}
\alias{sr_compose}
\title{Compose a film from light layers on the GPU}
\description{Compose a film from light layers on the GPU}
\usage{sr_compose(layers, out, story = "network", quality = "draft", ...)}
\arguments{
\item{layers}{directory with `scene_meta.json` and `*.f32` layers. @param out output .mp4 (use a NEW name).}
\item{story}{"network" (data-driven, see docs/BRING_YOUR_OWN_DATA.md) or "poudre". @param quality "draft", "review" or "final".}
\item{...}{extra command-line flags, e.g. `"--t0", "0", "--t1", "5"`.}
}
