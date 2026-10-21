\name{sr_water_preset}
\alias{sr_water_preset}
\title{Illustrative water optics (per metre), to start from}
\description{Rough, *uncalibrated* absorption and scattering coefficients (red, green, blue, per metre) for four common looks. They are starting points for a picture, not measurements: calibrate against your own kd, Secchi depth, Chl-a, DOC or turbidity data.}
\usage{sr_water_preset(type = c("clear", "algal", "brown", "turbid"))}
\arguments{
\item{type}{one of "clear", "algal", "brown" (high dissolved organic carbon) or "turbid" (suspended sediment).}
}
\value{a list with `absorption` and `scattering`, to merge into the `water` list of [sr_upload_mesh()] (set `metres_per_unit` too).}
