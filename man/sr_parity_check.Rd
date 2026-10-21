\name{sr_parity_check}
\alias{sr_parity_check}
\title{Compare a `sonicray` Metal GPU layer against a `rayrender` CPU reference layer}
\description{Compare a `sonicray` Metal GPU layer against a `rayrender` CPU reference layer}
\usage{sr_parity_check(gpu_layer, cpu_layer)}
\arguments{
\item{gpu_layer,cpu_layer}{numeric vectors of the same length.}
}
\value{a list with `rmse`, `rel_mean_diff`, `cosine_similarity` and `passed`.}
