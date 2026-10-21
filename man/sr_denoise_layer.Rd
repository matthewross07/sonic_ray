\name{sr_denoise_layer}
\alias{sr_denoise_layer}
\title{Denoise a single HxWx3 layer using first-hit Albedo and Normal AOVs via Intel OIDN}
\description{Denoise a single HxWx3 layer using first-hit Albedo and Normal AOVs via Intel OIDN}
\usage{sr_denoise_layer(noisy_rgb, albedo_rgb, normal_xyz, width, height, demodulate_albedo = TRUE)}
\arguments{
\item{noisy_rgb,albedo_rgb,normal_xyz}{numeric vectors (width x height x 3, interleaved) for the noisy layer and its first-hit albedo and normal.}
\item{width,height}{image size in pixels.}
\item{demodulate_albedo}{divide by the albedo before denoising and multiply back after (keeps texture detail).}
}
\value{the denoised layer as a numeric vector.}
