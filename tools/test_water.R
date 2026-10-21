#!/usr/bin/env Rscript
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
# Closed-form checks of the water renderer on a flat lakebed (no external data). Exits with an error if any check is off by more than its tolerance.
suppressPackageStartupMessages(library(sonicray))
fres <- function(cos_i, n_i, n_t) { s2 <- (n_i / n_t)^2 * (1 - cos_i^2); if (s2 >= 1) return(1); ct <- sqrt(1 - s2)
  rs <- (n_i * cos_i - n_t * ct) / (n_i * cos_i + n_t * ct); rp <- (n_t * cos_i - n_i * ct) / (n_t * cos_i + n_i * ct); 0.5 * (rs^2 + rp^2) }
bed <- function(water) {
  mesh <- sr_mesh_from_heightfield(matrix(0, 4, 4), c(-100, 100), c(-100, 100), y0 = 0, vexag = 1)
  sr_upload_mesh(mesh, array(0.5, c(4, 4, 3)), water = water)
}
render <- function(gpu, lights, from, at, fov, spp, bounces = 1L) {
  sr_render_light_layers(gpu, lights, lookfrom = from, lookat = at, fov = fov, width = 64L, height = 64L, amb_spp = 64L, light_spp = spp, max_bounces = bounces, clamp = 1e6, batch_size = 1L)
}
centre <- function(v) { m <- array(v, c(3, 64, 64)); rowMeans(m[, 29:36, 29:36]) }
ok <- TRUE
check <- function(name, got, want, tol) { err <- max(abs(got / want - 1)); good <- err < tol; ok <<- ok && good
  cat(sprintf("%-46s %s  worst relative error %.2f%% (tolerance %.0f%%)\n", name, if (good) "PASS" else "FAIL", 100 * err, 100 * tol)) }

# ---- 1. Beer-Lambert + Fresnel: camera straight down on a lakebed under depth d; light at 45 degrees
d <- 2; sig <- c(0.5, 0.2, 0.05); light <- data.frame(name = "L", x = 60, y = 60, z = 0, r = 5, intensity = 1000)
from <- c(0, 40, 0.001); at <- c(0, 0, 0)
dry <- render(bed(NULL), light, from, at, 1, 4096L)$layers[[1]]
wet <- render(bed(list(level = d, absorption = sig)), light, from, at, 1, 4096L)$layers[[1]]
F0 <- fres(1, 1, 1.333); F45 <- fres(cos(pi / 4), 1, 1.333)
want <- (1 - F0) * exp(-sig * d) * (1 - F45) * exp(-sig * d * sqrt(2))
check("1. wet/dry radiance = (1-F0) e^-kd (1-F45) e^-kd.sqrt2", centre(wet) / centre(dry), want, 0.03)

# ---- 2. Fresnel reflection of an emitter at 60 degrees off a deep (opaque-looking) lake
gpu2 <- bed(list(level = 5, absorption = rep(50, 3)))
dir_in <- c(0, -cos(pi / 3), -sin(pi / 3)); hitp <- c(0, 5, 0); from2 <- hitp - 40 * dir_in; dir_out <- c(0, -dir_in[2], dir_in[3])
em <- data.frame(name = "E", x = hitp[1] + 100 * dir_out[1], y = hitp[2] + 100 * dir_out[2], z = hitp[3] + 100 * dir_out[3], r = 30, intensity = 1000)
r2 <- render(gpu2, em, from2, hitp, 1, 8192L)$layers[[1]]
check("2. reflected emitter = F(60 degrees) x radiance", centre(r2), rep(fres(cos(pi / 3), 1, 1.333) * 1000, 3), 0.03)

# ---- 3. water-depth output equals the depth at normal incidence
dep <- render(bed(list(level = d, absorption = sig)), light, from, at, 1, 4L)$aovs$water_depth
got <- mean(array(dep, c(64, 64))[29:36, 29:36]); check("3. water_depth AOV = depth", got, d, 0.02)
# ---- 4. single scattering: a deep scattering lake seen from above; light 10 degrees off vertical, far away
#   L = (1-F0) . sigma_s/(4 pi) . Omega L_e . (1-F(10 deg)) / (sigma_t (1 + 1/cos 10 deg))      (bed negligible: sigma_t D = 10)
th <- 10 * pi / 180; Rr <- 1000; st <- 0.5; ss <- 0.3; D <- 20
gp4 <- bed(list(level = D, absorption = rep(st - ss, 3), scattering = rep(ss, 3)))
li <- data.frame(name = "S", x = Rr * sin(th), y = D + Rr * cos(th), z = 0, r = 5, intensity = 1000)
r4 <- render(gp4, li, c(0, D + 40, 0.001), c(0, 0, 0), 1, 16384L)$layers[[1]]
omega <- 2 * pi * (1 - sqrt(1 - (5 / Rr)^2))
want4 <- (1 - F0) * (ss / (4 * pi)) * omega * 1000 * (1 - fres(cos(th), 1, 1.333)) / (st * (1 + 1 / cos(th)))
check("4. single-scatter radiance (deep scattering lake)", centre(r4), rep(want4, 3), 0.05)
if (!ok) stop("water checks failed")
cat("all water checks passed\n")
