#!/usr/bin/env Rscript
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
# SYNTHETIC water demo: a cutaway lake with three submerged white spheres ("Secchi disks") at increasing depth, four water types side by side.
#   Rscript tools/water_demo.R [out.png]
suppressPackageStartupMessages(library(sonicray))
out <- { a <- commandArgs(TRUE); if (length(a)) a[1] else "water_demo.png" }
n <- 161L; xs <- seq(-10, 10, length.out = n)
# A basin lake (metres): bed about 0.3 m at the centre, banks rising to the rim; the water fills the middle
H <- 0.3 + 0.03 * outer(xs, xs, function(x, z) x^2 + z^2) + 0.05 * outer(xs, xs, function(x, z) sin(1.3 * x) * cos(1.1 * z))
level <- 2.2                                                                               # water surface (metres)
mesh <- sr_mesh_from_heightfield(H, c(-10, 10), c(-10, 10), y0 = 0, vexag = 1000)          # vexag 1000: one scene unit = one metre of height
alb <- array(0, c(512, 512, 3)); col <- c(0.50, 0.42, 0.30)^2.2
for (k in 1:3) alb[, , k] <- col[k]
disk <- list(c(-3.4, level - 0.45, 0.8), c(0.2, level - 0.95, -0.6), c(3.2, level - 1.5, 0.9))   # white spheres at 0.45, 0.95 and 1.5 m depth
uv <- c(0.002, 0.998); alb <- sr_paint_albedo(alb, uv, c(0.9, 0.9, 0.9)^2.2, half_px = 2L)
for (d in disk) mesh <- sr_mesh_append(mesh, sr_sphere_mesh(d, 0.34, uv = uv))
cam <- list(from = c(0, 10.5, 9.2), at = c(0, 0.8, -1.5), fov = 46); W <- 640L; Hh <- 360L
el <- 34 * pi / 180; az <- 262 * pi / 180                                                   # the sun is low in front of the camera: a glint on the water
sun <- data.frame(name = "sun", x = 300 * cos(el) * cos(az), y = 300 * sin(el), z = 300 * cos(el) * sin(az), r = 9, intensity = 1500)
types <- c("clear", "algal", "brown", "turbid")
tile <- function(type) {
  w <- c(sr_water_preset(type), list(level = level, cutaway = FALSE, roughness = 0.03, metres_per_unit = 1))
  gpu <- sr_upload_mesh(mesh, alb, water = w)
  r <- sr_render_light_layers(gpu, sun, lookfrom = cam$from, lookat = cam$at, fov = cam$fov, width = W, height = Hh, amb_spp = 256L, light_spp = 4096L, max_bounces = 4L, clamp = 60, batch_size = 1L)
  lin <- 14 * array(r$aovs$amb, c(3, W, Hh)) + 0.9 * array(r$layers[[1]], c(3, W, Hh))                    # moonlit sky fill + sun
  m <- array(r$aovs$mask, c(W, Hh)); lin <- aperm(lin, c(3, 2, 1)); mm <- t(m)
  x <- pmax(lin, 0); aces <- (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14); img <- pmin(pmax(aces, 0), 1)^(1 / 2.2)
  sky <- array(rep(c(0.05, 0.08, 0.16), each = Hh * W), c(Hh, W, 3)); for (k in 1:3) img[, , k] <- sky[, , k] * (1 - mm) + img[, , k] * mm
  img
}
tiles <- lapply(types, tile)
sheet <- array(0.04, c(2 * Hh + 3 * 6, 2 * W + 3 * 6, 3))
for (i in 1:4) { r0 <- ((i - 1) %/% 2) * (Hh + 6) + 6; c0 <- ((i - 1) %% 2) * (W + 6) + 6; sheet[r0 + seq_len(Hh), c0 + seq_len(W), ] <- tiles[[i]] }
png::writePNG(sheet, out); cat("wrote", out, "\n")
