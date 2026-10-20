#!/usr/bin/env Rscript
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
# Ground truth for a sphere light over a flat Lambertian plane, vs rayrender (CPU) and sonicray (GPU).
#   irradiance at nadir  E = pi * L * sin(alpha)^2   (alpha = asin(r/h)),  outgoing radiance  Lo = rho/pi * E  = rho * L * sin(alpha)^2
suppressPackageStartupMessages({ library(sonicray); library(rayrender) })
rho <- 0.5; Lrad <- 1000; h <- 3; rad <- 0.5; W <- 400L; Hh <- 300L
cam <- list(lookfrom = c(0, 6, 10), lookat = c(0, 0, 0), fov = 40)
# --- GPU: flat plane as a heightfield mesh
Hm <- matrix(1000, 60, 60); mesh <- sr_mesh_from_heightfield(Hm, c(-12, 12), c(-12, 12), y0 = 1000, vexag = 1)
gpu <- sr_upload_mesh(mesh, array(rho, c(8, 8, 3)))
lt <- data.frame(name = "L", x = 0, y = h, z = 0, r = rad, intensity = Lrad)
g <- sr_render_light_layers(gpu, lt, lookfrom = cam$lookfrom, lookat = cam$lookat, fov = cam$fov, width = W, height = Hh,
                            amb_spp = 4L, light_spp = 1024L, max_bounces = 1L, clamp = 1e4, batch_size = 1L, denoise = FALSE)$layers[[1]]
G <- aperm(array(g, c(3, W, Hh)), c(3, 2, 1))
# --- CPU: rayrender, same plane / light / camera
scene <- xz_rect(xwidth = 24, zwidth = 24, y = 0, material = diffuse(color = c(rho, rho, rho))) %>%
  add_object(sphere(x = 0, y = h, z = 0, radius = rad, material = light(color = "#ffffff", intensity = Lrad, importance_sample = TRUE, invisible = TRUE)))
C <- render_scene(scene, width = W, height = Hh, samples = 1024, lookfrom = cam$lookfrom, lookat = cam$lookat, fov = cam$fov,
                  ambient_light = FALSE, backgroundhigh = "#000000", backgroundlow = "#000000", tonemap = "raw", bloom = FALSE, denoise = FALSE,
                  return_raw_array = TRUE, preview = FALSE, verbose = FALSE, progress = FALSE, integrator_type = "nee", clamp_value = Inf)[, , 1:3]
# pixel of the plane point directly below the light (0,0,0) in this camera:
f <- cam$lookat - cam$lookfrom; f <- f / sqrt(sum(f^2)); r <- c(-f[3], 0, f[1]); r <- r / sqrt(sum(r^2)); u <- c(r[2]*f[3]-r[3]*f[2], r[3]*f[1]-r[1]*f[3], r[1]*f[2]-r[2]*f[1])
P <- c(0, 0, 0) - cam$lookfrom; xc <- sum(P * r); yc <- sum(P * u); zc <- sum(P * f); tf <- tan(cam$fov * pi / 360)
px <- round(((xc / (zc * tf * W / Hh)) * 0.5 + 0.5) * W); py <- round((0.5 - yc / (zc * tf) * 0.5) * Hh)
nadir <- function(A) mean(A[(py - 1):(py + 1), (px - 1):(px + 1), 2])
alpha <- asin(rad / h); expected <- rho * Lrad * sin(alpha)^2
cat(sprintf("nadir pixel (%d,%d)\n  analytic  Lo = rho*L*sin^2(alpha) = %.3f\n  rayrender (CPU)               = %.3f   (%.2fx analytic)\n  sonicray  (GPU)               = %.3f   (%.2fx analytic)\n",
            px, py, expected, nadir(C), nadir(C) / expected, nadir(G), nadir(G) / expected))
# falloff with distance along the plane (centerline): ratio GPU/CPU at several rows
rows <- seq(py - 60, py + 90, by = 30); cat("row  CPU    GPU    GPU/CPU\n"); for (i in rows) cat(sprintf("%3d  %.3f  %.3f  %.3f\n", i, mean(C[i, px, ]), mean(G[i, px, ]), mean(G[i, px, ]) / mean(C[i, px, ])))
