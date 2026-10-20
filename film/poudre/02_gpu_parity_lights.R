#!/usr/bin/env Rscript
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
# Light-layer parity: GPU (sonicray) vs CPU (rayrender) for the same single lights on the exact Poudre v2 (60 m) scene.
suppressPackageStartupMessages({ library(sonicray); library(png); library(jsonlite) })
VIS <- Sys.getenv("VISUALS")   # path to your visuals folder (set the VISUALS environment variable)
CACHE <- path.expand("~/.cache/poudre_3d_v2_yaw256_60m"); Sys.setenv(POUDRE3D_CACHE = CACHE)
W <- 1920L; H <- 1080L
S <- readRDS(file.path(CACHE, "scene.rds")); cam <- fromJSON(file.path(CACHE, "layers_1920x1080", "scene_meta.json"))$cam
# ---- the three test lights, defined exactly as in poudre_3d_v2/01_render_layers.R ----
s <- S$sites[S$sites$site == "pman", ]
lights <- data.frame(
  name = c("burst_toc_pman", "beacon_pman", "sun_2"),
  x = c(s$x - 2.4, s$x, 300 * cos(42 * pi / 180) * cos(152 * pi / 180)),
  y = c(s$y + 5.2 + 0.2, s$y + 2.5, 300 * sin(42 * pi / 180)),
  z = c(s$z + 1.0, s$z, 300 * cos(42 * pi / 180) * sin(152 * pi / 180)),
  r = c(0.55, 0.30, 9), intensity = c(2400, 1100, 1500), stringsAsFactors = FALSE)
print(lights)
# ---- GPU ----
mesh <- sr_mesh_from_heightfield(S$H, S$xr, S$zr, S$Y0, S$VEXAG)
gpu  <- sr_upload_mesh(mesh, readPNG(file.path(CACHE, "albedo.png"))[, , 1:3]^2.2)
for (b in c(1L, 2L)) {                                    # bounces: 1 = direct only, 2 = direct + 1 indirect (current kernel)
  t0 <- Sys.time()
  g <- sr_render_light_layers(gpu, lights, lookfrom = cam$lookfrom, lookat = cam$lookat, fov = cam$fov, width = W, height = H,
                              amb_spp = 8L, light_spp = 256L, max_bounces = b, batch_size = 3L, denoise = FALSE)
  cat(sprintf("GPU  max_bounces=%d: 3 layers @1080p 256 spp in %.2f s\n", b, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  assign(paste0("gpu_b", b), g$layers)
}
saveRDS(list(gpu_b1 = gpu_b1, gpu_b2 = gpu_b2, lights = lights), "/tmp/poudre_parity_lights_gpu.rds")
# ---- CPU (rayrender), same recipe as the film pipeline (160 spp, adaptive) ----
suppressMessages(source(file.path(VIS, "poudre_3d_v2", "lib_scene.R")))
terr <- terrain_obj()
cpu <- list()
for (i in seq_len(nrow(lights))) {
  l <- sphere(x = lights$x[i], y = lights$y[i], z = lights$z[i], radius = lights$r[i],
              material = light(color = "#ffffff", intensity = lights$intensity[i], importance_sample = TRUE, invisible = TRUE))
  t0 <- Sys.time()
  a <- rr(terr %>% add_object(l), W, H, 160L, ambient_light = FALSE, backgroundhigh = "#000000", backgroundlow = "#000000",
          clamp_value = if (i == 3) 25 else 12, min_variance = 1.5e-5)
  cat(sprintf("CPU  %-16s 160 spp adaptive @1080p: %.1f s\n", lights$name[i], as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  cpu[[lights$name[i]]] <- a[, , 1:3]
}
saveRDS(cpu, "/tmp/poudre_parity_lights_cpu.rds")
to_arr <- function(v) aperm(array(v, c(3, W, H)), c(3, 2, 1))
cat("\n=== parity on lit pixels (CPU > 1% of its max) ===\n")
for (i in seq_len(nrow(lights))) for (b in 1:2) {
  g <- to_arr(get(paste0("gpu_b", b))[[lights$name[i]]]); c <- cpu[[lights$name[i]]]
  sel <- apply(c, c(1, 2), max) > 0.01 * max(c)
  x <- g[sel]; y <- c[sel]
  cat(sprintf("%-16s bounces=%d  lit px %7d | total energy GPU/CPU %.3f | corr %.4f | RMSE/mean %.3f\n", lights$name[i], b, sum(sel), sum(x) / sum(y), cor(x, y), sqrt(mean((x - y)^2)) / mean(y)))
}
