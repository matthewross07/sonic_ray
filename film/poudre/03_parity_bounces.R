#!/usr/bin/env Rscript
# Re-test GPU light layers vs the saved rayrender CPU layers for several bounce counts.
.libPaths(c(path.expand("~/Developer/rlib"), .libPaths()))
suppressPackageStartupMessages({ library(sonicray); library(png); library(jsonlite) })
CACHE <- path.expand("~/.cache/poudre_3d_v2_yaw256_60m"); W <- 1920L; H <- 1080L
S <- readRDS(file.path(CACHE, "scene.rds")); cam <- fromJSON(file.path(CACHE, "layers_1920x1080", "scene_meta.json"))$cam
G0 <- readRDS("/tmp/poudre_parity_lights_gpu.rds"); lights <- G0$lights; cpu <- readRDS("/tmp/poudre_parity_lights_cpu.rds")
mesh <- sr_mesh_from_heightfield(S$H, S$xr, S$zr, S$Y0, S$VEXAG); gpu <- sr_upload_mesh(mesh, readPNG(file.path(CACHE, "albedo.png"))[, , 1:3]^2.2)
to_arr <- function(v) aperm(array(v, c(3, W, H)), c(3, 2, 1))
cat(sprintf("%-16s %-8s %7s | %9s | %6s | %9s | %12s\n", "layer", "bounces", "sec", "GPU/CPU E", "corr", "RMSE/mean", "median ratio"))
for (b in c(1L, 3L, 6L)) {
  t0 <- Sys.time()
  g <- sr_render_light_layers(gpu, lights, lookfrom = cam$lookfrom, lookat = cam$lookat, fov = cam$fov, width = W, height = H,
                              amb_spp = 8L, light_spp = 256L, max_bounces = b, clamp = 50, batch_size = 3L, denoise = FALSE)$layers
  sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  for (i in seq_len(nrow(lights))) {
    G <- to_arr(g[[lights$name[i]]]); C <- cpu[[lights$name[i]]]
    sel <- apply(C, c(1, 2), max) > 0.01 * max(C); x <- G[sel]; y <- C[sel]
    cat(sprintf("%-16s %-8d %7.2f | %9.3f | %6.4f | %9.3f | %12.3f\n", lights$name[i], b, sec, sum(x) / sum(y), cor(x, y), sqrt(mean((x - y)^2)) / mean(y), median(x / pmax(y, 1e-9))))
  }
}
