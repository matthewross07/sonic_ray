#!/usr/bin/env Rscript
# Parity test: the EXACT Cache la Poudre v2 (60 m) mesh + texture on the Metal GPU vs the CPU/rayrender layers that made the film.
.libPaths(c(path.expand("~/Developer/rlib"), .libPaths()))
suppressPackageStartupMessages({ library(sonicray); library(png); library(jsonlite) })
CACHE <- path.expand(Sys.getenv("POUDRE_CACHE", "~/.cache/poudre_3d_v2_yaw256_60m")); REF <- file.path(CACHE, "layers_1920x1080")
W <- 1920L; H <- 1080L
t_ <- function(label, expr) { t0 <- Sys.time(); v <- force(expr); cat(sprintf("%-34s %6.2f s\n", label, as.numeric(difftime(Sys.time(), t0, units = "secs")))); v }
load_f32 <- function(n, ch = 3) aperm(array(readBin(file.path(REF, paste0(n, ".f32")), "numeric", n = W * H * ch, size = 4), c(ch, W, H)), c(3, 2, 1))

S    <- t_("read scene.rds", readRDS(file.path(CACHE, "scene.rds")))
meta <- fromJSON(file.path(REF, "scene_meta.json")); cam <- meta$cam
mesh <- t_("build mesh (identical to CPU)", sr_mesh_from_heightfield(S$H, S$xr, S$zr, S$Y0, S$VEXAG))
cat(sprintf("   %d vertices, %d triangles\n", nrow(mesh$vertices), nrow(mesh$indices)))
tex  <- t_("read 4608x4856 albedo PNG", readPNG(file.path(CACHE, "albedo.png")))
cat("   texture dims:", dim(tex), "\n")
alb  <- t_("linearise texture (^2.2)", tex[, , 1:3]^2.2)
gpu  <- t_("upload mesh + texture, build BVH", sr_upload_mesh(mesh, alb))

res <- t_("render ambient + AOVs @1080p 64spp", sonicray:::cpp_render_ambient_and_aovs(gpu$gpu_handle, as.numeric(cam$lookfrom), as.numeric(cam$lookat), cam$fov, W, H, 64L, 4.0))
to_arr <- function(v, ch) aperm(array(v, c(ch, W, H)), c(3, 2, 1))
g_amb <- to_arr(res$amb, 3); g_alb <- to_arr(res$albedo, 3); g_mask <- matrix(res$mask, H, W, byrow = TRUE)
c_amb <- load_f32("amb"); c_alb <- load_f32("alb"); c_mask <- load_f32("mask", 1)[, , 1]
sel <- c_mask > 0.99 & g_mask > 0.99
cmp <- function(g, c, nm) {
  gm <- as.vector(g[cbind(which(sel, arr.ind = TRUE), 1)]); 
  for (k in 1:3) { x <- g[, , k][sel]; y <- c[, , k][sel]
    cat(sprintf("%-8s ch%d  mean GPU %.4f | CPU %.4f | ratio %.3f | corr %.4f | RMSE/mean %.3f\n", nm, k, mean(x), mean(y), mean(x) / mean(y), cor(x, y), sqrt(mean((x - y)^2)) / mean(y))) }
}
cat(sprintf("\ninterior pixels compared: %d  (mask agreement: GPU covered %.1f%% vs CPU %.1f%%)\n", sum(sel), 100 * mean(g_mask > 0.5), 100 * mean(c_mask > 0.5)))
cmp(g_alb, c_alb, "albedo"); cmp(g_amb, c_amb, "ambient")
saveRDS(list(g_amb = g_amb, c_amb = c_amb, g_alb = g_alb, c_alb = c_alb, sel = sel), "/tmp/poudre_parity_amb.rds")
