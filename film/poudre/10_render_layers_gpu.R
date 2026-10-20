#!/usr/bin/env Rscript
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
# =============================================================================
# film/poudre/10_render_layers_gpu.R
# Re-creates every light layer of the Cache la Poudre v2 (60 m, yaw 256) film on the Metal GPU with sonicray, in the SAME .f32 format
# that poudre_3d_v2/02_compose_video.py reads -- so the existing compositor / Sonic Pi mux run unchanged on GPU layers.
#
#   Rscript film/poudre/10_render_layers_gpu.R [out_dir] [light_spp] [amb_spp]
# Inputs : the CPU pipeline's scene cache (scene.rds, albedo.png, rivers.png, edge.png, uv_u.png, uv_v.png) -- identical geometry/camera/textures.
# Reused : sky_0..6 and stars (skymodelr atmosphere, not path tracing) and scene_meta.json are copied from the reference cache.
# =============================================================================
suppressPackageStartupMessages({ library(sonicray); library(png); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
OUT      <- path.expand(if (length(args) >= 1) args[1] else "~/.cache/sonicray_poudre/layers_1920x1080")
LIGHT_SPP <- if (length(args) >= 2) as.integer(args[2]) else 512L
AMB_SPP   <- if (length(args) >= 3) as.integer(args[3]) else 1024L
DENOISE   <- identical(Sys.getenv("SONICRAY_DENOISE", "0"), "1")
CACHE <- path.expand(Sys.getenv("POUDRE_CACHE", "~/.cache/poudre_3d_v2_yaw256_60m")); REF <- file.path(CACHE, "layers_1920x1080")
W <- 1920L; H <- 1080L; dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
T0 <- Sys.time(); stamp <- function(label, t0) cat(sprintf("  %-44s %7.2f s\n", label, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
tick <- function(label, expr) { t0 <- Sys.time(); v <- force(expr); stamp(label, t0); invisible(v) }
save_layer <- function(arr_flat, name) writeBin(as.numeric(arr_flat), file.path(OUT, paste0(name, ".f32")), size = 4)

S <- readRDS(file.path(CACHE, "scene.rds")); meta <- fromJSON(file.path(REF, "scene_meta.json")); cam <- meta$cam
mesh <- tick("mesh (identical to the CPU pipeline)", sr_mesh_from_heightfield(S$H, S$xr, S$zr, S$Y0, S$VEXAG))
tex  <- tick("read albedo PNG", readPNG(file.path(CACHE, "albedo.png"))[, , 1:3])
gpu  <- tick("upload mesh + texture, build BVH", sr_upload_mesh(mesh, tex^2.2))

# ---- ambient + AOVs -------------------------------------------------------------------------------------------
aov <- tick(sprintf("ambient + AOVs @%d spp", AMB_SPP), sonicray:::cpp_render_ambient_and_aovs(gpu$gpu_handle, as.numeric(cam$lookfrom), as.numeric(cam$lookat), cam$fov, W, H, AMB_SPP, 4.0))
mask <- aov$mask
save_layer(aov$amb, "amb"); save_layer(mask, "mask"); save_layer(aov$albedo, "alb")

# ---- static emissive-texture layers (edge / rivers / position maps), sampled at each pixel's (u, v) ----------------
sample_tex <- function(img, u, v) {                       # bilinear, v = 1 is the top image row (same convention as the mesh uvs)
  h <- dim(img)[1]; w <- dim(img)[2]
  x <- pmin(pmax(u * w - 0.5, 0), w - 1.001); y <- pmin(pmax((1 - v) * h - 0.5, 0), h - 1.001)
  x0 <- floor(x); y0 <- floor(y); fx <- x - x0; fy <- y - y0
  out <- matrix(0, length(u), 3)
  for (k in 1:3) { m <- img[, , k]
    out[, k] <- (1 - fy) * ((1 - fx) * m[cbind(y0 + 1, x0 + 1)] + fx * m[cbind(y0 + 1, x0 + 2)]) + fy * ((1 - fx) * m[cbind(y0 + 2, x0 + 1)] + fx * m[cbind(y0 + 2, x0 + 2)]) }
  out
}
uvm <- matrix(aov$uv, ncol = 2, byrow = TRUE)
tick("edge / rivers / uv_u / uv_v layers", for (nm in c("edge", "riv", "uv_u", "uv_v")) {
  f <- c(edge = "edge.png", riv = "rivers.png", uv_u = "uv_u.png", uv_v = "uv_v.png")[[nm]]
  v <- sample_tex(readPNG(file.path(CACHE, f))[, , 1:3], uvm[, 1], uvm[, 2])^2.2 * mask        # rayrender decodes 8-bit textures with ~2.2 gamma
  save_layer(t(v), nm)                                                                          # (3 x npix) -> pixel-major rgb
})

# ---- light layers ---------------------------------------------------------------------------------------------
BEA <- list(h = 2.5, r = 0.30, I = 1100); FWK <- list(h = 5.2, r = 0.55, I = 2400); SUN_R <- 300
BURSTS <- list(toc = c(dx = -2.4, dz = 1.0, dh = 0.2), chla = c(dx = 2.4, dz = 1.0, dh = 0.7), alk = c(dx = 0.0, dz = -2.0, dh = -0.3))
s <- S$sites; cells <- S$cells
L <- list()
for (i in seq_len(nrow(s))) for (k in names(BURSTS)) { o <- BURSTS[[k]]
  L[[length(L) + 1]] <- data.frame(name = paste0("fwk_", k, "_", s$site[i]), x = s$x[i] + o[["dx"]], y = s$y[i] + FWK$h + o[["dh"]], z = s$z[i] + o[["dz"]], r = FWK$r, intensity = FWK$I) }
for (i in seq_len(nrow(cells))) L[[length(L) + 1]] <- data.frame(name = paste0("fwk_cell_", cells$id[i]), x = cells$x[i], y = cells$y[i] + FWK$h, z = cells$z[i], r = FWK$r, intensity = FWK$I)
for (i in seq_len(nrow(s))) L[[length(L) + 1]] <- data.frame(name = paste0("bea_", s$site[i]), x = s$x[i], y = s$y[i] + BEA$h, z = s$z[i], r = BEA$r, intensity = BEA$I)
SUN_EL <- c(6.0, 22.0, 42.0, 54.0, 26.0, 8.5, 2.0); AZ_COMP <- c(100, 122, 152, 185, 235, 264, 272)
for (k in 0:6) { e <- SUN_EL[k + 1] * pi / 180; a <- AZ_COMP[k + 1] * pi / 180
  L[[length(L) + 1]] <- data.frame(name = sprintf("sun_%d", k), x = SUN_R * cos(e) * cos(a), y = SUN_R * sin(e), z = SUN_R * cos(e) * sin(a), r = 9, intensity = 1500) }
lights <- do.call(rbind, L); rownames(lights) <- NULL
cat(sprintf("  %d light layers to render (%d site bursts, %d cells, %d beacons, 7 sun)\n", nrow(lights), 3 * nrow(s), nrow(cells), nrow(s)))
t_l <- Sys.time()
res <- sr_render_light_layers(gpu, lights, lookfrom = cam$lookfrom, lookat = cam$lookat, fov = cam$fov, width = W, height = H, amb_spp = 4L,
                              light_spp = LIGHT_SPP, max_bounces = 4L, clamp = 50, batch_size = 32L, denoise = DENOISE, out_dir = OUT)
stamp(sprintf("%d light layers @%d spp (batches of 32)", nrow(lights), LIGHT_SPP), t_l)
file.remove(file.path(OUT, "amb.f32"), file.path(OUT, "mask.f32"))   # sr_render_light_layers also wrote a 4-spp amb/mask: keep ours above
save_layer(aov$amb, "amb"); save_layer(mask, "mask")

# ---- reuse non-path-traced layers + metadata from the reference cache -------------------------------------------
for (f in c(sprintf("sky_%d.f32", 0:6), "stars.f32", "scene_meta.json")) file.copy(file.path(REF, f), file.path(OUT, f), overwrite = TRUE)
cat(sprintf("\nDONE: %d layer files in %s\nTOTAL GPU layer time: %.1f s (the CPU/rayrender render of the same layers took 39.3 min by file timestamps)\n", length(list.files(OUT)), OUT, as.numeric(difftime(Sys.time(), T0, units = "secs"))))
