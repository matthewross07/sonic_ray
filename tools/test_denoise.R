#!/usr/bin/env Rscript
# Does OIDN (via sonicray) actually reduce error?  noisy(8 spp) vs reference(512 spp), with and without denoising.
.libPaths(c(path.expand("~/Developer/rlib"), .libPaths()))
suppressPackageStartupMessages({ library(terra); library(sonicray) })
info <- sr_device_info(); print(unlist(info))
stopifnot(isTRUE(info$oidn_available))
r <- terra::rast(nrows = 600, ncols = 600, xmin = -106, xmax = -105, ymin = 40.3, ymax = 40.9)
xy <- terra::xyFromCell(r, 1:terra::ncell(r))
terra::values(r) <- 1500 + 1400 * pmax(0, -xy[, 1] - 105.1) + 280 * sin(xy[, 2] * 35) * cos(xy[, 1] * 25)
f <- tempfile(fileext = ".tif"); terra::writeRaster(r, f, overwrite = TRUE)
gpu <- sr_upload_scene(sr_scene_from_dem(f, mesh_cols = 820L, tex_width = 2048L, az_deg = 256))
lights <- data.frame(name = "L1", x = -2, y = 7, z = 10, r = 0.55, intensity = 2400)
W <- 960L; H <- 540L
run <- function(spp, dn) sr_render_light_layers(gpu, lights, width = W, height = H, amb_spp = spp, light_spp = spp, denoise = dn)
ref   <- run(512L, FALSE); noisy <- run(8L, FALSE); den <- run(8L, TRUE)
a <- ref$layers[[1]]; b <- noisy$layers[[1]]; d <- den$layers[[1]]
rmse <- function(x, y) sqrt(mean((x - y)^2)); sel <- a > 0.01 * max(a)
cat(sprintf("\nlight layer, lit pixels only (%d):\n  noisy 8spp   vs 512spp reference: RMSE %.4f\n  OIDN 8spp    vs 512spp reference: RMSE %.4f   (%.1fx lower)\n  mean: ref %.4f | noisy %.4f | denoised %.4f\n",
            sum(sel), rmse(b[sel], a[sel]), rmse(d[sel], a[sel]), rmse(b[sel], a[sel]) / rmse(d[sel], a[sel]), mean(a[sel]), mean(b[sel]), mean(d[sel])))
