#' Run the whole pipeline on SYNTHETIC data (about a minute at draft quality)
#'
#' Invented terrain + invented sensors and samples -> GPU path-traced light layers -> GPU compositor -> mp4. The data are synthetic;
#' the point is to show the pipeline and the file formats. For real data see `docs/BRING_YOUR_OWN_DATA.md`.
#' @param out_dir where layers and the film go (created).
#' @param quality "draft" (960x540), "review" (1280x720) or "final" (1920x1080).
#' @param compose set FALSE to stop after the light layers.
#' @param seed random seed of the synthetic data.
#' @return invisibly, the path of the mp4 (or the layers directory when `compose = FALSE`).
#' @export
sr_demo <- function(out_dir = "sonicray_demo", quality = c("draft", "review", "final"), compose = TRUE, seed = 7L) {
  quality <- match.arg(quality)
  P <- list(draft = list(w = 960L, h = 540L, amb = 64L, lig = 32L), review = list(w = 1280L, h = 720L, amb = 256L, lig = 128L),
            final = list(w = 1920L, h = 1080L, amb = 1024L, lig = 512L))[[quality]]
  if (!sr_metal_available()) stop("needs an Apple Silicon Mac with Metal ray tracing; run sr_doctor()")
  dir <- path.expand(out_dir); layers <- file.path(dir, paste0("layers_", P$w, "x", P$h)); dir.create(layers, recursive = TRUE, showWarnings = FALSE)
  tm <- function(label, expr) { t0 <- Sys.time(); v <- force(expr); message(sprintf("[sonicray demo] %-34s %6.1f s", label, as.numeric(difftime(Sys.time(), t0, units = "secs")))); v }
  message("[sonicray demo] SYNTHETIC data: invented terrain, sensors and samples (not real measurements)")
  d <- tm("synthetic terrain + sensors", sr_demo_data(seed = seed, duration_s = 18))
  mesh <- tm("mesh", sr_mesh_from_heightfield(d$H, d$xr, d$zr, d$y0, d$vexag))
  gpu  <- tm("upload + acceleration structure", sr_upload_mesh(mesh, d$albedo))
  bursts <- list(param_a = c(dx = -2.4, dz = 1.0, dh = 0.2), param_b = c(dx = 2.4, dz = 1.0, dh = 0.7), param_c = c(dx = 0.0, dz = -2.0, dh = -0.3))
  s <- d$sensors; L <- list()
  for (i in seq_len(nrow(s))) {
    for (k in names(bursts)) { o <- bursts[[k]]
      L[[length(L) + 1]] <- data.frame(name = paste0("fwk_", k, "_", s$site[i]), x = s$x[i] + o[["dx"]], y = s$y[i] + 5.2 + o[["dh"]], z = s$z[i] + o[["dz"]], r = 0.55, intensity = 2400) }
    L[[length(L) + 1]] <- data.frame(name = paste0("bea_", s$site[i]), x = s$x[i], y = s$y[i] + 2.5, z = s$z[i], r = 0.30, intensity = 1100)
  }
  lights <- do.call(rbind, L)
  res <- tm(sprintf("%d light layers", nrow(lights)), sr_render_light_layers(gpu, lights, lookfrom = d$cam$lookfrom, lookat = d$cam$lookat, fov = d$cam$fov,
              width = P$w, height = P$h, amb_spp = P$amb, light_spp = P$lig, max_bounces = 4L, clamp = 50, batch_size = 32L, denoise = isTRUE(sr_device_info()$oidn_available), out_dir = layers))
  # river mask (R = river, G = all water) and a sparse starfield, written in the compositor's layer format
  uv <- matrix(res$aovs$uv, ncol = 2, byrow = TRUE); mask <- as.numeric(res$aovs$mask)
  riv <- cbind(sr_sample_texture(d$river, uv[, 1], uv[, 2]), sr_sample_texture(d$water, uv[, 1], uv[, 2]), 0) * mask
  writeBin(as.numeric(t(riv)), file.path(layers, "riv.f32"), size = 4)
  set.seed(seed + 1); st <- numeric(P$w * P$h * 3); idx <- sample.int(P$w * P$h, round(P$w * P$h * 0.0008)); st[rep((idx - 1) * 3, each = 3) + 1:3] <- stats::runif(length(idx), 0.01, 0.045)
  st[rep((seq_len(P$w * P$h) - 1) * 3, each = 3) + 1:3][rep(mask > 0.5, each = 3)] <- 0
  writeBin(st, file.path(layers, "stars.f32"), size = 4)
  sr_write_story_files(layers, d, P$w, P$h, bursts)
  message("[sonicray demo] layers: ", layers)
  if (!compose) return(invisible(layers))
  out <- file.path(dir, sprintf("demo_%s.mp4", quality))
  tm("compose on the GPU", sr_compose(layers, out, story = "network", quality = quality))
  message("[sonicray demo] wrote ", out, "  (SYNTHETIC data)")
  invisible(out)
}
