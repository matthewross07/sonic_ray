# Helpers that connect the GPU path tracer (R) to the GPU compositor (Python): camera projection, texture sampling, layer files.

#' Project world points to pixel coordinates (same convention as the compositor)
#' @param xyz n x 3 matrix of scene coordinates (km). @param lookfrom,lookat camera. @param fov vertical field of view (degrees).
#' @param width,height image size. @return n x 2 matrix (px, py), y down.
#' @export
sr_project <- function(xyz, lookfrom, lookat, fov, width, height) {
  xyz <- matrix(xyz, ncol = 3)
  f <- lookat - lookfrom; f <- f / sqrt(sum(f^2))
  r <- c(f[2] * 0 - f[3] * 1, f[3] * 0 - f[1] * 0, f[1] * 1 - f[2] * 0); r <- r / sqrt(sum(r^2))   # cross(f, up = (0, 1, 0))
  u <- c(r[2] * f[3] - r[3] * f[2], r[3] * f[1] - r[1] * f[3], r[1] * f[2] - r[2] * f[1])
  P <- sweep(xyz, 2, lookfrom)
  xc <- P %*% r; yc <- P %*% u; zc <- pmax(P %*% f, 1e-3); tf <- tan(fov * pi / 360)
  cbind(px = (xc / (zc * tf * width / height) * 0.5 + 0.5) * width, py = (0.5 - yc / (zc * tf) * 0.5) * height)
}

#' Bilinear sample of a matrix at texture coordinates (v = 1 is the top row, like the mesh uvs)
#' @export
sr_sample_texture <- function(m, u, v) {
  h <- nrow(m); w <- ncol(m)
  x <- pmin(pmax(u * w - 0.5, 0), w - 1.001); y <- pmin(pmax((1 - v) * h - 0.5, 0), h - 1.001)
  x0 <- floor(x); y0 <- floor(y); fx <- x - x0; fy <- y - y0
  (1 - fy) * ((1 - fx) * m[cbind(y0 + 1, x0 + 1)] + fx * m[cbind(y0 + 1, x0 + 2)]) +
    fy * ((1 - fx) * m[cbind(y0 + 2, x0 + 1)] + fx * m[cbind(y0 + 2, x0 + 2)])
}

#' Path to the `sonicray` command-line tool (Python GPU compositor)
#' @export
sr_cli <- function() {
  p <- system.file("bin", "sonicray", package = "sonicray")
  if (!nzchar(p)) p <- normalizePath(file.path("inst", "bin", "sonicray"), mustWork = FALSE)
  if (!file.exists(p)) stop("cannot find the sonicray command line tool; is the package installed?")
  p
}

#' Check this Mac and print exact fixes (R side + Python/ffmpeg/Metal side)
#' @param json if TRUE, print machine-readable JSON (for AI assistants).
#' @return invisibly, TRUE if everything needed is in place.
#' @export
sr_doctor <- function(json = FALSE) {
  info <- tryCatch(sr_device_info(), error = function(e) NULL)
  r_ok <- !is.null(info) && isTRUE(info$available)
  if (!json) {
    cat("R side\n")
    cat(sprintf("  %s Metal GPU: %s\n", if (r_ok) "ok  " else "FAIL", if (r_ok) paste(info$name, if (isTRUE(info$supports_raytracing)) "(hardware ray tracing)" else "(no ray tracing)") else "not available (needs an Apple Silicon Mac)"))
    cat(sprintf("  %s Intel OIDN denoiser: %s\n", if (isTRUE(info$oidn_available)) "ok  " else "info", if (isTRUE(info$oidn_available)) "found" else "optional; brew install open-image-denoise, then reinstall sonicray"))
    cat("Python / GPU compositor side\n")
  }
  rc <- system2(sr_cli(), c("doctor", if (json) "--json"))
  invisible(r_ok && rc == 0)
}

#' Print the questions an assistant should ask the user before rendering (colours, quality, speed, ...)
#' @param json machine-readable output.
#' @export
sr_intake <- function(json = FALSE) invisible(system2(sr_cli(), c("intake", if (json) "--json")))

#' Compose a film from light layers on the GPU
#' @param layers directory with `scene_meta.json` and `*.f32` layers. @param out output .mp4 (use a NEW name).
#' @param story "network" (data-driven, see docs/BRING_YOUR_OWN_DATA.md) or "poudre". @param quality "draft", "review" or "final".
#' @param ... extra command-line flags, e.g. `"--t0", "0", "--t1", "5"`.
#' @export
sr_compose <- function(layers, out, story = "network", quality = "draft", ...) {
  rc <- system2(sr_cli(), c("compose", "--layers", shQuote(path.expand(layers)), "--story", story, "--out", shQuote(path.expand(out)), "--quality", quality, ...))
  if (rc != 0) stop("sonicray compose failed; run sr_doctor()")
  invisible(path.expand(out))
}

#' Write the scene description the compositor needs (width, height, camera, sensors with screen positions)
#' @keywords internal
sr_write_story_files <- function(dir, d, width, height, bursts, bea_h = 2.5) {
  cam <- d$cam; s <- d$sensors
  base <- sr_project(cbind(s$x, s$y, s$z), cam$lookfrom, cam$lookat, cam$fov, width, height)
  top  <- sr_project(cbind(s$x, s$y + bea_h * 1.6, s$z), cam$lookfrom, cam$lookat, cam$fov, width, height)
  sites <- lapply(seq_len(nrow(s)), function(i) list(site = s$site[i], x = s$x[i], y = s$y[i], z = s$z[i], px = base[i, 1], py = base[i, 2],
                  beam = max(base[i, 2] - top[i, 2], 8), lighthouse = FALSE,
                  bursts = lapply(names(bursts), function(k) list(color = k, x = s$x[i] + bursts[[k]][["dx"]], y = s$y[i] + 5.2 + bursts[[k]][["dh"]], z = s$z[i] + bursts[[k]][["dz"]]))))
  jsonlite::write_json(list(width = width, height = height, cam = cam, fwk_h = 5.2, sites = sites, cells = list(), sun_path = list()),
                       file.path(dir, "scene_meta.json"), auto_unbox = TRUE, digits = 8)
  jsonlite::write_json(list(duration_s = d$duration_s,
                            parameters = list(param_a = list(plane = 0, glow = c(1.00, 0.14, 0.12)), param_b = list(plane = 1, glow = c(0.20, 0.96, 0.32)),
                                              param_c = list(plane = 2, glow = c(0.36, 0.14, 1.00)))),
                       file.path(dir, "story.json"), auto_unbox = TRUE)
  utils::write.csv(s[, c("site", "online_s")], file.path(dir, "sensors.csv"), row.names = FALSE, quote = FALSE)
  utils::write.csv(d$events, file.path(dir, "events.csv"), row.names = FALSE, quote = FALSE)
  invisible(dir)
}
