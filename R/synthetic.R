# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
# ==============================================================================
# sonicray/R/synthetic.R
# SYNTHETIC demo data: invented terrain, invented sensors, invented samples.
# Nothing here is a real place or a real measurement. It exists so that a new user (or an AI assistant) can run the whole
# pipeline in about a minute with no external data. To use real data see docs/BRING_YOUR_OWN_DATA.md.
# ==============================================================================

smoothstep <- function(e0, e1, x) { t <- pmin(pmax((x - e0) / (e1 - e0), 0), 1); t * t * (3 - 2 * t) }

# bilinear resize of a numeric matrix (rows = image rows)
resize_mat <- function(m, nr, nc) {
  lin <- function(n_out, n_in) {
    p <- (seq_len(n_out) - 0.5) / n_out * n_in + 0.5
    i0 <- pmin(pmax(floor(p), 1L), n_in - 1L); f <- pmin(pmax(p - i0, 0), 1)
    W <- matrix(0, n_out, n_in)
    W[cbind(seq_len(n_out), i0)] <- 1 - f; W[cbind(seq_len(n_out), i0 + 1L)] <- f
    W
  }
  lin(nr, nrow(m)) %*% m %*% t(lin(nc, ncol(m)))
}

# spectral-synthesis fractal noise on an n x n grid (n even), scaled to [0, 1]
fbm_noise <- function(n, beta) {
  k <- c(0:(n / 2), -(n / 2 - 1):-1)
  K <- sqrt(outer(k^2, k^2, "+")); K[1, 1] <- 1
  z <- Re(stats::fft(stats::fft(matrix(stats::rnorm(n * n), n, n)) * K^(-beta / 2), inverse = TRUE))
  (z - min(z)) / (max(z) - min(z))
}

#' Synthetic demo data (invented terrain, sensors and samples)
#'
#' Generates a fractal river-valley terrain, a river and lakes, \code{n_sensors} invented sensors along the river, and a table of
#' invented "samples" (three made-up parameters, \code{param_a/b/c}) with made-up seasonal values. **Everything is synthetic**;
#' it is only an example of the *shape* of data the package expects. See \code{docs/BRING_YOUR_OWN_DATA.md} to replace it with real
#' sensors, real coordinates and a real digital elevation model.
#'
#' @param seed random seed (the demo is fully reproducible).
#' @param n_sensors number of invented sensors.
#' @param duration_s length of the film in seconds (sets the event times).
#' @param n terrain grid size (even number).
#' @param tex albedo texture size in pixels.
#' @return a list of class \code{sonicray_demo_data} with elements \code{H} (elevation matrix, metres), \code{xr}, \code{zr}
#'   (km), \code{y0}, \code{vexag}, \code{albedo} (h x w x 3, linear), \code{river} and \code{water} (masks, \code{n x n}),
#'   \code{sensors} (data frame), \code{events} (data frame), \code{cam}, \code{synthetic = TRUE}.
#' @export
sr_demo_data <- function(seed = 7L, n_sensors = 12L, duration_s = 18, n = 360L, tex = 2048L) {
  set.seed(seed)
  r  <- matrix(rep(seq(0, 1, length.out = n), times = n), n, n)   # row fraction: 0 = far, 1 = near the camera
  cx <- matrix(rep(seq(0, 1, length.out = n), each = n), n, n)
  river_x <- function(rr) 0.5 + 0.13 * sin(2 * pi * (1.25 * rr + 0.15)) * (0.4 + 0.6 * rr)
  d   <- abs(cx - river_x(r))
  fb  <- fbm_noise(n, 2.4); rd <- 1 - abs(2 * fbm_noise(n, 2.0) - 1)
  env <- pmin(1, 0.18 + 1.15 * (1 - r)^1.5 + 1.6 * (2 * abs(cx - 0.5))^1.4)
  H <- 1500 * env * (0.55 * fb + 0.45 * rd) + 120 * fb
  H <- 60 + H * (0.10 + 0.90 * smoothstep(0.015, 0.20, d))          # low, flat valley floor around the river
  lakes <- Reduce(`+`, lapply(list(c(0.24, 0.82), c(0.76, 0.72), c(0.33, 0.52)), function(p) exp(-((cx - p[1])^2 / 0.0016 + (r - p[2])^2 / 0.0030))))
  H <- H - 90 * pmin(lakes, 1)
  river <- 1 - smoothstep(0.0045 + 0.008 * r, 0.0075 + 0.011 * r, d)
  water <- pmax(river, smoothstep(0.35, 0.55, lakes))

  # albedo texture (sRGB-ish colours -> linear)
  e <- (H - min(H)) / (max(H) - min(H))
  gy <- (H[c(2:n, n), ] - H[c(1, 1:(n - 1)), ]); gx <- (H[, c(2:n, n)] - H[, c(1, 1:(n - 1))])
  slope <- pmin(sqrt(gx^2 + gy^2) / 60, 1)
  mix3 <- function(a, b, w) array(c(outer(1 - w, a) + outer(w, b)), c(dim(w), 3))
  col <- mix3(c(0.17, 0.24, 0.12), c(0.30, 0.26, 0.19), smoothstep(0.05, 0.30, e))
  rockw <- pmax(smoothstep(0.30, 0.70, e), slope * 0.9)
  for (k in 1:3) col[, , k] <- col[, , k] * (1 - rockw) + c(0.42, 0.38, 0.35)[k] * rockw * (0.8 + 0.4 * fb)
  snow <- smoothstep(0.62, 0.78, e) * (1 - 0.6 * slope)
  for (k in 1:3) col[, , k] <- col[, , k] * (1 - snow) + c(0.88, 0.90, 0.94)[k] * snow
  for (k in 1:3) col[, , k] <- col[, , k] * (1 - water) + c(0.04, 0.14, 0.38)[k] * water
  albedo <- array(0, c(tex, tex, 3))
  for (k in 1:3) albedo[, , k] <- pmin(pmax(resize_mat(col[, , k], tex, tex), 0), 1)^2.2

  xr <- c(-30, 30); zr <- c(-30, 30); y0 <- min(H); vexag <- 2.25
  rs <- seq(0.93, 0.10, length.out = n_sensors)
  xs <- river_x(rs) + 0.018 * sin(seq_len(n_sensors) * 2.1)
  hy <- vapply(seq_len(n_sensors), function(i) H[round(rs[i] * (n - 1)) + 1, round(xs[i] * (n - 1)) + 1], 0)
  sensors <- data.frame(site = sprintf("s%02d", seq_len(n_sensors)), x = xr[1] + xs * diff(xr), z = zr[1] + rs * diff(zr),
                        y = (hy - y0) / 1000 * vexag, online_s = 1.2 + (seq_len(n_sensors) - 1) * 0.55, stringsAsFactors = FALSE)
  params <- c("param_a", "param_b", "param_c")
  ev <- do.call(rbind, lapply(seq_len(n_sensors), function(i) {
    m <- max(3L, round((duration_s - sensors$online_s[i] - 3) * 0.7))
    t <- sort(stats::runif(m, sensors$online_s[i] + 0.8, duration_s - 2.2))
    p <- sample(params, m, replace = TRUE)
    ph <- c(param_a = 0, param_b = 2.1, param_c = 4.2)[p]
    v <- pmin(pmax(0.5 + 0.4 * sin(2 * pi * 1.4 * t / duration_s + ph + i) + 0.12 * stats::rnorm(m), 0), 1)
    data.frame(site = sensors$site[i], parameter = p, time_s = round(t, 3), value = round(v, 3), stringsAsFactors = FALSE)
  }))
  ev <- ev[order(ev$time_s), ]; rownames(ev) <- NULL
  structure(list(H = H, xr = xr, zr = zr, y0 = y0, vexag = vexag, albedo = albedo, river = river, water = water, sensors = sensors, events = ev,
                 parameters = params, duration_s = duration_s, cam = list(lookfrom = c(0, 9.5, 46), lookat = c(0, 1.2, -4), fov = 26),
                 synthetic = TRUE), class = "sonicray_demo_data")
}

#' @export
print.sonicray_demo_data <- function(x, ...) {
  cat("<sonicray_demo_data> SYNTHETIC: invented terrain, sensors and samples (not real measurements)\n")
  cat(sprintf("  terrain %d x %d, %d sensors, %d samples over %.0f s\n", nrow(x$H), ncol(x$H), nrow(x$sensors), nrow(x$events), x$duration_s))
  invisible(x)
}
