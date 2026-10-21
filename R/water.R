# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
# Water: options for sr_upload_mesh(water = ...), plus small helpers for submerged objects.

# Turn the user's water list into the 10-number vector the C++ side takes: level, ior, roughness, cutaway, sigma_t[3], sigma_s[3].
water_vector <- function(water) {
  if (is.null(water)) return(numeric(0))
  if (is.null(water$level)) stop("water$level (the world y of the water surface) is required")
  mpu <- if (is.null(water$metres_per_unit)) 1 else water$metres_per_unit
  ab <- if (!is.null(water$absorption)) water$absorption else rep(if (is.null(water$kd)) 0 else water$kd, 3)
  sc <- if (is.null(water$scattering)) rep(0, 3) else water$scattering
  stopifnot(length(ab) == 3, length(sc) == 3, all(ab >= 0), all(sc >= 0), mpu > 0)
  ior <- if (is.null(water$ior)) 1.333 else water$ior
  if (ior <= 1) stop("water$ior must be greater than 1")
  c(water$level, ior, if (is.null(water$roughness)) 0 else water$roughness, if (isTRUE(water$cutaway)) 1 else 0, (ab + sc) * mpu, sc * mpu)
}

#' Illustrative water optics (per metre), to start from
#'
#' Rough, *uncalibrated* absorption and scattering coefficients (red, green, blue, per metre) for four common looks. They are starting points for
#' a picture, not measurements: calibrate against your own kd, Secchi depth, Chl-a, DOC or turbidity data.
#' @param type one of "clear", "algal", "brown" (high dissolved organic carbon) or "turbid" (suspended sediment).
#' @return a list with `absorption` and `scattering`, to merge into the `water` list of [sr_upload_mesh()] (set `metres_per_unit` too).
#' @export
sr_water_preset <- function(type = c("clear", "algal", "brown", "turbid")) {
  type <- match.arg(type)
  switch(type,
    clear  = list(absorption = c(0.30, 0.06, 0.015), scattering = c(0.002, 0.003, 0.005)),
    algal  = list(absorption = c(0.55, 0.15, 0.45),  scattering = c(0.20, 0.30, 0.25)),
    brown  = list(absorption = c(0.50, 0.35, 1.50),  scattering = c(0.05, 0.05, 0.05)),
    turbid = list(absorption = c(0.40, 0.20, 0.20),  scattering = c(1.50, 1.50, 1.50)))
}

#' Append a mesh (for example a submerged object) to another mesh
#' @param a,b meshes with `vertices`, `normals`, `uvs`, `indices` (0-based triangles), as returned by [sr_mesh_from_heightfield()] and [sr_sphere_mesh()].
#' @return the combined mesh. Objects take their colour from the albedo texture at their uv; see [sr_paint_albedo()].
#' @export
sr_mesh_append <- function(a, b) {
  off <- nrow(a$vertices)
  list(vertices = rbind(a$vertices, b$vertices), normals = rbind(a$normals, b$normals), uvs = rbind(a$uvs, b$uvs),
       indices = rbind(a$indices, b$indices + off), mesh_dims = a$mesh_dims)
}

#' A sphere mesh whose every vertex samples one point of the albedo texture
#' @param center length-3 centre in scene units. @param radius radius in scene units. @param uv texture coordinate (u, v) whose texel gives the colour.
#' @param n_lat,n_lon sphere resolution.
#' @return a mesh for [sr_mesh_append()].
#' @export
sr_sphere_mesh <- function(center, radius, uv = c(0.5, 0.5), n_lat = 16L, n_lon = 24L) {
  th <- seq(0, pi, length.out = n_lat + 1); ph <- seq(0, 2 * pi, length.out = n_lon + 1)
  g <- expand.grid(i = seq_len(n_lat + 1), j = seq_len(n_lon + 1))
  nx <- sin(th[g$i]) * cos(ph[g$j]); ny <- cos(th[g$i]); nz <- sin(th[g$i]) * sin(ph[g$j])
  nrm <- cbind(nx, ny, nz); vert <- sweep(nrm * radius, 2, center, "+")
  id <- function(i, j) (j - 1L) * (n_lat + 1L) + i - 1L
  ii <- rep(seq_len(n_lat), times = n_lon); jj <- rep(seq_len(n_lon), each = n_lat)
  tri <- rbind(cbind(id(ii, jj), id(ii + 1L, jj), id(ii, jj + 1L)), cbind(id(ii + 1L, jj), id(ii + 1L, jj + 1L), id(ii, jj + 1L)))
  storage.mode(tri) <- "integer"
  list(vertices = unname(vert), normals = unname(nrm), uvs = matrix(uv, nrow(vert), 2, byrow = TRUE), indices = tri, mesh_dims = c(0L, 0L))
}

#' Paint a small square of colour into the albedo texture (so an appended object has a colour)
#' @param albedo numeric array (h x w x 3, linear). @param uv texture coordinate (u, v) to paint at (v = 1 is the top row).
#' @param rgb linear colour. @param half_px half-size of the painted square in pixels.
#' @return the modified albedo array.
#' @export
sr_paint_albedo <- function(albedo, uv, rgb, half_px = 3L) {
  h <- dim(albedo)[1]; w <- dim(albedo)[2]
  r <- round((1 - uv[2]) * h); cc <- round(uv[1] * w)
  ri <- pmin(pmax((r - half_px):(r + half_px), 1L), h); ci <- pmin(pmax((cc - half_px):(cc + half_px), 1L), w)
  for (k in 1:3) albedo[ri, ci, k] <- rgb[k]
  albedo
}
