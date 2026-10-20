# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
# ==============================================================================
# sonicray/R/scene.R
# DEM -> Slope-Gated Heightfield Mesh, Smooth Vertex Normals, Highlighted River
# Masks, and Zero-Copy Upload to Apple Silicon Metal Acceleration Structures.
# ==============================================================================

#' Check if Apple Silicon Metal 3 Hardware Ray Tracing is available
#' @export
sr_metal_available <- function() {
  info <- sr_device_info()
  isTRUE(info$available) && isTRUE(info$supports_raytracing)
}

#' Query Apple Silicon GPU, Unified Memory, and Intel OIDN capabilities
#' @export
sr_device_info <- function() {
  cpp_metal_device_info()
}

#' Build a triangle mesh from a heightfield matrix (same vertex/uv/triangle layout as the CPU rayrender pipelines)
#'
#' Row 1 of `H` is the first z (north-most / far) row. Vertices are ordered column-major, uv = ((col-0.5)/nc, 1-(row-0.5)/nr),
#' and each grid cell becomes two triangles with the winding used by `poudre_3d_v2/lib_scene.R`, so that a GPU render and a
#' rayrender render of the same scene see identical geometry.
#' @param H numeric matrix (nr x nc) of elevations in metres.
#' @param xr,zr length-2 ranges of the x and z axes (km).
#' @param y0 elevation (m) mapped to y = 0.   @param vexag vertical exaggeration.
#' @export
sr_mesh_from_heightfield <- function(H, xr, zr, y0 = min(H), vexag = 1) {
  nr <- nrow(H); nc <- ncol(H)
  xs <- seq(xr[1], xr[2], length.out = nc); zs <- seq(zr[1], zr[2], length.out = nr)
  ys <- (H - y0) / 1000 * vexag
  idx <- matrix(seq_len(nr * nc), nr, nc)
  vertices <- cbind(rep(xs, each = nr), as.vector(ys), rep(zs, times = nc))
  uvs <- cbind(rep((seq_len(nc) - 0.5) / nc, each = nr), 1 - rep((seq_len(nr) - 0.5) / nr, times = nc))
  a <- as.vector(idx[-nr, -nc]); b <- as.vector(idx[-nr, -1]); c <- as.vector(idx[-1, -nc]); d <- as.vector(idx[-1, -1])
  tri <- rbind(cbind(a, c, b), cbind(b, c, d)) - 1L          # 0-based for Metal
  storage.mode(tri) <- "integer"
  list(vertices = vertices, normals = cpp_vertex_normals(vertices, tri), uvs = uvs, indices = tri, mesh_dims = c(nr, nc))
}

#' Upload a mesh + an albedo image to a persistent Metal acceleration structure
#'
#' @param mesh output of [sr_mesh_from_heightfield()].
#' @param albedo numeric array (h x w x 3) of *linear* albedo in [0, 1]. For rayrender parity pass `png::readPNG(f)^2.2`
#'   (rayrender decodes 8-bit image textures with a ~2.2 gamma).
#' @export
sr_upload_mesh <- function(mesh, albedo) {
  shader_path <- system.file("metal", "sonicray_pathtracer.metal", package = "sonicray")
  if (!nzchar(shader_path)) shader_path <- file.path("inst", "metal", "sonicray_pathtracer.metal")
  src <- paste(readLines(shader_path, warn = FALSE), collapse = "\n")
  d <- dim(albedo); th <- d[1]; tw <- d[2]
  rgba <- array(1, c(4, tw, th))                                   # channel fastest, then x, then y (row-major image)
  for (k in 1:3) rgba[k, , ] <- t(albedo[, , k])
  xptr <- cpp_upload_scene(mesh$vertices, mesh$normals, mesh$uvs, mesh$indices, as.numeric(rgba), as.integer(tw), as.integer(th),
                           rep(0, tw * th * 2), src)
  structure(list(gpu_handle = xptr, n_vertices = nrow(mesh$vertices), n_triangles = nrow(mesh$indices), tex_dims = c(tw, th)),
            class = "sonicray_gpu_scene")
}
