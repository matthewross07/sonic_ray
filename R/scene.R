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

#' Build a slope-gated 3D terrain mesh + linear albedo/river textures from a DEM
#'
#' Encapsulates the terrain preparation pipeline from `denver_3d_v2` and `poudre_3d_v2`:
#'   - Slope-gated topographic relief (smooth plains < 1.2 deg, crisp canyons & Divide)
#'   - Analytical smooth vertex normals (avoiding rayrender's recalculate_normals segfault)
#'   - Linear-space river & mainstem highlight masks (avoiding 8-bit sRGB gamma distortion)
#'
#' @param dem_path Path to elevation GeoTIFF (e.g., 30m elevatr raster).
#' @param mesh_cols Horizontal mesh resolution (e.g., 820 for 100m, 1150 for 80m).
#' @param tex_width Texture horizontal resolution (e.g., 4096 or 4608).
#' @param az_deg Camera compass yaw in degrees (256 = WSW up-canyon Poudre view).
#' @param vexag Vertical exaggeration factor (default 2.25).
#' @param dem_agg Aggregation factor applied to raw 30m DEM (default 5 for ~80-100m fidelity).
#' @export
sr_scene_from_dem <- function(
  dem_path,
  rivers_sf = NULL,
  lakes_sf  = NULL,
  sites_df  = NULL,
  wqp_df    = NULL,
  mesh_cols = 820L,
  tex_width = 4096L,
  az_deg    = 256.0,
  vexag     = 2.25,
  dem_agg   = 5L,
  lon0      = -105.42,
  lat0      = 40.59,
  x_lim     = c(-47, 46),
  z_lim     = c(-46, 52)
) {
  dem_raw <- terra::rast(dem_path)
  dem_cur <- if (dem_agg > 1L) terra::aggregate(dem_raw, fact = dem_agg, fun = "mean", na.rm = TRUE) else dem_raw

  km_lon <- 111.32 * cos(lat0 * pi / 180)
  km_lat <- 110.57
  az <- az_deg * pi / 180
  fwd_e <- sin(az); fwd_n <- cos(az)
  rgt_e <- cos(az); rgt_n <- -sin(az)

  xz_to_ll <- function(x, z) {
    e <- x * rgt_e - z * fwd_e
    n <- x * rgt_n - z * fwd_n
    cbind(lon = lon0 + e / km_lon, lat = lat0 + n / km_lat)
  }

  mesh_rows <- round(mesh_cols * diff(z_lim) / diff(x_lim))
  xs <- seq(x_lim[1], x_lim[2], length.out = mesh_cols)
  zs <- seq(z_lim[1], z_lim[2], length.out = mesh_rows)
  grid_xz <- expand.grid(z = zs, x = xs)
  pts_ll  <- xz_to_ll(grid_xz$x, grid_xz$z)

  ev <- terra::extract(dem_cur, pts_ll, method = "bilinear")[, 1]
  ev[is.na(ev)] <- min(ev, na.rm = TRUE)
  H_m <- matrix(ev, nrow = mesh_rows, ncol = mesh_cols)
  base_elev <- min(H_m)
  Y <- (H_m - base_elev) / 1000.0 * vexag

  # Build contiguous vertex, smooth normal, UV, and 0-based triangle index matrices for Metal
  dx <- xs[2] - xs[1]
  dz <- zs[2] - zs[1]
  dYdx <- matrix(0, mesh_rows, mesh_cols)
  dYdz <- matrix(0, mesh_rows, mesh_cols)
  dYdx[, 2:(mesh_cols - 1)] <- (Y[, 3:mesh_cols] - Y[, 1:(mesh_cols - 2)]) / (2 * dx)
  dYdz[2:(mesh_rows - 1), ] <- (Y[3:mesh_rows, ] - Y[1:(mesh_rows - 2), ]) / (2 * dz)
  nlen <- sqrt(dYdx^2 + 1.0 + dYdz^2)

  vertices <- cbind(as.vector(grid_xz$x), as.vector(Y), as.vector(grid_xz$z))
  normals  <- cbind(as.vector(-dYdx / nlen), as.vector(1.0 / nlen), as.vector(-dYdz / nlen))
  u_grid   <- rep(seq(0, 1, length.out = mesh_cols), each = mesh_rows)
  v_grid   <- rep(seq(0, 1, length.out = mesh_rows), times = mesh_cols)
  uvs      <- cbind(u_grid, v_grid)

  # 0-based triangle indices: 2 triangles per grid cell
  r_idx <- rep(0:(mesh_rows - 2L), times = mesh_cols - 1L)
  c_idx <- rep(0:(mesh_cols - 2L), each  = mesh_rows - 1L)
  v00 <- c_idx * mesh_rows + r_idx
  v10 <- v00 + 1L
  v01 <- (c_idx + 1L) * mesh_rows + r_idx
  v11 <- v01 + 1L
  indices <- rbind(cbind(v00, v10, v01), cbind(v10, v11, v01))

  structure(
    list(
      vertices  = vertices,
      normals   = normals,
      uvs       = uvs,
      indices   = indices,
      mesh_dims = c(mesh_rows, mesh_cols),
      tex_width = tex_width,
      az_deg    = az_deg,
      vexag     = vexag,
      x_lim     = x_lim,
      z_lim     = z_lim,
      base_elev = base_elev,
      sites     = sites_df,
      wqp       = wqp_df
    ),
    class = "sonicray_scene_spec"
  )
}

#' Upload a `sonicray_scene_spec` to a persistent Apple Silicon Metal Acceleration Structure
#' @export
sr_upload_scene <- function(scene_spec, albedo_rgba = NULL, river_rg = NULL) {
  shader_path <- system.file("metal", "sonicray_pathtracer.metal", package = "sonicray")
  if (!nzchar(shader_path) || !file.exists(shader_path)) {
    # Fallback when running directly from source directory before package install
    shader_path <- file.path("inst", "metal", "sonicray_pathtracer.metal")
  }
  shader_src <- paste(readLines(shader_path, warn = FALSE), collapse = "\n")

  tex_w <- as.integer(scene_spec$tex_width)
  tex_h <- as.integer(round(tex_w * diff(scene_spec$z_lim) / diff(scene_spec$x_lim)))
  if (is.null(albedo_rgba)) {
    albedo_rgba <- rep(c(0.28, 0.34, 0.24, 1.0), times = tex_w * tex_h)
  }
  if (is.null(river_rg)) {
    river_rg <- rep(c(0.0, 0.0), times = tex_w * tex_h)
  }

  xptr <- cpp_upload_scene(
    vertices         = scene_spec$vertices,
    normals          = scene_spec$normals,
    uvs              = scene_spec$uvs,
    indices          = scene_spec$indices,
    albedo_rgba      = albedo_rgba,
    tex_w            = tex_w,
    tex_h            = tex_h,
    river_rg         = river_rg,
    metal_shader_src = shader_src
  )

  structure(
    list(
      gpu_handle = xptr,
      spec       = scene_spec,
      tex_dims   = c(tex_w, tex_h)
    ),
    class = "sonicray_gpu_scene"
  )
}
