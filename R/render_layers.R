# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
# ==============================================================================
# sonicray/R/render_layers.R
# Batched Multi-Light GPU Path Tracing + Albedo-Demodulated Intel OIDN Denoising
# ==============================================================================

#' Render all light layers (Ambient + AOVs + Sun Arc + Sonde/WQP Multi-Light Batches) on Metal GPU
#'
#' @param gpu_scene A `sonicray_gpu_scene` returned by [sr_upload_scene()].
#' @param light_table Data frame of lights with columns `name`, `x`, `y`, `z`, `r`, `intensity`.
#' @param lookfrom,lookat Camera 3-vectors in scene coordinates (the defaults frame the Cache la Poudre scene; set your own).
#' @param fov Vertical field-of-view in degrees (default 22.8).
#' @param width,height Output layer resolution (default 1920 x 1080).
#' @param amb_spp,light_spp Samples per pixel for ambient and point/sphere light batches.
#' @param max_bounces Surface vertices per path with next-event estimation (1 = direct light only, 4 = three indirect bounces).
#' @param clamp Per-sample brightness clamp (fireflies); the CPU film pipeline used 12, which also clips genuine near-light highlights.
#' @param batch_size Number of lights evaluated simultaneously per BVH traversal (1..32, default 16).
#' @param denoise Logical; run albedo-demodulated Intel OIDN on each layer (default TRUE).
#' @param out_dir Optional directory to write `.f32` layer files compatible with `02_compose_video.py`.
#' @export
sr_render_light_layers <- function(
  gpu_scene,
  light_table,
  lookfrom   = c(-0.5, 15.8, 70.0),
  lookat     = c(-0.5,  1.5, -2.0),
  fov        = 22.8,
  width      = 1920L,
  height     = 1080L,
  amb_spp    = 128L,
  light_spp  = 48L,
  max_bounces = 4L,
  clamp      = 50,
  batch_size = 16L,
  denoise    = TRUE,
  out_dir    = NULL
) {
  stopifnot(inherits(gpu_scene, "sonicray_gpu_scene"))
  if (!is.null(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  t0 <- Sys.time()
  message(sprintf("[sonicray] Pass 1: Ambient + First-Hit AOVs (%dx%d @ %d spp)...", width, height, amb_spp))
  aov_res <- cpp_render_ambient_and_aovs(
    scene_xptr = gpu_scene$gpu_handle,
    lookfrom   = lookfrom,
    lookat     = lookat,
    fov_deg    = fov,
    width      = width,
    height     = height,
    samples    = amb_spp,
    clamp_val  = 4.0
  )

  if (isTRUE(denoise)) {
    aov_res$amb <- cpp_oidn_denoise_layer(
      noisy_rgb         = aov_res$amb,
      albedo_rgb        = aov_res$albedo,
      normal_xyz        = aov_res$normal,
      width             = width,
      height            = height,
      demodulate_albedo = TRUE
    )
  }

  if (!is.null(out_dir)) {
    writeBin(as.numeric(aov_res$amb),  file.path(out_dir, "amb.f32"),  size = 4)
    writeBin(as.numeric(aov_res$mask), file.path(out_dir, "mask.f32"), size = 4)
  }

  # Pass 2: Batched Multi-Light NEE Path Tracing (up to 32 lights per pass, zero RGB cross-talk)
  n_lights <- nrow(light_table)
  n_batches <- ceiling(n_lights / batch_size)
  npix3 <- as.integer(width) * as.integer(height) * 3L
  rendered_layers <- list()

  for (b in seq_len(n_batches)) {
    idx <- ((b - 1L) * batch_size + 1L):min(b * batch_size, n_lights)
    sub <- light_table[idx, , drop = FALSE]
    mat <- as.matrix(sub[, c("x", "y", "z", "r", "intensity")])

    message(sprintf("[sonicray] Batch %d/%d: %d lights simultaneously (%d spp)...",
                    b, n_batches, nrow(sub), light_spp))
    raw_batch <- cpp_render_multilight_batch(
      scene_xptr            = gpu_scene$gpu_handle,
      lookfrom              = lookfrom,
      lookat                = lookat,
      fov_deg               = fov,
      width                 = width,
      height                = height,
      samples               = light_spp,
      max_bounces           = max_bounces,
      clamp_val             = clamp,
      light_pos_r_intensity = mat
    )

    for (k in seq_len(nrow(sub))) {
      offset <- (k - 1L) * npix3
      lyr <- raw_batch[(offset + 1L):(offset + npix3)]
      if (isTRUE(denoise)) {
        lyr <- cpp_oidn_denoise_layer(
          noisy_rgb         = lyr,
          albedo_rgb        = aov_res$albedo,
          normal_xyz        = aov_res$normal,
          width             = width,
          height            = height,
          demodulate_albedo = TRUE
        )
      }
      lname <- as.character(sub$name[k])
      rendered_layers[[lname]] <- lyr
      if (!is.null(out_dir)) {
        writeBin(as.numeric(lyr), file.path(out_dir, paste0(lname, ".f32")), size = 4)
      }
    }
  }

  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  message(sprintf("[sonicray] Completed %d light layers + AOVs in %.2f s (%.2f s/light).",
                  n_lights, dt, dt / max(n_lights, 1L)))
  list(aovs = aov_res, layers = rendered_layers, elapsed_sec = dt)
}

#' Denoise a single HxWx3 layer using first-hit Albedo and Normal AOVs via Intel OIDN
#' @export
sr_denoise_layer <- function(noisy_rgb, albedo_rgb, normal_xyz, width, height, demodulate_albedo = TRUE) {
  cpp_oidn_denoise_layer(noisy_rgb, albedo_rgb, normal_xyz, width, height, demodulate_albedo)
}

#' Compare a `sonicray` Metal GPU layer against a `rayrender` CPU reference layer
#' @export
sr_parity_check <- function(gpu_layer, cpu_layer) {
  diff <- as.numeric(gpu_layer) - as.numeric(cpu_layer)
  rmse <- sqrt(mean(diff^2))
  rel_mean <- (mean(gpu_layer) - mean(cpu_layer)) / max(mean(cpu_layer), 1e-6)
  list(
    rmse             = rmse,
    rel_mean_diff    = rel_mean,
    cosine_similarity = sum(gpu_layer * cpu_layer) / max(sqrt(sum(gpu_layer^2)) * sqrt(sum(cpu_layer^2)), 1e-8),
    passed           = abs(rel_mean) < 0.05 && rmse < 0.03
  )
}
