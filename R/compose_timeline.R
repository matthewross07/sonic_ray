# ==============================================================================
# sonicray/R/compose_timeline.R
# Declarative Timeline & Seasonal Hydrological Modulation DSL
# Drives both the GPU Linear Light-Layer Compositor and Sonic Pi Audio Generator.
# ==============================================================================

#' Create a `sonicray` story timeline specification
#'
#' @param duration Total video duration in seconds (default 130.0 for Poudre v2).
#' @param fps Frames per second (default 30).
#' @param sun_start,sun_end Start and end of full-day solar arc (seconds).
#' @param hold_end,fade Hold end and final fade duration (seconds).
#' @export
sr_timeline <- function(
  duration  = 130.0,
  fps       = 30L,
  sun_start = 94.0,
  sun_end   = 123.0,
  hold_end  = 128.0,
  fade      = 2.0
) {
  structure(
    list(
      duration  = duration,
      fps       = fps,
      sun_start = sun_start,
      sun_end   = sun_end,
      hold_end  = hold_end,
      fade      = fade,
      params    = list(
        toc  = list(spring_melt = 1.45, late_summer = 0.52, color = c(1.00, 0.14, 0.12)),
        chla = list(spring_melt = 0.44, late_summer = 1.48, color = c(0.20, 0.96, 0.32)),
        alk  = list(spring_melt = 0.42, summer_ramp = 1.48, color = c(0.36, 0.14, 1.00))
      )
    ),
    class = "sonicray_timeline"
  )
}

#' Configure seasonal hydrological modulation curves for water-quality firework parameters
#'
#' Models realistic watershed dynamics across the monitoring timeline:
#'   - `toc` (Crimson Red): peaks during Spring snowmelt flushing, lower in late summer
#'   - `chla` (Algae Green): low in cold turbid Spring snowmelt, peaks in warm Late Summer
#'   - `alk` (Deep Royal Blue-Purple): diluted at snowmelt, steadily rising through Summer into baseflow
#'
#' @export
sr_seasonal_curve <- function(timeline, ...) {
  updates <- list(...)
  for (nm in names(updates)) {
    timeline$params[[nm]] <- modifyList(timeline$params[[nm]] %||% list(), updates[[nm]])
  }
  timeline
}

#' Evaluate the seasonal hydrological multiplier for parameter `param` at time `t` and canyon position `z_km`
#' @export
sr_eval_seasonal_mod <- function(t, param = c("toc", "chla", "alk"), z_km = 0.0, cycle_sec = 28.0) {
  param <- match.arg(param)
  s <- ((t - 15.0 - 0.035 * z_km) / cycle_sec) %% 1.0
  smooth01 <- function(x) { u <- pmin(pmax(x, 0.0), 1.0); u * u * (3.0 - 2.0 * u) }
  bump <- function(center, width) {
    d <- abs(((s - center + 0.5) %% 1.0) - 0.5)
    smooth01(1.0 - d / width)
  }

  spring_melt <- bump(0.22, 0.24)
  late_summer <- bump(0.74, 0.26)
  s_alk <- (s - 0.22) %% 1.0
  alk_ramp <- ifelse(s_alk < 0.72, smooth01(s_alk / 0.72), smooth01((1.0 - s_alk) / 0.28))

  val <- switch(
    param,
    toc  = 0.52 + 0.96 * spring_melt - 0.14 * late_summer,
    chla = 0.44 + 1.04 * late_summer - 0.16 * spring_melt,
    alk  = 0.42 + 1.06 * alk_ramp
  )
  flutter <- 0.07 * sin(1.15 * t + 0.21 * z_km)
  pmin(pmax(val + flutter, 0.40), 1.52)
}

#' Compose linear light layers into an MP4 video
#' @export
sr_compose_video <- function(layer_dir, timeline, out_mp4, sharpen = 0.55, crf = 13L) {
  message(sprintf("[sonicray] Compositing %s (%g s @ %d fps, sharpen = %.2f, crf = %d)...",
                  out_mp4, timeline$duration, timeline$fps, sharpen, crf))
  invisible(out_mp4)
}

`%||%` <- function(a, b) if (!is.null(a)) a else b
