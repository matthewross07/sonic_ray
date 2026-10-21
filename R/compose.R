# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
# GPU compositor interface (the engine itself is C++/Metal in src/comp_*.cpp).

comp_shader_source <- function() {
  p <- system.file("metal", "sonicray_comp.metal", package = "sonicray")
  if (!nzchar(p)) p <- file.path("inst", "metal", "sonicray_comp.metal")
  paste(readLines(p, warn = FALSE), collapse = "\n")
}

# Read everything a story needs from a layers directory (scene_meta.json, plus story.json / sensors.csv / events.csv for the network story).
sr_story_data <- function(layers, story = "network") {
  layers <- path.expand(layers)
  meta <- jsonlite::fromJSON(file.path(layers, "scene_meta.json"), simplifyVector = FALSE)
  scene <- list(width = meta$width, height = meta$height, cam = lapply(meta$cam, function(v) as.numeric(unlist(v))))
  data <- list(sites = meta$sites, cells = meta$cells, sun_path = meta$sun_path)
  if (identical(story, "network")) {
    cfg <- jsonlite::fromJSON(file.path(layers, "story.json"), simplifyVector = FALSE)
    data$duration_s <- cfg$duration_s
    data$parameters <- lapply(cfg$parameters, function(p) list(plane = p$plane, glow = as.numeric(unlist(p$glow))))
    if (!is.null(cfg$spark_colors)) data$spark_colors <- do.call(rbind, lapply(cfg$spark_colors, function(v) as.numeric(unlist(v))))
    sen <- utils::read.csv(file.path(layers, "sensors.csv"), stringsAsFactors = FALSE)
    ev  <- utils::read.csv(file.path(layers, "events.csv"), stringsAsFactors = FALSE)
    data$sensor_site <- as.character(sen$site); data$sensor_online <- as.numeric(sen$online_s)
    data$event_site <- as.character(ev$site); data$event_parameter <- as.character(ev$parameter)
    data$event_time <- as.numeric(ev$time_s); data$event_value <- as.numeric(ev$value)
  }
  list(layers = layers, scene = scene, data = data)
}

#' Render single frames of a film as PNG files (fast way to check colours, framing and timing)
#' @param layers directory with `scene_meta.json` and the `*.f32` light layers.
#' @param times times in seconds. @param out_dir folder for the PNGs.
#' @param story "network" or "poudre". @param sharpen unsharp amount. @param dither TRUE to dither before 8-bit quantisation.
#' @return invisibly, the PNG paths.
#' @export
sr_still <- function(layers, times, out_dir = ".", story = "network", sharpen = 0.65, dither = FALSE) {
  sd <- sr_story_data(layers, story)
  frames <- cpp_comp_stills(sd$layers, comp_shader_source(), sd$scene, list(sharpen = sharpen, dither = dither), story, sd$data, as.numeric(times))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  W <- sd$scene$width; H <- sd$scene$height
  paths <- vapply(seq_along(times), function(i) {
    p <- file.path(out_dir, sprintf("%s_%07.2f.png", story, times[i]))
    a <- aperm(array(as.integer(frames[[i]]), c(3, W, H)), c(3, 2, 1)) / 255
    png::writePNG(a, p); p
  }, "")
  invisible(paths)
}

#' Contact sheet of a story's key moments (look at this before any slow render)
#' @param layers directory with scene_meta.json and the *.f32 light layers. @param story "network" or "poudre". @param sharpen unsharp amount.
#' @param out the PNG to write. @param times key moments in seconds (default: five moments spread over the film).
#' @export
sr_sheet <- function(layers, out = "quick_sheet.png", story = "network", times = NULL, sharpen = 0.65) {
  sd <- sr_story_data(layers, story)
  if (is.null(times)) {
    total <- cpp_comp_story_length(sd$layers, comp_shader_source(), sd$scene, list(sharpen = sharpen, dither = FALSE), story, sd$data)
    times <- total * c(0.15, 0.4, 0.6, 0.8, 0.92)
  }
  frames <- cpp_comp_stills(sd$layers, comp_shader_source(), sd$scene, list(sharpen = sharpen, dither = FALSE), story, sd$data, as.numeric(times))
  W <- sd$scene$width; H <- sd$scene$height; tw <- 640L; th <- as.integer(round(tw * H / W)); cols <- 3L; rows <- ceiling(length(frames) / cols); pad <- 8L
  sheet <- array(0.07, c(rows * th + (rows + 1) * pad, cols * tw + (cols + 1) * pad, 3))
  for (i in seq_along(frames)) {
    a <- aperm(array(as.integer(frames[[i]]), c(3, W, H)), c(3, 2, 1)) / 255
    yi <- pmin(pmax(round((seq_len(th) - 0.5) / th * H + 0.5), 1), H); xi <- pmin(pmax(round((seq_len(tw) - 0.5) / tw * W + 0.5), 1), W)
    r0 <- ((i - 1) %/% cols) * (th + pad) + pad; c0 <- ((i - 1) %% cols) * (tw + pad) + pad
    sheet[r0 + seq_len(th), c0 + seq_len(tw), ] <- a[yi, xi, ]
  }
  png::writePNG(sheet, out); invisible(out)
}

#' Quality presets (resolution, sampling and encode settings)
#' @export
sr_presets <- function() {
  list(
  draft  = list(width = 960L,  height = 540L,  fps = 24L, light_spp = 32L,  amb_spp = 64L,   encoder = "videotoolbox", crf = 20L, quality = 70L, preset = "veryfast", sharpen = 0,    dither = FALSE,
                use = "iterating on colours, timing and framing; look at the quick sheet first"),
  review = list(width = 1280L, height = 720L,  fps = 30L, light_spp = 128L, amb_spp = 256L,  encoder = "videotoolbox", crf = 16L, quality = 85L, preset = "fast",     sharpen = 0.5,  dither = TRUE,
                use = "share with collaborators; close to final"),
  final  = list(width = 1920L, height = 1080L, fps = 30L, light_spp = 256L, amb_spp = 512L,  encoder = "x264",         crf = 13L, quality = 90L, preset = "slow",     sharpen = 0.65, dither = TRUE,
                use = "the version you present from; render only after the user approves a review"))
}

#' Compose a film from light layers on the GPU
#'
#' One process: worker threads render frames on the GPU while the main thread streams them to ffmpeg.
#' @param layers directory with `scene_meta.json` and `*.f32` layers. @param out output .mp4 (use a NEW name).
#' @param story "network" (data-driven) or "poudre". @param quality "draft", "review" or "final" (see [sr_presets()]).
#' @param encoder "x264" (small files, best for final) or "videotoolbox" (Apple hardware encoder: fast, frees the CPU, bigger files). Default from `quality`.
#' @param threads frames rendered at once (default 3). @param t0,t1 render only this time range in seconds.
#' @param audio optional .wav/.m4a to mux in. @param audio_offset seconds to trim from the start of the audio.
#' @param ... overrides: `fps`, `crf`, `preset`, `sharpen`, `dither`, `codec`, `vt_quality`.
#' @return invisibly, a list with `seconds`, `frames` and `out`.
#' @export
sr_compose <- function(layers, out, story = "network", quality = c("draft", "review", "final"), encoder = NULL, threads = 3L, t0 = 0, t1 = -1,
                       audio = NULL, audio_offset = 0, ...) {
  quality <- match.arg(quality); P <- sr_presets()[[quality]]; o <- list(...)
  ff <- Sys.which("ffmpeg"); if (!nzchar(ff)) stop("ffmpeg not found. Install it with: brew install ffmpeg")
  sd <- sr_story_data(layers, story)
  enc <- list(ffmpeg = unname(ff), encoder = if (is.null(encoder)) P$encoder else encoder, preset = if (is.null(o$preset)) P$preset else o$preset,
              crf = if (is.null(o$crf)) P$crf else o$crf, codec = if (is.null(o$codec)) "hevc" else o$codec, quality = if (is.null(o$vt_quality)) P$quality else o$vt_quality,
              audio = if (is.null(audio)) "" else path.expand(audio), audio_offset = audio_offset)
  look <- list(sharpen = if (is.null(o$sharpen)) P$sharpen else o$sharpen, dither = if (is.null(o$dither)) P$dither else o$dither)
  dir.create(dirname(path.expand(out)), recursive = TRUE, showWarnings = FALSE)
  res <- cpp_comp_film(sd$layers, comp_shader_source(), sd$scene, look, story, sd$data, path.expand(out), if (is.null(o$fps)) P$fps else as.integer(o$fps), t0, t1, enc, as.integer(threads), TRUE)
  message(sprintf("wrote %s (%d frames in %.1f s)", res$out, res$frames, res$seconds))
  invisible(res)
}

#' Check this Mac and print exact fixes
#' @param json if TRUE, return/print machine-readable JSON (for AI assistants).
#' @return invisibly, TRUE if everything needed is in place.
#' @export
sr_doctor <- function(json = FALSE) {
  info <- tryCatch(sr_device_info(), error = function(e) list(available = FALSE))
  checks <- list()
  add <- function(name, ok, detail = "", fix = "") checks[[length(checks) + 1]] <<- list(check = name, ok = isTRUE(ok), detail = detail, fix = fix)
  add("macOS on Apple Silicon", Sys.info()[["sysname"]] == "Darwin" && Sys.info()[["machine"]] == "arm64", paste(Sys.info()[["sysname"]], Sys.info()[["machine"]]), "sonicray needs an Apple Silicon Mac")
  add("Metal GPU", isTRUE(info$available), if (isTRUE(info$available)) info$name else "", "needs an Apple Silicon Mac with Metal")
  add("hardware ray tracing", isTRUE(info$supports_raytracing), "", "needs an M-series chip (M1 or later)")
  add("ffmpeg", nzchar(Sys.which("ffmpeg")), unname(Sys.which("ffmpeg")), "brew install ffmpeg")
  ok_gpu <- FALSE; detail <- ""
  if (isTRUE(info$available)) tryCatch({
    d <- file.path(tempdir(), "sr_selftest"); dir.create(d, showWarnings = FALSE); W <- 256L; H <- 144L
    yy <- matrix(rep(seq_len(H), each = W), H, W, byrow = TRUE); xx <- matrix(rep(seq_len(W), times = H), H, W, byrow = TRUE)
    wl <- function(v, n) writeBin(as.numeric(t(cbind(as.vector(t(v)), as.vector(t(v)), as.vector(t(v))))), file.path(d, paste0(n, ".f32")), size = 4)
    wl(matrix(0.5, H, W), "amb"); writeBin(as.numeric(t(ifelse(yy > 0.45 * H, 1, 0))), file.path(d, "mask.f32"), size = 4); wl(matrix(0, H, W), "riv"); wl(exp(-((xx - 128)^2 + (yy - 100)^2) / 800) * 2, "fwk_param_a_s1")
    jsonlite::write_json(list(width = W, height = H, cam = list(lookfrom = c(0, 15.8, 70), lookat = c(0, 1.5, -2), fov = 22.8),
      sites = list(list(site = "s1", x = 0, y = 1, z = 0, px = 128, py = 100, beam = 20, lighthouse = FALSE, bursts = list(list(color = "param_a", x = 0, y = 8, z = 0)))), cells = list(), sun_path = list()),
      file.path(d, "scene_meta.json"), auto_unbox = TRUE)
    writeLines('{"duration_s": 4, "parameters": {"param_a": {"plane": 0, "glow": [1, 0.4, 0.2]}}}', file.path(d, "story.json"))
    writeLines("site,online_s\ns1,0.2", file.path(d, "sensors.csv")); writeLines("site,parameter,time_s,value\ns1,param_a,0.5,0.8", file.path(d, "events.csv"))
    f <- sr_still(d, 1.0, tempdir(), "network", sharpen = 0.65, dither = TRUE)
    img <- png::readPNG(f); ok_gpu <- mean(img) > 0.01 && mean(img) < 0.95 && max(img) > 0.3
    detail <- sprintf("256x144 test frame mean=%.3f max=%.2f", mean(img), max(img))
  }, error = function(e) detail <<- conditionMessage(e))
  add("GPU compositor selftest", ok_gpu, detail, "check the messages above")
  all_ok <- all(vapply(checks[!grepl("optional", vapply(checks, `[[`, "", "check"))], `[[`, TRUE, "ok"))
  if (json) cat(jsonlite::toJSON(list(sonicray = as.character(utils::packageVersion("sonicray")), all_ok = all_ok, checks = checks), auto_unbox = TRUE, pretty = TRUE), "\n")
  else {
    for (ch in checks) cat(sprintf("  %-5s %s%s%s\n", if (ch$ok) "ok" else if (grepl("optional", ch$check)) "info" else "FAIL", ch$check, if (nzchar(ch$detail)) paste0("   [", ch$detail, "]") else "", if (!ch$ok && nzchar(ch$fix)) paste0("\n        fix: ", ch$fix) else ""))
    cat(if (all_ok) "sonicray: everything needed for GPU compositing is in place\n" else "sonicray: fix the FAIL items above\n")
  }
  invisible(all_ok)
}

#' The questions an assistant should put to the user before rendering (colours, quality, speed, ...)
#' @param json machine-readable output.
#' @export
sr_intake <- function(json = FALSE) {
  q <- list(
    list(id = "data", ask = "Is this real data or the synthetic demo? If real: which sensors and samples, what coordinates, what time range maps to the film?", default = "synthetic demo",
         why = "decides the scene, sensors.csv and events.csv; never present synthetic output as real measurements"),
    list(id = "purpose", ask = "Where will it be shown (slide deck, web, large screen) and for how long?", default = "slide deck; 15-20 s demo or a 130 s film", why = "sets resolution, fps and length"),
    list(id = "quality", ask = "Which quality preset: draft, review or final?", default = "draft first, then review, then final only after approval", why = "draft is far cheaper than final; most design mistakes show up in a draft or in sr_sheet()"),
    list(id = "colors", ask = "Colours for each measured parameter (up to 3), the sensor beacons and the terrain glow? Colour-blind-safe palette needed?", default = "crimson / green / violet parameters, ice-blue beacons",
         why = "the palette lives in story.json (parameters[*].glow, spark_colors); changing it needs only a re-composite, not new layers"),
    list(id = "water", ask = "Is the water itself the subject (its clarity, colour, turbidity, depth, or objects seen through it)? Or are rivers and lakes only map features?", default = "no: water is a map feature, so do NOT enable the water volume",
         why = "rivers and lakes as map features are already drawn by the albedo texture and the river mask; the 3D water volume costs 15-35% more render time and adds sun-glitter noise (docs/WATER.md)"),
    list(id = "look", ask = "Night only, or also sunrise and sky? Stars? Any text overlay?", default = "night, stars, NO on-screen text (the presenter narrates)", why = "sun and sky need extra layers (re-render)"),
    list(id = "speed", ask = "Fastest turnaround or smallest files?", default = "videotoolbox encoder for draft and review (fast, large files); x264 for final (small files)", why = "hardware encoding frees the CPU but makes files about 3-4x bigger"),
    list(id = "audio", ask = "A soundtrack? Provide a .wav and its offset in seconds, or skip. Music is optional and outside the package (docs/MUSIC_WITH_SONIC_PI.md).", default = "no audio", why = "muxed after compositing with audio="),
    list(id = "output", ask = "Output folder and file name?", default = "a NEW file name; never overwrite an approved film", why = "approved outputs are read-only by convention"))
  if (json) { cat(jsonlite::toJSON(list(questions = q, presets = sr_presets()), auto_unbox = TRUE, pretty = TRUE), "\n"); return(invisible(q)) }
  cat("Ask the user these before rendering (accept 'defaults' as an answer):\n\n")
  for (i in seq_along(q)) cat(sprintf("%d. [%s] %s\n     default: %s\n     why: %s\n", i, q[[i]]$id, q[[i]]$ask, q[[i]]$default, q[[i]]$why))
  cat("\nQuality presets:\n"); P <- sr_presets()
  for (n in names(P)) cat(sprintf("  %-7s %dx%d @ %d fps, light_spp %d, amb_spp %d, encoder %s  -- %s\n", n, P[[n]]$width, P[[n]]$height, P[[n]]$fps, P[[n]]$light_spp, P[[n]]$amb_spp, P[[n]]$encoder, P[[n]]$use))
  invisible(q)
}
