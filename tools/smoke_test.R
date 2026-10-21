#!/usr/bin/env Rscript
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
# End-to-end check on SYNTHETIC data: doctor, light layers, quick sheet, draft film. Exits with an error if anything fails.
suppressPackageStartupMessages(library(sonicray))
stopifnot(sr_doctor())
out <- file.path(tempdir(), "sonicray_smoke")
mp4 <- sr_demo(out, "draft")
stopifnot(file.exists(mp4), file.size(mp4) > 1e5)
sheet <- sr_sheet(file.path(out, "layers_960x540"), file.path(out, "sheet.png"), "network")
img <- png::readPNG(sheet); stopifnot(mean(img) > 0.02, max(img) > 0.5)
cat("smoke test passed:", mp4, "\n")
