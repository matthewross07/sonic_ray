#!/usr/bin/env Rscript
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
# Compose the Cache la Poudre v2 film (real-data case study) from the light layers that 10_render_layers_gpu.R wrote.
#   Rscript film/poudre/20_compose.R [layers_dir] [out.mp4] [quality]
# The layers (and the Poudre scene they come from) are not part of this repository.
suppressPackageStartupMessages(library(sonicray))
a <- commandArgs(TRUE)
layers  <- if (length(a) >= 1) a[1] else "~/.cache/sonicray_poudre/layers_1920x1080"
out     <- if (length(a) >= 2) a[2] else "poudre_network_3d_v2_cpp.mp4"
quality <- if (length(a) >= 3) a[3] else "final"
sr_sheet(layers, sub("\\.mp4$", "_sheet.png", out), story = "poudre", times = c(6.5, 21.3, 56, 82, 99, 120.5))   # look at this first
sr_compose(layers, out, story = "poudre", quality = quality)
