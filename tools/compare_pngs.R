#!/usr/bin/env Rscript
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
# Frame parity check: compare same-named PNG stills in two folders (for example sr_still() output before and after a change).
#   Rscript tools/compare_pngs.R DIR_A DIR_B
a <- commandArgs(TRUE)
if (length(a) != 2) stop("usage: Rscript tools/compare_pngs.R DIR_A DIR_B")
for (f in intersect(list.files(a[1], "\\.png$"), list.files(a[2], "\\.png$"))) {
  x <- png::readPNG(file.path(a[1], f)) * 255; y <- png::readPNG(file.path(a[2], f)) * 255
  d <- abs(x - y); mse <- mean(d^2)
  cat(sprintf("%-28s mean|diff| %.4f/255  max %3.0f  PSNR %6.1f dB  >8 levels: %.3f%% of pixels\n", f, mean(d), max(d), 10 * log10(255^2 / max(mse, 1e-9)), 100 * mean(apply(d, c(1, 2), max) > 8)))
}
