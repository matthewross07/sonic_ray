#!/bin/bash
# One-shot setup for sonicray on an Apple Silicon Mac. Safe to re-run.
set -euo pipefail
[ "$(uname)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] || { echo "sonicray needs an Apple Silicon Mac."; exit 1; }
xcode-select -p >/dev/null 2>&1 || { echo "Install the Xcode Command Line Tools first:  xcode-select --install"; exit 1; }
command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
brew list open-image-denoise >/dev/null 2>&1 || brew install open-image-denoise      # optional denoiser; ok if it fails
python3 -m pip install --user numpy pillow
Rscript -e 'for (p in c("Rcpp","jsonlite","png")) if (!requireNamespace(p, quietly=TRUE)) install.packages(p, repos="https://cloud.r-project.org")'
R CMD INSTALL "$(cd "$(dirname "$0")" && pwd)"
Rscript -e 'library(sonicray); sr_doctor()'
