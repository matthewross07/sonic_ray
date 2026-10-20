#!/bin/bash
# Build the GPU compositor host library (needs only the Xcode Command Line Tools; shaders compile at runtime).
set -euo pipefail
cd "$(dirname "$0")"
clang++ -O2 -std=c++17 -fobjc-arc -ObjC++ -dynamiclib -fPIC \
  -framework Metal -framework Foundation \
  -o libsonicray_comp.dylib sonicray_comp.mm
echo "built $(pwd)/libsonicray_comp.dylib"
