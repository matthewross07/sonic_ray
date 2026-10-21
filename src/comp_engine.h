// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
// GPU compositor: FrameSpec -> uint8 RGB frame, entirely on the Apple GPU (Metal). Pure C++ interface (Metal types stay in the .cpp).
#pragma once
#include <memory>
#include <set>
#include <string>
#include "comp_types.h"

namespace sr {

struct Geometry {                  // derived from the terrain mask; stories use it for the sun path and horizon band
  double y_top = 0, y_bot = 0, ox0 = 0, ox1 = 0, y_edge = 0, orb_r = 0, orb_y_top = 0, orb_y_set = 0;
};

class Compositor {
 public:
  // n_contexts = number of frames that may render at the same time (each owns ~0.5 GB of working buffers)
  Compositor(const std::string& layers_dir, const std::string& metal_source, const SceneInfo& scene, const Look& look, int n_contexts);
  ~Compositor();
  int width() const;
  int height() const;
  const Geometry& geometry() const;
  Look& look();
  bool has_layer(const std::string& name) const;
  const std::set<std::string>& layer_names() const;
  // Render one frame with working context `ctx` (0 .. n_contexts-1). Thread-safe for distinct contexts. `out` holds W*H*3 bytes.
  void render(int ctx, const FrameSpec& spec, uint8_t* out);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sr
