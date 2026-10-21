// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
// Firework sparks: CPU geometry (primitives in draw order, binned into 16x16-pixel tiles) for the GPU `spark_raster` kernel.
#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "comp_types.h"

namespace sr {

struct Prim { float a, b, c, d, w, fill; int32_t plane, kind; };   // kind 0 = thick line (a,b)-(c,d); 1 = ellipse in bbox (a,b,c,d)
static_assert(sizeof(Prim) == 32, "Prim must match the Metal struct");
constexpr int TILE = 16;                                           // = the Metal threadgroup size, so a whole group shares one tile list

// numpy-compatible legacy RandomState (MT19937, polar normal, 53-bit doubles): reproduces the original film's sparks exactly
class NumpyRng {
 public:
  explicit NumpyRng(uint32_t seed);
  double uniform01();
  double normal();                // standard normal, with numpy's cached second value
  double uniform(double lo, double hi) { return lo + (hi - lo) * uniform01(); }
  // numpy.random.RandomState.choice(n, p=...) for one draw
  int choice(const std::vector<double>& p);
 private:
  uint32_t next32();
  uint32_t key_[624];
  int pos_;
  bool has_gauss_ = false;
  double gauss_ = 0;
};

class SparkSystem {
 public:
  SparkSystem(const SceneInfo& scene, double slow);
  // all primitives of this frame, in painter's order
  void primitives(double t, const std::vector<Burst>& bursts, std::vector<Prim>& out);
  // counting-sort the primitives into tiles (order preserved per tile). Returns tiles per row.
  int bin_tiles(const std::vector<Prim>& prims, std::vector<uint32_t>& tile_off, std::vector<uint32_t>& tile_idx) const;
 private:
  struct Static { std::vector<double> v, sp, life, phase; };       // v: N*3
  std::shared_ptr<const Static> spark_static(const Burst& b);
  void project(const double* P, int n, double* sx, double* sy) const;
  int W_, H_;
  double slow_, px_, tf_;
  double lf_[3], f_[3], r_[3], u_[3];
  int w_tail_, w_mid_, w_head_;
  std::mutex mu_;
  std::unordered_map<int, std::shared_ptr<const Static>> cache_;
};

}  // namespace sr
