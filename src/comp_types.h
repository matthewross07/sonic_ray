// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
// Plain-data description of ONE film frame (the contract between a "story" and the GPU compositor). No Metal or R types in here.
#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace sr {

constexpr int PLANE_TOC = 0, PLANE_CHLA = 1, PLANE_ALK = 2, PLANE_SHIMMER = 3, PLANE_WHITE = 4;

struct Term {                 // out += layer * weight (mode 0: RGB, 1: red broadcast, 2: green broadcast)
  std::string layer;
  float w[3];
  int mode = 0;
};

struct SatScan {
  bool active = false;
  double t = 0, half_width = 0.18, duration = 4.0, tau = 1.76;
  std::vector<std::pair<double, double>> events;   // (start time, u centre)
  std::string second_layer;                        // empty = none
  float color[3] = {0.84f, 0.93f, 1.0f};
};

struct Burst {                // one firework (sphere of ballistic sparks, or the rising rocket when dt < 0)
  int uid = 0, plane = 0;
  double t0 = 0;
  double origin[3] = {0, 0, 0};
  double launch_y = 0;
  int n_sparks = 64;
  double gain = 1, v_max = 5, rise = 0.5, life0 = 1.0, life1 = 2.0, trail_dt = 0.6;
};

struct SkyDay {
  bool active = false;
  std::string layer_a, layer_b;
  double f = 0, sky_exp = 0;
  double zenith[3] = {0, 0, 0}, horizon_tint[3] = {0, 0, 0};
  double dawn_band = 0, set_band = 0;
  double venus[3] = {0.92, 0.46, 0.30}, sunset[3] = {1.00, 0.42, 0.16};
};

struct Orb {                  // mode 1: noon corona, 2: descending disc + aureole
  int mode = 0;
  double x = 0, y = 0, radius = 0, visibility = 0, dusk = 0, amount = 0;
  double color[3] = {1.0, 0.94, 0.80};
};

struct Beacon { double x, y, beam, pulse; };

struct FrameSpec {
  double t = 0;
  std::vector<Term> terms, beacon_terms;
  SatScan sat;
  std::vector<Burst> bursts;
  SkyDay sky;
  double star_amount = 0;
  Orb orb;
  double aerial = 0;
  std::vector<Beacon> beacons;
  double fade = 1;
  uint32_t seed = 0;
};

struct Look {
  double sharpen = 0.0;
  bool dither = false;
  float beacon_tint[3] = {0.20f, 0.50f, 1.00f};
  double spark_slow = 1.5;
  float spark_colors[3][3] = {{1.00f, 0.22f, 0.18f}, {0.20f, 0.96f, 0.32f}, {0.46f, 0.20f, 1.00f}};
};

struct SceneInfo {
  int W = 0, H = 0;
  double lookfrom[3] = {0, 0, 0}, lookat[3] = {0, 0, 0}, fov = 30;
};

}  // namespace sr
