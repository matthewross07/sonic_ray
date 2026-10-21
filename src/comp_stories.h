// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
// Stories turn time into a FrameSpec. All creative decisions (what lights up when, colours, seasons, sun path) live here, never in the compositor.
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "comp_engine.h"
#include "comp_types.h"

namespace sr {

struct SiteBurst { std::string color; double x = 0, y = 0, z = 0; };
struct Site { std::string name; double x = 0, y = 0, z = 0, px = 0, py = 0, beam = 0; bool lighthouse = false; std::vector<SiteBurst> bursts; };
struct Cell { std::string id; double x = 0, y = 0, z = 0, y0 = 0; int n_toc = 0, n_chla = 0, n_alk = 0; };
struct SunSample { double minute = 0, el = 0; };
struct Param { int plane = 0; float glow[3] = {1, 1, 1}; };
struct Event { std::string site, parameter; double t = 0, value = 0; int uid = 0; };

struct StoryData {
  std::vector<Site> sites;
  std::vector<Cell> cells;
  std::vector<SunSample> sun_path;
  // data-driven "network" story
  double duration_s = 16;
  std::map<std::string, Param> parameters;
  std::vector<std::pair<std::string, double>> sensors;     // (site, online_s)
  std::vector<Event> events;
  bool has_spark_colors = false;
  float spark_colors[3][3] = {};
};

class Story {
 public:
  virtual ~Story() {}
  virtual FrameSpec frame_spec(double t) const = 0;       // thread-safe (read-only)
  virtual double total() const = 0;
};

// name: "network" (data-driven sensors + samples) or "poudre" (the Cache la Poudre film). Throws on unknown names.
std::unique_ptr<Story> make_story(const std::string& name, Compositor& comp, const StoryData& data);

}  // namespace sr
