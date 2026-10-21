// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
#include "comp_stories.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include "comp_sparks.h"

namespace sr {
namespace {

double smooth(double x) { x = std::min(std::max(x, 0.0), 1.0); return x * x * (3 - 2 * x); }
double pymod1(double x) { return x - std::floor(x); }                                   // python: x % 1.0
double clampd(double x, double lo, double hi) { return std::min(std::max(x, lo), hi); }
constexpr double SLOW = 1.5, PEAK = 1.5;

// ============================================================================================ network (data-driven)
class NetworkStory : public Story {
 public:
  NetworkStory(Compositor& comp, const StoryData& d) : c_(comp), W_(comp.width()), H_(comp.height()), d_(d) {
    for (const Site& s : d_.sites) site_[s.name] = &s;                      // point into OUR copy (d_), never into the argument
    for (const auto& so : d.sensors) if (site_.count(so.first) && comp.has_layer("bea_" + so.first)) online_.push_back(so);
    for (const Event& e : d.events) if (site_.count(e.site) && d.parameters.count(e.parameter)) events_.push_back(e);
    std::stable_sort(events_.begin(), events_.end(), [](const Event& a, const Event& b) { return a.t < b.t; });
    for (const Site& s : d.sites) {
      double neigh = 0;
      for (const Site& q : d.sites) neigh += std::exp(-((s.x - q.x) * (s.x - q.x) + (s.z - q.z) * (s.z - q.z)) / (2.0 * 7.5 * 7.5));
      scale_[s.name] = (1.0 / std::pow(neigh, 0.72)) * 1.42;
    }
    if (d.has_spark_colors) { Look& L = comp.look(); for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) L.spark_colors[i][j] = d.spark_colors[i][j]; }
  }
  double total() const override { return d_.duration_s; }
  FrameSpec frame_spec(double t) const override {
    FrameSpec spec; spec.t = t; spec.seed = (uint32_t)((long long)(t * 1000) % 2147483648LL);
    spec.fade = std::min(1.0, t / 0.6) * (t <= d_.duration_s - 1.5 ? 1.0 : std::max(0.0, (d_.duration_s - t) / 1.5));
    spec.terms.push_back({"amb", {1, 1, 1}, 0});
    spec.star_amount = c_.has_layer("stars") ? 0.75 : 0.0;
    const double scale = W_ / 640.0;
    for (const auto& so : online_) {                                           // sensor beacons
      const Site& s = *site_.at(so.first);
      double dt = t - so.second;
      if (dt < 0) continue;
      double ramp = 1 - std::exp(-dt / (0.28 * SLOW)), flare = 1.0 + 1.8 * std::exp(-dt / (0.55 * SLOW));
      double w_b = ramp * flare * 1.2 * (1.0 + 0.05 * std::sin(2 * M_PI * 0.4 * t + s.x));
      float wv = (float)(w_b * 0.175 * scale_.at(s.name));
      spec.beacon_terms.push_back({"bea_" + s.name, {wv, wv, wv}, 0});
      if (-12 * scale <= s.px && s.px <= W_ + 12 * scale && -12 * scale <= s.py && s.py <= H_ + 12 * scale) {
        double flare_mult = clampd(w_b / std::max(ramp, 1e-3), 0.85, 1.45);
        double pulse = 0.70 * ramp * (0.95 + 0.05 * std::sin(2 * M_PI * 0.4 * t + s.x)) * (0.82 + 0.18 * flare_mult);
        spec.beacons.push_back({s.px, s.py, std::max(s.beam, 8.0 * scale), pulse});
      }
    }
    std::map<std::pair<std::string, std::string>, double> glow;
    for (const Event& e : events_) {                                           // sample fireworks
      double dt = t - e.t;
      if (!(-0.75 < dt && dt < 6.6)) continue;
      const Site& s = *site_.at(e.site);
      const Param& pc = d_.parameters.at(e.parameter);
      double m = 0.55 + 0.9 * e.value;
      const SiteBurst* b = nullptr;
      for (const SiteBurst& sb : s.bursts) if (sb.color == e.parameter) { b = &sb; break; }
      if (!b && !s.bursts.empty()) b = &s.bursts[0];
      if (!b) continue;
      double dens = clampd(std::pow(scale_.at(e.site), 0.28), 0.74, 1.10);
      Burst br; br.uid = e.uid; br.plane = pc.plane; br.t0 = e.t; br.origin[0] = b->x; br.origin[1] = b->y; br.origin[2] = b->z; br.launch_y = s.y;
      br.n_sparks = std::max(48, (int)std::lround(128 * (0.55 + 0.45 * m))); br.gain = 0.92 * dens * (0.48 + 0.58 * m);
      br.v_max = 5.6 * dens * (0.58 + 0.42 * m); br.rise = 0.45 * SLOW; br.life0 = 1.05; br.life1 = 1.95; br.trail_dt = 0.44 * SLOW;
      spec.bursts.push_back(br);
      if (dt >= 0 && c_.has_layer("fwk_" + e.parameter + "_" + e.site))
        glow[{e.parameter, e.site}] += (1 - std::exp(-dt / (0.035 * SLOW))) * std::exp(-dt / (0.24 * SLOW * PEAK)) * 0.72 * m;
    }
    for (const auto& g : glow) {
      if (g.second <= 1e-3) continue;
      const Param& pc = d_.parameters.at(g.first.first);
      double wt = 0.056 * 0.65 * scale_.at(g.first.second);
      spec.terms.push_back({"fwk_" + g.first.first + "_" + g.first.second, {(float)(pc.glow[0] * g.second * wt), (float)(pc.glow[1] * g.second * wt), (float)(pc.glow[2] * g.second * wt)}, 0});
    }
    return spec;
  }
 private:
  Compositor& c_; int W_, H_; StoryData d_;
  std::map<std::string, const Site*> site_;
  std::vector<std::pair<std::string, double>> online_;
  std::vector<Event> events_;
  std::map<std::string, double> scale_;
};

// ============================================================================================ Cache la Poudre v2
const float BCOL[3][3] = {{1.00f, 0.14f, 0.12f}, {0.20f, 0.96f, 0.32f}, {0.36f, 0.14f, 1.00f}};   // toc, chla, alk terrain glow
const char* PNAME[3] = {"toc", "chla", "alk"};
constexpr double CAND_WEIGHT = 0.056, CELL_WEIGHT = 0.62 * 0.056, CELL_SPARK = 0.52, MONTH = 7.0;
constexpr double SUN_START = 94.0, SUN_END = 123.0, DW_FIRST = 20.5, SAT_FIRST = 15.5, SAT_DUR = 4.0, SAT_TAU = 1.76, SAT_HALF = 0.18;
constexpr double ERA2_START = 46.0, ERA3_START = 72.0, ERA3_STEP = 0.38, HOLD_END = 128.0, FADE = 2.0;
const double SAT_CENTERS[3] = {0.22, 0.50, 0.78};

double seasonal_mod(double t, int param, double z_km) {
  double s = pymod1((t - 15.0 - 0.035 * z_km) / 28.0);
  auto bump = [&](double center, double width) { double d = std::fabs(pymod1(s - center + 0.5) - 0.5); double u = std::max(0.0, 1.0 - d / width); return u * u * (3.0 - 2.0 * u); };
  double spring = bump(0.22, 0.24), late = bump(0.74, 0.26);
  double s_alk = pymod1(s - 0.22), alk_ramp = s_alk < 0.72 ? smooth(s_alk / 0.72) : smooth((1.0 - s_alk) / 0.28);
  double val = param == 0 ? 0.52 + 0.96 * spring - 0.14 * late : (param == 1 ? 0.44 + 1.04 * late - 0.16 * spring : 0.42 + 1.06 * alk_ramp);
  return clampd(val + 0.07 * std::sin(1.15 * t + 0.21 * z_km), 0.40, 1.52);
}
std::set<int> active_steps(int bar) {
  if (bar < 3) return {0, 8};
  if (bar < 6) return {0, 4, 8, 12};
  if (bar < 10) return {0, 2, 4, 8, 10, 12};
  if (bar < 16) return {0, 2, 4, 6, 8, 10, 12, 14};
  return {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14};
}
int param_for_step(int bar, int s16) {            // 0 toc, 1 chla, 2 alk
  if (bar < 3) return s16 == 0 ? 0 : ((bar % 2 == 0) ? 1 : 2);
  if (bar < 6) return (s16 == 0 || s16 == 8) ? 0 : (s16 == 4 ? 1 : 2);
  if (bar < 16) return (s16 == 0 || s16 == 8) ? 0 : ((s16 == 4 || s16 == 12) ? 2 : 1);
  if (s16 % 4 == 0) return 0;
  return (s16 % 2 == 1) ? 1 : 2;
}

class PoudreStory : public Story {
 public:
  PoudreStory(Compositor& comp, const StoryData& d) : c_(comp), W_(comp.width()), H_(comp.height()), d_(d) {
    for (const Site& s : d_.sites) by_[s.name] = &s;                         // point into OUR copy (d_), never into the argument
    for (const Cell& c : d_.cells) cells_[c.id] = &c;
    for (const SunSample& q : d.sun_path) { sun_mins_.push_back(q.minute); sun_els_.push_back(q.el); }
    const size_t ns = d.sun_path.size();
    for (size_t k = 0; k < ns; ++k) { sun_.push_back(has("sun_" + std::to_string(k)) ? "sun_" + std::to_string(k) : ""); sky_.push_back(has("sky_" + std::to_string(k)) ? "sky_" + std::to_string(k) : ""); }
    std::vector<const Site*> sorted; for (const Site& s : d.sites) sorted.push_back(&s);
    std::stable_sort(sorted.begin(), sorted.end(), [](const Site* a, const Site* b) { return -a->z < -b->z; });
    for (const Site* s : sorted) { all_.push_back(s->name); if (!s->lighthouse) rest_.push_back(s->name); }
    for (const Site& s : d.sites) {
      double neigh = 0;
      for (const Site& q : d.sites) neigh += std::exp(-((s.x - q.x) * (s.x - q.x) + (s.z - q.z) * (s.z - q.z)) / (2.0 * 7.5 * 7.5));
      scale_[s.name] = (1.0 / std::pow(neigh, 0.72)) * 1.42;
    }
    for (int m = 0; m < 40; ++m) { double v = DW_FIRST + MONTH * m; if (v < SUN_START - 6.0) fire_times_.push_back(v); }
    for (int k = 0; k < 40; ++k) { double v = SAT_FIRST + MONTH * k; if (v < SUN_START - 8.0) sat_times_.push_back(v); }
    lh_ignite_["pman"] = ERA2_START + 1.0; lh_ignite_["pbr"] = ERA2_START + 1.0 + 0.7 * SLOW * 1.4; lh_ignite_["archery"] = ERA2_START + 1.0 + 2 * 0.7 * SLOW * 1.4;
    cell_schedule();
    int uid = 0;                                                                // site fireworks: same numbering as the original film
    for (double t0 : fire_times_) for (size_t si = 0; si < all_.size(); ++si) for (int k = 0; k < 3; ++k)
      fw_.push_back({all_[si], k, t0 + 0.015 * si + 0.125 * k, uid++, false});
    for (const Fw& f : cell_events_) fw_.push_back(f);
  }
  double total() const override { return HOLD_END + FADE; }

  FrameSpec frame_spec(double t) const override {
    std::vector<const Fw*> fw;
    for (const Fw& f : fw_) {
      double dt = t - f.t0;
      if (!(-0.75 < dt && dt < 6.6)) continue;
      std::string layer = f.cell ? "fwk_cell_" + f.site : std::string("fwk_") + PNAME[f.param] + "_" + f.site;
      if (!has(layer)) continue;
      fw.push_back(&f);
    }
    std::map<std::pair<std::string, int>, double> w_fwk;
    for (const Fw* f : fw) {
      double dt = t - f->t0;
      if (dt < 0) continue;
      double z = f->cell ? cells_.at(f->site)->z : by_.at(f->site)->z, sm = seasonal_mod(f->t0, f->param, z), w;
      if (f->cell) w = (1 - std::exp(-dt / 0.025)) * std::exp(-dt / 0.22) * 1.15 * sm;
      else w = (1 - std::exp(-dt / (0.035 * SLOW))) * std::exp(-dt / (0.24 * SLOW * PEAK)) * 0.72 * sm;
      w_fwk[{(f->cell ? "cell:" : "") + f->site, f->param}] += w;
    }
    std::map<std::string, double> ign(lh_ignite_.begin(), lh_ignite_.end());
    for (size_t k = 0; k < rest_.size(); ++k) ign[rest_[k]] = ERA3_START + 0.5 + ERA3_STEP * k;
    std::map<std::string, double> on, w_bea;
    for (const auto& iv : ign) {
      if (!by_.count(iv.first)) continue;
      double dt = t - iv.second;
      if (dt < 0 || !has("bea_" + iv.first)) continue;
      double ramp = 1 - std::exp(-dt / (0.28 * SLOW)), flare = 1.0 + 1.8 * std::exp(-dt / (0.55 * SLOW)), boost = 1.0 + 0.45 * smooth((t - ERA3_START) / 4.0);
      on[iv.first] = ramp;
      w_bea[iv.first] = ramp * flare * boost * (1.0 + 0.05 * std::sin(2 * M_PI * 0.4 * t + by_.at(iv.first)->x));
    }
    const double p = smooth((t - SUN_START) / (SUN_END - SUN_START));
    const double fade = t <= HOLD_END ? 1.0 : std::max(0.0, 1 - (t - HOLD_END) / FADE);
    const double dusk = t > SUN_END ? smooth((t - SUN_END) / std::max(HOLD_END - SUN_END, 1.0)) : 0.0;
    FrameSpec spec; spec.t = t; spec.fade = fade; spec.seed = (uint32_t)((long long)(t * 1000) % 2147483648LL);
    const Geometry& geo = c_.geometry();
    double amb_lvl = 0.88 * smooth(p * 1.8) * (1.0 - 0.42 * smooth(std::max(0.0, (p - 0.75) / 0.25)) - 0.25 * dusk);
    spec.terms.push_back({"amb", {(float)amb_lvl, (float)amb_lvl, (float)amb_lvl}, 0});
    if (p > 0 && sun_mins_.size() >= 2) {
      double m = sun_mins_.back() * p;
      int k = (int)(std::upper_bound(sun_mins_.begin(), sun_mins_.end(), m) - sun_mins_.begin()) - 1;   // searchsorted(side='right') - 1
      k = std::min(std::max(k, 0), (int)sun_mins_.size() - 2);
      double f = clampd((m - sun_mins_[k]) / (sun_mins_[k + 1] - sun_mins_[k]), 0, 1);
      double el_now = sun_els_[k] * (1 - f) + sun_els_[k + 1] * f, hi_sun = std::min(std::max(el_now - 2.0, 0.0) / 28.0, 1.0);
      double sunset_w = smooth(std::max(0.0, (p - 0.68) / 0.32));
      double low[3] = {1.0, 0.60 * (1 - sunset_w) + 0.46 * sunset_w, 0.36 * (1 - sunset_w) + 0.24 * sunset_w}, hi[3] = {1.0, 0.96, 0.88}, warm[3];
      double hp = std::pow(hi_sun, 0.65);
      for (int i = 0; i < 3; ++i) warm[i] = low[i] * (1.0 - hp) + hi[i] * hp;
      double sun_gain = smooth(p * 6.0) * (0.66 - 0.14 * sunset_w) * (1.0 - 0.55 * dusk);
      if (!sun_[k].empty()) spec.terms.push_back({sun_[k], {(float)(warm[0] * sun_gain * (1 - f)), (float)(warm[1] * sun_gain * (1 - f)), (float)(warm[2] * sun_gain * (1 - f))}, 0});
      if (!sun_[k + 1].empty()) spec.terms.push_back({sun_[k + 1], {(float)(warm[0] * sun_gain * f), (float)(warm[1] * sun_gain * f), (float)(warm[2] * sun_gain * f)}, 0});
      double rg = smooth(p * 2.5) * (1.0 - 0.35 * dusk);
      spec.terms.push_back({"riv", {(float)(0.04 * rg), (float)(0.30 * rg), (float)(0.65 * rg)}, 1});
      if (!sky_[k].empty() && !sky_[k + 1].empty()) {
        SkyDay& sk = spec.sky; sk.active = true; sk.layer_a = sky_[k]; sk.layer_b = sky_[k + 1]; sk.f = f; sk.sky_exp = 0.44 * smooth(p * 2.6) * (1.0 - 0.38 * dusk);
        const double z0[3] = {0.36, 0.60, 1.15}, z1[3] = {0.24, 0.36, 0.78}, h0[3] = {1.05, 0.92, 0.82}, h1[3] = {1.22, 0.74, 0.48};
        for (int i = 0; i < 3; ++i) { sk.zenith[i] = z0[i] * (1.0 - 0.35 * sunset_w) + z1[i] * (0.35 * sunset_w); sk.horizon_tint[i] = h0[i] * (1.0 - sunset_w) + h1[i] * sunset_w; }
        sk.dawn_band = smooth(p * 6.0) * std::max(0.0, 1.0 - p * 4.2); sk.set_band = smooth(std::max(0.0, (p - 0.68) / 0.32)) * (1.0 - 0.30 * dusk);
      }
      spec.aerial = (0.05 + 0.03 * smooth(std::max(0.0, (p - 0.70) / 0.30))) * smooth(p * 2.2);
      if (p < 0.42) {
        double noon_w = smooth(std::max(0.0, (p - 0.24) / 0.18));
        if (noon_w > 0) { spec.orb.mode = 1; spec.orb.x = geo.ox0; spec.orb.y = -0.08 * H_; spec.orb.amount = 0.28 * noon_w; }
      } else {
        double w_desc = clampd((p - 0.42) / 0.58, 0.0, 1.0), a = std::pow(w_desc, 1.2);
        Orb& o = spec.orb; o.mode = 2; o.x = geo.ox0 + (geo.ox1 - geo.ox0) * std::pow(w_desc, 0.85);
        o.y = geo.orb_y_top + (geo.orb_y_set - geo.orb_y_top) * std::pow(w_desc, 0.92) + 0.025 * H_ * dusk;
        o.radius = geo.orb_r * (1.0 + 0.12 * w_desc); o.visibility = smooth(std::min(w_desc * 4.5, 1.0)) * (1.0 - 0.75 * dusk); o.dusk = dusk;
        o.color[0] = 1.0; o.color[1] = 0.94 * (1.0 - a) + 0.42 * a; o.color[2] = 0.78 * (1.0 - a) + 0.14 * a;
      }
    }
    double star_fade = std::max(0.0, 1.0 - smooth(p * 4.0)) + 0.45 * dusk;
    spec.star_amount = star_fade > 0.01 ? 0.75 * star_fade : 0.0;
    for (const auto& kv : w_fwk) {
      if (kv.second <= 1e-3) continue;
      bool cell = kv.first.first.rfind("cell:", 0) == 0;
      std::string site = cell ? kv.first.first.substr(5) : kv.first.first;
      std::string layer = cell ? "fwk_cell_" + site : std::string("fwk_") + PNAME[kv.first.second] + "_" + site;
      double wt = cell ? CELL_WEIGHT : CAND_WEIGHT * 0.65 * scale_.at(site);
      spec.terms.push_back({layer, {(float)(BCOL[kv.first.second][0] * kv.second * wt), (float)(BCOL[kv.first.second][1] * kv.second * wt), (float)(BCOL[kv.first.second][2] * kv.second * wt)}, 0});
    }
    for (const auto& kv : w_bea) { float wv = (float)(kv.second * 0.175 * scale_.at(kv.first)); spec.beacon_terms.push_back({"bea_" + kv.first, {wv, wv, wv}, 0}); }
    // satellite scans
    for (size_t kk = 0; kk < sat_times_.size(); ++kk) {
      double ts = sat_times_[kk];
      if (ts - 0.2 < t && t < ts + SAT_DUR + 7 * SAT_TAU) spec.sat.events.push_back({ts, SAT_CENTERS[kk % 3]});
    }
    if (!spec.sat.events.empty()) {
      spec.sat.active = true; spec.sat.t = t; spec.sat.half_width = SAT_HALF; spec.sat.duration = SAT_DUR; spec.sat.tau = SAT_TAU;
      spec.sat.second_layer = sun_.size() > 1 ? sun_[1] : "";
    }
    // beacon sprites
    const double scale = W_ / 640.0;
    for (const auto& kv : on) {
      if (kv.second <= 1e-3) continue;
      const Site& s = *by_.at(kv.first);
      if (s.px < -12 * scale || s.px > W_ + 12 * scale || s.py < -12 * scale || s.py > H_ + 12 * scale) continue;
      double w_b = w_bea.count(kv.first) ? w_bea.at(kv.first) : kv.second;
      double flare = clampd(w_b / std::max(kv.second, 1e-3), 0.85, 1.45);
      double pulse = 0.70 * kv.second * (0.95 + 0.05 * std::sin(2 * M_PI * 0.4 * t + s.x)) * (0.82 + 0.18 * flare);
      spec.beacons.push_back({s.px, s.py, std::max(s.beam, 8.0 * scale), pulse});
    }
    // sparks
    for (const Fw* f : fw) {
      Burst b; b.uid = f->uid; b.plane = f->param; b.t0 = f->t0;
      if (f->cell) {
        const Cell& cc = *cells_.at(f->site);
        double sm = seasonal_mod(f->t0, f->param, cc.z);
        b.origin[0] = cc.x; b.origin[1] = cc.y; b.origin[2] = cc.z; b.launch_y = cc.y0; b.n_sparks = std::max(24, (int)std::lround(68 * (0.55 + 0.45 * sm)));
        b.gain = CELL_SPARK * 1.15 * (0.50 + 0.56 * sm); b.v_max = 4.5 * (0.58 + 0.42 * sm); b.rise = 0.28; b.life0 = 0.72; b.life1 = 1.30; b.trail_dt = 0.36 * SLOW;
      } else {
        const Site& s = *by_.at(f->site);
        const SiteBurst* sb = nullptr; for (const SiteBurst& x : s.bursts) if (x.color == PNAME[f->param]) { sb = &x; break; }
        if (!sb) continue;
        double dens = clampd(std::pow(scale_.at(f->site), 0.28), 0.74, 1.10), sm = seasonal_mod(f->t0, f->param, sb->z);
        b.origin[0] = sb->x; b.origin[1] = sb->y; b.origin[2] = sb->z; b.launch_y = s.y; b.n_sparks = std::max(48, (int)std::lround(128 * (0.55 + 0.45 * sm)));
        b.gain = 0.92 * dens * (0.48 + 0.58 * sm); b.v_max = 5.6 * dens * (0.58 + 0.42 * sm); b.rise = 0.45 * SLOW; b.life0 = 1.05; b.life1 = 1.95; b.trail_dt = 0.44 * SLOW;
      }
      spec.bursts.push_back(b);
    }
    (void)geo; return spec;
  }

 private:
  struct Fw { std::string site; int param; double t0; int uid; bool cell; };
  bool has(const std::string& n) const { return c_.has_layer(n); }
  void cell_schedule() {                       // real WQP cells: slow and sparse from 3.5 s, building into the 8 Hz groove
    std::vector<std::string> ids; for (const auto& kv : cells_) ids.push_back(kv.first);
    std::sort(ids.begin(), ids.end());
    std::vector<double> p[3]; double sum[3] = {0, 0, 0};
    for (const std::string& id : ids) {
      const Cell& c = *cells_.at(id);
      double v[3] = {std::sqrt(std::max(c.n_toc, 0)) + 0.15, std::sqrt(std::max(c.n_chla, 0)) + 0.15, std::sqrt(std::max(c.n_alk, 0) * 0.25) + 0.15};
      for (int k = 0; k < 3; ++k) { p[k].push_back(v[k]); sum[k] += v[k]; }
    }
    for (int k = 0; k < 3; ++k) for (auto& x : p[k]) x /= sum[k];
    NumpyRng rng(2026);
    int uid = 10000; std::vector<std::string> last;
    const int n_steps = (int)((SUN_START - 6.0 - 3.5) / 0.125);
    for (int i = 0; i < n_steps; ++i) {
      int bar = i / 16, s16 = i % 16;
      if (!active_steps(bar).count(s16)) continue;
      int col = param_for_step(bar, s16);
      std::string cid;
      for (int tries = 0; tries < 6; ++tries) {
        cid = ids[rng.choice(p[col])];
        bool recent = false; for (size_t j = last.size() > 4 ? last.size() - 4 : 0; j < last.size(); ++j) if (last[j] == cid) recent = true;
        if (!recent) break;
      }
      last.push_back(cid);
      cell_events_.push_back({cid, col, std::round((3.5 + i * 0.125) * 1e4) / 1e4, uid++, true});
    }
  }
  Compositor& c_; int W_, H_; StoryData d_;
  std::map<std::string, const Site*> by_; std::map<std::string, const Cell*> cells_;
  std::vector<double> sun_mins_, sun_els_, fire_times_, sat_times_;
  std::vector<std::string> sun_, sky_, all_, rest_;
  std::map<std::string, double> scale_, lh_ignite_;
  std::vector<Fw> cell_events_, fw_;
};

}  // namespace

std::unique_ptr<Story> make_story(const std::string& name, Compositor& comp, const StoryData& data) {
  if (name == "network") return std::unique_ptr<Story>(new NetworkStory(comp, data));
  if (name == "poudre") return std::unique_ptr<Story>(new PoudreStory(comp, data));
  throw std::runtime_error("unknown story '" + name + "' (available: network, poudre)");
}

}  // namespace sr
