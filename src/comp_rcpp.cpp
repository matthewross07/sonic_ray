// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 Matthew Ross. Written with AI assistance (Claude Sonnet 5.5); see AI_ATTRIBUTION.md.
// R interface to the C++ GPU compositor: stills and full films (worker threads render frames; the main thread streams them to ffmpeg).
#include <Rcpp.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <map>
#include <mutex>
#include <thread>
#include "comp_engine.h"
#include "comp_stories.h"

using namespace Rcpp;

namespace {

double num(const List& l, const char* k, double def = 0.0) { return l.containsElementNamed(k) && !Rf_isNull(l[k]) ? as<double>(l[k]) : def; }
std::string str(const List& l, const char* k, const std::string& def = "") { return l.containsElementNamed(k) && !Rf_isNull(l[k]) ? as<std::string>(l[k]) : def; }

sr::SceneInfo scene_from(const List& s) {
  sr::SceneInfo si; si.W = (int)num(s, "width"); si.H = (int)num(s, "height");
  List cam = s["cam"]; NumericVector lf = cam["lookfrom"], la = cam["lookat"];
  for (int i = 0; i < 3; ++i) { si.lookfrom[i] = lf[i]; si.lookat[i] = la[i]; }
  si.fov = num(cam, "fov", 30);
  return si;
}
sr::Look look_from(const List& l) {
  sr::Look k; k.sharpen = num(l, "sharpen", 0.0); k.dither = num(l, "dither", 0.0) != 0.0;
  return k;
}
sr::StoryData data_from(const List& d) {
  sr::StoryData sd;
  if (d.containsElementNamed("sites")) for (SEXP so : (List)d["sites"]) {
    List s(so); sr::Site x; x.name = str(s, "site"); x.x = num(s, "x"); x.y = num(s, "y"); x.z = num(s, "z"); x.px = num(s, "px"); x.py = num(s, "py"); x.beam = num(s, "beam");
    x.lighthouse = num(s, "lighthouse", 0.0) != 0.0;
    if (s.containsElementNamed("bursts")) for (SEXP bo : (List)s["bursts"]) { List b(bo); x.bursts.push_back({str(b, "color"), num(b, "x"), num(b, "y"), num(b, "z")}); }
    sd.sites.push_back(x);
  }
  if (d.containsElementNamed("cells")) for (SEXP co : (List)d["cells"]) {
    List c(co); sr::Cell x; x.id = str(c, "id"); x.x = num(c, "x"); x.y = num(c, "y"); x.z = num(c, "z"); x.y0 = num(c, "y0");
    x.n_toc = (int)num(c, "n_toc"); x.n_chla = (int)num(c, "n_chla"); x.n_alk = (int)num(c, "n_alk"); sd.cells.push_back(x);
  }
  if (d.containsElementNamed("sun_path")) for (SEXP qo : (List)d["sun_path"]) { List q(qo); sd.sun_path.push_back({num(q, "min"), num(q, "el")}); }
  sd.duration_s = num(d, "duration_s", 16.0);
  if (d.containsElementNamed("parameters")) {
    List ps = d["parameters"]; CharacterVector nm = ps.names();
    for (int i = 0; i < ps.size(); ++i) {
      List p = ps[i]; sr::Param x; x.plane = (int)num(p, "plane"); NumericVector g = p["glow"]; for (int j = 0; j < 3; ++j) x.glow[j] = (float)g[j];
      sd.parameters[as<std::string>(nm[i])] = x;
    }
  }
  if (d.containsElementNamed("sensor_site")) {
    CharacterVector s = d["sensor_site"]; NumericVector o = d["sensor_online"];
    for (int i = 0; i < s.size(); ++i) sd.sensors.push_back({as<std::string>(s[i]), o[i]});
  }
  if (d.containsElementNamed("event_site")) {
    CharacterVector s = d["event_site"], p = d["event_parameter"]; NumericVector t = d["event_time"], v = d["event_value"];
    for (int i = 0; i < s.size(); ++i) sd.events.push_back({as<std::string>(s[i]), as<std::string>(p[i]), t[i], std::min(std::max((double)v[i], 0.0), 1.0), i});
  }
  if (d.containsElementNamed("spark_colors")) {
    NumericMatrix m = d["spark_colors"]; sd.has_spark_colors = true;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) sd.spark_colors[i][j] = (float)m(i, j);
  }
  return sd;
}

std::string sq(const std::string& s) {                       // single-quote for /bin/sh
  std::string o = "'"; for (char c : s) { if (c == '\'') o += "'\\''"; else o += c; } return o + "'";
}

}  // namespace

// [[Rcpp::export]]
List cpp_comp_stills(std::string layers_dir, std::string metal_src, List scene, List look, std::string story, List data, NumericVector times) {
  try {
    sr::SceneInfo si = scene_from(scene);
    sr::Compositor comp(layers_dir, metal_src, si, look_from(look), 1);
    auto st = sr::make_story(story, comp, data_from(data));
    List out(times.size());
    for (int i = 0; i < times.size(); ++i) {
      RawVector f((size_t)si.W * si.H * 3);
      comp.render(0, st->frame_spec(times[i]), (uint8_t*)RAW(f));
      out[i] = f;
    }
    return out;
  } catch (...) { throw; }                        // Rcpp turns std::exception into an R error
}

// [[Rcpp::export]]
double cpp_comp_story_length(std::string layers_dir, std::string metal_src, List scene, List look, std::string story, List data) {
  try {
    sr::Compositor comp(layers_dir, metal_src, scene_from(scene), look_from(look), 1);
    return sr::make_story(story, comp, data_from(data))->total();
  } catch (...) { throw; }
}

// Render a film. enc: ffmpeg (path), encoder ("x264" | "videotoolbox"), preset, crf, codec ("h264" | "hevc"), quality (videotoolbox -q:v), audio, audio_offset
// [[Rcpp::export]]
List cpp_comp_film(std::string layers_dir, std::string metal_src, List scene, List look, std::string story, List data, std::string out_path,
                   int fps, double t0, double t1, List enc, int threads, bool progress) {
  using clock = std::chrono::steady_clock;
  FILE* pipe = nullptr;
  std::vector<std::thread> workers;
  std::atomic<bool> stop_flag(false);
  std::mutex mu; std::condition_variable cv;
  std::map<int, std::vector<uint8_t>> done;
  std::string error; int written = 0;
  auto join_all = [&]() { stop_flag = true; cv.notify_all(); for (auto& w : workers) if (w.joinable()) w.join(); workers.clear(); };
  try {
    sr::SceneInfo si = scene_from(scene);
    threads = std::max(1, threads);
    sr::Compositor comp(layers_dir, metal_src, si, look_from(look), threads);
    auto st = sr::make_story(story, comp, data_from(data));
    const double total = st->total();
    const int f0 = (int)std::lround(t0 * fps), f1 = (int)std::lround((t1 >= 0 ? t1 : total) * fps), n = f1 - f0;
    if (n <= 0) throw std::runtime_error("nothing to render (t1 <= t0)");

    const std::string ffmpeg = str(enc, "ffmpeg", "ffmpeg"), encoder = str(enc, "encoder", "x264"), audio = str(enc, "audio");
    const std::string silent = audio.empty() ? out_path : out_path + ".silent.mp4";
    std::string vargs;
    if (encoder == "videotoolbox") {
      std::string codec = str(enc, "codec", "hevc");
      vargs = "-c:v " + codec + "_videotoolbox -q:v " + std::to_string((int)num(enc, "quality", 85)) + (codec == "hevc" ? " -tag:v hvc1" : "") + " -pix_fmt yuv420p";
    } else {
      vargs = "-c:v libx264 -preset " + str(enc, "preset", "slow") + " -crf " + std::to_string((int)num(enc, "crf", 13)) + " -pix_fmt yuv420p";
    }
    std::string cmd = sq(ffmpeg) + " -y -loglevel error -f rawvideo -pix_fmt rgb24 -s " + std::to_string(si.W) + "x" + std::to_string(si.H) + " -r " + std::to_string(fps) +
                      " -i - " + vargs + " -movflags +faststart " + sq(silent);
    pipe = popen(cmd.c_str(), "w");
    if (!pipe) throw std::runtime_error("cannot start ffmpeg (" + ffmpeg + "); install it with: brew install ffmpeg");

    std::atomic<int> next(0);
    const int window = std::max(8, threads * 4);
    const size_t fbytes = (size_t)si.W * si.H * 3;
    for (int w = 0; w < threads; ++w) workers.emplace_back([&, w]() {
      try {
        std::vector<uint8_t> buf(fbytes);
        while (!stop_flag) {
          int i = next.fetch_add(1);
          if (i >= n) break;
          { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return stop_flag || i - written < window; }); if (stop_flag) break; }
          comp.render(w, st->frame_spec((f0 + i) / (double)fps), buf.data());
          { std::lock_guard<std::mutex> lk(mu); done[i] = buf; }
          cv.notify_all();
        }
      } catch (std::exception& e) { std::lock_guard<std::mutex> lk(mu); if (error.empty()) error = e.what(); stop_flag = true; cv.notify_all(); }
    });

    const auto start = clock::now();
    for (int i = 0; i < n; ++i) {
      std::vector<uint8_t> frame;
      {
        std::unique_lock<std::mutex> lk(mu);
        while (!done.count(i)) {
          if (!error.empty()) break;
          cv.wait_for(lk, std::chrono::milliseconds(100));
          lk.unlock(); try { checkUserInterrupt(); } catch (...) { join_all(); if (pipe) pclose(pipe); pipe = nullptr; throw; } lk.lock();
        }
        if (!error.empty()) break;
        frame = std::move(done[i]); done.erase(i); written = i + 1;
      }
      cv.notify_all();
      if (fwrite(frame.data(), 1, frame.size(), pipe) != frame.size()) { error = "ffmpeg closed the pipe (check the encoder settings)"; break; }
      if (progress && i % 120 == 0) Rcout << "frame " << i << "/" << n << "  (" << std::chrono::duration_cast<std::chrono::seconds>(clock::now() - start).count() << " s)\n";
    }
    join_all();
    int rc = pclose(pipe); pipe = nullptr;
    if (!error.empty()) throw std::runtime_error(error);
    if (rc != 0) throw std::runtime_error("ffmpeg failed (exit code " + std::to_string(rc) + ")");
    if (!audio.empty()) {
      std::string c2 = sq(ffmpeg) + " -y -loglevel error -i " + sq(silent) + " -ss " + std::to_string(num(enc, "audio_offset", 0.0)) + " -i " + sq(audio) +
                       " -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 320k -t " + std::to_string((double)n / fps) + " -movflags +faststart " + sq(out_path);
      if (system(c2.c_str()) != 0) throw std::runtime_error("audio mux failed");
      remove(silent.c_str());
    }
    double secs = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - start).count() / 1000.0;
    return List::create(_["seconds"] = secs, _["frames"] = n, _["out"] = out_path);
  } catch (...) {
    join_all(); if (pipe) pclose(pipe);
    throw;                                          // keep the original exception; Rcpp converts it to an R error
  }
}

// What does a story produce at time t? (debugging aid: counts and the beacon sprites)
// [[Rcpp::export]]
List cpp_comp_describe(std::string layers_dir, std::string metal_src, List scene, List look, std::string story, List data, double t) {
  sr::Compositor comp(layers_dir, metal_src, scene_from(scene), look_from(look), 1);
  sr::FrameSpec f = sr::make_story(story, comp, data_from(data))->frame_spec(t);
  NumericMatrix bc(f.beacons.size(), 4);
  for (size_t i = 0; i < f.beacons.size(); ++i) { bc(i, 0) = f.beacons[i].x; bc(i, 1) = f.beacons[i].y; bc(i, 2) = f.beacons[i].beam; bc(i, 3) = f.beacons[i].pulse; }
  CharacterVector tn; for (const auto& x : f.terms) tn.push_back(x.layer);
  return List::create(_["terms"] = tn, _["beacon_terms"] = (int)f.beacon_terms.size(), _["beacons"] = bc, _["bursts"] = (int)f.bursts.size(), _["sky"] = f.sky.active,
                      _["orb_mode"] = f.orb.mode, _["star_amount"] = f.star_amount, _["fade"] = f.fade, _["aerial"] = f.aerial, _["sat_events"] = (int)f.sat.events.size());
}
